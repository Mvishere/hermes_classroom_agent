"""Entry point for the Classroom polling agent.

Runs an initial sync and then schedules periodic polling.
"""

import logging
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler

import config
from auth.google_auth import get_classroom_service, get_drive_service
from classroom.announcements import process_announcements
from classroom.classroom_client import ClassroomClient
from classroom.coursework import process_coursework
from classroom.materials import process_materials
from rag.embeddings import EmbeddingModel
from rag.llm import LocalLLM
from rag.pipeline import RagPipeline
from rag.topic_extractor import TopicExtractor
from recommendation.recommender import Recommender
from scheduler.polling_scheduler import PollingScheduler
from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.recommendation_store import RecommendationStore
from storage.quiz_storage import QuizStorage
from storage.state_manager import StateManager
from storage.topics_store import TopicsStore
from storage.user_status import UserStatusManager
from student.inference_rules import InferenceRules
from student.knowledge_store import KnowledgeStore
from student.knowledge_tracker import KnowledgeTracker
from student.mastery_engine import MasteryEngine
from student.topic_graph import TopicGraph
from student.topic_graph_builder import TopicGraphBuilder
from student.topic_mapper import TopicMapper
from quiz_extractor.browser.playwright_client import PlaywrightBrowserClient


def setup_logging() -> None:
    """Configure console and file logging."""
    config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIRECTORY / "agent.log"
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
        ),
    ]
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
    )


def run_polling_cycle(
    client: ClassroomClient,
    json_store: JsonStore,
    state_manager: StateManager,
    file_storage: FileStorage,
    drive_service,
    browser_client,
    quiz_storage,
    rag_pipeline: RagPipeline,
    user_status: UserStatusManager,
    knowledge_tracker: KnowledgeTracker | None = None,
) -> None:
    """Run a single polling cycle and persist any new content."""
    try:
        logging.info("Polling cycle started.")
        courses = client.list_courses()
        json_store.upsert_courses(courses)

        new_assignments = process_coursework(
            courses,
            client,
            json_store,
            state_manager,
            file_storage,
            drive_service,
            browser_client,
            quiz_storage,
        )
        new_materials = process_materials(
            courses, client, json_store, state_manager, file_storage, drive_service
        )
        new_announcements = process_announcements(
            courses, client, json_store, state_manager, file_storage, drive_service
        )

        summaries_created = rag_pipeline.process_new_materials(json_store, new_materials)

        # Refresh submission state for existing assignments so completion changes are detected
        status_refreshed = 0
        if config.FETCH_SUBMISSIONS:
            try:
                all_assignments = json_store.get_all_items("assignments")
                for assignment in all_assignments:
                    course_id = assignment.get("course_id")
                    item_id = assignment.get("id")
                    if not course_id or not item_id:
                        continue
                    try:
                        subs = client.list_student_submissions(course_id, item_id, user_id="me")
                        state = subs[0].get("state", "") if subs else ""
                        if state and state != assignment.get("submission_state", ""):
                            assignment["submission_state"] = state
                            json_store.upsert_item("assignments", assignment)
                            status_refreshed += 1
                    except Exception:
                        logging.exception("Failed to refresh submissions for %s", item_id)
            except Exception:
                logging.exception("Failed to refresh assignment submission states")

        assignment_ids: list[str] = []
        material_ids: list[str] = []
        announcement_ids: list[str] = []
        if knowledge_tracker is not None:
            assignment_ids = _merge_ids(
                new_assignments,
                _collect_missing_topic_ids(json_store, knowledge_tracker.topics_store, "assignments"),
            )
            material_ids = _merge_ids(
                new_materials,
                _collect_missing_topic_ids(json_store, knowledge_tracker.topics_store, "materials"),
            )
            announcement_ids = _merge_ids(
                new_announcements,
                _collect_missing_topic_ids(
                    json_store, knowledge_tracker.topics_store, "announcements"
                ),
            )

            if assignment_ids or material_ids or announcement_ids:
                knowledge_tracker.process_new_items(
                    json_store, assignment_ids, material_ids, announcement_ids
                )
                knowledge_tracker.generate_recommendations(json_store, material_ids)

        assignments = json_store.get_all_items("assignments")
        materials = json_store.get_all_items("materials")
        knowledge_updates = 0
        if knowledge_tracker is not None:
            knowledge_updates = knowledge_tracker.update_from_assignments(assignments)
        # user_status.update_from_assignments will mark known/completed items
        status_updates = user_status.update_from_assignments(assignments, materials)
        # include any submission refresh count in logs
        status_updates += status_refreshed

        logging.info(
            "Polling cycle complete. New assignments: %s, materials: %s, announcements: %s, summaries: %s, knowledge updates: %s, status updates: %s",
            len(new_assignments),
            len(new_materials),
            len(new_announcements),
            summaries_created,
            knowledge_updates,
            status_updates,
        )
    except Exception:
        logging.exception("Polling cycle failed.")


def _merge_ids(*lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for seq in lists:
        for item_id in seq or []:
            if item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item_id)
    return merged


def _collect_missing_topic_ids(
    json_store: JsonStore, topics_store: TopicsStore, item_type: str
) -> list[str]:
    missing: list[str] = []
    items = json_store.get_all_items(item_type)
    for item in items:
        course_id = item.get("course_id", "")
        item_id = item.get("id", "")
        if not course_id or not item_id:
            continue
        if not topics_store.has_topics(item_type, course_id, item_id):
            missing.append(item_id)
    return missing


def main() -> int:
    setup_logging()
    try:
        config.ensure_directories()
        config.validate_config()
    except Exception as exc:
        logging.error("Configuration error: %s", exc)
        return 1

    json_store = JsonStore(config.DATA_DIRECTORY)
    state_manager = StateManager(config.STATE_PATH)
    user_status = UserStatusManager(config.USER_STATUS_PATH)
    quiz_storage = QuizStorage(config.QUIZ_STORAGE_PATH)
    file_storage = FileStorage(
        config.BASE_DIR, config.DOWNLOAD_DIRECTORY, config.DATA_DIRECTORY
    )
    rag_pipeline = RagPipeline()

    topic_graph = TopicGraph(config.DATA_DIRECTORY / "topic_graph.json")
    knowledge_store = KnowledgeStore(config.DATA_DIRECTORY / "knowledge_state.json")
    topics_store = TopicsStore(config.DATA_DIRECTORY / "topics")
    recommendation_store = RecommendationStore(
        config.DATA_DIRECTORY / "recommendations.json"
    )
    mastery_engine = MasteryEngine()
    topic_mapper = TopicMapper(topic_graph)
    inference_rules = InferenceRules(mastery_engine, topic_mapper)
    graph_builder = TopicGraphBuilder(topic_graph)

    embedding_model = None
    if config.EMBEDDING_MODEL_PATH:
        embedding_path = Path(config.EMBEDDING_MODEL_PATH)
        if embedding_path.exists():
            embedding_model = EmbeddingModel(
                str(embedding_path), device=config.RAG_DEVICE
            )
        else:
            logging.warning(
                "Embedding model path not found for topic extraction: %s",
                config.EMBEDDING_MODEL_PATH,
            )

    topic_llm = None
    if config.TOPIC_EXTRACT_LLM_ENABLED and config.LLM_MODEL_PATH:
        llm_path = Path(config.LLM_MODEL_PATH)
        if llm_path.exists():
            topic_llm = LocalLLM(
                str(llm_path),
                device=config.RAG_DEVICE,
                max_new_tokens=config.RAG_MAX_NEW_TOKENS,
                temperature=config.RAG_TEMPERATURE,
            )
        else:
            logging.warning(
                "LLM model path not found for topic extraction: %s",
                config.LLM_MODEL_PATH,
            )

    topic_extractor = TopicExtractor(
        config.BASE_DIR,
        topic_graph,
        llm=topic_llm,
        embedding_model=embedding_model,
        keyword_limit=config.TOPIC_EXTRACT_KEYWORD_LIMIT,
        max_chars=config.TOPIC_EXTRACT_MAX_CHARS,
    )
    recommender = Recommender(topic_graph, knowledge_store)
    knowledge_tracker = KnowledgeTracker(
        topics_store,
        knowledge_store,
        inference_rules,
        topic_extractor,
        recommender,
        recommendation_store,
        graph_builder,
        enabled=True,
    )

    classroom_service = get_classroom_service()
    drive_service = get_drive_service()
    client = ClassroomClient(classroom_service)

    def _run_with_browser_client() -> None:
        with PlaywrightBrowserClient(
            config.QUIZ_BROWSER_PROFILE_DIR,
            headless=config.QUIZ_BROWSER_HEADLESS,
            timeout_ms=config.QUIZ_BROWSER_TIMEOUT_MS,
        ) as browser_client:
            run_polling_cycle(
                client,
                json_store,
                state_manager,
                file_storage,
                drive_service,
                browser_client,
                quiz_storage,
                rag_pipeline,
                user_status,
                knowledge_tracker,
            )

    _run_with_browser_client()

    scheduler = PollingScheduler(
        interval_minutes=config.POLL_INTERVAL_MINUTES,
        job_func=_run_with_browser_client,
    )
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutdown requested.")
    finally:
        scheduler.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
