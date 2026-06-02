"""Entry point for the Classroom polling agent + RAG chat."""

import logging
import time
import os

import config

from auth.google_auth import get_classroom_service, get_drive_service

from classroom.announcements import process_announcements
from classroom.classroom_client import ClassroomClient
from classroom.coursework import process_coursework
from classroom.materials import process_materials

from rag.pipeline import RagPipeline
from rag.embeddings import EmbeddingModel
from rag.llm import LLM

from recommendation.recommender import Recommender

from scheduler.polling_scheduler import PollingScheduler

from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.recommendation_store import RecommendationStore
from storage.quiz_storage import QuizStorage
from storage.state_manager import StateManager
from storage.topics_store import TopicsStore
from storage.user_status import UserStatusManager

from student.knowledge_store import KnowledgeStore
from student.knowledge_tracker import KnowledgeTracker
from student.knowledge_updater import KnowledgeUpdater

from student.topic_graph import TopicGraph
from student.topic_graph_builder import TopicGraphBuilder

from topic_graph.extractor import TopicExtractor

from quiz_extractor.browser.playwright_client import PlaywrightBrowserClient

from vector_store.chroma_store import ChromaStore

from chat.cli import RagChatCLI   # ✅ ADD THIS


# =========================================================
# LOGGING
# =========================================================
def setup_logging():
    config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# =========================================================
# PIPELINE
# =========================================================
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
):
    try:
        logging.info("Polling cycle started.")

        courses = client.list_courses()
        json_store.upsert_courses(courses)

        new_assignments = process_coursework(
            courses, client, json_store, state_manager,
            file_storage, drive_service, browser_client, quiz_storage
        )

        new_materials = process_materials(
            courses, client, json_store, state_manager,
            file_storage, drive_service
        )

        new_announcements = process_announcements(
            courses, client, json_store, state_manager,
            file_storage, drive_service
        )

        # ---------------- RAG ----------------
        rag_pipeline.process_new_materials(json_store, new_materials)

        # ---------------- KNOWLEDGE ----------------
        if knowledge_tracker and (new_assignments or new_materials or new_announcements):
            knowledge_tracker.process_new_items(
                json_store,
                new_assignments,
                new_materials,
                new_announcements,
            )

            knowledge_tracker.generate_recommendations(
                json_store,
                new_materials,
            )

        # ---------------- STATUS ----------------
        assignments = json_store.get_all_items("assignments")
        materials = json_store.get_all_items("materials")

        updates = user_status.update_from_assignments(assignments, materials)

        logging.info(
            "Cycle complete: A=%s M=%s Ann=%s updates=%s",
            len(new_assignments),
            len(new_materials),
            len(new_announcements),
            updates,
        )

    except Exception:
        logging.exception("Polling cycle failed.")


# =========================================================
# CHAT MODE
# =========================================================
def run_chat(rag_pipeline):
    chat = RagChatCLI(rag_pipeline)
    chat.chat()


# =========================================================
# MAIN
# =========================================================
def main():
    setup_logging()

    config.ensure_directories()
    config.validate_config()

    # ---------------- STORAGE ----------------
    json_store = JsonStore(config.DATA_DIRECTORY)
    state_manager = StateManager(config.STATE_PATH)
    user_status = UserStatusManager(config.USER_STATUS_PATH)
    quiz_storage = QuizStorage(config.QUIZ_STORAGE_PATH)

    file_storage = FileStorage(
        config.BASE_DIR,
        config.DOWNLOAD_DIRECTORY,
        config.DATA_DIRECTORY,
    )

    # ---------------- RAG ----------------
    rag_pipeline = RagPipeline()

    # ---------------- KNOWLEDGE ----------------
    knowledge_store = KnowledgeStore(config.DATA_DIRECTORY / "knowledge_state.json")
    topics_store = TopicsStore(config.DATA_DIRECTORY / "topics")
    recommendation_store = RecommendationStore(config.DATA_DIRECTORY / "recommendations.json")

    topic_graph = TopicGraph(config.DATA_DIRECTORY / "topic_graph.json")

    embedding_model = EmbeddingModel(model_name="mxbai-embed-large")

    chroma_store = ChromaStore(persist_dir=config.CHROMA_PERSIST_DIR)

    # ---------------- LLM ----------------
    llm = LLM(
        str(config.LLM_MODEL_PATH),
        device=config.RAG_DEVICE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )

    topic_extractor = TopicExtractor(llm=llm)

    recommender = Recommender(topic_graph, knowledge_store)

    knowledge_updater = KnowledgeUpdater(llm=llm)

    knowledge_tracker = KnowledgeTracker(
        topics_store=topics_store,
        knowledge_store=knowledge_store,
        topic_extractor=topic_extractor,
        knowledge_updater=knowledge_updater,
        recommender=recommender,
        recommendation_store=recommendation_store,
        graph_generator=TopicGraphBuilder(
            topic_graph,
            embedding_model=embedding_model,
            min_edge_weight=config.TOPIC_GRAPH_MIN_EDGE_WEIGHT,
            max_related=config.TOPIC_GRAPH_MAX_RELATED,
            debug=config.TOPIC_GRAPH_DEBUG,
        ),
        enabled=True,
    )

    # ---------------- CLASSROOM ----------------
    classroom_service = get_classroom_service()
    drive_service = get_drive_service()
    client = ClassroomClient(classroom_service)

    def run_cycle():
        with PlaywrightBrowserClient(
            config.QUIZ_BROWSER_PROFILE_DIR,
            headless=config.QUIZ_BROWSER_HEADLESS,
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

    # ---------------- MODE SWITCH ----------------
    if os.getenv("CHAT_MODE", "0") == "1":
        logging.info("Starting CHAT MODE (RAG only)")
        run_chat(rag_pipeline)
        return 0

    # initial run
    run_cycle()

    scheduler = PollingScheduler(
        interval_minutes=config.POLL_INTERVAL_MINUTES,
        job_func=run_cycle,
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