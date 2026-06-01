import json
import tempfile
from pathlib import Path

from chat.router import Router
from router.intent_classifier import IntentClassifier
from router.orchestrator import QueryOrchestrator


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_intent_classifier_routes_queries() -> None:
    classifier = IntentClassifier()

    structured = classifier.classify("How many materials do I have?")
    latest = classifier.classify("What is the latest assignment?")
    topic = classifier.classify("What are the prerequisites for responsive design?")

    assert structured.intent == "structured_count"
    assert structured.document_type == "materials"
    assert latest.intent == "latest_item"
    assert topic.intent == "prerequisite"


def test_intent_classifier_prefers_material_metadata_lookup() -> None:
    classifier = IntentClassifier()

    classification = classifier.classify("What is the title of the material related to HTML and CSS notes?")

    assert classification.intent == "metadata_lookup"
    assert classification.document_type == "materials"


def test_intent_classifier_prefers_summary_over_latest_for_recent_material() -> None:
    classifier = IntentClassifier()

    classification = classifier.classify("Summarize the most recent material for me.")

    assert classification.intent == "summary"
    assert classification.document_type == "materials"


def test_router_only_short_circuits_explicit_knowledge_queries() -> None:
    class DummyAssistant:
        def __init__(self) -> None:
            self.calls = 0

        def student_profile_summary(self) -> str:
            self.calls += 1
            return "Known topics: HTML"

    assistant = DummyAssistant()
    router = Router(assistant, data_dir=".")

    route = router.route("What topics appear to be foundation topics for JavaScript events?")

    assert assistant.calls == 0
    assert route.engine != "knowledge_state"


def test_orchestrator_uses_specialized_engines() -> None:
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        write_json(
            data_dir / "courses" / "courses.json",
            {
                "courses": {
                    "c1": {"id": "c1", "name": "hermes classroom", "updated_at": "2026-01-01T00:00:00Z"}
                }
            },
        )
        write_json(
            data_dir / "announcements" / "announcements.json",
            {
                "courses": {
                    "c1": {
                        "course_id": "c1",
                        "course_name": "hermes classroom",
                        "items": [{"id": "a1", "title": "Welcome to class", "updated_at": "2026-01-02T00:00:00Z"}],
                    }
                }
            },
        )
        write_json(
            data_dir / "assignments" / "assignments.json",
            {
                "courses": {
                    "c1": {
                        "course_id": "c1",
                        "course_name": "hermes classroom",
                        "items": [{"id": "q1", "title": "Responsive Design Quiz", "updated_at": "2026-01-03T00:00:00Z"}],
                    }
                }
            },
        )
        write_json(
            data_dir / "materials" / "materials.json",
            {
                "courses": {
                    "c1": {
                        "course_id": "c1",
                        "course_name": "hermes classroom",
                        "items": [
                            {"id": "m1", "title": "Responsive Design Notes", "description": "CSS and media queries"},
                            {"id": "m2", "title": "Html css notes", "description": "HTML basics and CSS selectors"},
                        ],
                    }
                }
            },
        )
        write_json(
            data_dir / "rag" / "materials_index.json",
            {
                "items": [
                    {
                        "item_id": "m1",
                        "title": "Responsive Design Notes",
                        "text": "Responsive layout techniques and media queries.",
                    },
                    {
                        "item_id": "m2",
                        "title": "Html css notes",
                        "text": "HTML structure and CSS fundamentals.",
                    },
                ]
            },
        )
        write_json(
            data_dir / "topic_graph.json",
            {
                "Responsive Design": {
                    "prerequisites": ["CSS"],
                    "related_topics": [{"topic": "CSS Grid", "weight": 0.83}],
                }
            },
        )

        orchestrator = QueryOrchestrator(str(data_dir))

        course_answer = orchestrator.handle("Which courses am I enrolled in?")
        announcement_answer = orchestrator.handle("Show me the latest announcement")
        topic_answer = orchestrator.handle("What are the prerequisites for responsive design?")
        metadata_answer = orchestrator.handle("What is the title of the material related to html?")

        assert "hermes classroom" in course_answer.lower()
        assert "Welcome to class" in announcement_answer
        assert "CSS" in topic_answer
        assert "Html css notes" in metadata_answer
        assert "Responsive Design Notes" not in metadata_answer
