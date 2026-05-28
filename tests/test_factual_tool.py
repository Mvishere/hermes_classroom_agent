import json
import tempfile
from pathlib import Path

from chat.tools import factual_tool


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_announcement_titles():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        # create announcements data
        anns = {
            "courses": {
                "c1": {
                    "course_id": "c1",
                    "course_name": "test",
                    "items": [
                        {"id": "a1", "title": "Welcome to class", "description": ""}
                    ],
                }
            }
        }
        write_json(data_dir / "announcements" / "announcements.json", anns)

        resp = factual_tool.handle("Tell me the title of all annoucements", str(data_dir))
        assert resp is not None
        assert "Welcome to class" in resp


def test_quiz_title():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        assigns = {
            "courses": {
                "c1": {
                    "course_id": "c1",
                    "course_name": "test",
                    "items": [
                        {"id": "asg1", "title": "Web Development quiz", "updated_at": "2026-01-01T00:00:00Z"}
                    ],
                }
            }
        }
        write_json(data_dir / "assignments" / "assignments.json", assigns)

        resp = factual_tool.handle("What is the title for my recent quiz", str(data_dir))
        assert resp is not None
        assert "Web Development quiz" in resp


def test_mention_count_and_topic_graph_questions():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        materials = {
            "courses": {
                "c1": {
                    "course_id": "c1",
                    "course_name": "test",
                    "items": [
                        {"id": "m1", "title": "HTML and CSS notes", "description": "Responsive design with flexbox"},
                        {"id": "m2", "title": "JavaScript events", "description": "DOM and event listeners"},
                    ],
                }
            }
        }
        assignments = {
            "courses": {
                "c1": {
                    "course_id": "c1",
                    "course_name": "test",
                    "items": [
                        {"id": "a1", "title": "Quiz 1", "description": "forms and quizzes"},
                        {"id": "a2", "title": "Project", "description": "forms practice"},
                    ],
                }
            }
        }
        topic_graph = {
            "Responsive Design": {
                "prerequisites": ["CSS Basics"],
                "related_topics": [{"topic": "CSS Grid", "weight": 0.83}],
            }
        }
        write_json(data_dir / "materials" / "materials.json", materials)
        write_json(data_dir / "assignments" / "assignments.json", assignments)
        write_json(data_dir / "topic_graph.json", topic_graph)

        mention_resp = factual_tool.handle("How many materials mention HTML or CSS?", str(data_dir))
        topic_resp = factual_tool.handle("What topics are related to responsive design?", str(data_dir))
        prereq_resp = factual_tool.handle("What are the prerequisites for responsive design?", str(data_dir))

        assert mention_resp is not None
        assert "1 material" in mention_resp.lower()
        assert topic_resp is not None
        assert "CSS Grid" in topic_resp
        assert prereq_resp is not None
        assert "CSS Basics" in prereq_resp
