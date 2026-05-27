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
