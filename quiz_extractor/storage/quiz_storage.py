"""Persistent storage for extracted quiz JSON."""

from __future__ import annotations

from datetime import datetime
import json
import threading
from pathlib import Path
from typing import Any


class QuizStorage:
    """Stores extracted quiz payloads in structured JSON."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.storage_path.exists():
            return {"courses": {}}
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return {"courses": {}}
        if not isinstance(data, dict):
            return {"courses": {}}
        data.setdefault("courses", {})
        return data

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def upsert_quiz(self, quiz_data: dict[str, Any]) -> None:
        """Persist a quiz payload keyed by course and quiz identifier."""
        course_id = str(quiz_data.get("course_id") or "unassigned")
        quiz_id = str(
            quiz_data.get("quiz_id")
            or quiz_data.get("form_id")
            or quiz_data.get("source_url")
            or ""
        )
        if not quiz_id:
            return

        with self._lock:
            courses = self._payload.setdefault("courses", {})
            course_entry = courses.setdefault(
                course_id,
                {
                    "course_id": course_id,
                    "course_name": quiz_data.get("course_name", ""),
                    "items": {},
                },
            )
            if quiz_data.get("course_name"):
                course_entry["course_name"] = quiz_data.get("course_name", "")

            course_entry["items"][quiz_id] = {
                "quiz_id": quiz_id,
                "assignment_id": quiz_data.get("assignment_id", ""),
                "form_id": quiz_data.get("form_id", ""),
                "source_url": quiz_data.get("source_url", ""),
                "title": quiz_data.get("title", ""),
                "questions": quiz_data.get("questions", []),
                "section_titles": quiz_data.get("section_titles", []),
                "quiz_text": quiz_data.get("quiz_text", ""),
                "fetch_status": quiz_data.get("fetch_status", ""),
                "source": quiz_data.get("source", ""),
                "page_url": quiz_data.get("page_url", ""),
                "extracted_at": quiz_data.get("extracted_at", ""),
            }
            courses[course_id] = course_entry
            self._save()

    def get_quiz(self, course_id: str, quiz_id: str) -> dict[str, Any]:
        if not course_id or not quiz_id:
            return {}
        course = self._payload.get("courses", {}).get(course_id, {})
        return course.get("items", {}).get(quiz_id, {})
