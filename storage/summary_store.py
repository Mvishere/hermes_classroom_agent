"""Stores RAG summaries grouped by course."""

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import List


class SummaryStore:
    """Persists summaries for materials in structured JSON."""

    def __init__(self, summary_path: Path):
        self.summary_path = Path(summary_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.summary_path.exists():
            return {"courses": {}}
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {"courses": {}}

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def has_summary(self, material: dict) -> bool:
        course_id = material.get("course_id")
        item_id = material.get("id")
        if not course_id or not item_id:
            return False
        course = self._payload.get("courses", {}).get(course_id, {})
        return item_id in course.get("items", {})

    def get_summary(self, course_id: str, item_id: str) -> str:
        if not course_id or not item_id:
            return ""
        course = self._payload.get("courses", {}).get(course_id, {})
        entry = course.get("items", {}).get(item_id, {})
        summary = entry.get("summary")
        if not isinstance(summary, str):
            return ""
        return summary

    def upsert_summary(
        self,
        material: dict,
        summary: str,
        contexts: List[dict],
        attachment_formats: List[str],
    ) -> None:
        course_id = material.get("course_id")
        item_id = material.get("id")
        if not course_id or not item_id:
            return
        with self._lock:
            courses = self._payload.setdefault("courses", {})
            course_entry = courses.setdefault(
                course_id,
                {
                    "course_id": course_id,
                    "course_name": material.get("course_name", ""),
                    "items": {},
                },
            )
            if material.get("course_name"):
                course_entry["course_name"] = material.get("course_name", "")

            course_entry["items"][item_id] = {
                "item_id": item_id,
                "title": material.get("title", ""),
                "summary": summary,
                "attachment_formats": attachment_formats,
                "contexts": [
                    {
                        "item_id": entry.get("item_id"),
                        "course_id": entry.get("course_id"),
                        "score": entry.get("score"),
                        "text": entry.get("text"),
                    }
                    for entry in contexts
                ],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            courses[course_id] = course_entry
            self._save()
