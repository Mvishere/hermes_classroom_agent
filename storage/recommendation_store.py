"""Stores recommendation messages for items."""

from datetime import datetime
import json
from pathlib import Path
import threading


class RecommendationStore:
    """Persists recommendations grouped by course."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"courses": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"courses": {}}
        if not isinstance(data, dict):
            return {"courses": {}}
        data.setdefault("courses", {})
        return data

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def upsert_recommendation(self, item: dict, recommendation: dict) -> None:
        course_id = item.get("course_id")
        item_id = item.get("id")
        if not course_id or not item_id:
            return

        with self._lock:
            courses = self._payload.setdefault("courses", {})
            course_entry = courses.setdefault(
                course_id,
                {
                    "course_id": course_id,
                    "course_name": item.get("course_name", ""),
                    "items": {},
                },
            )
            if item.get("course_name"):
                course_entry["course_name"] = item.get("course_name", "")

            course_entry["items"][item_id] = {
                "item_id": item_id,
                "item_type": recommendation.get("item_type", ""),
                "readiness": recommendation.get("readiness", ""),
                "message": recommendation.get("message", ""),
                "known_ratio": recommendation.get("known_ratio", 0.0),
                "missing_topics": recommendation.get("missing_topics", []),
                "weak_topics": recommendation.get("weak_topics", []),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            courses[course_id] = course_entry
            self._save()
