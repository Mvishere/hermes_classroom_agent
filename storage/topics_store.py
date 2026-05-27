"""Topic extraction storage utilities."""

from datetime import datetime
import json
import threading
from pathlib import Path
from typing import Optional


class TopicsStore:
    """Persists per-item topic extraction results."""

    def __init__(self, topics_dir: Path):
        self.topics_dir = Path(topics_dir)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _item_path(self, item_type: str, course_id: str, item_id: str) -> Path:
        safe_course = course_id or "unknown"
        safe_item = item_id or "unknown"
        return self.topics_dir / item_type / f"{safe_course}_{safe_item}.json"

    def has_topics(self, item_type: str, course_id: str, item_id: str) -> bool:
        path = self._item_path(item_type, course_id, item_id)
        return path.exists()

    def get_topics(self, item_type: str, course_id: str, item_id: str) -> Optional[dict]:
        path = self._item_path(item_type, course_id, item_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def upsert_topics(self, payload: dict) -> Path:
        item_type = payload.get("item_type", "unknown")
        course_id = payload.get("course_id", "unknown")
        item_id = payload.get("item_id", "unknown")
        path = self._item_path(item_type, course_id, item_id)
        payload = dict(payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        return path

    def list_payloads(self) -> list[dict]:
        payloads = []
        for path in self.topics_dir.rglob("*.json"):
            if path.name.startswith("_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                payloads.append(data)
        return payloads
