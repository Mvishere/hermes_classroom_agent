"""Persistent storage for student knowledge state."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Dict, Optional


class KnowledgeStore:
    """Stores knowledge state for topics in a JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"topics": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"topics": {}}
        if not isinstance(data, dict):
            return {"topics": {}}
        data.setdefault("topics", {})
        return data

    def save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with self._lock:
            self.path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )

    def get_topic(self, topic: str) -> Optional[dict]:
        return self._payload.get("topics", {}).get(topic)

    def set_topic(self, topic: str, entry: dict) -> None:
        self._payload.setdefault("topics", {})[topic] = entry

    def all_topics(self) -> Dict[str, dict]:
        return dict(self._payload.get("topics", {}))
