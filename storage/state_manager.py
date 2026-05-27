"""State manager for deduplicating processed Classroom items."""

from datetime import datetime
import json
import threading
from pathlib import Path


class StateManager:
    """Tracks which items have already been seen by this agent."""

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"processed": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"processed": {}}
        if not isinstance(data, dict):
            return {"processed": {}}
        data.setdefault("processed", {})
        return data

    def _save_state(self) -> None:
        payload = dict(self._state)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _key(self, item_type: str, item_id: str, course_id: str) -> str:
        return f"{item_type}:{course_id}:{item_id}"

    def is_seen(self, item_type: str, item_id: str, course_id: str) -> bool:
        """Return True if the item was already processed by this agent."""
        key = self._key(item_type, item_id, course_id)
        with self._lock:
            return key in self._state.get("processed", {})

    def mark_seen(self, item_type: str, item_id: str, course_id: str) -> bool:
        """Return True if the item is new, False if already processed."""
        key = self._key(item_type, item_id, course_id)
        with self._lock:
            if key in self._state.get("processed", {}):
                return False
            self._state.setdefault("processed", {})[key] = {
                "item_type": item_type,
                "item_id": item_id,
                "course_id": course_id,
                "processed_at": datetime.utcnow().isoformat() + "Z",
            }
            self._save_state()
            return True

    def mark_hermes_processed(self, item_type: str, item_id: str, course_id: str) -> None:
        """Allow Hermes to mark items as processed later."""
        key = self._key(item_type, item_id, course_id)
        with self._lock:
            if key not in self._state.get("processed", {}):
                self._state.setdefault("processed", {})[key] = {
                    "item_type": item_type,
                    "item_id": item_id,
                    "course_id": course_id,
                }
            self._state["processed"][key]["hermes_processed_at"] = (
                datetime.utcnow().isoformat() + "Z"
            )
            self._save_state()
