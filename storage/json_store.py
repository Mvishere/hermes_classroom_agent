"""JSON storage for Classroom content grouped by course."""

from datetime import datetime
import json
import threading
from pathlib import Path
from typing import Iterable, Optional, List


class JsonStore:
    """Persists courses and items into structured JSON files."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self, path: Path, default: dict) -> dict:
        if not path.exists():
            return dict(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)
        if not isinstance(data, dict):
            return dict(default)
        return data

    def _write(self, path: Path, payload: dict) -> None:
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def upsert_courses(self, courses: Iterable[dict]) -> None:
        path = self.data_dir / "courses" / "courses.json"
        with self._lock:
            payload = self._load(path, {"courses": {}})
            payload.setdefault("courses", {})
            for course in courses:
                course_id = course.get("id")
                if not course_id:
                    continue
                payload["courses"][course_id] = {
                    "id": course_id,
                    "name": course.get("name", ""),
                    "section": course.get("section", ""),
                    "description_heading": course.get("descriptionHeading", ""),
                    "description": course.get("description", ""),
                    "updated_at": course.get("updateTime", ""),
                }
            self._write(path, payload)

    def upsert_item(self, item_type: str, record: dict, raw_payload: Optional[dict] = None) -> None:
        path = self.data_dir / item_type / f"{item_type}.json"
        course_id = record.get("course_id")
        if not course_id:
            return
        with self._lock:
            payload = self._load(path, {"courses": {}})
            payload.setdefault("courses", {})
            course_entry = payload["courses"].setdefault(
                course_id,
                {
                    "course_id": course_id,
                    "course_name": record.get("course_name", ""),
                    "items": [],
                },
            )
            if record.get("course_name"):
                course_entry["course_name"] = record.get("course_name", "")

            item = dict(record)
            if raw_payload is not None:
                item["raw_payload"] = raw_payload

            items = course_entry.get("items", [])
            existing_index = None
            for index, existing in enumerate(items):
                if existing.get("id") == record.get("id"):
                    existing_index = index
                    break
            if existing_index is None:
                items.append(item)
            else:
                items[existing_index] = item
            course_entry["items"] = items
            payload["courses"][course_id] = course_entry
            self._write(path, payload)

    def get_all_items(self, item_type: str) -> List[dict]:
        path = self.data_dir / item_type / f"{item_type}.json"
        with self._lock:
            payload = self._load(path, {"courses": {}})
            results = []
            for course_entry in payload.get("courses", {}).values():
                results.extend(course_entry.get("items", []))
            return results

    def get_items_by_ids(self, item_type: str, ids: List[str]) -> List[dict]:
        if not ids:
            return []
        target_ids = set(ids)
        path = self.data_dir / item_type / f"{item_type}.json"
        with self._lock:
            payload = self._load(path, {"courses": {}})
            results = []
            for course_entry in payload.get("courses", {}).values():
                for item in course_entry.get("items", []):
                    if item.get("id") in target_ids:
                        results.append(item)
            return results
