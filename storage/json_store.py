"""CSV storage for Classroom content."""

import csv
import json
import threading
from pathlib import Path
from typing import Iterable, Optional, List


class JsonStore:
    """
    Drop-in replacement for the old JsonStore.

    Stores:
        data/
        ├── courses.csv
        ├── announcements.csv
        ├── assignments.csv
        └── materials.csv
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _csv_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.csv"

    def _load_rows(self, path: Path) -> List[dict]:
        if not path.exists():
            return []

        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_rows(self, path: Path, rows: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            return

        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())

        fieldnames = sorted(fieldnames)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                normalized = {
                    key: (
                        json.dumps(value)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
                writer.writerow(normalized)

    def upsert_courses(self, courses: Iterable[dict]) -> None:
        path = self._csv_path("courses")

        with self._lock:
            rows = self._load_rows(path)

            existing = {
                row.get("id"): row
                for row in rows
                if row.get("id")
            }

            for course in courses:
                course_id = course.get("id")
                if not course_id:
                    continue

                existing[course_id] = {
                    "id": course_id,
                    "name": course.get("name", ""),
                    "section": course.get("section", ""),
                    "description_heading": course.get(
                        "descriptionHeading", ""
                    ),
                    "description": course.get("description", ""),
                    "updated_at": course.get("updateTime", ""),
                }

            self._write_rows(
                path,
                list(existing.values()),
            )

    def upsert_item(
        self,
        item_type: str,
        record: dict,
        raw_payload: Optional[dict] = None,
    ) -> None:
        path = self._csv_path(item_type)

        with self._lock:
            rows = self._load_rows(path)

            item = dict(record)

            if raw_payload is not None:
                item["raw_payload"] = raw_payload

            item_id = item.get("id")
            if not item_id:
                return

            found = False

            for index, row in enumerate(rows):
                if row.get("id") == item_id:
                    rows[index] = item
                    found = True
                    break

            if not found:
                rows.append(item)

            self._write_rows(path, rows)

    def get_all_items(self, item_type: str) -> List[dict]:
        path = self._csv_path(item_type)

        with self._lock:
            rows = self._load_rows(path)

            for row in rows:
                for key, value in list(row.items()):
                    if not isinstance(value, str):
                        continue

                    value = value.strip()

                    if (
                        value.startswith("[")
                        and value.endswith("]")
                    ) or (
                        value.startswith("{")
                        and value.endswith("}")
                    ):
                        try:
                            row[key] = json.loads(value)
                        except Exception:
                            pass

            return rows

    def get_items_by_ids(
        self,
        item_type: str,
        ids: List[str],
    ) -> List[dict]:
        if not ids:
            return []

        target_ids = set(ids)

        return [
            item
            for item in self.get_all_items(item_type)
            if item.get("id") in target_ids
        ]