"""Tracks user learning progress in JSON."""

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import List


class UserStatusManager:
    """Stores progress signals like known assignments/materials."""

    def __init__(self, status_path: Path):
        self.status_path = Path(status_path)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.status_path.exists():
            return {"courses": {}}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return {"courses": {}}

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.status_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def update_from_assignments(
        self, assignments: List[dict], materials: List[dict]
    ) -> int:
        """Mark assignments and related materials as known when completed."""
        updates = 0
        materials_by_topic = {}
        for material in materials:
            topic_id = material.get("topic_id")
            if topic_id:
                materials_by_topic.setdefault(topic_id, []).append(material)

        with self._lock:
            for assignment in assignments:
                if not self._is_completed(assignment):
                    continue
                course_id = assignment.get("course_id")
                if not course_id:
                    continue
                course_entry = self._ensure_course(course_id, assignment.get("course_name", ""))
                updates += self._mark_assignment_known(course_entry, assignment)

                topic_id = assignment.get("topic_id")
                if topic_id:
                    for material in materials_by_topic.get(topic_id, []):
                        updates += self._mark_material_known(course_entry, material, assignment)

            if updates:
                self._save()

        return updates

    def _ensure_course(self, course_id: str, course_name: str) -> dict:
        courses = self._payload.setdefault("courses", {})
        course_entry = courses.setdefault(
            course_id,
            {
                "course_id": course_id,
                "course_name": course_name,
                "assignments": {},
                "materials": {},
            },
        )
        if course_name:
            course_entry["course_name"] = course_name
        return course_entry

    def _is_completed(self, assignment: dict) -> bool:
        submission_state = assignment.get("submission_state", "").upper()
        if submission_state in {"TURNED_IN", "RETURNED"}:
            return True
        return assignment.get("completed") is True

    def _mark_assignment_known(self, course_entry: dict, assignment: dict) -> int:
        assignment_id = assignment.get("id")
        if not assignment_id:
            return 0
        assignments = course_entry.setdefault("assignments", {})
        if assignment_id in assignments and assignments[assignment_id].get("status") == "known":
            return 0
        assignments[assignment_id] = {
            "assignment_id": assignment_id,
            "title": assignment.get("title", ""),
            "status": "known",
            "topic_id": assignment.get("topic_id", ""),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        return 1

    def _mark_material_known(self, course_entry: dict, material: dict, assignment: dict) -> int:
        material_id = material.get("id")
        if not material_id:
            return 0
        materials = course_entry.setdefault("materials", {})
        if material_id in materials and materials[material_id].get("status") == "known":
            return 0
        materials[material_id] = {
            "material_id": material_id,
            "title": material.get("title", ""),
            "status": "known",
            "topic_id": material.get("topic_id", ""),
            "source_assignment_id": assignment.get("id", ""),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        return 1
