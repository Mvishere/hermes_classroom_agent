"""Processor for Classroom courseWorkMaterial items.

Detects new study materials and stores metadata plus attachments.
"""

import logging
from typing import List

from classroom.classroom_client import ClassroomClient
from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.state_manager import StateManager


def process_materials(
    courses: List[dict],
    client: ClassroomClient,
    json_store: JsonStore,
    state_manager: StateManager,
    file_storage: FileStorage,
    drive_service,
) -> List[str]:
    """Fetch course materials for each course and store only new items."""
    new_items = []
    for course in courses:
        course_id = course.get("id")
        if not course_id:
            continue
        course_name = course.get("name", "")
        try:
            items = client.list_coursework_materials(course_id)
        except Exception:
            logging.exception("Failed to fetch materials for course %s", course_id)
            continue

        for item in items:
            item_id = item.get("id")
            if not item_id:
                continue
            if state_manager.is_seen("material", item_id, course_id):
                continue

            record = {
                "id": item_id,
                "course_id": course_id,
                "course_name": course_name,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "created_at": item.get("creationTime", ""),
                "updated_at": item.get("updateTime", ""),
                "topic_id": item.get("topicId", ""),
                "attachment_paths": [],
                "processed": 0,
            }

            attachments = file_storage.download_attachments(
                item.get("materials", []),
                "materials",
                course_id,
                item_id,
                drive_service,
            )
            record["attachment_paths"] = attachments

            json_store.upsert_item("materials", record, raw_payload=item)
            state_manager.mark_seen("material", item_id, course_id)
            new_items.append(item_id)

    return new_items
