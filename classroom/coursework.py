"""Processor for Classroom coursework items.

Detects new assignments and stores metadata plus attachments.
"""

import logging
from typing import List

import config
from classroom.classroom_client import ClassroomClient
from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.state_manager import StateManager


def process_coursework(
    courses: List[dict],
    client: ClassroomClient,
    json_store: JsonStore,
    state_manager: StateManager,
    file_storage: FileStorage,
    drive_service,
) -> List[str]:
    """Fetch coursework for each course and store only new items."""
    new_items = []
    for course in courses:
        course_id = course.get("id")
        if not course_id:
            continue
        course_name = course.get("name", "")
        try:
            items = client.list_coursework(course_id)
        except Exception:
            logging.exception("Failed to fetch coursework for course %s", course_id)
            continue

        for item in items:
            item_id = item.get("id")
            if not item_id:
                continue
            if state_manager.is_seen("assignment", item_id, course_id):
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
                "submission_state": "",
                "attachment_paths": [],
                "processed": 0,
            }

            if config.FETCH_SUBMISSIONS:
                try:
                    # Request only the current user's submissions to determine completion
                    submissions = client.list_student_submissions(course_id, item_id, user_id="me")
                    if submissions:
                        record["submission_state"] = submissions[0].get("state", "")
                except Exception:
                    logging.exception(
                        "Failed to fetch submissions for coursework %s", item_id
                    )

            attachments = file_storage.download_attachments(
                item.get("materials", []),
                "assignments",
                course_id,
                item_id,
                drive_service,
            )
            record["attachment_paths"] = attachments

            json_store.upsert_item("assignments", record, raw_payload=item)
            state_manager.mark_seen("assignment", item_id, course_id)
            new_items.append(item_id)

    return new_items
