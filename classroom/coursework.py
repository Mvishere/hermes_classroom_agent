"""Processor for Classroom coursework items.

Detects new assignments and stores metadata plus attachments.
"""

import logging
from typing import List

import config
from classroom.classroom_client import ClassroomClient
from classroom.forms import fetch_form_details
from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.quiz_storage import QuizStorage
from storage.state_manager import StateManager


def process_coursework(
    courses: List[dict],
    client: ClassroomClient,
    json_store: JsonStore,
    state_manager: StateManager,
    file_storage: FileStorage,
    drive_service,
    browser_client=None,
    quiz_storage: QuizStorage | None = None,
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

            form_details = _extract_form_details(item, browser_client)
            is_seen = state_manager.is_seen("assignment", item_id, course_id)
            if is_seen and not (
                form_details.get("form_questions") or form_details.get("form_text")
            ):
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
            record.update(form_details)

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
            if quiz_storage is not None and (
                record.get("form_questions") or record.get("form_text")
            ):
                quiz_storage.upsert_quiz(
                    {
                        "quiz_id": record.get("form_id") or record.get("id") or item_id,
                        "assignment_id": item_id,
                        "course_id": course_id,
                        "course_name": course_name,
                        "title": record.get("form_title") or record.get("title", ""),
                        "source_url": record.get("form_url", ""),
                        "questions": record.get("form_questions", []),
                        "section_titles": record.get("section_titles", []),
                        "quiz_text": record.get("form_text", ""),
                        "fetch_status": record.get("form_fetch_status", ""),
                        "source": record.get("form_source", "playwright_dom"),
                        "page_url": record.get("form_page_url", record.get("form_url", "")),
                        "extracted_at": record.get("form_extracted_at", ""),
                    }
                )
            if not is_seen:
                state_manager.mark_seen("assignment", item_id, course_id)
                new_items.append(item_id)

    return new_items


def _extract_form_details(item: dict, browser_client) -> dict:
    materials = item.get("materials", []) or []
    for material in materials:
        form = material.get("form") or {}
        form_url = form.get("formUrl") or ""
        if not form_url:
            continue

        fallback_title = form.get("title") or item.get("title", "")
        details = fetch_form_details(browser_client, form_url, fallback_title=fallback_title)
        if details.get("form_questions"):
            logging.info(
                "Fetched %s question(s) from Google Form for assignment %s.",
                len(details.get("form_questions", [])),
                item.get("id", ""),
            )
        else:
            logging.info(
                "Google Form detected for assignment %s but no questions were returned.",
                item.get("id", ""),
            )
        return details

    return {
        "form_url": "",
        "form_id": "",
        "form_title": "",
        "form_questions": [],
        "form_text": "",
    }
