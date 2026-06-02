"""Google Classroom API client with retries and pagination."""

import logging
import os
import random
import time
from typing import Callable, List
from auth.google_auth import get_credentials

from googleapiclient.errors import HttpError

from config import API_BACKOFF_BASE_SECONDS, API_MAX_RETRIES, API_THROTTLE_SECONDS, SCOPES


class ClassroomClient:
    """Wrapper around Classroom API with retry-safe list helpers."""

    def __init__(self, classroom_service):
        self.service = classroom_service

    def _execute_with_retries(self, request, request_name: str):
        for attempt in range(API_MAX_RETRIES):
            try:
                return request.execute()
            except HttpError as exc:
                status = getattr(exc.resp, "status", None)
                if status in (429, 500, 502, 503, 504):
                    sleep_for = API_BACKOFF_BASE_SECONDS * (2**attempt)
                    sleep_for += random.uniform(0, 1)
                    logging.warning(
                        "API error %s on %s. Retry in %.1fs.",
                        status,
                        request_name,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue
                logging.exception("Non-retriable API error on %s.", request_name)
                raise
            except Exception:
                logging.exception("Unexpected error on %s.", request_name)
                if attempt == API_MAX_RETRIES - 1:
                    raise
                sleep_for = API_BACKOFF_BASE_SECONDS * (2**attempt)
                time.sleep(sleep_for)
        raise RuntimeError(f"Max retries exceeded for {request_name}")

    def _list_with_pagination(
        self,
        request_builder: Callable[[str], object],
        result_key: str,
        request_name: str,
    ) -> List[dict]:
        items = []
        page_token = None
        while True:
            request = request_builder(page_token)
            response = self._execute_with_retries(request, request_name)
            items.extend(response.get(result_key, []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
            time.sleep(API_THROTTLE_SECONDS)
        return items

    def list_courses(self) -> List[dict]:
        return self._list_with_pagination(
            lambda page_token: self.service.courses().list(
                pageToken=page_token, pageSize=100, courseStates="ACTIVE"
            ),
            "courses",
            "courses.list",
        )

    def list_coursework(self, course_id: str) -> List[dict]:
        return self._list_with_pagination(
            lambda page_token: self.service.courses()
            .courseWork()
            .list(courseId=course_id, pageToken=page_token, pageSize=100),
            "courseWork",
            "courses.courseWork.list",
        )

    def list_coursework_materials(self, course_id: str) -> List[dict]:
        return self._list_with_pagination(
            lambda page_token: self.service.courses()
            .courseWorkMaterials()
            .list(courseId=course_id, pageToken=page_token, pageSize=100),
            "courseWorkMaterial",
            "courses.courseWorkMaterials.list",
        )

    def list_announcements(self, course_id: str) -> List[dict]:
        return self._list_with_pagination(
            lambda page_token: self.service.courses()
            .announcements()
            .list(courseId=course_id, pageToken=page_token, pageSize=100),
            "announcements",
            "courses.announcements.list",
        )

    def list_student_submissions(
        self, course_id: str, coursework_id: str, user_id: str | None = None
    ) -> List[dict]:
        def builder(page_token: str):
            params = {
                "courseId": course_id,
                "courseWorkId": coursework_id,
                "pageToken": page_token,
                "pageSize": 100,
            }
            if user_id:
                params["userId"] = user_id
            return self.service.courses().courseWork().studentSubmissions().list(**params)

        return self._list_with_pagination(builder, "studentSubmissions", "courses.courseWork.studentSubmissions.list")

if __name__ == "__main__":
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    SERVICE_ACCOUNT_FILE = "token.json"

    credentials = get_credentials()
    classroom_service = build("classroom", "v1", credentials=credentials)
    client = ClassroomClient(classroom_service)

    courses = client.list_courses()
    materials = client.list_coursework_materials(courses[0]["id"])
    print(f"Found {len(materials)} coursework materials. {materials[0]}")