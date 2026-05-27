"""Helpers for reading Google Form attachments from Classroom assignments."""

from __future__ import annotations

import re
from typing import Any
from quiz_extractor.parser.form_parser import render_quiz_text

_FORM_ID_PATTERNS = (
    re.compile(r"docs\.google\.com/forms/d/e/([^/?#]+)/"),
    re.compile(r"docs\.google\.com/forms/d/([^/?#]+)/"),
)


def extract_form_id(form_url: str) -> str:
    """Extract the Google Form ID from a form URL."""
    if not form_url:
        return ""
    for pattern in _FORM_ID_PATTERNS:
        match = pattern.search(form_url)
        if match:
            return match.group(1)
    return ""


def fetch_form_details(browser_client, form_url: str, fallback_title: str = "") -> dict[str, Any]:
    """Fetch quiz data from a rendered Google Form using the authenticated browser session."""
    form_id = extract_form_id(form_url)
    if browser_client is None:
        return {
            "form_url": form_url,
            "form_id": form_id,
            "form_title": fallback_title,
            "form_questions": [],
            "form_text": "",
            "form_fetch_status": "browser_unavailable",
            "quiz_id": form_id or form_url,
            "title": fallback_title,
            "questions": [],
            "section_titles": [],
            "quiz_text": "",
        }

    quiz_data = browser_client.extract_form(
        form_url=form_url,
        form_id=form_id,
        fallback_title=fallback_title,
    )
    quiz_data.setdefault("form_url", form_url)
    quiz_data.setdefault("form_id", form_id)
    quiz_data.setdefault("form_title", fallback_title)
    quiz_data.setdefault("form_questions", quiz_data.get("questions", []))
    quiz_data.setdefault("form_text", quiz_data.get("quiz_text", ""))
    quiz_data.setdefault("form_fetch_status", quiz_data.get("fetch_status", "rendered_dom"))
    quiz_data.setdefault("form_source", quiz_data.get("source", "playwright_dom"))
    quiz_data.setdefault("form_page_url", quiz_data.get("page_url", form_url))
    quiz_data.setdefault("form_extracted_at", quiz_data.get("extracted_at", ""))
    return quiz_data


def extract_form_questions(form_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility helper for normalized question payloads."""
    if not isinstance(form_payload, dict):
        return []

    if isinstance(form_payload.get("questions"), list):
        questions = form_payload.get("questions", [])
        normalized: list[dict[str, Any]] = []
        for question in questions:
            if not isinstance(question, dict):
                continue
            normalized.append(
                {
                    "title": question.get("question", ""),
                    "kind": question.get("type", "question"),
                    "options": question.get("options", []),
                    "required": question.get("required"),
                }
            )
        return normalized

    items = form_payload.get("items", [])
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        question_item = item.get("questionItem") or {}
        question = question_item.get("question") or {}
        choice_question = question.get("choiceQuestion") or {}
        grid_question = question.get("grid") or {}

        if title:
            entry: dict[str, Any] = {"title": title}
            if question:
                entry["required"] = bool(question.get("required", False))
                if "choiceQuestion" in question:
                    entry["kind"] = "choice"
                elif "textQuestion" in question:
                    entry["kind"] = "text"
                elif "scaleQuestion" in question:
                    entry["kind"] = "scale"
                elif "dateQuestion" in question:
                    entry["kind"] = "date"
                elif "timeQuestion" in question:
                    entry["kind"] = "time"
                elif "fileUploadQuestion" in question:
                    entry["kind"] = "file_upload"
                else:
                    entry["kind"] = "question"
                options = []
                for option in choice_question.get("options", []):
                    if isinstance(option, dict):
                        value = option.get("value") or option.get("label") or option.get("text")
                        if value:
                            options.append(str(value))
                    elif option:
                        options.append(str(option))
                if options:
                    entry["options"] = options
                rows = []
                for row in grid_question.get("rows", []):
                    if isinstance(row, dict):
                        value = row.get("value") or row.get("label") or row.get("text")
                        if value:
                            rows.append(str(value))
                    elif row:
                        rows.append(str(row))
                if rows:
                    entry["rows"] = rows
                columns = []
                for column in grid_question.get("columns", []):
                    if isinstance(column, dict):
                        value = column.get("value") or column.get("label") or column.get("text")
                        if value:
                            columns.append(str(value))
                    elif column:
                        columns.append(str(column))
                if columns:
                    entry["columns"] = columns
            elif item.get("pageBreakItem") is not None:
                entry["kind"] = "section_title"
            elif item.get("textItem") is not None:
                entry["kind"] = "text"

            normalized.append(entry)
    return normalized


def build_form_text(form_title: str, form_url: str, questions: list[dict[str, Any]]) -> str:
    """Compatibility wrapper that renders quiz text from structured questions."""
    normalized_questions = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        normalized_questions.append(
            {
                "question": question.get("title", question.get("question", "")),
                "type": question.get("kind", question.get("type", "question")),
                "options": question.get("options", []),
                "section_title": question.get("section_title", ""),
            }
        )
    lines = []
    if form_title:
        lines.append(f"Form Title: {form_title}")
    if form_url:
        lines.append(f"Form URL: {form_url}")
    rendered = render_quiz_text(form_title, normalized_questions, [])
    if rendered:
        lines.append(rendered)
    return "\n".join(lines).strip()
