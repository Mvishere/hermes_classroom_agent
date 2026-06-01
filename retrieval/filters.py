"""Document type filters and de-duplication helpers."""

from __future__ import annotations

import re
from typing import Iterable


class DocumentFilter:
    DOCUMENT_RULES = {
        "announcements": ["announcement", "announc", "notice", "posted"],
        "assignments": ["assignment", "quiz", "homework", "submission"],
        "materials": ["material", "lecture", "slide", "note", "reading", "resource"],
        "courses": ["course", "class", "enrolled", "section"],
    }

    def infer_document_type(self, question: str) -> str:
        text = question.lower()
        for document_type, markers in self.DOCUMENT_RULES.items():
            if any(marker in text for marker in markers):
                return document_type
        return "all"

    def filter_items(self, items: Iterable[dict], document_type: str) -> list[dict]:
        if document_type in {"", "all", None}:
            return list(items)
        filtered = []
        for item in items:
            item_type = str(item.get("item_type") or item.get("type") or item.get("category") or "").lower()
            if document_type in item_type:
                filtered.append(item)
                continue
            title = str(item.get("title", "")).lower()
            text = " ".join(str(item.get(field, "")) for field in ("description", "text", "content")).lower()
            haystack = f"{title} {text}"
            if any(marker in haystack for marker in self.DOCUMENT_RULES.get(document_type, [])):
                filtered.append(item)
        return filtered

    def dedupe_texts(self, items: Iterable[dict]) -> list[dict]:
        seen = set()
        result = []
        for item in items:
            text = self._item_text(item)
            fingerprint = re.sub(r"\s+", " ", text.lower()).strip()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            result.append(item)
        return result

    def _item_text(self, item: dict) -> str:
        return " ".join(
            str(item.get(field, "")) for field in ("title", "description", "text", "content")
        )
