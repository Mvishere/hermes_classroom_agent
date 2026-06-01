"""Stores RAG summaries grouped by course."""

from datetime import datetime
import json
from pathlib import Path
import threading
import re
from typing import List


class SummaryStore:
    """Persists summaries for materials in structured JSON."""

    def __init__(self, summary_path: Path):
        self.summary_path = Path(summary_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.summary_path.exists():
            return {"courses": {}}
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {"courses": {}}

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def has_summary(self, material: dict) -> bool:
        course_id = material.get("course_id")
        item_id = material.get("id")
        if not course_id or not item_id:
            return False
        course = self._payload.get("courses", {}).get(course_id, {})
        return item_id in course.get("items", {})

    def get_summary(self, course_id: str, item_id: str) -> str:
        if not course_id or not item_id:
            return ""
        course = self._payload.get("courses", {}).get(course_id, {})
        entry = course.get("items", {}).get(item_id, {})
        summary = entry.get("summary")
        if not isinstance(summary, str):
            return ""
        return summary

    def iter_summaries(self) -> List[dict]:
        """Return a flat list of stored summary entries."""
        entries: List[dict] = []
        courses = self._payload.get("courses", {})
        for course_id, course in courses.items():
            items = (course or {}).get("items", {})
            for item_id, entry in items.items():
                if not isinstance(entry, dict):
                    continue
                entries.append(
                    {
                        "course_id": course_id,
                        "item_id": item_id,
                        "course_name": course.get("course_name", "") if isinstance(course, dict) else "",
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "updated_at": entry.get("updated_at", ""),
                    }
                )
        return entries

    def search_topic(self, topic: str, limit: int = 5) -> List[dict]:
        """Find summaries whose title or text closely matches a requested topic.

        The matcher is intentionally strict for demo reliability: for multi-word
        topics, we only return entries when all topic terms are represented in the
        summary title or stored summary text. This prevents unrelated summaries
        from being returned just because one word happened to match.
        """
        query = str(topic or "").strip().lower()
        if not query:
            return []
        terms = [term for term in re.findall(r"[a-z0-9]+", query) if len(term) > 1]
        if not terms:
            return []

        scored: List[tuple[float, dict]] = []
        for entry in self.iter_summaries():
            title = str(entry.get("title", "")).lower()
            summary = str(entry.get("summary", "")).lower()
            if not self.is_usable_summary(summary):
                continue

            haystack = f"{title} {summary}"
            score = 0.0
            if query in title:
                score += 3.0
            elif query in haystack:
                score += 2.0

            matched_terms = sum(1 for term in terms if term in title or term in summary)
            if len(terms) == 1:
                if matched_terms >= 1:
                    score += 1.5
            else:
                if matched_terms == len(terms):
                    score += 1.5

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("updated_at", "")), str(pair[1].get("title", ""))))
        return [entry for _, entry in scored[: max(1, limit)]]

    def is_usable_summary(self, summary: str) -> bool:
        text = str(summary or "").strip().lower()
        if not text:
            return False
        fallback_markers = (
            "i don't have enough information in the local course data to answer that reliably",
            "i don't have enough grounded evidence to answer that reliably",
            "i could not find any local materials to answer from",
            "no matching grounded evidence was found",
        )
        return not any(marker in text for marker in fallback_markers)

    def upsert_summary(
        self,
        material: dict,
        summary: str,
        contexts: List[dict],
        attachment_formats: List[str],
    ) -> None:
        course_id = material.get("course_id")
        item_id = material.get("id")
        if not course_id or not item_id:
            return
        with self._lock:
            courses = self._payload.setdefault("courses", {})
            course_entry = courses.setdefault(
                course_id,
                {
                    "course_id": course_id,
                    "course_name": material.get("course_name", ""),
                    "items": {},
                },
            )
            if material.get("course_name"):
                course_entry["course_name"] = material.get("course_name", "")

            course_entry["items"][item_id] = {
                "item_id": item_id,
                "title": material.get("title", ""),
                "summary": summary,
                "attachment_formats": attachment_formats,
                "contexts": [
                    {
                        "item_id": entry.get("item_id"),
                        "course_id": entry.get("course_id"),
                        "score": entry.get("score"),
                        "text": entry.get("text"),
                    }
                    for entry in contexts
                ],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            courses[course_id] = course_entry
            self._save()
