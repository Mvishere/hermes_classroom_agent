"""Canonical naming helpers for topic graph concepts."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


_ALIASES = {
    "functions": "function",
    "function": "function",
    "queries": "query",
    "query": "query",
    "events": "event",
    "event": "event",
    "responsive web design": "responsive design",
    "media query": "media queries",
    "media queries": "media queries",
    "css basics": "css basics",
    "css basic": "css basics",
    "css basis": "css basics",
    "dom manipulation": "dom manipulation",
    "dom": "dom",
    "flex box": "flexbox",
    "css grid": "css grid",
    "grid layout": "css grid",
    "html": "html",
    "css": "css",
    "javascript": "javascript",
    "js": "javascript",
    "ui": "user interface",
    "ux": "user experience",
    "html css": "html css",
    "css html": "html css",
}

_DISPLAY_NAMES = {
    "html": "HTML",
    "css": "CSS",
    "dom": "DOM",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "api": "API",
    "sql": "SQL",
    "ui": "UI",
    "ux": "UX",
    "flexbox": "Flexbox",
    "css grid": "CSS Grid",
    "media queries": "Media Queries",
    "responsive design": "Responsive Design",
    "dom manipulation": "DOM Manipulation",
    "javascript events": "JavaScript Events",
    "html css": "HTML/CSS",
    "css basics": "CSS Basics",
    "function": "Function",
    "query": "Query",
    "event": "Event",
    "prerequisite": "Prerequisite",
}


class TopicNormalizer:
    """Normalizes topic strings into stable canonical forms."""

    def canonicalize(self, topic: str) -> str:
        value = re.sub(r"\s+", " ", str(topic or "").strip().lower())
        if not value:
            return ""
        value = value.replace("/", " ")
        value = re.sub(r"[^a-z0-9+\-\s]", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            return ""
        if value in _ALIASES:
            return _ALIASES[value]
        if value.endswith("ies") and len(value) > 4:
            value = value[:-3] + "y"
        elif value.endswith("ses") and len(value) > 4:
            value = value[:-2]
        elif value.endswith("s") and len(value) > 3 and not value.endswith(("ss", "us")):
            value = value[:-1]
        return _ALIASES.get(value, value)

    def display_name(self, topic: str) -> str:
        canonical = self.canonicalize(topic)
        if not canonical:
            return ""
        if canonical in _DISPLAY_NAMES:
            return _DISPLAY_NAMES[canonical]
        return " ".join(part.capitalize() for part in canonical.split())

    def merge(self, topics: Iterable[str]) -> List[str]:
        seen = set()
        merged: List[str] = []
        for topic in topics or []:
            canonical = self.canonicalize(topic)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            merged.append(self.display_name(canonical))
        return merged

    def merge_with_canonical(self, topics: Iterable[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for topic in topics or []:
            canonical = self.canonicalize(topic)
            if not canonical:
                continue
            mapping.setdefault(canonical, self.display_name(canonical))
        return mapping
