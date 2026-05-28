"""Topic cleaning and candidate filtering for educational concept extraction."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, List


_CUSTOM_STOPWORDS = {
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "who",
    "whom",
    "whose",
    "test",
    "page",
    "topic",
    "section",
    "layout",
    "screen",
    "screens",
    "page",
    "pages",
    "section",
    "sections",
    "content",
    "item",
    "items",
    "thing",
    "things",
    "problem",
    "problems",
    "question",
    "questions",
    "answer",
    "answers",
    "assignment",
    "assignments",
    "material",
    "materials",
    "announcement",
    "announcements",
    "lesson",
    "lessons",
    "module",
    "modules",
    "unit",
    "units",
    "class",
    "course",
    "student",
    "teacher",
    "learn",
    "learning",
    "study",
    "studying",
    "use",
    "using",
    "used",
    "make",
    "makes",
    "made",
    "create",
    "created",
    "creating",
    "show",
    "shows",
    "shown",
    "tell",
    "tells",
    "asked",
    "ask",
    "asking",
}

_GENERIC_VERBS = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "get",
    "gets",
    "got",
    "go",
    "goes",
    "went",
    "see",
    "saw",
    "seen",
    "write",
    "writes",
    "wrote",
    "read",
    "reading",
    "run",
    "runs",
    "running",
}

_UI_WORDS = {
    "button",
    "menu",
    "page",
    "screen",
    "tab",
    "dialog",
    "popup",
    "form",
    "input",
    "field",
    "label",
    "layout",
    "section",
    "panel",
    "window",
    "view",
}

_FILLER_WORDS = {
    "important",
    "basic",
    "basics",
    "intro",
    "introduction",
    "overview",
    "general",
    "simple",
    "example",
    "examples",
    "practice",
    "review",
    "discussion",
    "focus",
    "details",
    "concept",
    "concepts",
}

_ALLOWED_SHORT_FORMS = {
    "ai",
    "ux",
    "ui",
    "api",
    "sql",
    "html",
    "css",
    "dom",
    "json",
    "xml",
    "git",
    "oop",
    "ml",
    "nlp",
    "js",
    "ts",
}

_IRREGULAR_LEMMAS = {
    "queries": "query",
    "events": "event",
    "functions": "function",
    "classes": "class",
    "properties": "property",
    "conditions": "condition",
    "forms": "form",
    "queries": "query",
    "media": "media",
    "methods": "method",
    "prerequisites": "prerequisite",
    "responsive": "responsive",
    "layouts": "layout",
    "screens": "screen",
    "pages": "page",
    "topics": "topic",
}


class TopicCleaner:
    """Filters noisy words and normalizes candidate educational concepts."""

    def __init__(self, min_frequency: int = 2, debug: bool = False) -> None:
        self.min_frequency = max(1, int(min_frequency))
        self.debug = debug

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", str(text).lower())

    def normalize_token(self, token: str) -> str:
        value = str(token).strip().lower()
        if not value:
            return ""
        if value in _IRREGULAR_LEMMAS:
            return _IRREGULAR_LEMMAS[value]
        if value.endswith("ies") and len(value) > 4:
            return value[:-3] + "y"
        if value.endswith("ses") and len(value) > 4:
            return value[:-2]
        if value.endswith("s") and len(value) > 3 and not value.endswith(("ss", "us")):
            return value[:-1]
        return value

    def is_noise_token(self, token: str) -> bool:
        value = self.normalize_token(token)
        if not value:
            return True
        if value in _CUSTOM_STOPWORDS:
            return True
        if value in _GENERIC_VERBS:
            return True
        if value in _UI_WORDS:
            return True
        if value in _FILLER_WORDS:
            return True
        if len(value) <= 2 and value not in _ALLOWED_SHORT_FORMS:
            return True
        return False

    def clean_phrase(self, phrase: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_+\-\s]", " ", str(phrase))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        tokens = [self.normalize_token(token) for token in self.tokenize(text)]
        tokens = [token for token in tokens if token and not self.is_noise_token(token)]
        if not tokens:
            return ""

        if len(tokens) == 1 and tokens[0] not in _ALLOWED_SHORT_FORMS and len(tokens[0]) < 4:
            return ""

        return " ".join(tokens)

    def filter_candidates(self, candidates: Iterable[str]) -> List[str]:
        cleaned = [self.clean_phrase(candidate) for candidate in candidates]
        cleaned = [candidate for candidate in cleaned if candidate]
        if not cleaned:
            return []

        counts = Counter(cleaned)
        ranked = [
            candidate
            for candidate in cleaned
            if counts[candidate] >= self.min_frequency or len(candidate.split()) > 1
        ]
        seen = set()
        result: List[str] = []
        for candidate in ranked:
            if candidate in seen:
                continue
            seen.add(candidate)
            result.append(candidate)
        if self.debug:
            return result
        return result

    def looks_like_noun_phrase(self, phrase: str) -> bool:
        tokens = [token for token in self.tokenize(phrase) if not self.is_noise_token(token)]
        if not tokens:
            return False
        if len(tokens) >= 2:
            return True
        token = tokens[0]
        return token in _ALLOWED_SHORT_FORMS or len(token) >= 4
