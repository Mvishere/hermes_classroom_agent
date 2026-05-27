"""Topic extraction pipeline for Classroom items."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from rag.attachments import AttachmentTextExtractor
from rag.embedding_matcher import EmbeddingMatcher
from rag.embeddings import EmbeddingModel
from rag.llm import LocalLLM
from student.topic_graph import TopicGraph


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "into",
    "over",
    "under",
    "about",
    "your",
    "you",
    "are",
    "was",
    "were",
    "will",
    "shall",
    "can",
    "could",
    "should",
    "would",
    "to",
    "of",
    "in",
    "on",
    "by",
    "as",
    "an",
    "a",
    "is",
    "it",
    "be",
    "or",
    "at",
    "if",
    "we",
    "our",
    "they",
    "their",
    "them",
    "he",
    "she",
    "his",
    "her",
    "not",
    "no",
    "yes",
    "use",
    "using",
    "used",
    "learn",
    "learning",
    "study",
    "studying",
    "material",
    "assignment",
    "announcement",
    "title",
    "description",
    "welcome",
    "people",
    "class",
    "course",
    "getting",
    "started",
}

_SHORT_ALLOWLIST = {
    "ai",
    "ml",
    "nlp",
    "api",
    "sql",
    "ui",
    "ux",
    "css",
    "html",
    "js",
    "web",
}


class TopicExtractor:
    """Extracts topics, concepts, and keywords from Classroom items."""

    def __init__(
        self,
        base_dir: Path,
        topic_graph: TopicGraph,
        llm: Optional[LocalLLM] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        keyword_limit: int = 12,
        max_chars: int = 4000,
        embedding_threshold: float = 0.72,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.topic_graph = topic_graph
        self.llm = llm
        self.keyword_limit = keyword_limit
        self.max_chars = max_chars
        self.attachment_extractor = AttachmentTextExtractor(
            self.base_dir, max_chars=config.TOPIC_EXTRACT_MAX_CHARS
        )
        self.embedding_matcher = (
            EmbeddingMatcher(embedding_model, embedding_threshold)
            if embedding_model
            else None
        )

    def extract(self, item: dict, item_type: str) -> dict:
        """Extract topics and related metadata for a Classroom item."""
        attachments = item.get("attachment_paths", [])
        extracted_text = ""
        formats: List[str] = []
        if config.PDF_EXTRACT_ENABLED and attachments:
            extracted_text, formats = self.attachment_extractor.extract(attachments)

        source_text = self._build_source_text(item, extracted_text)
        if self.max_chars and len(source_text) > self.max_chars:
            source_text = source_text[: self.max_chars]

        keywords = self._extract_keywords(source_text)
        llm_payload = {}
        if self.llm and config.TOPIC_EXTRACT_LLM_ENABLED:
            llm_payload = self._extract_with_llm(source_text)

        topics = self._merge_topics(keywords, llm_payload.get("topics"))
        concepts = self._merge_topics(topics, llm_payload.get("concepts"))
        skills = self._normalize_topics(llm_payload.get("skills", []))
        subject_areas = self._normalize_topics(llm_payload.get("subject_areas", []))

        topics = self._normalize_topics(topics)
        concepts = self._normalize_topics(concepts)

        if self.embedding_matcher:
            topics = self.embedding_matcher.map_to_known(
                topics, self.topic_graph.all_topics()
            )

        difficulty = llm_payload.get("difficulty") or self._infer_difficulty(source_text)

        return {
            "item_id": item.get("id", ""),
            "item_type": item_type,
            "course_id": item.get("course_id", ""),
            "course_name": item.get("course_name", ""),
            "title": item.get("title", ""),
            "topics": topics,
            "concepts": concepts,
            "skills": skills,
            "subject_areas": subject_areas,
            "keywords": keywords,
            "difficulty": difficulty,
            "attachment_formats": formats,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "method": "hybrid" if llm_payload else "keywords",
        }

    def _build_source_text(self, item: dict, extracted_text: str) -> str:
        parts = [
            f"Title: {item.get('title', '')}",
            f"Description: {item.get('description', '')}",
        ]
        if extracted_text:
            parts.append("Attachment Text:\n" + extracted_text)
        return "\n".join(parts).strip()

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
        freq: Dict[str, int] = {}
        for token in tokens:
            if token in _STOPWORDS:
                continue
            if len(token) < 4 and token not in _SHORT_ALLOWLIST:
                continue
            freq[token] = freq.get(token, 0) + 1

        ranked = sorted(freq.items(), key=lambda pair: (-pair[1], -len(pair[0])))
        keywords = [word for word, _ in ranked[: self.keyword_limit]]
        return keywords

    def _extract_with_llm(self, text: str) -> dict:
        prompt = (
            "Extract topics from the text. Return JSON with keys: topics, concepts, skills, "
            "subject_areas, difficulty. Each list should be short (max 8 items). "
            "Use plain strings only and lowercase difficulty (low, medium, high, unknown).\n\n"
            "Text:\n"
            f"{text}\n"
        )
        try:
            response = self.llm.generate(prompt)
        except Exception:
            logging.exception("Topic LLM extraction failed.")
            return {}

        payload = self._parse_json(response)
        if not isinstance(payload, dict):
            return {}

        return {
            "topics": self._normalize_topics(payload.get("topics", [])),
            "concepts": self._normalize_topics(payload.get("concepts", [])),
            "skills": self._normalize_topics(payload.get("skills", [])),
            "subject_areas": self._normalize_topics(payload.get("subject_areas", [])),
            "difficulty": self._normalize_difficulty(payload.get("difficulty", "unknown")),
        }

    def _parse_json(self, response: str) -> dict:
        if not response:
            return {}
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        snippet = response[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}

    def _merge_topics(self, base: List[str], extra: Optional[List[str]]) -> List[str]:
        topics = list(base)
        if extra:
            topics.extend(extra)
        return self._normalize_topics(topics)

    def _normalize_topics(self, topics: List[str]) -> List[str]:
        cleaned = []
        for topic in topics or []:
            value = str(topic).strip()
            if not value:
                continue
            value = re.sub(r"\s+", " ", value)
            cleaned.append(value)
        return list(dict.fromkeys(cleaned))

    def _normalize_difficulty(self, value: str) -> str:
        lowered = str(value).strip().lower()
        if lowered in {"low", "medium", "high", "unknown"}:
            return lowered
        return "unknown"

    def _infer_difficulty(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("advanced", "graduate", "complex", "proof")):
            return "high"
        if any(word in lowered for word in ("intro", "basic", "beginner", "fundamental")):
            return "low"
        if "intermediate" in lowered or "moderate" in lowered:
            return "medium"
        return "unknown"
