"""Vector store for semantic retrieval over source classroom content only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import re

import numpy as np

import config
from rag.embeddings import EmbeddingModel
from storage.json_store import JsonStore
from .filters import DocumentFilter


@dataclass(slots=True)
class RetrievalHit:
    item: dict
    score: float
    evidence: str


class VectorStore:
    """Builds a searchable corpus from raw classroom JSON documents.

    Only source content is embedded. Generated summaries, assistant outputs,
    and routing/debug strings are never added to the corpus.
    """

    def __init__(self, data_dir: str, embedding_model_path: Optional[str] = None, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.json_store = JsonStore(self.data_dir)
        self.filter = DocumentFilter()
        self.embedding_model_path = embedding_model_path or config.EMBEDDING_MODEL_PATH
        self.device = device
        self.embedding_model = None
        if self.embedding_model_path:
            try:
                self.embedding_model = EmbeddingModel(self.embedding_model_path, device=device)
            except Exception:
                self.embedding_model = None
        self._items: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._fingerprints: set[str] = set()

    def build(self, document_type: str = "all") -> None:
        items: list[dict] = []
        for item_type in ("announcements", "assignments", "materials", "courses"):
            items.extend(self.json_store.get_all_items(item_type) if item_type != "courses" else self._load_courses())
        items = self.filter.filter_items(items, document_type)
        items = self.filter.dedupe_texts(items)

        self._items = []
        texts: list[str] = []
        fingerprints: set[str] = set()
        for item in items:
            text = self._item_text(item)
            fingerprint = self._fingerprint(text)
            if not text or fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            self._items.append(item)
            texts.append(text)

        self._fingerprints = fingerprints
        if not texts or self.embedding_model is None:
            self._vectors = None
            return

        vectors = self.embedding_model.encode(texts)
        self._vectors = np.array(vectors, dtype=float)

    def search(self, query: str, document_type: str = "all", top_k: int = 5) -> list[RetrievalHit]:
        if not self._items:
            self.build(document_type=document_type)

        if not self._items:
            return []

        query_terms = re.findall(r"[a-z0-9]+", query.lower())
        keyword_scores = [self._keyword_score(query_terms, self._item_text(item)) for item in self._items]
        semantic_scores = self._semantic_scores(query) if self._vectors is not None else None

        hits: list[RetrievalHit] = []
        for index, item in enumerate(self._items):
            score = float(keyword_scores[index])
            if semantic_scores is not None:
                score += float(semantic_scores[index]) * 2.0
            hits.append(
                RetrievalHit(
                    item=item,
                    score=round(score, 4),
                    evidence=self._item_text(item)[:500],
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _load_courses(self) -> list[dict]:
        payload = self.json_store._load(self.data_dir / "courses" / "courses.json", {"courses": {}})
        courses = []
        for course_id, course in (payload.get("courses") or {}).items():
            courses.append(
                {
                    "id": course_id,
                    "item_type": "courses",
                    "course_id": course_id,
                    "course_name": course.get("name", ""),
                    "title": course.get("name", ""),
                    "description": course.get("description", ""),
                    "text": f"{course.get('name', '')} {course.get('description', '')} {course.get('description_heading', '')}".strip(),
                    "updated_at": course.get("updated_at", ""),
                }
            )
        return courses

    def _item_text(self, item: dict) -> str:
        title = str(item.get("title", ""))
        description = str(item.get("description", item.get("text", "")))
        return f"{title}\n{description}".strip()

    def _keyword_score(self, terms: list[str], text: str) -> float:
        text_lower = text.lower()
        return float(sum(1 for term in terms if term and term in text_lower))

    def _semantic_scores(self, query: str) -> list[float]:
        query_vector = np.array(self.embedding_model.encode([query])[0], dtype=float)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-9
        scores = (self._vectors @ query_vector) / norms
        return scores.tolist()

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")).hexdigest()
