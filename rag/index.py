"""Persistent embedding index for local retrieval."""

from datetime import datetime
import json
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from rag.embeddings import EmbeddingModel


class EmbeddingIndex:
    """Stores embeddings on disk and provides cosine similarity search."""

    def __init__(self, index_path: Path, embedding_model: EmbeddingModel):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.index_path.exists():
            return {"items": []}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}

    def _save(self) -> None:
        payload = dict(self._payload)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _key(self, course_id: str, item_id: str) -> str:
        return f"{course_id}:{item_id}"

    def upsert_items(
        self, items: List[dict], text_builder: Callable[[dict], str]
    ) -> int:
        """Add embeddings for new items. Returns how many were added."""
        existing = {entry.get("key") for entry in self._payload.get("items", [])}
        pending: List[Tuple[str, dict, str]] = []
        for item in items:
            course_id = item.get("course_id")
            item_id = item.get("id")
            if not course_id or not item_id:
                continue
            key = self._key(course_id, item_id)
            if key in existing:
                continue
            text = text_builder(item)
            if not text.strip():
                continue
            pending.append((key, item, text))

        if not pending:
            return 0

        texts = [entry[2] for entry in pending]
        embeddings = self.embedding_model.encode(texts)

        for (key, item, text), embedding in zip(pending, embeddings):
            self._payload.setdefault("items", []).append(
                {
                    "key": key,
                    "item_id": item.get("id"),
                    "course_id": item.get("course_id"),
                    "course_name": item.get("course_name", ""),
                    "text": text,
                    "embedding": embedding,
                }
            )

        self._save()
        return len(pending)

    def search(self, query: str, top_k: int) -> List[dict]:
        items = self._payload.get("items", [])
        if not items:
            return []
        query_vec = np.array(self.embedding_model.encode([query])[0])
        matrix = np.array([item.get("embedding", []) for item in items], dtype=float)
        if matrix.size == 0:
            return []
        scores = matrix @ query_vec
        top_k = min(top_k, len(items))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for index in top_indices:
            entry = dict(items[int(index)])
            entry["score"] = float(scores[int(index)])
            results.append(entry)
        return results
