"""Embedding-based topic matcher."""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

from rag.embeddings import EmbeddingModel


class EmbeddingMatcher:
    """Maps candidate topics to known topics using embeddings."""

    def __init__(self, embedding_model: EmbeddingModel, similarity_threshold: float = 0.72):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def map_to_known(
        self, candidates: Iterable[str], known_topics: Iterable[str]
    ) -> List[str]:
        candidate_list = [
            str(candidate).strip()
            for candidate in candidates
            if candidate and str(candidate).strip()
        ]
        known_list = [
            str(topic).strip()
            for topic in known_topics
            if topic and str(topic).strip()
        ]
        if not candidate_list:
            return []
        if not known_list:
            return list(dict.fromkeys(candidate_list))

        try:
            embeddings = self.embedding_model.encode(known_list + candidate_list)
            known_vecs = np.array(embeddings[: len(known_list)], dtype=float)
            candidate_vecs = np.array(embeddings[len(known_list) :], dtype=float)

            mapped: List[str] = []
            for index, candidate in enumerate(candidate_list):
                scores = known_vecs @ candidate_vecs[index]
                best_index = int(np.argmax(scores))
                best_score = float(scores[best_index])
                if best_score >= self.similarity_threshold:
                    mapped.append(known_list[best_index])
                else:
                    mapped.append(candidate)

            return list(dict.fromkeys(mapped))
        except Exception:
            logging.exception("Embedding topic matching failed.")
            return list(dict.fromkeys(candidate_list))
