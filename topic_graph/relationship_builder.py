"""Semantic relationship scoring for topic graphs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from itertools import combinations
from typing import Dict, Iterable, List, Optional

import numpy as np

from rag.embeddings import EmbeddingModel
from topic_graph.normalizer import TopicNormalizer


@dataclass(frozen=True)
class RelatedTopicEdge:
    topic: str
    weight: float


class RelationshipBuilder:
    """Builds weighted semantic relationships between topic nodes."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        normalizer: Optional[TopicNormalizer] = None,
        min_weight: float = 0.65,
        max_related: int = 6,
        debug: bool = False,
    ) -> None:
        self.embedding_model = embedding_model
        self.normalizer = normalizer or TopicNormalizer()
        self.min_weight = float(min_weight)
        self.max_related = int(max_related)
        self.debug = debug

    def build(self, topics: Iterable[str], hints: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[RelatedTopicEdge]]:
        topic_list = self.normalizer.merge(topics)
        hints = hints or {}
        if not topic_list:
            return {}

        edges: Dict[str, Dict[str, float]] = {topic: {} for topic in topic_list}
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(topic_list)
                vectors = np.array(embeddings, dtype=float)
                for left_index, right_index in combinations(range(len(topic_list)), 2):
                    left_topic = topic_list[left_index]
                    right_topic = topic_list[right_index]
                    similarity = float(np.dot(vectors[left_index], vectors[right_index]))
                    if similarity < self.min_weight:
                        continue
                    edges[left_topic][right_topic] = max(edges[left_topic].get(right_topic, 0.0), similarity)
                    edges[right_topic][left_topic] = max(edges[right_topic].get(left_topic, 0.0), similarity)
            except Exception:
                logging.exception("Semantic topic relationship scoring failed.")

        for topic, related_topics in hints.items():
            canonical_topic = self.normalizer.display_name(topic) if topic else ""
            if not canonical_topic or canonical_topic not in edges:
                continue
            for related_topic in related_topics or []:
                canonical_related = self.normalizer.display_name(related_topic)
                if not canonical_related or canonical_related == canonical_topic:
                    continue
                edges[canonical_topic][canonical_related] = max(
                    edges[canonical_topic].get(canonical_related, 0.0), 0.8
                )
                edges.setdefault(canonical_related, {})[canonical_topic] = max(
                    edges.get(canonical_related, {}).get(canonical_topic, 0.0), 0.8
                )

        result: Dict[str, List[RelatedTopicEdge]] = {}
        for topic, scored_topics in edges.items():
            ranked = sorted(scored_topics.items(), key=lambda item: (-item[1], item[0].lower()))
            result[topic] = [
                RelatedTopicEdge(topic=related_topic, weight=round(weight, 2))
                for related_topic, weight in ranked[: self.max_related]
                if weight >= self.min_weight
            ]
        if self.debug:
            logging.info("Relationship builder created edges for %s topics.", len(result))
        return result
