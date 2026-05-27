"""Builds topic relationships from extracted topics and mastery data."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List

import config
from storage.topics_store import TopicsStore
from student.knowledge_store import KnowledgeStore
from student.topic_graph import TopicGraph


class TopicGraphBuilder:
    """Auto-generates topic relationships using co-occurrence and mastery deltas."""

    def __init__(
        self,
        graph: TopicGraph,
        min_cooccurrence: int = 2,
        max_related: int = 6,
        mastery_delta: float = 0.2,
    ) -> None:
        self.graph = graph
        self.min_cooccurrence = min_cooccurrence
        self.max_related = max_related
        self.mastery_delta = mastery_delta

    def rebuild(self, topics_store: TopicsStore, knowledge_store: KnowledgeStore) -> None:
        payloads = topics_store.list_payloads()
        if not payloads:
            return

        cooc = defaultdict(lambda: defaultdict(int))
        prereq = defaultdict(lambda: defaultdict(int))
        mastery = self._mastery_scores(knowledge_store)

        for payload in payloads:
            item_type = payload.get("item_type")
            if item_type == "announcements":
                continue
            if item_type not in {"materials", "assignments"}:
                continue
            topics = self._normalize_topics(payload.get("topics", []))
            if len(topics) < 2:
                continue
            for i, topic in enumerate(topics):
                for other in topics[i + 1 :]:
                    cooc[topic][other] += 1
                    cooc[other][topic] += 1

                    mastery_topic = mastery.get(topic, 0.0)
                    mastery_other = mastery.get(other, 0.0)
                    if mastery_topic - mastery_other >= self.mastery_delta:
                        prereq[other][topic] += 1
                    elif mastery_other - mastery_topic >= self.mastery_delta:
                        prereq[topic][other] += 1

        graph = {}
        for topic, related_counts in cooc.items():
            related_topics = self._top_related(related_counts)
            prereqs = self._top_prereqs(prereq.get(topic, {}))
            graph[topic] = {
                "prerequisites": prereqs,
                "related_topics": related_topics,
            }

        if graph:
            logging.info("Topic graph updated with %s topics.", len(graph))
            self.graph.replace(graph)

    def _mastery_scores(self, knowledge_store: KnowledgeStore) -> Dict[str, float]:
        scores = {}
        for topic, entry in knowledge_store.all_topics().items():
            try:
                scores[topic] = float(entry.get("mastery_score", 0.0))
            except Exception:
                scores[topic] = 0.0
        return scores

    def _top_related(self, related_counts: Dict[str, int]) -> List[str]:
        filtered = [
            (topic, count)
            for topic, count in related_counts.items()
            if count >= self.min_cooccurrence
        ]
        filtered.sort(key=lambda pair: (-pair[1], pair[0].lower()))
        return [topic for topic, _ in filtered[: self.max_related]]

    def _top_prereqs(self, prereq_counts: Dict[str, int]) -> List[str]:
        filtered = [
            (topic, count)
            for topic, count in prereq_counts.items()
            if count >= 1
        ]
        filtered.sort(key=lambda pair: (-pair[1], pair[0].lower()))
        return [topic for topic, _ in filtered[: self.max_related]]

    def _normalize_topics(self, topics: Iterable[str]) -> List[str]:
        cleaned = []
        for topic in topics or []:
            text = str(topic).strip()
            if text:
                cleaned.append(text)
        return list(dict.fromkeys(cleaned))
