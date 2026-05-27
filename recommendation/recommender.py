"""Personalized recommendation engine."""

from __future__ import annotations

from typing import Dict, List

from student.knowledge_store import KnowledgeStore
from student.topic_graph import TopicGraph


class Recommender:
    """Generates readiness recommendations for new items."""

    def __init__(self, topic_graph: TopicGraph, knowledge_store: KnowledgeStore):
        self.topic_graph = topic_graph
        self.knowledge_store = knowledge_store

    def recommend(self, item: dict, topics: List[str], item_type: str) -> dict:
        if not topics:
            return {
                "item_type": item_type,
                "readiness": "ready",
                "message": "No prerequisite topics detected. You look ready to start.",
                "known_ratio": 1.0,
                "missing_topics": [],
                "weak_topics": [],
            }

        prerequisites = self._gather_prerequisites(topics)
        knowledge = self.knowledge_store.all_topics()

        missing = []
        weak = []
        known = 0
        for topic in prerequisites:
            entry = knowledge.get(topic)
            status = (entry or {}).get("status", "unknown")
            if status == "weak":
                weak.append(topic)
            elif status in {"known", "learning"}:
                known += 1
            else:
                missing.append(topic)

        total = max(len(prerequisites), 1)
        known_ratio = known / total

        readiness = "ready"
        if weak or len(missing) > max(1, total // 2):
            readiness = "not_ready"
        elif missing:
            readiness = "partially_ready"

        message = self._build_message(readiness, missing, weak, known_ratio)
        return {
            "item_type": item_type,
            "readiness": readiness,
            "message": message,
            "known_ratio": round(known_ratio, 2),
            "missing_topics": missing,
            "weak_topics": weak,
        }

    def _gather_prerequisites(self, topics: List[str]) -> List[str]:
        prerequisites = []
        for topic in topics:
            prerequisites.extend(self.topic_graph.prerequisites(topic))
        return list(dict.fromkeys(prerequisites))

    def _build_message(
        self, readiness: str, missing: List[str], weak: List[str], known_ratio: float
    ) -> str:
        if readiness == "not_ready":
            focus = weak or missing
            return (
                "You may struggle with this item because it relies on: "
                + ", ".join(focus)
                + "."
            )
        if readiness == "partially_ready":
            review = missing or weak
            return (
                "You know some prerequisites, but consider reviewing: "
                + ", ".join(review)
                + "."
            )
        return f"You already know most prerequisites ({known_ratio:.0%})."
