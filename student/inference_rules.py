"""Inference rules for updating student knowledge."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from student.knowledge_store import KnowledgeStore
from student.mastery_engine import MasteryEngine
from student.topic_mapper import TopicMapper


class InferenceRules:
    """Applies inference rules to update student knowledge."""

    def __init__(self, mastery_engine: MasteryEngine, topic_mapper: TopicMapper):
        self.mastery_engine = mastery_engine
        self.topic_mapper = topic_mapper

    def apply_assignment_completion(
        self,
        knowledge_store: KnowledgeStore,
        topics: Iterable[str],
        evidence: str,
        score: float | None = None,
    ) -> int:
        updates = 0
        topics = list(topics or [])
        for topic in topics:
            updates += self._apply_topic(knowledge_store, topic, evidence, score, 1.0)

        prerequisites = self.topic_mapper.prerequisites_for_topics(topics)
        for topic in prerequisites:
            updates += self._apply_topic(knowledge_store, topic, evidence, score, 0.4)

        related = self.topic_mapper.related_for_topics(topics)
        for topic in related:
            updates += self._apply_topic(knowledge_store, topic, evidence, score, 0.2)

        return updates

    def apply_explanation_request(
        self,
        knowledge_store: KnowledgeStore,
        topics: Iterable[str],
        evidence: str,
    ) -> int:
        updates = 0
        for topic in topics:
            entry = knowledge_store.get_topic(topic)
            updated = self.mastery_engine.apply_confidence_penalty(entry, evidence)
            knowledge_store.set_topic(topic, updated)
            updates += 1
        return updates

    def apply_decay(self, knowledge_store: KnowledgeStore, decay_days: int = 30) -> int:
        updates = 0
        now = datetime.utcnow()
        for topic, entry in knowledge_store.all_topics().items():
            last_updated = entry.get("last_updated")
            if not last_updated:
                continue
            try:
                last_dt = datetime.fromisoformat(last_updated.replace("Z", ""))
            except Exception:
                continue
            days_since = (now - last_dt).total_seconds() / 86400.0
            if days_since < decay_days:
                continue
            updated = self.mastery_engine.apply_decay(entry, days_since)
            knowledge_store.set_topic(topic, updated)
            updates += 1
        return updates

    def _apply_topic(
        self,
        knowledge_store: KnowledgeStore,
        topic: str,
        evidence: str,
        score: float | None,
        weight: float,
    ) -> int:
        entry = knowledge_store.get_topic(topic)
        updated = self.mastery_engine.apply_completion(entry, evidence, score, weight)
        knowledge_store.set_topic(topic, updated)
        return 1
