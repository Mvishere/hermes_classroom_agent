"""Deterministic topic graph engine for prerequisite and relation queries."""

from __future__ import annotations

from pathlib import Path

from engines.common import EngineResult
from student.topic_graph import TopicGraph


class TopicGraphEngine:
    GROUNDED_FALLBACK = "I don't yet have enough grounded topic relationships to answer that reliably."

    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        self.graph = TopicGraph(self.graph_path)

    def answer(self, question: str) -> EngineResult:
        topic = self._best_topic_match(question)
        if not topic:
            return self._fallback("topic:not_found")

        node = self.graph.get(topic)
        if not node:
            return self._fallback(f"topic:{topic}:missing")

        lowered = question.lower()
        prerequisites = node.get("prerequisites", []) or []
        related_topics = self._related_topics(node)
        parent_topic = node.get("parent_topic", "")
        subtopics = node.get("subtopics", []) or []

        if any(marker in lowered for marker in ("prerequisite", "before", "what should i learn before", "foundational")):
            if not prerequisites:
                return self._fallback(f"topic:{topic}:prerequisite:insufficient")
            chain = self._walk_prerequisites(topic, seen=set())
            answer = f"Before {topic}, review: {', '.join(chain)}."
            return EngineResult(answer=answer, confidence=0.95, evidence_source=f"topic_graph:{topic}:prerequisite", matched_documents=[node], engine="topic_graph")

        if any(marker in lowered for marker in ("related", "connected", "associated", "similar")):
            if not related_topics:
                return self._fallback(f"topic:{topic}:related:insufficient")
            return EngineResult(
                answer=f"Related to {topic}: {', '.join(related_topics)}.",
                confidence=0.9,
                evidence_source=f"topic_graph:{topic}:related",
                matched_documents=[node],
                engine="topic_graph",
            )

        if any(marker in lowered for marker in ("learning path", "what next", "recommend", "next concept")):
            path = self._learning_path(topic)
            if not path:
                return self._fallback(f"topic:{topic}:path:insufficient")
            return EngineResult(
                answer=f"A grounded path for {topic}: {', '.join(path)}.",
                confidence=0.86,
                evidence_source=f"topic_graph:{topic}:learning_path",
                matched_documents=[node],
                engine="topic_graph",
            )

        if parent_topic or subtopics:
            details = []
            if parent_topic:
                details.append(f"parent topic is {parent_topic}")
            if subtopics:
                details.append("subtopics include " + ", ".join(subtopics[:6]))
            return EngineResult(
                answer=f"For {topic}, {', '.join(details)}.",
                confidence=0.7,
                evidence_source=f"topic_graph:{topic}:metadata",
                matched_documents=[node],
                engine="topic_graph",
            )

        return self._fallback(f"topic:{topic}:no_rule")

    def _fallback(self, source: str) -> EngineResult:
        return EngineResult(
            answer=self.GROUNDED_FALLBACK,
            confidence=0.25,
            evidence_source=source,
            matched_documents=[],
            engine="topic_graph",
        )

    def _best_topic_match(self, question: str) -> str:
        lowered = question.lower()
        best_topic = ""
        best_score = 0
        for topic in self.graph.all_topics():
            topic_lower = topic.lower()
            score = 0
            if topic_lower in lowered:
                score += 3
            score += sum(1 for token in topic_lower.split() if len(token) > 2 and token in lowered)
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic

    def _related_topics(self, node: dict) -> list[str]:
        related = node.get("related_topics", []) or []
        if related and isinstance(related[0], dict):
            return [item.get("topic") for item in related if item.get("topic")]
        return [str(item) for item in related if item]

    def _walk_prerequisites(self, topic: str, seen: set[str]) -> list[str]:
        if topic in seen:
            return []
        seen.add(topic)
        node = self.graph.get(topic)
        if not node:
            return []
        direct = node.get("prerequisites", []) or []
        ordered: list[str] = []
        for prereq in direct:
            if prereq not in ordered:
                ordered.append(prereq)
            child_path = self._walk_prerequisites(prereq, seen)
            for child in child_path:
                if child not in ordered:
                    ordered.append(child)
        return ordered[:10]

    def _learning_path(self, topic: str) -> list[str]:
        node = self.graph.get(topic)
        if not node:
            return []
        path = []
        parent = node.get("parent_topic", "")
        prerequisites = node.get("prerequisites", []) or []
        if parent:
            path.append(parent)
        path.extend(prerequisites[:5])
        path.extend(self._related_topics(node)[:3])
        return [item for item in path if item]
