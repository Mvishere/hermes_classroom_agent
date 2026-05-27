"""Utilities for mapping topics to related concepts."""

from __future__ import annotations

from typing import Iterable, List, Set

from student.topic_graph import TopicGraph


class TopicMapper:
    """Expands topics using a topic relationship graph."""

    def __init__(self, graph: TopicGraph):
        self.graph = graph

    def prerequisites_for_topics(self, topics: Iterable[str]) -> List[str]:
        prerequisites: Set[str] = set()
        for topic in topics:
            prerequisites.update(self.graph.prerequisites(topic))
        return sorted(prerequisites)

    def related_for_topics(self, topics: Iterable[str]) -> List[str]:
        related: Set[str] = set()
        for topic in topics:
            related.update(self.graph.related_topics(topic))
        return sorted(related)
