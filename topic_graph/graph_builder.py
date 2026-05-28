"""Builds a semantic educational topic graph from extracted item payloads."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import config
from rag.embeddings import EmbeddingModel
from storage.topics_store import TopicsStore
from student.knowledge_store import KnowledgeStore
from topic_graph.graph_storage import TopicGraphStore
from topic_graph.normalizer import TopicNormalizer
from topic_graph.ontology_mapper import OntologyMapper
from topic_graph.relationship_builder import RelationshipBuilder


class SemanticTopicGraphBuilder:
    """Aggregates structured topic payloads into a hierarchical weighted graph."""

    def __init__(
        self,
        graph: TopicGraphStore,
        embedding_model: Optional[EmbeddingModel] = None,
        min_edge_weight: float = 0.65,
        max_related: int = 6,
        debug: bool = False,
    ) -> None:
        self.graph = graph
        self.normalizer = TopicNormalizer()
        self.ontology = OntologyMapper(self.normalizer)
        self.relationship_builder = RelationshipBuilder(
            embedding_model=embedding_model,
            normalizer=self.normalizer,
            min_weight=min_edge_weight,
            max_related=max_related,
            debug=debug,
        )
        self.debug = debug

    def rebuild(self, topics_store: TopicsStore, knowledge_store: KnowledgeStore) -> None:
        payloads = topics_store.list_payloads()
        if not payloads:
            return

        nodes: Dict[str, dict] = {}
        topic_counts = Counter()
        for payload in payloads:
            item_type = payload.get("item_type")
            if item_type == "announcements":
                continue
            if item_type not in {"materials", "assignments"}:
                continue

            primary = self._primary_topic(payload)
            if not primary:
                continue
            node = nodes.setdefault(
                primary,
                {
                    "canonical_name": self.normalizer.canonicalize(primary),
                    "display_name": primary,
                    "domain": payload.get("domain") or self.ontology.domain_for(primary),
                    "difficulty": payload.get("difficulty", "unknown"),
                    "parent_topic": self._parent_topic(primary, payload),
                    "prerequisites": [],
                    "subtopics": [],
                    "related_topics": [],
                    "related_topics_raw": [],
                    "source_topics": [],
                    "evidence_count": 0,
                },
            )
            node["evidence_count"] += 1
            node["source_topics"].append(payload.get("item_id", ""))
            topic_counts[primary] += 1

            for topic in self._clean_topic_list(payload.get("prerequisites", [])):
                self._add_topic(node["prerequisites"], topic)
            for topic in self._clean_topic_list(payload.get("subtopics", [])):
                self._add_topic(node["subtopics"], topic)
            for topic in self._clean_topic_list(payload.get("related_topics", [])):
                self._add_topic(node["related_topics_raw"], topic)

            for related in payload.get("related_topic_edges", []) or []:
                if isinstance(related, dict) and related.get("topic"):
                    node["related_topics"].append(
                        {"topic": self.normalizer.display_name(related.get("topic", "")), "weight": float(related.get("weight", 0.0))}
                    )

            if payload.get("domain") and node["domain"] == "General":
                node["domain"] = payload.get("domain")
            if payload.get("difficulty") and node["difficulty"] == "unknown":
                node["difficulty"] = payload.get("difficulty")

        if not nodes:
            return

        relationship_hints = {
            topic: node.get("related_topics_raw", []) + node.get("subtopics", [])
            for topic, node in nodes.items()
        }
        weighted_relations = self.relationship_builder.build(nodes.keys(), relationship_hints)
        for topic, edges in weighted_relations.items():
            node = nodes.get(topic)
            if not node:
                continue
            existing = {item["topic"]: item["weight"] for item in node.get("related_topics", []) if isinstance(item, dict)}
            for edge in edges:
                existing[edge.topic] = max(existing.get(edge.topic, 0.0), edge.weight)
            node["related_topics"] = [
                {"topic": related_topic, "weight": round(weight, 2)}
                for related_topic, weight in sorted(existing.items(), key=lambda item: (-item[1], item[0].lower()))
                if weight >= self.relationship_builder.min_weight
            ][: self.relationship_builder.max_related]

        graph_payload = {
            topic: self._finalize_node(topic, node)
            for topic, node in nodes.items()
        }
        if graph_payload:
            logging.info("Semantic topic graph rebuilt with %s topics.", len(graph_payload))
            if self.debug:
                logging.info("Semantic topic graph topics: %s", ", ".join(sorted(graph_payload.keys())))
            self.graph.replace(graph_payload)

    def _primary_topic(self, payload: dict) -> str:
        for key in ("primary_topic", "title"):
            value = payload.get(key)
            if value:
                mapped = self.ontology.map_topic(value)
                if mapped:
                    return mapped
        topics = payload.get("topics", []) or []
        if topics:
            return self.ontology.map_topic(topics[0])
        return ""

    def _parent_topic(self, topic: str, payload: dict) -> str:
        parent = self.ontology.parent_for(topic)
        if parent:
            return parent
        subject_areas = payload.get("subject_areas", []) or []
        if subject_areas:
            return subject_areas[0]
        return payload.get("domain", "") or ""

    def _finalize_node(self, topic: str, node: dict) -> dict:
        prerequisites = self.normalizer.merge(node.get("prerequisites", []))
        subtopics = self.normalizer.merge(node.get("subtopics", []))
        related_topics = self._normalize_related(node.get("related_topics", []))
        if not related_topics and node.get("related_topics_raw"):
            related_topics = [
                {"topic": related, "weight": 0.75}
                for related in self.normalizer.merge(node.get("related_topics_raw", []))
            ]
        return {
            "canonical_name": self.normalizer.canonicalize(topic),
            "display_name": topic,
            "domain": node.get("domain", "General") or "General",
            "difficulty": node.get("difficulty", "unknown") or "unknown",
            "parent_topic": node.get("parent_topic", "") or "",
            "prerequisites": prerequisites,
            "subtopics": subtopics,
            "related_topics": related_topics,
            "related_topics_raw": self.normalizer.merge(node.get("related_topics_raw", [])),
            "source_topics": self.normalizer.merge(node.get("source_topics", [])),
            "evidence_count": int(node.get("evidence_count", 0) or 0),
        }

    def _normalize_related(self, related: Iterable) -> List[dict]:
        normalized: List[dict] = []
        for item in related or []:
            if isinstance(item, dict) and item.get("topic"):
                topic = self.normalizer.display_name(item.get("topic", ""))
                if not topic:
                    continue
                try:
                    weight = float(item.get("weight", 0.0))
                except Exception:
                    weight = 0.0
                if weight >= self.relationship_builder.min_weight:
                    normalized.append({"topic": topic, "weight": round(weight, 2)})
            elif item:
                topic = self.normalizer.display_name(item)
                if topic:
                    normalized.append({"topic": topic, "weight": 0.75})
        seen = set()
        result: List[dict] = []
        for item in sorted(normalized, key=lambda edge: (-edge["weight"], edge["topic"].lower())):
            if item["topic"] in seen:
                continue
            seen.add(item["topic"])
            result.append(item)
        return result[: self.relationship_builder.max_related]

    def _clean_topic_list(self, topics: Iterable[str]) -> List[str]:
        return self.normalizer.merge(topics)

    def _add_topic(self, target: List[str], topic: str) -> None:
        if topic and topic not in target:
            target.append(topic)
