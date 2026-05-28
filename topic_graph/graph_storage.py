"""Storage format for semantic topic graphs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from topic_graph.normalizer import TopicNormalizer


class TopicGraphStore:
    """Loads and stores a hierarchical topic graph with weighted relationships."""

    def __init__(self, graph_path: Path, normalizer: Optional[TopicNormalizer] = None):
        self.graph_path = Path(graph_path)
        self.normalizer = normalizer or TopicNormalizer()
        self._payload = self._load()

    def _load(self) -> dict:
        if not self.graph_path.exists():
            return self._empty_payload()
        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_payload()

        if isinstance(data, dict) and "topics" in data:
            topics = data.get("topics", {})
            metadata = {k: v for k, v in data.items() if k != "topics"}
        elif isinstance(data, dict):
            topics = data
            metadata = {}
        else:
            return self._empty_payload()

        normalized_topics: Dict[str, dict] = {}
        for topic, details in topics.items():
            normalized_topics[self.normalizer.display_name(topic)] = self._normalize_node(details)
        payload = self._empty_payload()
        payload.update(metadata)
        payload["topics"] = normalized_topics
        return payload

    def _empty_payload(self) -> dict:
        return {
            "version": 2,
            "generated_at": "",
            "topics": {},
        }

    def _normalize_node(self, details: dict) -> dict:
        if not isinstance(details, dict):
            details = {}
        prerequisites = self._normalize_list(details.get("prerequisites", []))
        subtopics = self._normalize_list(details.get("subtopics", []))
        related_topics = self._normalize_related(details.get("related_topics", []))
        canonical = self.normalizer.canonicalize(details.get("canonical_name", ""))
        if not canonical and details.get("display_name"):
            canonical = self.normalizer.canonicalize(details.get("display_name", ""))
        node = {
            "canonical_name": canonical,
            "display_name": details.get("display_name", ""),
            "domain": str(details.get("domain", "General") or "General"),
            "difficulty": str(details.get("difficulty", "unknown") or "unknown"),
            "parent_topic": str(details.get("parent_topic", "") or ""),
            "prerequisites": prerequisites,
            "subtopics": subtopics,
            "related_topics": related_topics,
            "related_topics_raw": self._normalize_list(details.get("related_topics_raw", [])),
            "source_topics": self._normalize_list(details.get("source_topics", [])),
            "evidence_count": int(details.get("evidence_count", 0) or 0),
        }
        return node

    def _normalize_related(self, values: Iterable) -> List[dict]:
        normalized: List[dict] = []
        for value in values or []:
            if isinstance(value, dict):
                topic = self.normalizer.display_name(value.get("topic", ""))
                if not topic:
                    continue
                try:
                    weight = float(value.get("weight", 0.0))
                except Exception:
                    weight = 0.0
                normalized.append({"topic": topic, "weight": round(max(0.0, min(1.0, weight)), 2)})
            else:
                topic = self.normalizer.display_name(value)
                if topic:
                    normalized.append({"topic": topic, "weight": 0.75})
        seen = set()
        result: List[dict] = []
        for item in normalized:
            key = item["topic"]
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _normalize_list(self, values: Iterable[str]) -> List[str]:
        cleaned = []
        for value in values or []:
            text = self.normalizer.display_name(value)
            if text:
                cleaned.append(text)
        return list(dict.fromkeys(cleaned))

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._payload["generated_at"] = datetime.utcnow().isoformat() + "Z"
        self.graph_path.write_text(
            json.dumps(self._payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def replace(self, graph: Dict[str, dict]) -> None:
        self._payload = self._empty_payload()
        self._payload["topics"] = {
            self.normalizer.display_name(topic): self._normalize_node(details)
            for topic, details in graph.items()
        }
        self.save()

    def get(self, topic: str) -> dict:
        topic_name = self.normalizer.display_name(topic)
        return dict(self._payload.get("topics", {}).get(topic_name, {}))

    def prerequisites(self, topic: str) -> List[str]:
        return list(self.get(topic).get("prerequisites", []))

    def related_topics(self, topic: str) -> List[str]:
        node = self.get(topic)
        related = node.get("related_topics", [])
        if related and isinstance(related[0], dict):
            return [item["topic"] for item in related]
        return list(related)

    def all_topics(self) -> List[str]:
        return sorted(self._payload.get("topics", {}).keys())

    def nodes(self) -> Dict[str, dict]:
        return dict(self._payload.get("topics", {}))
