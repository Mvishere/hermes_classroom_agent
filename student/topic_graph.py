"""Topic relationship graph storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class TopicGraph:
    """Loads and provides access to topic relationships."""

    def __init__(self, graph_path: Path):
        self.graph_path = Path(graph_path)
        self._graph = self._load()

    def _load(self) -> Dict[str, dict]:
        if not self.graph_path.exists():
            return {}
        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        if isinstance(data, dict) and isinstance(data.get("topics"), dict):
            data = data.get("topics", {})
        if not isinstance(data, dict):
            return {}

        normalized: Dict[str, dict] = {}
        for topic, details in data.items():
            if not isinstance(details, dict):
                details = {}
            normalized[topic] = {
                "prerequisites": self._normalize_list(details.get("prerequisites", [])),
                "related_topics": self._normalize_list(details.get("related_topics", [])),
            }
        return normalized

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(
            json.dumps(self._graph, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def replace(self, graph: Dict[str, dict]) -> None:
        self._graph = dict(graph)
        self.save()

    def get(self, topic: str) -> dict:
        return dict(self._graph.get(topic, {}))

    def prerequisites(self, topic: str) -> List[str]:
        return list(self._graph.get(topic, {}).get("prerequisites", []))

    def related_topics(self, topic: str) -> List[str]:
        return list(self._graph.get(topic, {}).get("related_topics", []))

    def all_topics(self) -> List[str]:
        return sorted(self._graph.keys())

    def _normalize_list(self, values: List[str]) -> List[str]:
        cleaned = []
        for value in values or []:
            text = str(value).strip()
            if text:
                cleaned.append(text)
        return list(dict.fromkeys(cleaned))
