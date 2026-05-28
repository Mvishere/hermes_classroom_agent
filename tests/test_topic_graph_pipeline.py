from __future__ import annotations

import json
from pathlib import Path

from rag.topic_extractor import TopicExtractor
from storage.topics_store import TopicsStore
from student.knowledge_store import KnowledgeStore
from student.topic_graph import TopicGraph
from student.topic_graph_builder import TopicGraphBuilder
from topic_graph.cleaner import TopicCleaner
from topic_graph.normalizer import TopicNormalizer


class FakeEmbeddingModel:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(word in lowered for word in ("responsive", "media", "css", "flexbox", "grid", "javascript", "dom", "event")):
                vectors.append([1.0, 1.0, 1.0])
            else:
                vectors.append([0.1, 0.1, 0.1])
        return vectors


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "primary_topic": "Responsive Web Design",
                "subtopics": ["Media Queries", "Flexbox"],
                "prerequisites": ["CSS Basics"],
                "related_topics": ["CSS Grid"],
                "domain": "Web Development",
                "difficulty": "medium",
            }
        )


def test_topic_cleaner_filters_generic_terms() -> None:
    cleaner = TopicCleaner(min_frequency=1)
    candidates = ["what", "when", "Functions", "queries", "page", "Responsive Design", "JavaScript Events"]

    cleaned = cleaner.filter_candidates(candidates)

    assert "what" not in cleaned
    assert "when" not in cleaned
    assert "page" not in cleaned
    assert "function" in cleaned
    assert "query" in cleaned
    assert "responsive design" in cleaned
    assert "javascript event" in cleaned


def test_topic_normalizer_merges_plural_variants() -> None:
    normalizer = TopicNormalizer()

    merged = normalizer.merge(["function", "functions", "query", "queries", "event", "events"])

    assert merged == ["Function", "Query", "Event"]


def test_topic_extractor_builds_structured_payload(tmp_path: Path) -> None:
    extractor = TopicExtractor(
        tmp_path,
        None,
        llm=FakeLLM(),
        embedding_model=FakeEmbeddingModel(),
        keyword_limit=6,
        max_chars=2000,
    )

    item = {
        "id": "m1",
        "course_id": "c1",
        "course_name": "Web Design",
        "title": "Responsive web design and media queries",
        "description": "Build responsive layouts with CSS and flexbox.",
        "attachment_paths": [],
    }

    payload = extractor.extract(item, "materials")

    assert payload["primary_topic"] == "Responsive Design"
    assert payload["domain"] == "Web Development"
    assert "Media Queries" in payload["subtopics"]
    assert "Flexbox" in payload["subtopics"]
    assert "CSS Basics" in payload["prerequisites"]
    assert payload["related_topic_edges"]
    assert payload["topics"]


def test_topic_graph_builder_creates_weighted_hierarchy(tmp_path: Path) -> None:
    topics_dir = tmp_path / "topics"
    graph_path = tmp_path / "topic_graph.json"
    knowledge_path = tmp_path / "knowledge.json"

    topics_store = TopicsStore(topics_dir)
    knowledge_store = KnowledgeStore(knowledge_path)
    topic_graph = TopicGraph(graph_path)
    builder = TopicGraphBuilder(topic_graph, embedding_model=FakeEmbeddingModel(), debug=True)

    topics_store.upsert_topics(
        {
            "item_id": "m1",
            "item_type": "materials",
            "course_id": "c1",
            "course_name": "Web Design",
            "title": "Responsive web design and media queries",
            "primary_topic": "Responsive Design",
            "topics": ["Responsive Design", "Media Queries", "Flexbox"],
            "prerequisites": ["CSS Basics"],
            "subtopics": ["Media Queries", "Flexbox"],
            "related_topics": ["CSS Grid"],
            "related_topic_edges": [{"topic": "CSS Grid", "weight": 0.83}],
            "domain": "Web Development",
            "difficulty": "medium",
        }
    )

    builder.rebuild(topics_store, knowledge_store)

    node = topic_graph.get("Responsive Design")
    assert node["domain"] == "Web Development"
    assert node["difficulty"] == "medium"
    assert "CSS Basics" in node["prerequisites"]
    assert "Media Queries" in node["subtopics"]
    assert any(edge["topic"] == "CSS Grid" and edge["weight"] >= 0.8 for edge in node["related_topics"])
    assert graph_path.exists()
