"""Query-focused RAG pipeline: intent detection, retrieval, and response generation.

This file provides a pipeline safe to import alongside the existing
`rag.pipeline` summarization module. It is focused on interactive queries
and ensures answers are generated dynamically from local JSON data.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import re

from storage.json_store import JsonStore
from student.topic_graph import TopicGraph
import config

try:
    from rag.embeddings import EmbeddingModel
    import numpy as np
except Exception:
    EmbeddingModel = None  # type: ignore
    np = None  # type: ignore


class IntentDetector:
    @staticmethod
    def detect(question: str) -> str:
        q = question.lower()
        if any(token in q for token in ("enroll", "which courses", "what courses", "am i enrolled")):
            return "list_courses"
        if "how many" in q and any(x in q for x in ("announcement", "assignment", "material", "materials")):
            return "count_items"
        if any(x in q for x in ("latest announcement", "most recent announcement", "latest material", "most recent material")):
            return "latest_item"
        if "mention" in q and "how many" in q:
            return "mention_count"
        if any(marker in q for marker in ("prerequisite", "prerequisites", "related", "ready", "before learning", "before")):
            return "topic_query"
        return "search"


class Retriever:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.store = JsonStore(self.data_dir)
        self.embedding = None
        if EmbeddingModel and config.EMBEDDING_MODEL_PATH:
            try:
                self.embedding = EmbeddingModel(config.EMBEDDING_MODEL_PATH, device=config.RAG_DEVICE)
            except Exception:
                self.embedding = None

    def _item_text(self, item: dict) -> str:
        pieces = []
        for field in ("title", "description", "text", "content", "summary"):
            v = item.get(field)
            if isinstance(v, str) and v:
                pieces.append(v)
        return " ".join(pieces)

    def _keyword_score(self, query_terms: List[str], text: str) -> int:
        text_lower = text.lower()
        return sum(1 for t in query_terms if t in text_lower)

    def _semantic_score(self, query: str, texts: List[str]) -> Optional[List[float]]:
        if not self.embedding or not np:
            return None
        try:
            q_vec = self.embedding.encode([query])[0]
            t_vecs = self.embedding.encode(texts)
            qv = np.array(q_vec, dtype=float)
            tv = np.array(t_vecs, dtype=float)
            denom = (np.linalg.norm(qv) * np.linalg.norm(tv, axis=1))
            denom[denom == 0] = 1e-9
            sims = (tv @ qv) / denom
            return sims.tolist()
        except Exception:
            return None

    def search(self, query: str, item_types: Optional[List[str]] = None, top_k: int = 5) -> List[Tuple[dict, float]]:
        item_types = item_types or ["courses", "assignments", "materials", "announcements"]
        items: List[dict] = []
        for it in item_types:
            items.extend(self.store.get_all_items(it))

        texts = [self._item_text(i) for i in items]
        query_terms = re.findall(r"[a-z0-9]+", query.lower())

        keyword_scores = [self._keyword_score(query_terms, t) for t in texts]
        semantic_scores = self._semantic_score(query, texts)

        combined: List[Tuple[dict, float]] = []
        for idx, item in enumerate(items):
            k = float(keyword_scores[idx])
            s = float(semantic_scores[idx]) if semantic_scores is not None else 0.0
            score = k * 1.0 + s * 2.0
            combined.append((item, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class ResponseGenerator:
    @staticmethod
    def list_courses(store: JsonStore) -> str:
        path = store.data_dir / "courses" / "courses.json"
        payload = store._load(path, {"courses": {}})
        names = [c.get("name") for c in (payload.get("courses") or {}).values() if c.get("name")]
        if not names:
            return "I could not find any courses in the local data."
        return f"Found {len(names)} course(s): {', '.join(names)}."

    @staticmethod
    def count_items(store: JsonStore, item_type: str) -> str:
        items = store.get_all_items(item_type)
        return f"{len(items)} {item_type} found locally."

    @staticmethod
    def latest_item(store: JsonStore, item_type: str) -> str:
        items = store.get_all_items(item_type)
        if not items:
            return f"No {item_type} found in local data."
        latest = sorted(items, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        title = latest.get("title") or latest.get("name") or "Untitled"
        return f"Latest {item_type[:-1] if item_type.endswith('s') else item_type}: {title}."

    @staticmethod
    def mention_counts(items: List[dict], terms: List[str], label: str) -> str:
        counts = sum(1 for item in items if any(t in (str(item.get(k, "")).lower()) for k in ("title", "description", "text") for t in terms))
        return f"{counts} {label} mention {', '.join(terms)}."

    @staticmethod
    def format_search_results(results: List[Tuple[dict, float]]) -> str:
        if not results:
            return "No matching items found."
        parts = []
        for item, score in results[:5]:
            title = item.get("title") or item.get("name") or "Untitled"
            course = item.get("course_name") or item.get("course", "")
            parts.append(f"{title} (score={score:.2f}{', course='+course if course else ''})")
        return "Matches: " + ", ".join(parts) + "."


class RagQueryPipeline:
    def __init__(self, data_dir: str):
        self.retriever = Retriever(data_dir)
        self.store = self.retriever.store

    def handle(self, question: str) -> Optional[str]:
        intent = IntentDetector.detect(question)
        q = question.lower()
        if intent == "list_courses":
            return ResponseGenerator.list_courses(self.store)
        if intent == "count_items":
            for it in ("announcements", "materials", "assignments"):
                if it.rstrip('s') in q or it in q:
                    return ResponseGenerator.count_items(self.store, it)
            counts = {it: len(self.store.get_all_items(it)) for it in ("announcements", "materials", "assignments")}
            return ", ".join(f"{v} {k}" for k, v in counts.items())
        if intent == "latest_item":
            if "announcement" in q:
                return ResponseGenerator.latest_item(self.store, "announcements")
            return ResponseGenerator.latest_item(self.store, "materials")
        if intent == "mention_count":
            match = re.search(r"mention(?:s)?(?:\s+\w+)*\s+(.+)$", question, re.I)
            terms = []
            if match:
                terms = [t for t in re.findall(r"[a-z0-9]+", match.group(1).lower()) if t not in {"how","many","and","or","the","a","an","of","for","to","with","in"}]
            item_type = "announcements" if "announcement" in q else ("materials" if "material" in q else ("assignments" if "assignment" in q else "announcements"))
            items = self.store.get_all_items(item_type)
            label = item_type[:-1] if item_type.endswith('s') else item_type
            if not terms:
                return ResponseGenerator.count_items(self.store, item_type)
            return ResponseGenerator.mention_counts(items, terms, label)
        if intent == "topic_query":
            graph = TopicGraph(self.store.data_dir / "topic_graph.json")
            topics = graph.all_topics()
            best = self._best_topic_match(question, topics)
            if not best:
                return None
            node = graph.get(best)
            if not node:
                return None
            if "related" in q:
                related = [r.get("topic") for r in node.get("related_topics", []) if isinstance(r, dict)]
                return f"Related to {best}: {', '.join(related)}." if related else f"No related topics for {best}."
            if any(marker in q for marker in ("prerequisite", "prerequisites", "before")):
                prereqs = node.get("prerequisites", []) or []
                return f"Prerequisites for {best}: {', '.join(prereqs)}." if prereqs else f"No prerequisites recorded for {best}."
            return None

        results = self.retriever.search(question, top_k=config.RAG_TOP_K)
        return ResponseGenerator.format_search_results(results)

    def _best_topic_match(self, question: str, topics: List[str]) -> Optional[str]:
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        best = None
        best_score = 0
        for t in topics:
            tokens = set(re.findall(r"[a-z0-9]+", t.lower()))
            score = len(q_tokens & tokens)
            if t.lower() in question.lower():
                score += 3
            if score > best_score:
                best_score = score
                best = t
        return best
