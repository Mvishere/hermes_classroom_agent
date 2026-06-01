"""Modular RAG pipeline: intent detection, retrieval, and response generation.

This module provides a small, extensible pipeline used by the factual tool
to perform dynamic, data-driven answers without hardcoded user data.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import math
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
    """Detects high-level intent from a user question."""

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
        # fallback to search
        return "search"


class Retriever:
    """Fetches and ranks items from the JsonStore using keyword and optional semantic matching.

    This class is intentionally small but extensible. It never hardcodes any data
    and only returns items found in local JSON files.
    """

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
            # cosine similarity
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
            # weight keyword higher when present; semantic helps reorder
            score = k * 1.0 + s * 2.0
            combined.append((item, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class ResponseGenerator:
    """Builds human-readable responses from retrieved results.

    All responses are generated from data only; no static answers about user data
    are embedded in the code.
    """

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


class RagPipeline:
    def __init__(self, data_dir: str):
        self.retriever = Retriever(data_dir)
        self.store = self.retriever.store

    def handle(self, question: str) -> Optional[str]:
        intent = IntentDetector.detect(question)
        q = question.lower()
        if intent == "list_courses":
            return ResponseGenerator.list_courses(self.store)
        if intent == "count_items":
            # infer item type
            for it in ("announcements", "materials", "assignments"):
                if it.rstrip('s') in q or it in q:
                    return ResponseGenerator.count_items(self.store, it)
            # fallback to aggregated
            counts = {it: len(self.store.get_all_items(it)) for it in ("announcements", "materials", "assignments")}
            return ", ".join(f"{v} {k}" for k, v in counts.items())
        if intent == "latest_item":
            if "announcement" in q:
                return ResponseGenerator.latest_item(self.store, "announcements")
            return ResponseGenerator.latest_item(self.store, "materials")
        if intent == "mention_count":
            # extract terms after 'mention'
            match = re.search(r"mention(?:s)?(?:\s+\w+)*\s+(.+)$", question, re.I)
            terms = []
            if match:
                terms = [t for t in re.findall(r"[a-z0-9]+", match.group(1).lower()) if t not in {"how","many","and","or","the","a","an","of","for","to","with","in"}]
            # default to announcements if not explicit
            item_type = "announcements" if "announcement" in q else ("materials" if "material" in q else ("assignments" if "assignment" in q else "announcements"))
            items = self.store.get_all_items(item_type)
            label = item_type[:-1] if item_type.endswith('s') else item_type
            if not terms:
                return ResponseGenerator.count_items(self.store, item_type)
            return ResponseGenerator.mention_counts(items, terms, label)
        if intent == "topic_query":
            graph = TopicGraph(self.store.data_dir / "topic_graph.json")
            # delegate to TopicGraph-based logic for prerequisites/related
            # simple reuse: find best topic and return its data
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

        # fallback search
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
"""RAG pipeline for summarizing new materials offline."""

import logging
import time
from pathlib import Path
from typing import List

import config
from rag.attachments import AttachmentTextExtractor
from rag.embeddings import EmbeddingModel
from rag.index import EmbeddingIndex
from rag.llm import LocalLLM
from storage.json_store import JsonStore
from storage.summary_store import SummaryStore


class RagPipeline:
    """Coordinates retrieval and summarization for Classroom materials."""

    def __init__(self):
        self.enabled = config.RAG_ENABLED
        self.embedding_model = None
        self.embedding_index = None
        self.llm = None
        self.summary_store = SummaryStore(config.RAG_SUMMARIES_PATH)
        self.attachment_extractor = AttachmentTextExtractor(
            config.BASE_DIR, max_chars=config.PDF_EXTRACT_MAX_CHARS
        )

        if not self.enabled:
            logging.info("RAG pipeline disabled.")
            return

        if not config.EMBEDDING_MODEL_PATH or not config.LLM_MODEL_PATH:
            logging.warning("RAG model paths are not configured.")
            self.enabled = False
            return

        if not Path(config.EMBEDDING_MODEL_PATH).exists():
            logging.warning("Embedding model path not found: %s", config.EMBEDDING_MODEL_PATH)
            self.enabled = False
            return

        # For backends such as Ollama the model is referenced by name (e.g. "qwen2.5:7b")
        # and will not have a local filesystem path. Skip existence check in that case.
        if config.LLM_BACKEND != "ollama" and not Path(config.LLM_MODEL_PATH).exists():
            logging.warning("LLM model path not found: %s", config.LLM_MODEL_PATH)
            self.enabled = False
            return

        try:
            self.embedding_model = EmbeddingModel(
                config.EMBEDDING_MODEL_PATH, device=config.RAG_DEVICE
            )
            self.embedding_index = EmbeddingIndex(
                config.RAG_INDEX_PATH, self.embedding_model
            )
            self.llm = LocalLLM(
                config.LLM_MODEL_PATH,
                device=config.RAG_DEVICE,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
        except Exception:
            logging.exception("Failed to initialize RAG models.")
            self.enabled = False

    def process_new_materials(
        self, json_store: JsonStore, material_ids: List[str]
    ) -> int:
        if not self.enabled:
            return 0
        if not material_ids:
            return 0

        materials = json_store.get_items_by_ids("materials", material_ids)
        if not materials:
            return 0

        added = self.embedding_index.upsert_items(materials, self._build_text)
        if added:
            logging.info("RAG index updated with %s new materials.", added)

        summaries_created = 0
        for material in materials:
            if self.summary_store.has_summary(material):
                continue
            logging.info(
                "Summarizing material %s (%s)",
                material.get("title", ""),
                material.get("id", ""),
            )
            query = self._build_query(material)
            contexts = self.embedding_index.search(query, config.RAG_TOP_K)
            summary, formats = self._summarize_material(material, contexts)
            self.summary_store.upsert_summary(material, summary, contexts, formats)
            summaries_created += 1

        return summaries_created

    def _build_text(self, material: dict) -> str:
        attachments = material.get("attachment_paths", [])
        attachment_text = ", ".join(attachments)
        extracted_text = ""
        if config.PDF_EXTRACT_ENABLED:
            extracted_text, _ = self.attachment_extractor.extract(attachments)
        return (
            f"Course: {material.get('course_name', '')}\n"
            f"Title: {material.get('title', '')}\n"
            f"Description: {material.get('description', '')}\n"
            f"Attachments: {attachment_text}\n"
            f"Extracted Text: {extracted_text}"
        )

    def _build_query(self, material: dict) -> str:
        return f"{material.get('title', '')} {material.get('description', '')}".strip()

    def _build_prompt(self, material: dict, contexts: List[dict]) -> tuple:
        context_block = self._build_context_block(contexts)
        attachments = material.get("attachment_paths", [])
        extracted_text = ""
        formats = []
        if config.PDF_EXTRACT_ENABLED:
            extracted_text, formats = self.attachment_extractor.extract(attachments)
        prompt = self._build_prompt_text(
            material,
            formats,
            extracted_text,
            context_block,
            "You are a study assistant. Summarize the material in 3-5 bullets.\n"
            "Keep it concise and focus on actionable points.",
        )
        return prompt, formats

    def _summarize_material(self, material: dict, contexts: List[dict]) -> tuple[str, List[str]]:
        started_at = time.perf_counter()
        attachments = material.get("attachment_paths", [])
        formats: List[str] = []
        chunk_texts: List[str] = []
        if config.PDF_EXTRACT_ENABLED:
            chunk_texts, formats = self.attachment_extractor.extract_chunks(
                attachments,
                chunk_size=config.RAG_CHUNK_SIZE_CHARS,
                overlap=config.RAG_CHUNK_OVERLAP_CHARS,
                max_chars=config.RAG_EXTRACT_MAX_CHARS,
            )

        if not chunk_texts:
            logging.info(
                "No chunked attachment text for %s; summarizing from prompt context only.",
                material.get("title", ""),
            )
            prompt, formats = self._build_prompt(material, contexts)
            summary = self.llm.generate(prompt)
            logging.info(
                "Summarized %s in %.2fs using non-chunked prompt.",
                material.get("title", ""),
                time.perf_counter() - started_at,
            )
            return summary, formats

        total_chunks = len(chunk_texts)
        chunk_summaries: List[str] = []
        for index, chunk in enumerate(chunk_texts, start=1):
            chunk_started_at = time.perf_counter()
            logging.info(
                "Summarizing chunk %s/%s for %s (%s chars)",
                index,
                total_chunks,
                material.get("title", ""),
                len(chunk),
            )
            prompt = self._build_chunk_prompt(material, formats, chunk, index, total_chunks)
            summary = self.llm.generate(prompt)
            if summary:
                chunk_summaries.append(summary)
            logging.info(
                "Finished chunk %s/%s for %s in %.2fs",
                index,
                total_chunks,
                material.get("title", ""),
                time.perf_counter() - chunk_started_at,
            )

        if not chunk_summaries:
            prompt, formats = self._build_prompt(material, contexts)
            summary = self.llm.generate(prompt)
            logging.info(
                "Summarized %s in %.2fs using fallback prompt.",
                material.get("title", ""),
                time.perf_counter() - started_at,
            )
            return summary, formats

        final_summary = self._reduce_summaries(material, formats, contexts, chunk_summaries)
        logging.info(
            "Completed summarizing %s in %.2fs",
            material.get("title", ""),
            time.perf_counter() - started_at,
        )
        return final_summary, formats

    def _build_context_block(self, contexts: List[dict]) -> str:
        if not contexts:
            return ""
        context_lines = []
        for entry in contexts:
            context_lines.append(
                f"- {entry.get('course_name', '')}: {entry.get('text', '')}"
            )
        context_block = "\n".join(context_lines)
        max_chars = config.RAG_CONTEXT_MAX_CHARS
        if max_chars and len(context_block) > max_chars:
            context_block = context_block[:max_chars]
        return context_block

    def _build_prompt_text(
        self,
        material: dict,
        formats: List[str],
        extracted_text: str,
        context_block: str,
        instructions: str,
    ) -> str:
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
        ]
        if extracted_text:
            parts.append(f"Extracted Attachment Text:\n{extracted_text}")
        if context_block:
            parts.append("Retrieved Context:\n" + context_block)
        parts.append("Summary:\n")
        return "\n\n".join(parts)

    def _build_chunk_prompt(
        self,
        material: dict,
        formats: List[str],
        chunk_text: str,
        index: int,
        total: int,
    ) -> str:
        instructions = (
            "You are a study assistant. Summarize this excerpt in 3-5 bullets.\n"
            "Focus on key details without repeating boilerplate."
        )
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
            f"Excerpt {index}/{total}:\n{chunk_text}",
            "Summary:\n",
        ]
        return "\n\n".join(parts)

    def _build_reduce_prompt(
        self,
        material: dict,
        formats: List[str],
        summary_text: str,
        context_block: str,
    ) -> str:
        instructions = (
            "You are a study assistant. Combine the bullet summaries into 3-5 bullets.\n"
            "Keep it concise and remove duplicates."
        )
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
            "Chunk Summaries:\n" + summary_text,
        ]
        if context_block:
            parts.append("Retrieved Context:\n" + context_block)
        parts.append("Summary:\n")
        return "\n\n".join(parts)

    def _reduce_summaries(
        self,
        material: dict,
        formats: List[str],
        contexts: List[dict],
        summaries: List[str],
    ) -> str:
        started_at = time.perf_counter()
        max_chars = config.RAG_COMBINE_MAX_CHARS
        current = [summary.strip() for summary in summaries if summary and summary.strip()]
        if not current:
            return ""

        while len(current) > 1 and max_chars and len(self._summaries_to_text(current)) > max_chars:
            batches = self._batch_summaries(current, max_chars)
            reduced = []
            logging.info(
                "Reducing %s chunk summaries into %s batch(es) for %s",
                len(current),
                len(batches),
                material.get("title", ""),
            )
            for batch in batches:
                prompt = self._build_reduce_prompt(
                    material,
                    formats,
                    self._summaries_to_text(batch),
                    "",
                )
                batch_started_at = time.perf_counter()
                reduced_summary = self.llm.generate(prompt)
                if reduced_summary:
                    reduced.append(reduced_summary)
                logging.info(
                    "Finished reduction batch for %s in %.2fs",
                    material.get("title", ""),
                    time.perf_counter() - batch_started_at,
                )
            current = reduced
            if not current:
                return ""

        context_block = self._build_context_block(contexts)
        final_prompt = self._build_reduce_prompt(
            material,
            formats,
            self._summaries_to_text(current),
            context_block,
        )
        final_summary = self.llm.generate(final_prompt)
        logging.info(
            "Final reduction for %s completed in %.2fs",
            material.get("title", ""),
            time.perf_counter() - started_at,
        )
        return final_summary

    def _summaries_to_text(self, summaries: List[str]) -> str:
        lines = []
        for summary in summaries:
            text = summary.strip()
            if not text:
                continue
            if text.startswith("-") or text.startswith("*"):
                lines.append(text)
            else:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def _batch_summaries(self, summaries: List[str], max_chars: int) -> List[List[str]]:
        batches: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for summary in summaries:
            entry = summary.strip()
            if not entry:
                continue
            entry_len = len(entry) + 3
            if current and current_len + entry_len > max_chars:
                batches.append(current)
                current = [entry]
                current_len = entry_len
            else:
                current.append(entry)
                current_len += entry_len
        if current:
            batches.append(current)
        return batches

if __name__ == "__main__":
    pipeline = RagPipeline()
    pipeline.process_new_materials(JsonStore(config.BASE_DIR), [])