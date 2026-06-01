"""Semantic educational topic extraction for Classroom materials and assignments."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config
from rag.attachments import AttachmentTextExtractor
from rag.embeddings import EmbeddingModel
from rag.llm import LocalLLM
from topic_graph.cleaner import TopicCleaner
from topic_graph.normalizer import TopicNormalizer
from topic_graph.ontology_mapper import OntologyMapper


class SemanticTopicExtractor:
    """Extracts educational concepts, hierarchy, and prerequisites from source text."""

    def __init__(
        self,
        base_dir: Path,
        topic_graph: Optional[object] = None,
        llm: Optional[LocalLLM] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        keyword_limit: int = 12,
        max_chars: int = 4000,
        debug: bool = False,
        cleaner: Optional[TopicCleaner] = None,
        normalizer: Optional[TopicNormalizer] = None,
        ontology_mapper: Optional[OntologyMapper] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.topic_graph = topic_graph
        self.llm = llm
        self.keyword_limit = keyword_limit
        self.max_chars = max_chars
        self.debug = debug
        self.normalizer = normalizer or TopicNormalizer()
        self.cleaner = cleaner or TopicCleaner(min_frequency=2, debug=debug)
        self.ontology_mapper = ontology_mapper or OntologyMapper(self.normalizer)
        self.attachment_extractor = AttachmentTextExtractor(
            self.base_dir, max_chars=config.TOPIC_EXTRACT_MAX_CHARS
        )
        self.embedding_model = embedding_model
        self._known_topics = set()
        if self.topic_graph is not None and hasattr(self.topic_graph, "all_topics"):
            try:
                self._known_topics = set(getattr(self.topic_graph, "all_topics")())
            except Exception:
                logging.debug("Failed to read known topics from graph hint.", exc_info=True)

    def extract(self, item: dict, item_type: str) -> dict:
        attachments = item.get("attachment_paths", [])
        extracted_text = ""
        formats: List[str] = []
        if config.PDF_EXTRACT_ENABLED and attachments:
            extracted_text, formats = self.attachment_extractor.extract(attachments)

        source_text = self._build_source_text(item, extracted_text)
        if self.max_chars and len(source_text) > self.max_chars:
            source_text = source_text[: self.max_chars]

        candidate_phrases = self._extract_candidates(source_text)
        heuristic_payload = self._build_structured_payload(source_text, candidate_phrases, item)
        llm_payload = self._extract_with_llm(source_text, heuristic_payload)
        payload = self._merge_payloads(heuristic_payload, llm_payload)

        primary_topic = payload.get("primary_topic", "")
        domain = payload.get("domain") or self.ontology_mapper.domain_for(primary_topic)
        difficulty = payload.get("difficulty") or self._infer_difficulty(source_text)
        subtopics = self._deduplicate_topics(payload.get("subtopics", []))
        prerequisites = self._deduplicate_topics(payload.get("prerequisites", []))
        related_topics = self._deduplicate_topics(payload.get("related_topics", []))

        if primary_topic:
            prerequisites = self._deduplicate_topics(
                self.ontology_mapper.prerequisites_for(primary_topic) + prerequisites
            )
            subtopics = self._deduplicate_topics(
                self.ontology_mapper.subtopics_for(primary_topic) + subtopics
            )
            related_topics = self._deduplicate_topics(
                related_topics + subtopics + prerequisites
            )

        mapped_primary = self.ontology_mapper.map_topic(primary_topic) if primary_topic else ""
        mapped_subtopics = self._deduplicate_topics(self.ontology_mapper.map_topic(topic) for topic in subtopics)
        mapped_prerequisites = self._deduplicate_topics(self.ontology_mapper.map_topic(topic) for topic in prerequisites)
        mapped_related = self._deduplicate_topics(self.ontology_mapper.map_topic(topic) for topic in related_topics)

        mapped_primary = mapped_primary if self._keep_topic(mapped_primary) else ""
        mapped_subtopics = [topic for topic in mapped_subtopics if self._keep_topic(topic)]
        mapped_prerequisites = [topic for topic in mapped_prerequisites if self._keep_topic(topic)]
        mapped_related = [topic for topic in mapped_related if self._keep_topic(topic)]

        topics = self._deduplicate_topics([mapped_primary] + mapped_subtopics)
        concepts = self._deduplicate_topics(topics + mapped_prerequisites + mapped_related)
        skills = self._skills_from_topics(concepts)
        subject_areas = [domain] if domain else []
        related_edges = self._weighted_related_topics(mapped_primary, mapped_related, source_text)

        return {
            "item_id": item.get("id", ""),
            "item_type": item_type,
            "course_id": item.get("course_id", ""),
            "course_name": item.get("course_name", ""),
            "title": item.get("title", ""),
            "primary_topic": mapped_primary,
            "topics": topics,
            "concepts": concepts,
            "subtopics": mapped_subtopics,
            "prerequisites": mapped_prerequisites,
            "related_topics": mapped_related,
            "related_topic_edges": related_edges,
            "skills": skills,
            "subject_areas": subject_areas,
            "domain": domain,
            "difficulty": difficulty,
            "keywords": self._top_keywords(candidate_phrases),
            "candidate_topics": candidate_phrases,
            "attachment_formats": formats,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "method": "semantic_llm" if llm_payload else "semantic_heuristic",
        }

    def _build_source_text(self, item: dict, extracted_text: str) -> str:
        parts = [
            f"Title: {item.get('title', '')}",
            f"Description: {item.get('description', '')}",
        ]
        if extracted_text:
            parts.append("Attachment Text:\n" + extracted_text)
        form_text = self._build_form_text(item)
        if form_text:
            parts.append("Form Text:\n" + form_text)
        return "\n".join(parts).strip()

    def _build_form_text(self, item: dict) -> str:
        form_text = item.get("form_text", "")
        if form_text:
            return str(form_text)

        questions = item.get("form_questions", [])
        if not isinstance(questions, list) or not questions:
            raw_payload = item.get("raw_payload", {}) or {}
            materials = raw_payload.get("materials", []) if isinstance(raw_payload, dict) else []
            for material in materials:
                form = material.get("form") or {}
                form_title = form.get("title") or ""
                form_url = form.get("formUrl") or ""
                if form_title or form_url:
                    return "\n".join(
                        part
                        for part in (
                            f"Form Title: {form_title}" if form_title else "",
                            f"Form URL: {form_url}" if form_url else "",
                        )
                        if part
                    )
            return ""

        lines: list[str] = []
        for index, question in enumerate(questions, start=1):
            if isinstance(question, dict):
                title = str(question.get("title", "")).strip()
                kind = str(question.get("kind", "question")).strip()
                lines.append(f"Question {index}: {title}")
                lines.append(f"Type: {kind}")
                options = question.get("options") or []
                if options:
                    lines.append("Options: " + ", ".join(str(option) for option in options))
            elif question:
                lines.append(f"Question {index}: {question}")
        return "\n".join(lines).strip()

    def _extract_candidates(self, text: str) -> List[str]:
        spans = []
        for sentence in re.split(r"[\n\.\?\!;]+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            spans.extend(self._sentence_candidates(sentence))

        if not spans:
            spans = self._keyword_candidates(text)

        cleaned = self.cleaner.filter_candidates(spans)
        mapped = [self.ontology_mapper.map_topic(candidate) for candidate in cleaned]
        if self.embedding_model:
            mapped = self._merge_by_semantics(mapped)
        mapped = [topic for topic in mapped if self._keep_topic(topic)]
        return self._deduplicate_topics(mapped)[: self.keyword_limit]

    def _sentence_candidates(self, sentence: str) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", sentence)
        if not words:
            return []

        candidates: List[str] = []
        start = 0
        for index, word in enumerate(words + ["."]):
            token = word.lower()
            if token in {"and", "or", "but", "with", "for", "from", "to", "of", "in", "on", "by", "as", "at"} or word == ".":
                if index > start:
                    chunk = words[start:index]
                    candidates.extend(self._phrase_variants(chunk))
                start = index + 1
        if start < len(words):
            candidates.extend(self._phrase_variants(words[start:]))
        return candidates

    def _phrase_variants(self, tokens: List[str]) -> List[str]:
        variants: List[str] = []
        if not tokens:
            return variants
        for size in range(1, min(4, len(tokens)) + 1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                if self.cleaner.looks_like_noun_phrase(phrase):
                    variants.append(phrase)
        return variants

    def _keyword_candidates(self, text: str) -> List[str]:
        tokens = self.cleaner.tokenize(text)
        counts = Counter(tokens)
        ranked = [token for token, _ in counts.most_common(self.keyword_limit * 3)]
        return [token for token in ranked if not self.cleaner.is_noise_token(token)]

    def _top_keywords(self, candidates: List[str]) -> List[str]:
        return [topic for topic in self._deduplicate_topics(candidates) if self._keep_topic(topic)][: self.keyword_limit]

    def _build_structured_payload(self, text: str, candidates: List[str], item: dict) -> dict:
        primary_topic = self._choose_primary_topic(text, candidates, item)
        domain = self.ontology_mapper.domain_for(primary_topic) if primary_topic else "General"
        subtopics = self._infer_subtopics(primary_topic, candidates)
        prerequisites = self._infer_prerequisites(primary_topic, candidates)
        related_topics = [candidate for candidate in candidates if candidate != primary_topic]
        return {
            "primary_topic": primary_topic,
            "domain": domain,
            "difficulty": self._infer_difficulty(text),
            "subtopics": subtopics,
            "prerequisites": prerequisites,
            "related_topics": related_topics,
        }

    def _choose_primary_topic(self, text: str, candidates: List[str], item: dict) -> str:
        title = str(item.get("title", ""))
        title_candidates = self._sentence_candidates(title)
        candidate_pool = self._deduplicate_topics(
            [*title_candidates, *candidates, *self._keyword_candidates(text)]
        )
        if not candidate_pool:
            return ""

        scored = []
        for candidate in candidate_pool:
            display = self.ontology_mapper.map_topic(candidate)
            score = 0.0
            if self.ontology_mapper.is_known(candidate):
                score += 2.0
            if display in self._known_topics:
                score += 1.0
            if len(display.split()) > 1:
                score += 1.0
            if candidate in self._deduplicate_topics(title_candidates):
                score += 0.25
            if any(word in candidate.lower() for word in ("design", "query", "event", "manipulation", "responsive", "grid", "flexbox", "javascript", "css", "html")):
                score += 0.8
            score += min(1.0, len(candidate.split()) * 0.25)
            scored.append((display, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].lower()))
        best = scored[0][0]
        return best if self._keep_topic(best) else ""

    def _infer_subtopics(self, primary_topic: str, candidates: List[str]) -> List[str]:
        if not primary_topic:
            return candidates[: max(0, self.keyword_limit - 1)]
        primary_lower = primary_topic.lower()
        subtopics = [candidate for candidate in candidates if candidate.lower() != primary_lower]
        ontology_subtopics = self.ontology_mapper.subtopics_for(primary_topic)
        return self._deduplicate_topics(ontology_subtopics + subtopics)[: self.keyword_limit]

    def _infer_prerequisites(self, primary_topic: str, candidates: List[str]) -> List[str]:
        prerequisites = []
        if primary_topic:
            prerequisites.extend(self.ontology_mapper.prerequisites_for(primary_topic))
        for candidate in candidates:
            if candidate in prerequisites or candidate == primary_topic:
                continue
            lower = candidate.lower()
            if any(token in lower for token in ("basic", "foundation", "intro", "html", "css", "javascript", "dom")):
                prerequisites.append(candidate)
        return self._deduplicate_topics(prerequisites)[: self.keyword_limit]

    def _extract_with_llm(self, text: str, heuristic_payload: dict) -> dict:
        if not self.llm or not config.TOPIC_EXTRACT_LLM_ENABLED:
            return {}
        from rag.prompts import topic_extraction_prompt

        prompt = topic_extraction_prompt(text, heuristic_payload)
        try:
            response = self.llm.generate(prompt)
        except Exception:
            logging.exception("Topic LLM extraction failed.")
            return {}

        payload = self._parse_json(response)
        if not isinstance(payload, dict):
            return {}

        # Validate and normalize schema
        main = payload.get("main_topics") or []
        sub = payload.get("subtopics") or []
        prereq = payload.get("prerequisites") or []
        domain = str(payload.get("domain") or "").strip()
        difficulty = self._normalize_difficulty(payload.get("difficulty", "unknown"))

        return {
            "primary_topic": self.ontology_mapper.map_topic(main[0]) if main else "",
            "subtopics": self._deduplicate_topics(sub),
            "prerequisites": self._deduplicate_topics(prereq),
            "related_topics": [],
            "domain": domain,
            "difficulty": difficulty,
        }

    def _parse_json(self, response: str) -> dict:
        if not response:
            return {}
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        snippet = response[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}

    def _merge_payloads(self, heuristic_payload: dict, llm_payload: dict) -> dict:
        merged = dict(heuristic_payload)
        if not llm_payload:
            return merged
        for key in ("primary_topic", "domain", "difficulty"):
            value = llm_payload.get(key)
            if value:
                merged[key] = value
        for key in ("subtopics", "prerequisites", "related_topics"):
            merged[key] = self._deduplicate_topics([
                *merged.get(key, []),
                *llm_payload.get(key, []),
            ])
        return merged

    def _keep_topic(self, topic: str) -> bool:
        """Reject heading-like or document-label phrases before they reach storage."""
        topic = self.ontology_mapper.map_topic(topic)
        if not topic:
            return False
        if self.ontology_mapper.is_known(topic):
            return True

        tokens = [token for token in self.cleaner.tokenize(topic) if token]
        if not tokens:
            return False

        if len(tokens) == 1:
            return tokens[0] in {"api", "html", "css", "dom", "js", "ux", "ui", "sql", "ml", "nlp", "oop"}

        concept_tokens = {
            "api",
            "html",
            "css",
            "javascript",
            "js",
            "dom",
            "responsive",
            "grid",
            "flexbox",
            "media",
            "query",
            "queries",
            "function",
            "event",
            "fetch",
            "data",
            "integration",
            "conditional",
            "manipulation",
            "forms",
            "form",
            "tables",
        }
        if not any(token in concept_tokens for token in tokens):
            return False

        heading_tokens = {
            "title",
            "text",
            "note",
            "notes",
            "quiz",
            "assignment",
            "material",
            "announcement",
            "course",
            "classroom",
            "lesson",
            "module",
            "unit",
        }
        if any(token in heading_tokens for token in tokens):
            return False
        if len(tokens) == 1 and tokens[0] in heading_tokens:
            return False
        if len(tokens) == 2 and set(tokens).issubset({"web", "development", "title", "quiz", "text", "note", "notes"}):
            return False
        return True

    def _merge_by_semantics(self, topics: List[str]) -> List[str]:
        if not self.embedding_model or len(topics) < 2:
            return topics
        try:
            embeddings = self.embedding_model.encode(topics)
        except Exception:
            logging.exception("Embedding candidate merging failed.")
            return topics
        vectors = [list(vector) for vector in embeddings]
        merged: List[str] = []
        used = set()
        for index, topic in enumerate(topics):
            if index in used:
                continue
            group = [topic]
            for other_index in range(index + 1, len(topics)):
                if other_index in used:
                    continue
                similarity = float(sum(a * b for a, b in zip(vectors[index], vectors[other_index])))
                if similarity >= 0.86 and self.normalizer.canonicalize(topic) == self.normalizer.canonicalize(topics[other_index]):
                    group.append(topics[other_index])
                    used.add(other_index)
            merged.append(group[0])
        return self._deduplicate_topics(merged)

    def _weighted_related_topics(self, primary_topic: str, related_topics: List[str], text: str) -> List[dict]:
        if not primary_topic:
            return []
        related = []
        for topic in related_topics[: self.keyword_limit]:
            if topic == primary_topic:
                continue
            weight = self._related_weight(primary_topic, topic, text)
            if weight >= 0.65:
                related.append({"topic": topic, "weight": round(weight, 2)})
        related.sort(key=lambda item: (-item["weight"], item["topic"].lower()))
        return related[: self.keyword_limit]

    def _related_weight(self, left: str, right: str, text: str) -> float:
        score = 0.55
        if self.embedding_model:
            try:
                vectors = self.embedding_model.encode([left, right, text])
                left_vec, right_vec = vectors[0], vectors[1]
                score = float(sum(a * b for a, b in zip(left_vec, right_vec)))
            except Exception:
                logging.exception("Related topic similarity failed.")
        if self.ontology_mapper.entry_for(right):
            score += 0.08
        if self.normalizer.canonicalize(left) == self.normalizer.canonicalize(right):
            score = 1.0
        return max(0.0, min(1.0, score))

    def _skills_from_topics(self, concepts: List[str]) -> List[str]:
        skill_words = []
        for concept in concepts:
            lower = concept.lower()
            if any(token in lower for token in ("design", "layout", "event", "query", "function", "responsive", "media", "dom", "flexbox", "grid")):
                skill_words.append(concept)
        return self._deduplicate_topics(skill_words)

    def _normalize_difficulty(self, value: str) -> str:
        lowered = str(value).strip().lower()
        if lowered in {"low", "medium", "high", "unknown"}:
            return lowered
        return "unknown"

    def _infer_difficulty(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("advanced", "graduate", "complex", "proof", "optimization")):
            return "high"
        if any(word in lowered for word in ("intro", "basic", "beginner", "fundamental", "foundation")):
            return "low"
        if any(word in lowered for word in ("intermediate", "moderate", "practice", "build", "apply")):
            return "medium"
        return "unknown"

    def _deduplicate_topics(self, topics: Iterable[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for topic in topics or []:
            text = self.ontology_mapper.map_topic(topic) if topic else ""
            if not text:
                continue
            canonical = self.normalizer.canonicalize(text)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            deduped.append(self.normalizer.display_name(text))
        return deduped
