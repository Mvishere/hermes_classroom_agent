"""Semantic retrieval engine for summaries and conceptual explanations."""

from __future__ import annotations

import json
from pathlib import Path
import re

import config
from engines.common import EngineResult
from retrieval.filters import DocumentFilter
from retrieval.vector_store import VectorStore
from storage.json_store import JsonStore
from storage.summary_store import SummaryStore


class SemanticEngine:
    SAFE_FALLBACK = "No matching grounded evidence was found."

    def __init__(self, data_dir: str, llm=None, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.llm = llm
        self.store = JsonStore(self.data_dir)
        self.summary_store = SummaryStore(self.data_dir / "rag" / "summaries.json")
        self.vector_store = VectorStore(self.data_dir, embedding_model_path=config.EMBEDDING_MODEL_PATH, device=device)
        self.filter = DocumentFilter()

    def answer(self, question: str, document_type: str = "all") -> EngineResult:
        summary_hits = self._summary_topic_hits(question)
        if summary_hits:
            formatted = self._format_summary_hits(summary_hits)
            return EngineResult(
                answer=formatted,
                confidence=0.82,
                evidence_source="semantic:summary:summary_store",
                matched_documents=summary_hits,
                engine="semantic",
            )

        raw_material_hits = self._raw_material_hits(question)
        if raw_material_hits:
            formatted = self._summarize_raw_material_hits(question, raw_material_hits)
            return EngineResult(
                answer=formatted,
                confidence=0.65,
                evidence_source="semantic:materials:raw_text_summary",
                matched_documents=raw_material_hits,
                engine="semantic",
            )

        lowered = question.lower()
        if any(marker in lowered for marker in ("summary", "summarize", "overview", "recap")):
            topic = self._extract_topic_phrase(question)
            topic_hits = self.vector_store.search(question, document_type=document_type, top_k=config.RAG_TOP_K * 2)
            filtered_hits = [hit for hit in topic_hits if self._hit_matches_topic(hit, topic)]
            if filtered_hits:
                snippets = [self._format_hit(hit) for hit in filtered_hits[:3]]
                return EngineResult(
                    answer="\n".join(snippets),
                    confidence=min(0.8, 0.42 + 0.1 * len(filtered_hits)),
                    evidence_source=f"semantic:{document_type}:summary_topic_fallback",
                    matched_documents=[hit.item for hit in filtered_hits[:3]],
                    engine="semantic",
                )
            return self._fallback(document_type, confidence=0.2)

        docs = self.vector_store.search(question, document_type=document_type, top_k=config.RAG_TOP_K)
        if not docs:
            return self._fallback(document_type)

        unique_docs = self._dedupe_hits(docs)
        if not unique_docs:
            return self._fallback(document_type)

        best_hit = unique_docs[0]
        confidence = min(0.92, max(0.35, best_hit.score / 4.0))
        lowered = question.lower()
        if any(marker in lowered for marker in ("compare", "difference", "similar", "likely difficulty", "why", "if i only have")):
            snippets = [self._format_hit(hit) for hit in unique_docs[:3]]
            return EngineResult(
                answer="\n".join(snippets) if snippets else self.SAFE_FALLBACK,
                confidence=max(confidence, 0.4),
                evidence_source=f"semantic:{document_type}:retrieval_only",
                matched_documents=[hit.item for hit in unique_docs[:3]],
                engine="semantic",
            )

        if confidence < 0.35 and not any(marker in lowered for marker in ("summary", "summarize", "overview")):
            return self._fallback(document_type, confidence=confidence)

        snippets = [self._format_hit(hit) for hit in unique_docs[:3]]
        if not snippets:
            return self._fallback(document_type, confidence=confidence)

        # Keep semantic responses grounded and fast by using retrieved evidence only.
        # This avoids recursive generation and prevents hallucinated explanations.
        if self.llm is None:
            return EngineResult(
                answer="\n".join(snippets),
                confidence=confidence,
                evidence_source=f"semantic:{document_type}:retrieval_only",
                matched_documents=[hit.item for hit in unique_docs[:3]],
                engine="semantic",
            )

        prompt = self._build_prompt(question, unique_docs[:3], document_type)
        answer = self.llm.generate(prompt).strip()
        if not answer:
            return self._fallback(document_type, confidence=confidence)
        return EngineResult(
            answer=self._sanitize(answer),
            confidence=confidence,
            evidence_source=f"semantic:{document_type}:llm",
            matched_documents=[hit.item for hit in unique_docs[:3]],
            engine="semantic",
        )

    def _dedupe_hits(self, hits):
        seen = set()
        deduped = []
        for hit in hits:
            fingerprint = self.filter._item_text(hit.item).lower().strip()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(hit)
        return deduped

    def _format_hit(self, hit) -> str:
        title = hit.item.get("title") or hit.item.get("name") or "Untitled"
        snippet = hit.evidence.replace("\n", " ")[:180]
        return f"{title}: {snippet}".strip()

    def _build_prompt(self, question: str, hits, document_type: str) -> str:
        context_block = "\n\n".join(self._format_hit(hit) for hit in hits)
        return (
            "You are a grounded classroom assistant. Use only the provided source evidence.\n"
            "Do not invent course facts, titles, or reasoning not present in the evidence.\n"
            "If the evidence is insufficient, say you do not have enough grounded evidence.\n\n"
            f"Document type: {document_type}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context_block}\n\n"
            "Answer concisely and only with grounded evidence:"
        )

    def _fallback(self, document_type: str, confidence: float = 0.2) -> EngineResult:
        return EngineResult(
            answer="I don't have enough information in the local course data to answer that reliably.",
            confidence=confidence,
            evidence_source=f"semantic:{document_type}:fallback",
            matched_documents=[],
            engine="semantic",
        )

    def _sanitize(self, text: str) -> str:
        cleaned = text.strip()
        for marker in ("Assistant:", "Student:", "Question:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned

    def _summary_topic_hits(self, question: str) -> list[dict]:
        lowered = question.lower()
        if not any(marker in lowered for marker in ("summary", "summarize", "overview", "recap")):
            return []

        topic = self._extract_topic_phrase(question)
        if not topic:
            return []
        hits = self.summary_store.search_topic(topic, limit=config.RAG_TOP_K)
        return [hit for hit in hits if self.summary_store.is_usable_summary(str(hit.get("summary", "")))]

    def _extract_topic_phrase(self, question: str) -> str:
        lowered = question.lower()
        patterns = [
            r"(?:summary|summarize|overview|recap)(?:\s+of)?\s+(.+)$",
            r"(?:tell me|show me|give me)(?:\s+the)?(?:\s+summary(?:\s+of)?)?\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if match:
                return match.group(1).strip(" ?.!:,;")
        return ""

    def _format_summary_hits(self, hits: list[dict]) -> str:
        if not hits:
            return self.SAFE_FALLBACK
        lines = []
        for hit in hits:
            title = hit.get("title") or "Untitled"
            course = hit.get("course_name") or ""
            summary = hit.get("summary") or "No stored summary available."
            if course:
                lines.append(f"{title} ({course}): {summary}")
            else:
                lines.append(f"{title}: {summary}")
        return "\n".join(lines)

    def explain_summary_context(self, question: str) -> str:
        """Return the exact summary context that would be used for a summary query."""
        hits = self._summary_topic_hits(question)
        if not hits:
            raw_hits = self._raw_material_hits(question)
            if raw_hits:
                return self._summarize_raw_material_hits(question, raw_hits)
            return self.SAFE_FALLBACK
        return self._format_summary_hits(hits)

    def _raw_material_hits(self, question: str) -> list[dict]:
        topic = self._extract_topic_phrase(question).lower().strip()
        query = question.lower().strip()
        terms = [term for term in re.findall(r"[a-z0-9]+", topic or query) if len(term) > 1]
        materials = self.store.get_all_items("materials")
        scored: list[tuple[float, dict]] = []
        for item in materials:
            title = str(item.get("title", "")).lower()
            raw_payload = item.get("raw_payload") or {}
            raw_text = json.dumps(raw_payload, ensure_ascii=True).lower() if raw_payload else ""
            haystack = f"{title} {raw_text}"

            score = 0.0
            if topic and topic == title:
                score += 3.0
            elif topic and topic in title:
                score += 2.5
            elif topic and topic in haystack:
                score += 2.0

            matched_terms = sum(1 for term in terms if term in haystack)
            if terms and matched_terms == len(terms):
                score += 1.5
            elif matched_terms:
                score += matched_terms * 0.25

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("updated_at", "")), str(pair[1].get("title", ""))))
        if topic:
            exact_matches = [item for _, item in scored if str(item.get("title", "")).lower().strip() == topic]
            if exact_matches:
                return exact_matches[:1]
            title_matches = [item for _, item in scored if topic in str(item.get("title", "")).lower()]
            if title_matches:
                return title_matches[:1]
        return [item for _, item in scored[:1]]

    def _format_raw_material_hits(self, hits: list[dict]) -> str:
        if not hits:
            return self.SAFE_FALLBACK
        if len(hits) == 1:
            return json.dumps(hits[0], ensure_ascii=True, indent=2)
        return json.dumps(hits, ensure_ascii=True, indent=2)

    def _summarize_raw_material_hits(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return self.SAFE_FALLBACK
        best = hits[0]
        text = self._lookup_material_text(best)
        if not text:
            return self._format_raw_material_hits(hits)

        prompt = (
            "Summarize the classroom material below for a student. "
            "Return 3-5 short bullet points and keep it grounded in the text.\n\n"
            f"Question: {question}\n"
            f"Material title: {best.get('title', 'Untitled')}\n\n"
            f"Material text:\n{text}\n"
        )

        if self.llm is not None:
            try:
                response = self.llm.generate(prompt).strip()
                if response:
                    return self._sanitize(response)
            except Exception:
                pass

        return self._heuristic_summary(best, text)

    def _lookup_material_text(self, material: dict) -> str:
        target_id = str(material.get("id") or material.get("item_id") or "").strip()
        target_title = str(material.get("title") or "").strip().lower()
        payload = self._load_materials_index()
        for item in payload.get("items", []):
            item_id = str(item.get("item_id") or "").strip()
            title = str(item.get("title") or "").strip().lower()
            if target_id and item_id == target_id:
                return str(item.get("text") or "")
            if target_title and title == target_title:
                return str(item.get("text") or "")
        return str(material.get("raw_payload", {}).get("text", ""))

    def _load_materials_index(self) -> dict:
        index_path = self.data_dir / "rag" / "materials_index.json"
        if not index_path.exists():
            return {"items": []}
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"items": []}

    def _heuristic_summary(self, material: dict, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return self._format_raw_material_hits([material])
        title = material.get("title") or "Untitled"
        bullets = []
        for line in lines:
            if line.lower().startswith(("course:", "title:", "attachments:", "test topic:")):
                continue
            bullets.append(line)
            if len(bullets) >= 5:
                break
        if not bullets:
            bullets = lines[:5]
        return "\n".join([f"- {title}"] + [f"- {bullet}" for bullet in bullets])

    def _hit_matches_topic(self, hit, topic: str) -> bool:
        query = str(topic or "").strip().lower()
        if not query:
            return False
        terms = [term for term in re.findall(r"[a-z0-9]+", query) if len(term) > 1]
        if not terms:
            return False

        title = str(hit.item.get("title", "")).lower()
        evidence = str(hit.evidence or "").lower()
        haystack = f"{title} {evidence}"

        if query in title:
            return True
        if len(terms) == 1:
            return terms[0] in haystack
        return all(term in haystack for term in terms)
