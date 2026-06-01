"""Deterministic engine for counts, titles, latest items, dates, and course names."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

from router.intent_classifier import QueryClassification
from engines.common import EngineResult
from storage.json_store import JsonStore


class StructuredEngine:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.store = JsonStore(self.data_dir)

    def answer(self, question: str, classification: QueryClassification) -> EngineResult:
        document_type = classification.document_type if classification.document_type != "all" else self._infer_document_type(question)
        lowered = question.lower()

        if "mention" in lowered and any(marker in lowered for marker in ("how many", "count", "number of")):
            return self._mention_count(question, document_type)

        if any(marker in lowered for marker in ("study first", "review first", "if i only have", "recommend", "most likely requires")):
            return self._recommend_item(question, document_type)

        if any(marker in lowered for marker in ("how many", "count", "number of")) and "," in question:
            return self._multi_count(question)

        if "course" in lowered and any(marker in lowered for marker in ("which", "what", "enrolled", "name")):
            return self._course_names()

        if any(marker in lowered for marker in ("how many", "count", "number of")):
            return self._count_items(document_type)

        if any(marker in lowered for marker in ("latest", "most recent", "newest")):
            return self._latest_item(document_type)

        if classification.intent == "metadata_lookup" or ("title" in lowered and document_type == "materials"):
            return self._lookup_titles(question, document_type)

        if "title" in lowered:
            return self._list_titles(document_type)

        if "date" in lowered:
            return self._list_dates(document_type)

        return EngineResult(
            answer="I don't have enough information in the local course data to answer that reliably.",
            confidence=0.15,
            evidence_source="structured_engine:fallback",
            matched_documents=[],
            engine="structured",
        )

    def _count_items(self, document_type: str) -> EngineResult:
        if document_type == "courses":
            count = len(self._courses())
            docs = self._courses()
        elif document_type == "all":
            docs = self.store.get_all_items("announcements") + self.store.get_all_items("materials") + self.store.get_all_items("assignments")
            count = len(docs)
        else:
            docs = self.store.get_all_items(document_type if document_type != "all" else "materials")
            count = len(docs)
        label = document_type if document_type != "all" else "items"
        return EngineResult(
            answer=f"Found {count} {label} locally.",
            confidence=0.95,
            evidence_source=f"structured:{label}:count",
            matched_documents=docs[:10],
            engine="structured",
        )

    def _latest_item(self, document_type: str) -> EngineResult:
        docs = self._documents_for_type(document_type)
        if not docs:
            return self._grounded_fallback(document_type)
        latest = sorted(docs, key=lambda item: item.get("updated_at") or item.get("created_at") or "")[-1]
        title = latest.get("title") or latest.get("name") or "Untitled"
        # If this is an assignment, include extracted questions and find related materials.
        if document_type == "assignments":
            questions = latest.get("questions") or latest.get("form_questions") or []
            if questions:
                lines = []
                related_docs = []
                for q in questions:
                    qtext = q.get("question") if isinstance(q, dict) else str(q)
                    related = self._find_related_material(qtext)
                    if related:
                        related_docs.append(related)
                        lines.append(f"- {qtext} (related: {related.get('title') or related.get('name')})")
                    else:
                        lines.append(f"- {qtext} (related: none found)")
                answer = f"Latest assignment: {title}.\nQuestions:\n" + "\n".join(lines)
                matched = [latest] + related_docs[:3]
                return EngineResult(
                    answer=answer,
                    confidence=0.9,
                    evidence_source=f"structured:{document_type}:latest",
                    matched_documents=matched,
                    engine="structured",
                )

        return EngineResult(
            answer=f"Latest {document_type[:-1] if document_type.endswith('s') else document_type}: {title}.",
            confidence=0.9,
            evidence_source=f"structured:{document_type}:latest",
            matched_documents=[latest],
            engine="structured",
        )

    def _find_related_material(self, question: str) -> dict | None:
        materials = self._documents_for_type("materials")
        if not materials:
            return None
        terms = self._extract_query_terms(question)
        scored = []
        for doc in materials:
            score = self._doc_lookup_score(doc, terms, question)
            scored.append((score, doc))
        scored.sort(key=lambda p: p[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
        return None

    def _list_titles(self, document_type: str) -> EngineResult:
        docs = self._documents_for_type(document_type)
        titles = [doc.get("title") or doc.get("name") for doc in docs if doc.get("title") or doc.get("name")]
        if not titles:
            return self._grounded_fallback(document_type)
        return EngineResult(
            answer="Titles: " + ", ".join(titles) + ".",
            confidence=0.9,
            evidence_source=f"structured:{document_type}:titles",
            matched_documents=docs[:10],
            engine="structured",
        )

    def _lookup_titles(self, question: str, document_type: str) -> EngineResult:
        docs = self._documents_for_type(document_type)
        if not docs:
            return self._grounded_fallback(document_type)

        query_terms = self._extract_query_terms(question)
        ranked = sorted(
            ((self._doc_lookup_score(doc, query_terms, question), doc) for doc in docs),
            key=lambda pair: (
                pair[0],
                pair[1].get("updated_at") or pair[1].get("created_at") or "",
                pair[1].get("title") or pair[1].get("name") or "",
            ),
            reverse=True,
        )
        ranked = [(score, doc) for score, doc in ranked if score > 0]
        if not ranked:
            return self._grounded_fallback(document_type)

        best_score, best_doc = ranked[0]
        title = best_doc.get("title") or best_doc.get("name") or "Untitled"
        if len(ranked) == 1:
            answer = f"Title: {title}."
        else:
            answer = "Titles: " + ", ".join((doc.get("title") or doc.get("name") or "Untitled") for _, doc in ranked[:3]) + "."

        return EngineResult(
            answer=answer,
            confidence=min(0.95, 0.65 + 0.1 * best_score),
            evidence_source=f"structured:{document_type}:metadata_lookup",
            matched_documents=[doc for _, doc in ranked[:3]],
            engine="structured",
        )

    def _list_dates(self, document_type: str) -> EngineResult:
        docs = self._documents_for_type(document_type)
        dates = []
        for doc in docs:
            date_value = doc.get("updated_at") or doc.get("created_at") or doc.get("due_date") or doc.get("scheduled_time")
            if date_value:
                dates.append(str(date_value))
        if not dates:
            return self._grounded_fallback(document_type)
        return EngineResult(
            answer="Dates: " + ", ".join(dates[:10]) + ".",
            confidence=0.8,
            evidence_source=f"structured:{document_type}:dates",
            matched_documents=docs[:10],
            engine="structured",
        )

    def _course_names(self) -> EngineResult:
        courses = self._courses()
        names = [course.get("name") for course in courses if course.get("name")]
        if not names:
            return self._grounded_fallback("courses")
        answer = "You are enrolled in: " + ", ".join(names) + "." if len(names) > 1 else f"You are enrolled in {names[0]}."
        return EngineResult(
            answer=answer,
            confidence=0.95,
            evidence_source="structured:courses:names",
            matched_documents=courses,
            engine="structured",
        )

    def _documents_for_type(self, document_type: str) -> list[dict]:
        if document_type == "courses":
            return self._courses()
        if document_type in {"announcements", "assignments", "materials"}:
            return self.store.get_all_items(document_type)
        return self.store.get_all_items("materials") + self.store.get_all_items("assignments") + self.store.get_all_items("announcements")

    def _courses(self) -> list[dict]:
        payload = self.store._load(self.data_dir / "courses" / "courses.json", {"courses": {}})
        return list((payload.get("courses") or {}).values())

    def _grounded_fallback(self, document_type: str) -> EngineResult:
        return EngineResult(
            answer="I don't have enough information in the local course data to answer that reliably.",
            confidence=0.2,
            evidence_source=f"structured:{document_type}:fallback",
            matched_documents=[],
            engine="structured",
        )

    def _mention_count(self, question: str, document_type: str) -> EngineResult:
        lowered = question.lower()
        if document_type == "all":
            document_type = self._infer_document_type(question)
        terms = self._extract_terms_after_mention(question)
        docs = self._documents_for_type(document_type)
        if not terms:
            return self._count_items(document_type)

        count = 0
        matched_documents = []
        for doc in docs:
            haystack = " ".join(str(doc.get(field, "")) for field in ("title", "description", "text")).lower()
            if any(term in haystack for term in terms):
                count += 1
                matched_documents.append(doc)

        label = document_type[:-1] if document_type.endswith("s") else document_type
        return EngineResult(
            answer=f"Found {count} {label}(s) mentioning {', '.join(terms)}.",
            confidence=0.9 if matched_documents else 0.5,
            evidence_source=f"structured:{document_type}:mention_count",
            matched_documents=matched_documents,
            engine="structured",
        )

    def _multi_count(self, question: str) -> EngineResult:
        lowered = question.lower()
        counts = []
        matched_documents = []
        if "pending" in lowered and "returned" in lowered and "assignment" in lowered:
            assignments = self._documents_for_type("assignments")
            pending = [a for a in assignments if str(a.get("submission_state", "")).upper() not in {"TURNED_IN", "RETURNED"}]
            returned = [a for a in assignments if str(a.get("submission_state", "")).upper() == "RETURNED"]
            return EngineResult(
                answer=f"{len(pending)} pending assignments; {len(returned)} returned assignments.",
                confidence=0.93,
                evidence_source="structured:assignments:pending_returned",
                matched_documents=assignments[:20],
                engine="structured",
            )
        for document_type, label in (("announcements", "announcement"), ("materials", "material"), ("assignments", "assignment"), ("courses", "course")):
            if label in lowered or document_type in lowered:
                docs = self._documents_for_type(document_type)
                counts.append(f"{len(docs)} {document_type}")
                matched_documents.extend(docs)
        if not counts:
            return self._count_items(self._infer_document_type(question))
        return EngineResult(
            answer="; ".join(counts).capitalize() + ".",
            confidence=0.88,
            evidence_source="structured:multi_count",
            matched_documents=matched_documents[:20],
            engine="structured",
        )

    def _infer_document_type(self, question: str) -> str:
        lowered = question.lower()
        if any(marker in lowered for marker in ("announcement", "announc", "notice")):
            return "announcements"
        if any(marker in lowered for marker in ("assignment", "quiz", "homework")):
            return "assignments"
        if any(marker in lowered for marker in ("material", "lecture", "slide", "note")):
            return "materials"
        if any(marker in lowered for marker in ("course", "class", "enrolled")):
            return "courses"
        return "all"

    def _recommend_item(self, question: str, document_type: str) -> EngineResult:
        lowered = question.lower()
        target_types = [document_type] if document_type in {"announcements", "assignments", "materials", "courses"} else ["materials", "assignments", "announcements", "courses"]
        docs = []
        for target_type in target_types:
            docs.extend(self._documents_for_type(target_type))
        if not docs:
            return self._grounded_fallback(document_type)

        terms = [term for term in self._extract_terms_after_mention(question)]
        if not terms:
            terms = [term for term in self._extract_keywords(question)]

        ranked = sorted(
            docs,
            key=lambda doc: (
                self._doc_score(doc, terms, lowered),
                doc.get("updated_at") or doc.get("created_at") or "",
            ),
            reverse=True,
        )
        best = ranked[0]
        title = best.get("title") or best.get("name") or "Untitled"
        evidence_terms = ", ".join(terms[:5]) if terms else "the available evidence"
        return EngineResult(
            answer=f"I would review {title} first because it best matches {evidence_terms}.",
            confidence=0.8,
            evidence_source=f"structured:{document_type}:recommendation",
            matched_documents=ranked[:5],
            engine="structured",
        )

    def _extract_query_terms(self, question: str) -> list[str]:
        stopwords = {
            "what", "which", "who", "where", "when", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "do", "does", "did",
            "title", "titles", "material", "materials", "item", "items", "related", "relation",
            "about", "with", "that", "this", "those", "these", "question", "questions", "find",
        }
        terms = [term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 1 and term not in stopwords]
        return terms

    def _doc_lookup_score(self, doc: dict, query_terms: list[str], question: str) -> float:
        title = str(doc.get("title") or doc.get("name") or "").lower()
        description = str(doc.get("description") or "").lower()
        raw_payload = doc.get("raw_payload") or {}
        raw_text = json.dumps(raw_payload, ensure_ascii=True).lower() if raw_payload else ""
        index_text = self._material_index_text(doc).lower()
        haystack = f"{title} {description} {raw_text} {index_text}"

        score = 0.0
        for term in query_terms:
            if term in title:
                score += 2.5
            elif term in description:
                score += 1.0
            elif term in haystack:
                score += 0.6

        normalized_question = question.lower().strip()
        if normalized_question and normalized_question in title:
            score += 2.0
        return score

    def _material_index_text(self, doc: dict) -> str:
        if doc.get("text"):
            return str(doc.get("text"))
        index_path = self.data_dir / "rag" / "materials_index.json"
        if not index_path.exists():
            return ""
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""

        target_id = str(doc.get("id") or doc.get("item_id") or "").strip()
        target_title = str(doc.get("title") or doc.get("name") or "").strip().lower()
        for item in payload.get("items", []):
            item_id = str(item.get("item_id") or "").strip()
            title = str(item.get("title") or "").strip().lower()
            if target_id and item_id == target_id:
                return str(item.get("text") or "")
            if target_title and title == target_title:
                return str(item.get("text") or "")
        return ""

    def _extract_terms_after_mention(self, question: str) -> list[str]:
        lowered = question.lower()
        if "mention" not in lowered:
            return []
        tail = lowered.split("mention", 1)[1]
        tail = tail.replace("s", " ", 1) if tail.startswith("s") else tail
        terms = [term for term in tail.split() if term not in {"how", "many", "and", "or", "the", "a", "an", "of", "for", "to", "with", "in", "do", "does", "mention", "mentions"}]
        return [term.strip("?,.!") for term in terms if term.strip("?,.!")]

    def _extract_keywords(self, question: str) -> list[str]:
        return [term for term in question.lower().split() if term.isalpha() and len(term) > 2]

    def _doc_score(self, doc: dict, terms: list[str], lowered: str) -> tuple[int, str]:
        haystack = " ".join(str(doc.get(field, "")) for field in ("title", "description", "text")).lower()
        score = sum(1 for term in terms if term and term in haystack)
        if any(marker in haystack for marker in ("quiz", "assignment", "material", "announcement")):
            score += 1
        if any(marker in lowered for marker in ("10 minutes", "first", "review", "study")):
            score += 1
        return score, str(doc.get("updated_at") or doc.get("created_at") or "")
