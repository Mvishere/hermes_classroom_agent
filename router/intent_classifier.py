"""Lightweight intent classifier for deterministic routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List


@dataclass(slots=True)
class QueryClassification:
    intent: str
    document_type: str = "all"
    confidence: float = 0.0
    matched_terms: List[str] = field(default_factory=list)


class IntentClassifier:
    """Rule-based intent classifier with easy extension points."""

    INTENT_RULES = {
        "structured_count": [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"],
        "latest_item": [r"\blatest\b", r"\bmost recent\b", r"\bnewest\b"],
        "summary": [r"\bsummarize\b", r"\bsummary\b", r"\boverview\b", r"\brecap\b"],
        "topic_relation": [r"\brelated\b", r"\brelation\b", r"\bconnected\b", r"\bassociate\b"],
        "prerequisite": [r"\bprerequisite\b", r"\bbefore learning\b", r"\bwhat should i learn before\b", r"\bbefore\b", r"\bready\b", r"\bseem ready\b", r"\bfoundational\b"],
        "recommendation": [r"\brecommend\b", r"\bsuggest\b", r"\bwhat next\b", r"\blearning path\b"],
        "metadata_lookup": [r"\btitle\b", r"\bdate\b", r"\bcourse name\b", r"\bcourse\b"],
        "semantic_search": [r"\bexplain\b", r"\bwhat is\b", r"\bmeaning\b", r"\bconcept\b"],
    }

    DOCUMENT_TYPE_RULES = {
        "announcements": [r"\bannouncement\w*\b", r"\bannounc\w*\b", r"\bposted\b", r"\bnotice\b"],
        "assignments": [r"\bassignment\w*\b", r"\bquiz\w*\b", r"\bhomework\b", r"\bsubmission\w*\b"],
        "materials": [r"\bmaterial\w*\b", r"\blecture\w*\b", r"\bslide\w*\b", r"\bnote\w*\b", r"\bresource\w*\b", r"\breading\w*\b"],
        "courses": [r"\bcourse\w*\b", r"\bclass\w*\b", r"\benroll\w*\b", r"\bsection\w*\b"],
    }

    def classify(self, question: str) -> QueryClassification:
        text = question.strip().lower()
        if not text:
            return QueryClassification(intent="semantic_search", confidence=0.1)

        document_type = self.infer_document_type(text)

        # If the user explicitly asks to summarize, prioritize summary intent
        # over "latest/most recent" phrasing so this routes to semantic summary.
        if any(term in text for term in ("summarize", "summary", "overview", "recap")):
            return QueryClassification(
                intent="summary",
                document_type=document_type,
                confidence=0.95,
                matched_terms=["summary"],
            )

        if any(
            phrase in text
            for phrase in (
                "what is the title of the material related to",
                "what is the title of the material that",
                "which material title",
                "which item title",
            )
        ):
            return QueryClassification(
                intent="metadata_lookup",
                document_type="materials",
                confidence=0.92,
                matched_terms=["title", "materials"],
            )

        if any(phrase in text for phrase in ("which courses am i enrolled in", "what courses am i enrolled in", "what courses am i taking", "which course am i enrolled in")):
            return QueryClassification(intent="structured_count", document_type="courses", confidence=0.95, matched_terms=["courses", "enrolled"])

        if "mention" in text and any(marker in text for marker in ("how many", "count", "number of")):
            return QueryClassification(intent="structured_count", document_type=document_type, confidence=0.88, matched_terms=["mention", document_type])

        if any(phrase in text for phrase in ("study first", "review first", "if i only have", "which item should i review first", "what should i review first", "most likely requires")):
            return QueryClassification(intent="recommendation", document_type=document_type, confidence=0.9, matched_terms=["recommendation", document_type])

        intent, intent_terms, intent_score = self._match_rule(text, self.INTENT_RULES)
        _, doc_terms, doc_score = self._match_rule(text, self.DOCUMENT_TYPE_RULES)

        if intent == "metadata_lookup" and any(term in text for term in ("what courses am i enrolled in", "which courses am i enrolled in")):
            intent = "structured_count"

        if intent == "semantic_search" and any(term in text for term in ("summary", "summarize", "overview")):
            intent = "summary"

        if intent == "semantic_search" and any(term in text for term in ("before", "prerequisite", "related", "recommend")):
            intent = "prerequisite"

        confidence = min(1.0, 0.35 + 0.2 * max(intent_score, doc_score))
        matched_terms = intent_terms + doc_terms
        if not matched_terms:
            confidence = 0.2
        return QueryClassification(
            intent=intent,
            document_type=document_type or "all",
            confidence=round(confidence, 2),
            matched_terms=matched_terms,
        )

    def infer_document_type(self, text: str) -> str:
        for document_type, patterns in self.DOCUMENT_TYPE_RULES.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return document_type
        return "all"

    def _match_rule(self, text: str, rules: dict[str, list[str]]) -> tuple[str, list[str], int]:
        best_label = "semantic_search"
        best_terms: list[str] = []
        best_score = 0
        for label, patterns in rules.items():
            matched_terms = []
            score = 0
            for pattern in patterns:
                if re.search(pattern, text):
                    score += 1
                    matched_terms.append(pattern)
            if score > best_score:
                best_label = label
                best_terms = matched_terms
                best_score = score
        return best_label, best_terms, best_score
