"""Multi-engine orchestration layer with intent-aware routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engines.common import EngineResult
from engines.semantic_engine import SemanticEngine
from engines.structured_engine import StructuredEngine
from engines.topic_graph_engine import TopicGraphEngine
from router.intent_classifier import IntentClassifier, QueryClassification


@dataclass(slots=True)
class RouterResult:
    answer: str
    intent: str
    document_type: str
    confidence: float
    engine: str
    evidence_source: str
    matched_documents: list[dict]


class QueryOrchestrator:
    """Routes questions to the specialized engine best suited to answer them."""

    def __init__(self, data_dir: str, llm=None, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.classifier = IntentClassifier()
        self.structured_engine = StructuredEngine(str(self.data_dir))
        self.semantic_engine = SemanticEngine(str(self.data_dir), llm=llm, device=device)
        self.topic_graph_engine = TopicGraphEngine(str(self.data_dir / "topic_graph.json"))

    def route(self, question: str) -> RouterResult:
        classification = self.classifier.classify(question)
        engine_result = self._dispatch(question, classification)
        if engine_result is None:
            engine_result = EngineResult(
                answer="I don't have enough information in the local course data to answer that reliably.",
                confidence=0.2,
                evidence_source="router:fallback",
                matched_documents=[],
                engine="router",
            )
        return RouterResult(
            answer=engine_result.answer,
            intent=classification.intent,
            document_type=classification.document_type,
            confidence=engine_result.confidence or classification.confidence,
            engine=engine_result.engine or classification.intent,
            evidence_source=engine_result.evidence_source,
            matched_documents=engine_result.matched_documents,
        )

    def handle(self, question: str) -> str:
        return self.route(question).answer

    def explain_context(self, question: str) -> str:
        classification = self.classifier.classify(question)
        if classification.intent != "summary":
            return ""
        return self.semantic_engine.explain_summary_context(question)

    def _dispatch(self, question: str, classification: QueryClassification) -> EngineResult | None:
        if classification.intent in {"structured_count", "latest_item", "metadata_lookup"}:
            return self.structured_engine.answer(question, classification)

        if classification.intent in {"topic_relation", "prerequisite", "recommendation"}:
            topic_result = self.topic_graph_engine.answer(question)
            if topic_result.confidence >= 0.3 and topic_result.answer != self.topic_graph_engine.GROUNDED_FALLBACK:
                return topic_result
            # If graph evidence is insufficient, do not let the LLM invent a topic answer.
            return topic_result

        if classification.intent == "summary":
            return self.semantic_engine.answer(question, document_type=classification.document_type)

        if classification.intent == "semantic_search":
            return self.semantic_engine.answer(question, document_type=classification.document_type)

        return self.semantic_engine.answer(question, document_type=classification.document_type)
