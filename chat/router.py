"""Intent-aware router backed by the modular multi-engine orchestrator."""

from __future__ import annotations

from router.orchestrator import QueryOrchestrator, RouterResult


class Router:
    def __init__(self, assistant, data_dir: str, threshold: float = 0.60):
        self.assistant = assistant
        self.data_dir = data_dir
        self.threshold = threshold
        self.orchestrator = QueryOrchestrator(
            data_dir,
            llm=getattr(assistant, "llm", None),
            device=getattr(assistant, "device", "cpu"),
        )

    def handle(self, question: str) -> str:
        return self.orchestrator.handle(question)

    def route(self, question: str):
        # Classify first so we can short-circuit knowledge-state queries.
        classification = self.orchestrator.classifier.classify(question)
        route_result = self.orchestrator.route(question)

        lowered = question.strip().lower()
        knowledge_triggers = (
            "what topics do i know",
            "what topics do i currently know",
            "what do i know",
            "knowledge state",
            "what am i good at",
            "my current topics",
            "what topics have i learned",
            "what topics am i learning",
        )
        if any(trigger in lowered for trigger in knowledge_triggers):
            profile = self.assistant.student_profile_summary()
            return RouterResult(
                answer=profile,
                intent=classification.intent,
                document_type=classification.document_type,
                confidence=0.95,
                engine="knowledge_state",
                evidence_source="knowledge_state:local",
                matched_documents=[{"file": "data/knowledge_state.json"}],
            )

        return route_result

    def explain_context(self, question: str) -> str:
        return self.orchestrator.explain_context(question)
