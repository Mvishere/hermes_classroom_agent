"""Factual question tool: answers direct queries from stored JSON data.

Delegates to the modular router orchestrator so structured queries are routed
to the deterministic engine and semantic/topic questions are routed safely.
"""
from typing import Optional

from router.orchestrator import QueryOrchestrator


def handle(question: str, data_dir: str) -> Optional[str]:
    pipeline = QueryOrchestrator(data_dir)
    try:
        return pipeline.handle(question)
    except Exception:
        # Fail silently to allow other tools/handlers to try
        return None
