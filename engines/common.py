"""Shared engine result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EngineResult:
    answer: str
    confidence: float = 0.0
    evidence_source: str = ""
    matched_documents: list[dict[str, Any]] = field(default_factory=list)
    engine: str = ""
