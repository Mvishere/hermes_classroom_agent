"""Mastery scoring logic for student knowledge."""

from __future__ import annotations

from datetime import datetime
from typing import Optional


class MasteryEngine:
    """Updates mastery and confidence scores based on events."""

    def __init__(
        self,
        base_increment: float = 0.08,
        confidence_increment: float = 0.05,
        decay_rate: float = 0.04,
    ) -> None:
        self.base_increment = base_increment
        self.confidence_increment = confidence_increment
        self.decay_rate = decay_rate

    def ensure_entry(self, entry: Optional[dict]) -> dict:
        entry = dict(entry or {})
        entry.setdefault("status", "unknown")
        entry.setdefault("mastery_score", 0.0)
        entry.setdefault("confidence_score", 0.0)
        entry.setdefault("evidence", [])
        entry.setdefault("last_updated", "")
        entry.setdefault("last_interaction", "")
        return entry

    def apply_completion(
        self,
        entry: Optional[dict],
        evidence: str,
        score: Optional[float] = None,
        weight: float = 1.0,
    ) -> dict:
        entry = self.ensure_entry(entry)
        delta = self.base_increment * self._score_weight(score) * max(weight, 0.0)
        mastery = float(entry.get("mastery_score", 0.0))
        confidence = float(entry.get("confidence_score", 0.0))

        entry["mastery_score"] = min(1.0, mastery + delta * (1.0 - mastery))
        entry["confidence_score"] = min(1.0, confidence + self.confidence_increment * weight)
        entry["status"] = self._status(entry["mastery_score"], entry["confidence_score"])
        entry["last_updated"] = datetime.utcnow().isoformat() + "Z"
        entry["last_interaction"] = entry["last_updated"]
        entry["evidence"] = self._merge_evidence(entry.get("evidence", []), evidence)
        return entry

    def apply_confidence_penalty(
        self, entry: Optional[dict], evidence: str, weight: float = 1.0
    ) -> dict:
        entry = self.ensure_entry(entry)
        confidence = float(entry.get("confidence_score", 0.0))
        entry["confidence_score"] = max(0.0, confidence - self.confidence_increment * weight)
        entry["status"] = self._status(entry.get("mastery_score", 0.0), entry["confidence_score"])
        entry["last_updated"] = datetime.utcnow().isoformat() + "Z"
        entry["last_interaction"] = entry["last_updated"]
        entry["evidence"] = self._merge_evidence(entry.get("evidence", []), evidence)
        return entry

    def apply_decay(self, entry: Optional[dict], days_since: float) -> dict:
        entry = self.ensure_entry(entry)
        mastery = float(entry.get("mastery_score", 0.0))
        decay = self.decay_rate * max(days_since, 0.0) / 30.0
        entry["mastery_score"] = max(0.0, mastery - decay)
        entry["status"] = self._status(entry["mastery_score"], entry.get("confidence_score", 0.0))
        entry["last_updated"] = datetime.utcnow().isoformat() + "Z"
        return entry

    def _score_weight(self, score: Optional[float]) -> float:
        if score is None:
            return 1.0
        try:
            normalized = float(score)
        except Exception:
            return 1.0
        if normalized > 1.0:
            normalized = normalized / 100.0
        return max(0.2, min(1.0, normalized))

    def _status(self, mastery: float, confidence: float) -> str:
        if confidence < 0.35:
            return "weak"
        if mastery >= 0.8:
            return "known"
        if mastery >= 0.4:
            return "learning"
        return "unknown"

    def _merge_evidence(self, evidence_list: list, evidence: str) -> list:
        evidence_list = list(evidence_list or [])
        if evidence and evidence not in evidence_list:
            evidence_list.append(evidence)
        return evidence_list
