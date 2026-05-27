"""Entry points for updating knowledge state."""

from __future__ import annotations

from typing import List

from student.knowledge_tracker import KnowledgeTracker


def update_from_assignments(
    knowledge_tracker: KnowledgeTracker, assignments: List[dict]
) -> int:
    """Update knowledge state from a list of assignments."""
    return knowledge_tracker.update_from_assignments(assignments)
