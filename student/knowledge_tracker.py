"""Knowledge tracking pipeline for Classroom items."""

from __future__ import annotations

import logging
from typing import Iterable, List

from rag.topic_extractor import TopicExtractor
from recommendation.recommender import Recommender
from storage.recommendation_store import RecommendationStore
from storage.topics_store import TopicsStore
from student.inference_rules import InferenceRules
from student.knowledge_store import KnowledgeStore
from student.topic_graph_builder import TopicGraphBuilder
from storage.json_store import JsonStore


class KnowledgeTracker:
    """Coordinates topic extraction and knowledge updates."""

    def __init__(
        self,
        topics_store: TopicsStore,
        knowledge_store: KnowledgeStore,
        inference_rules: InferenceRules,
        topic_extractor: TopicExtractor,
        recommender: Recommender,
        recommendation_store: RecommendationStore,
        graph_builder: TopicGraphBuilder,
        enabled: bool = True,
    ) -> None:
        self.topics_store = topics_store
        self.knowledge_store = knowledge_store
        self.inference_rules = inference_rules
        self.topic_extractor = topic_extractor
        self.recommender = recommender
        self.recommendation_store = recommendation_store
        self.graph_builder = graph_builder
        self.enabled = enabled

    def process_new_items(
        self,
        json_store: JsonStore,
        assignment_ids: List[str],
        material_ids: List[str],
        announcement_ids: List[str],
    ) -> None:
        if not self.enabled:
            return

        assignments = json_store.get_items_by_ids("assignments", assignment_ids)
        materials = json_store.get_items_by_ids("materials", material_ids)
        announcements = json_store.get_items_by_ids("announcements", announcement_ids)

        logging.info(
            "Topic extraction for new items. Assignments: %s, materials: %s, announcements: %s",
            len(assignments),
            len(materials),
            len(announcements),
        )

        self._extract_for_items(assignments, "assignments")
        self._extract_for_items(materials, "materials")
        self._extract_for_items(announcements, "announcements")
        self.graph_builder.rebuild(self.topics_store, self.knowledge_store)

    def generate_recommendations(
        self, json_store: JsonStore, material_ids: List[str]
    ) -> None:
        if not self.enabled:
            return
        materials = json_store.get_items_by_ids("materials", material_ids)
        self._recommend_for_items(materials, "materials")

    def update_from_assignments(self, assignments: List[dict]) -> int:
        if not self.enabled:
            return 0

        updates = 0
        for assignment in assignments:
            if not self._is_completed(assignment):
                continue
            course_id = assignment.get("course_id", "")
            item_id = assignment.get("id", "")
            topics_payload = self.topics_store.get_topics(
                "assignments", course_id, item_id
            )
            if not topics_payload:
                continue
            topics = topics_payload.get("topics", [])
            if not topics:
                continue
            score = self._extract_score(assignment)
            updates += self.inference_rules.apply_assignment_completion(
                self.knowledge_store, topics, evidence=item_id, score=score
            )

        updates += self.inference_rules.apply_decay(self.knowledge_store)
        if updates:
            self.knowledge_store.save()
            logging.info("Knowledge tracker updates applied: %s", updates)
            self.graph_builder.rebuild(self.topics_store, self.knowledge_store)
        return updates

    def _extract_for_items(self, items: Iterable[dict], item_type: str) -> None:
        for item in items:
            course_id = item.get("course_id", "")
            item_id = item.get("id", "")
            if not course_id or not item_id:
                continue
            if self.topics_store.has_topics(item_type, course_id, item_id):
                continue
            try:
                payload = self.topic_extractor.extract(item, item_type)
            except Exception:
                logging.exception("Topic extraction failed for %s", item_id)
                continue
            self.topics_store.upsert_topics(payload)

    def _recommend_for_items(self, items: Iterable[dict], item_type: str) -> None:
        for item in items:
            course_id = item.get("course_id", "")
            item_id = item.get("id", "")
            if not course_id or not item_id:
                continue
            topics_payload = self.topics_store.get_topics(item_type, course_id, item_id)
            if not topics_payload:
                continue
            topics = topics_payload.get("topics", [])
            recommendation = self.recommender.recommend(item, topics, item_type)
            if recommendation:
                self.recommendation_store.upsert_recommendation(item, recommendation)

    def _is_completed(self, assignment: dict) -> bool:
        submission_state = assignment.get("submission_state", "").upper()
        if submission_state in {"TURNED_IN", "RETURNED"}:
            return True
        return assignment.get("completed") is True

    def _extract_score(self, assignment: dict) -> float | None:
        score = assignment.get("assignedGrade")
        if score is None:
            score = assignment.get("draftGrade")
        if score is None:
            return None
        max_points = assignment.get("maxPoints")
        if max_points:
            try:
                return float(score) / float(max_points)
            except Exception:
                return None
        try:
            return float(score)
        except Exception:
            return None
