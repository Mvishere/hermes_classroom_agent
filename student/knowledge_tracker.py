"""Knowledge tracking pipeline for Classroom items."""

from __future__ import annotations

import logging
from typing import Iterable, List

from topic_graph.extractor import TopicExtractor
from recommendation.recommender import Recommender
from storage.recommendation_store import RecommendationStore
from storage.topics_store import TopicsStore
from student.knowledge_store import KnowledgeStore
from storage.json_store import JsonStore
from topic_graph.graph_generator import GraphGenerator
from student.knowledge_updater import KnowledgeUpdater


class KnowledgeTracker:
    """Coordinates topic extraction, knowledge updates, and graph generation."""

    def __init__(
        self,
        topics_store: TopicsStore,
        knowledge_store: KnowledgeStore,
        topic_extractor: TopicExtractor,
        knowledge_updater: KnowledgeUpdater,
        recommender: Recommender,
        recommendation_store: RecommendationStore,
        graph_generator: GraphGenerator,
        enabled: bool = True,
    ) -> None:
        self.topics_store = topics_store
        self.knowledge_store = knowledge_store
        self.topic_extractor = topic_extractor
        self.knowledge_updater = knowledge_updater
        self.recommender = recommender
        self.recommendation_store = recommendation_store
        self.graph_generator = graph_generator
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

        assignments = json_store.get_items_by_ids(
            "assignments",
            assignment_ids,
        )

        materials = json_store.get_items_by_ids(
            "materials",
            material_ids,
        )

        announcements = json_store.get_items_by_ids(
            "announcements",
            announcement_ids,
        )

        logging.info(
            (
                "Topic extraction for new items. "
                "Assignments=%s Materials=%s Announcements=%s"
            ),
            len(assignments),
            len(materials),
            len(announcements),
        )

        knowledge_updated = False
        knowledge_updated |= self._extract_for_items(
            assignments,
            "assignments",
            update_knowledge=True,
        )

        knowledge_updated |= self._extract_for_items(
            materials,
            "materials",
            update_knowledge=True,
        )

        knowledge_updated |= self._extract_for_items(
            announcements,
            "announcements",
            update_knowledge=False,
        )

        if knowledge_updated:
            self.knowledge_store.save()
            logging.info("Knowledge store updated.")

        # Rebuild semantic graph from all extracted payloads
        payloads = self.topics_store.list_payloads()

        if payloads:
            self.graph_generator.rebuild(self.topics_store, self.knowledge_store)
            logging.info(
                "Topic graph rebuilt using %s payloads.",
                len(payloads),
            )

    def generate_recommendations(
        self,
        json_store: JsonStore,
        material_ids: List[str],
    ) -> None:
        if not self.enabled:
            return

        materials = json_store.get_items_by_ids(
            "materials",
            material_ids,
        )

        self._recommend_for_items(
            materials,
            "materials",
        )

    def _extract_for_items(
        self,
        items: Iterable[dict],
        item_type: str,
        update_knowledge: bool,
    ) -> bool:
        """
        Extract topics and optionally update the knowledge store.

        Returns:
            bool: True if knowledge store was modified.
        """

        knowledge_updated = False
        logging.info("ENTER _extract_for_items: %s items", len(items))
        for item in items:
            
            course_id = item.get("course_id", "")
            item_id = item.get("id", "")
            logging.info("Processing item_id=%s", item_id)
            if not course_id or not item_id:
                continue

            if self.topics_store.has_topics(
                item_type,
                course_id,
                item_id,
            ):
                continue
            try:
                payload = self.topic_extractor.extract(
                    item,
                    item_type,
                )
                logging.info(
                    "Topic extraction done for %s: %s",
                    item_id,
                    payload
                )
            except Exception:
                logging.exception(
                    "Topic extraction failed for %s",
                    item_id,
                )
                continue

            self.topics_store.upsert_topics(payload)
            if update_knowledge:
                try:
                    self.knowledge_updater.process_payload(
                        payload,
                        self.knowledge_store,
                    )
                    knowledge_updated = True

                except Exception:
                    logging.exception(
                        "Knowledge update failed for %s",
                        item_id,
                    )
                    continue   # 🔥 IMPORTANT: DO NOT RETURN

        return knowledge_updated or False

    def _recommend_for_items(
        self,
        items: Iterable[dict],
        item_type: str,
    ) -> None:
        for item in items:
            course_id = item.get("course_id", "")
            item_id = item.get("id", "")

            if not course_id or not item_id:
                continue

            topics_payload = self.topics_store.get_topics(
                item_type,
                course_id,
                item_id,
            )

            if not topics_payload:
                continue

            topics = topics_payload.get("topics", [])

            recommendation = self.recommender.recommend(
                item,
                topics,
                item_type,
            )

            if recommendation:
                self.recommendation_store.upsert_recommendation(
                    item,
                    recommendation,
                )