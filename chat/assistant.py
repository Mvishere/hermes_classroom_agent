"""Local chat assistant backed by the RAG pipeline and local data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import config
from rag.embeddings import EmbeddingModel
from rag.llm import LocalLLM
from storage.json_store import JsonStore
from storage.summary_store import SummaryStore


@dataclass
class ChatDocument:
    """Represents a retrievable document for chat."""

    item_id: str
    item_type: str
    course_id: str
    course_name: str
    title: str
    text: str


class ChatAssistant:
    """Answers questions using local Classroom data and a local LLM."""

    def __init__(
        self,
        data_dir: str,
        llm_model_path: str,
        embedding_model_path: str,
        device: str = "cpu",
        top_k: int = 3,
        max_context_chars: int = 3500,
        max_history_turns: int = 3,
        history_limit: int = 50,
        similarity_threshold: float = 0.39,
    ) -> None:
        self.json_store = JsonStore(data_dir)
        self.summary_store = SummaryStore(config.RAG_SUMMARIES_PATH)
        self.embedding_model = EmbeddingModel(embedding_model_path, device=device)
        self.llm = LocalLLM(
            llm_model_path,
            device=device,
            max_new_tokens=config.RAG_MAX_NEW_TOKENS,
            temperature=config.RAG_TEMPERATURE,
        )
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_history_turns = max_history_turns
        self.history_limit = max(2, history_limit)
        self.similarity_threshold = similarity_threshold
        self._documents: List[ChatDocument] = []
        self._embeddings: np.ndarray | None = None
        self._history: List[Tuple[str, str]] = []

    def build_index(self) -> None:
        documents: List[ChatDocument] = []
        documents.extend(self._load_items("materials"))
        documents.extend(self._load_items("assignments"))
        documents.extend(self._load_items("announcements"))

        self._documents = documents
        if not documents:
            self._embeddings = None
            return

        texts = [doc.text for doc in documents]
        vectors = self.embedding_model.encode(texts)
        self._embeddings = np.array(vectors, dtype=float)

    def answer(self, question: str) -> str:
        if not question.strip():
            return "Please ask a question about your coursework or materials."

        structured = self._try_structured_answer(question)
        if structured:
            self._record_turn(question, structured)
            return structured

        if self._embeddings is None or not self._documents:
            self.build_index()

        if not self._documents or self._embeddings is None:
            return "I could not find any local materials to answer from."

        question_vec = np.array(self.embedding_model.encode([question])[0], dtype=float)
        scores = self._embeddings @ question_vec
        top_indices = np.argsort(scores)[-self.top_k :][::-1]

        best_score = scores[top_indices[0]]
        if best_score < self.similarity_threshold:
            response = (
                "I don't have enough information in your course materials to answer that question. "
                "Try asking about course materials, assignments, or announcements."
            )
            self._record_turn(question, response)
            return response

        context_lines = []
        for idx in top_indices:
            doc = self._documents[int(idx)]
            context_lines.append(
                f"[{doc.item_type}] {doc.title} ({doc.course_name}): {doc.text}"
            )
        context_block = "\n\n".join(context_lines)
        if self.max_context_chars and len(context_block) > self.max_context_chars:
            context_block = context_block[: self.max_context_chars]

        history_block = self._build_history(question)
        prompt = (
            "You are a helpful study assistant. CRITICAL RULES:\n"
            "1. Answer ONLY using information from the context below.\n"
            "2. Do NOT make up, infer, or hallucinate any information.\n"
            "3. If you cannot find the answer in the context, say: \"I don't have that information in your course materials.\"\n"
            "4. Do NOT add examples or information not explicitly in the context.\n\n"
        )
        if history_block:
            prompt += f"Recent conversation:\n{history_block}\n\n"
        prompt += f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer (use only context above):\n"

        response = self.llm.generate(prompt)
        response = self._sanitize_response(response)
        # guard: if LLM produced empty output, return a safe fallback
        if not response or not response.strip():
            fallback = (
                "I couldn't find an answer in your course materials. "
                "Try asking for specific assignment titles, announcements, or materials."
            )
            self._record_turn(question, fallback)
            return fallback

        self._record_turn(question, response)
        return response.strip()

    def _load_items(self, item_type: str) -> List[ChatDocument]:
        items = self.json_store.get_all_items(item_type)
        documents: List[ChatDocument] = []
        for item in items:
            summary = self._get_summary(item)
            text = self._build_item_text(item, summary)
            documents.append(
                ChatDocument(
                    item_id=item.get("id", ""),
                    item_type=item_type,
                    course_id=item.get("course_id", ""),
                    course_name=item.get("course_name", ""),
                    title=item.get("title", item.get("text", "")),
                    text=text,
                )
            )
        return documents

    def _get_summary(self, item: dict) -> str:
        return self.summary_store.get_summary(item.get("course_id", ""), item.get("id", ""))

    def _build_item_text(self, item: dict, summary: str) -> str:
        """Build context text from item. Prioritize extracted content over generated summaries."""
        description = item.get("description") or item.get("text") or ""
        parts = [f"Title: {item.get('title', '')}", f"Description: {description}"]
        if summary and len(summary) < 200 and not any(
            phrase in summary.lower()
            for phrase in [
                "actionable point",
                "download",
                "practice",
                "share your progress",
                "follow along",
            ]
        ):
            parts.append(f"Summary: {summary}")
        return "\n".join(parts).strip()

    def _build_history(self, question: str) -> str:
        if not self._history or not self._should_include_history(question):
            return ""
        turns = self._history[-self.max_history_turns :]
        lines = []
        for user_text, assistant_text in turns:
            lines.append(f"Student: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
        return "\n".join(lines)

    def _record_turn(self, question: str, response: str) -> None:
        self._history.append((question.strip(), response.strip()))
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]

    def _sanitize_response(self, response: str) -> str:
        if not response:
            return response
        text = response.strip()
        for marker in ("\nQuestion:", "\nStudent:", "Question:", "Student:"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        if text.lower().startswith("answer:"):
            text = text.split(":", 1)[1].strip()
        return text

    def _should_include_history(self, question: str) -> bool:
        lowered = question.lower()
        follow_up_markers = (
            "it",
            "that",
            "those",
            "these",
            "them",
            "they",
            "same",
            "previous",
            "earlier",
            "above",
            "also",
            "follow up",
            "follow-up",
            "compare",
            "instead",
            "then",
            "next",
        )
        return any(marker in lowered for marker in follow_up_markers)

    def _try_structured_answer(self, question: str) -> str | None:
        lowered = question.lower()
        if "how many" in lowered and "material" in lowered:
            count = len(self.json_store.get_all_items("materials"))
            return f"You have {count} materials in your Classroom data."
        if "how many" in lowered and "assignment" in lowered:
            count = len(self.json_store.get_all_items("assignments"))
            return f"You have {count} assignments in your Classroom data."
        if "how many" in lowered and "announcement" in lowered:
            count = len(self.json_store.get_all_items("announcements"))
            return f"You have {count} announcements in your Classroom data."
        if ("announcement" in lowered and "title" in lowered) or (
            "recent" in lowered and "announcement" in lowered
        ):
            anns = self.json_store.get_all_items("announcements")
            if not anns:
                return "You have no announcements in your Classroom data."
            if "title" in lowered:
                titles = [ann.get("title") for ann in anns if ann.get("title")]
                if not titles:
                    return "Could not find announcement titles."
                if len(titles) == 1:
                    return f"The announcement title is: {titles[0]}"
                return "The announcements are: " + ", ".join(titles) + "."
            if "recent" in lowered:
                latest = sorted(
                    anns, key=lambda a: a.get("updated_at") or a.get("created_at") or ""
                )[-1]
                title = latest.get("title", "Untitled")
                desc = latest.get("description", latest.get("text", ""))
                return f"Your most recent announcement is: {title}. {desc}"
            return f"Yes, you have {len(anns)} announcement(s)."
        if (
            "course" in lowered
            and (
                "name" in lowered
                or "what" in lowered
                or "which" in lowered
                or "am i" in lowered
                or "enrolled" in lowered
            )
        ):
            courses = self.json_store._load(
                self.json_store.data_dir / "courses" / "courses.json", {"courses": {}}
            )
            course_values = list((courses.get("courses") or {}).values())
            names = [entry.get("name", "") for entry in course_values if entry.get("name")]
            if not names:
                return "I could not find a course name in the local data."
            if len(names) == 1:
                return f"You are enrolled in {names[0]}."
            return "You are enrolled in: " + ", ".join(names) + "."
        return None
