"""Intent-based router that uses embeddings to classify questions.

This router computes embeddings for a small set of example phrases for
each intent (factual, summarize, general) using the assistant's
embedding model. Incoming questions are embedded and compared (dot
product on normalized embeddings) to each intent vector; the best match
above a threshold determines routing.
"""
from typing import Dict, List

import numpy as np

from chat.tools import factual_tool, summarization_tool, general_tool
import logging


class Router:
    def __init__(self, assistant, data_dir: str, threshold: float = 0.60):
        self.assistant = assistant
        self.data_dir = data_dir
        self.threshold = threshold

        # Define example phrases per intent
        self.intent_examples: Dict[str, List[str]] = {
            "factual": [
                "what courses am I in",
                "which courses am i enrolled in",
                "recent announcement",
                "list announcements",
                "announcement titles",
                "title of announcements",
                "title of all announcements",
                "tell me the title of all announcements",
                "annoucements",
                "what is the title for my recent quiz",
                "title for my recent quiz",
                "recent quiz title",
                "what is the title of my assignment",
                "how many assignments are pending",
                "show assignment titles",
            ],
            "summarize": [
                "summarize assignments",
                "give me a summary of the material",
                "summarize the recent lecture",
                "can you summarize this",
            ],
            "general": [
                "what are the prerequisites for this assignment",
                "what skills do i need",
                "how should i prepare",
                "explain the topic",
            ],
        }

        # Precompute averaged intent embeddings using the assistant's embedding model
        self.intent_vectors: Dict[str, np.ndarray] = {}
        self._build_intent_vectors()

    def _build_intent_vectors(self) -> None:
        model = self.assistant.embedding_model
        for intent, examples in self.intent_examples.items():
            try:
                vectors = model.encode(examples)
                arr = np.array(vectors, dtype=float)
                # average the normalized example vectors
                mean = np.mean(arr, axis=0)
                # re-normalize
                norm = np.linalg.norm(mean)
                if norm > 0:
                    mean = mean / norm
                self.intent_vectors[intent] = mean
            except Exception:
                # fallback: zero vector
                self.intent_vectors[intent] = np.zeros((model.encode(["."])[0].__len__(),), dtype=float)

    def classify(self, question: str) -> str:
        q = question.strip()
        if not q:
            return "general"

        q_vec = np.array(self.assistant.embedding_model.encode([q])[0], dtype=float)
        # vectors are normalized by the embedding model; ensure q_vec normalized
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        best_intent = "general"
        best_score = -1.0
        for intent, vec in self.intent_vectors.items():
            score = float(np.dot(q_vec, vec))
            logging.debug("Intent score: %s -> %.4f", intent, score)
            if score > best_score:
                best_score = score
                best_intent = intent
        logging.debug("Best intent=%s score=%.4f threshold=%.4f", best_intent, best_score, self.threshold)
        if best_score >= self.threshold:
            return best_intent
        return "general"

    def handle(self, question: str) -> str:
        kind = self.classify(question)
        if kind == "factual":
            resp = factual_tool.handle(question, str(self.data_dir))
            if resp:
                return resp
        # targeted heuristic fallback for announcement-related phrasing (covers misspellings)
        qlow = question.lower()
        if any(k in qlow for k in ("announcement", "announ", "annou", "annoucements", "announcements")):
            resp = factual_tool.handle(question, str(self.data_dir))
            if resp:
                return resp
        # targeted heuristic fallback for quiz/assignment title queries
        if "quiz" in qlow and "title" in qlow:
            resp = factual_tool.handle(question, str(self.data_dir))
            if resp:
                return resp
        if "title" in qlow and ("assignment" in qlow or "quiz" in qlow):
            resp = factual_tool.handle(question, str(self.data_dir))
            if resp:
                return resp
        if kind == "summarize":
            resp = summarization_tool.handle(question, str(self.data_dir))
            if resp:
                return resp
        # fallback to general RAG-based assistant
        return general_tool.handle(question, self.assistant)
