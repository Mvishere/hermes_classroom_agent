"""Embedding model wrapper for local retrieval."""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Loads a local embedding model and encodes text."""

    def __init__(self, model_path: str, device: str = "cpu"):
        if not model_path:
            raise ValueError("Embedding model path is required.")
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.tolist()
