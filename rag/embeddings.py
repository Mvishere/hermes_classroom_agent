import requests


class EmbeddingModel:
    """
    Ollama embedding wrapper (mxbai-embed-large)
    """

    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        for text in texts:
            res = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            res.raise_for_status()
            embeddings.append(res.json()["embedding"])

        return embeddings