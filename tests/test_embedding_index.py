from pathlib import Path

from rag.index import EmbeddingIndex


class FakeEmbeddingModel:
    def encode(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_embedding_index_upsert_and_search(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index = EmbeddingIndex(index_path, FakeEmbeddingModel())

    items = [
        {"id": "a", "course_id": "c1", "course_name": "Course", "text": "short"},
        {"id": "b", "course_id": "c1", "course_name": "Course", "text": "much longer"},
    ]

    added = index.upsert_items(items, lambda item: item["text"])
    assert added == 2

    results = index.search("longer", top_k=2)
    assert results[0]["item_id"] == "b"
    assert index_path.exists()
