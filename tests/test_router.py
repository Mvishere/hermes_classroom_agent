import tempfile
from types import SimpleNamespace
from pathlib import Path

from chat.router import Router


class FakeEmbedding:
    def encode(self, texts):
        out = []
        for t in texts:
            s = t.lower()
            if any(k in s for k in ("course", "announcement", "assignment", "title", "how many")):
                out.append([1.0, 0.0, 0.0])
            elif any(k in s for k in ("summarize", "summary")):
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


def test_router_classification_and_handle_factual():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        # create minimal announcements so factual tool can answer
        import json
        anns = {"courses": {"c1": {"course_id": "c1", "course_name": "test", "items": [{"id": "a1", "title": "Hello"}]}}}
        (data_dir / "announcements").mkdir(parents=True, exist_ok=True)
        (data_dir / "announcements" / "announcements.json").write_text(json.dumps(anns), encoding="utf-8")

        assistant = SimpleNamespace(embedding_model=FakeEmbedding())
        router = Router(assistant, data_dir)

        kind = router.classify("Tell me the title of all announcements")
        assert kind == "factual"

        resp = router.handle("Tell me the title of all announcements")
        assert "Hello" in resp
