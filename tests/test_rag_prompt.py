import config
from rag.pipeline import RagPipeline


def test_rag_prompt_includes_context(monkeypatch) -> None:
    monkeypatch.setattr(config, "PDF_EXTRACT_ENABLED", False)
    pipeline = RagPipeline.__new__(RagPipeline)

    material = {
        "title": "Unit 1",
        "description": "Intro",
        "attachment_paths": [],
    }
    contexts = [
        {"course_name": "Course A", "text": "First context"},
        {"course_name": "Course B", "text": "Second context"},
    ]

    prompt, formats = RagPipeline._build_prompt(pipeline, material, contexts)
    assert "Material Title: Unit 1" in prompt
    assert "Retrieved Context:" in prompt
    assert formats == []
