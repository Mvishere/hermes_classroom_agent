"""Prompt builders for grounded educational RAG workflows."""

from __future__ import annotations

from typing import Iterable

import config


def grounded_system_prompt() -> str:
    return config.LLM_SYSTEM_PROMPT


def educational_qa_prompt(question: str, context: str, task: str = "grounded_answer") -> str:
    return _build_prompt(
        task=task,
        question=question,
        context=context,
        instructions=(
            "Answer only from the retrieved course data. "
            "If the evidence is insufficient, explicitly say so. "
            "Do not invent educational relationships or course facts."
        ),
    )


def retrieval_summary_prompt(title: str, description: str, context: str, chunks: Iterable[str] | None = None) -> str:
    chunk_text = "\n".join(chunks or [])
    return _build_prompt(
        task="retrieval_summary",
        question=title,
        context=context,
        instructions=(
            "Summarize only the grounded retrieved material in 3-5 concise bullets. "
            "Focus on useful study points and avoid hallucinations."
        ),
        extra={
            "Material Title": title,
            "Material Description": description,
            "Chunk Summaries": chunk_text,
        },
    )


def topic_extraction_prompt(text: str, heuristic_payload: dict | None = None) -> str:
    """Prompt to extract canonical educational topics as strict JSON.

    The model must output JSON only using the schema:
    {
      "main_topics": [],
      "subtopics": [],
      "prerequisites": [],
      "domain": "",
      "difficulty": ""
    }

    The model must NOT return filenames, titles, attachments, or metadata.
    Provide few-shot examples to illustrate GOOD vs BAD outputs.
    """
    examples = (
        "Example GOOD: Text about CSS Grid and responsive layouts ->",
        '{"main_topics": ["CSS Grid"], "subtopics": ["Responsive Design", "Media Queries"], "prerequisites": ["HTML", "CSS"], "domain": "Web Development", "difficulty": "intermediate"}',
        "Example BAD: Text with attachment or title information ->",
        '{"main_topics": [], "subtopics": [], "prerequisites": [], "domain": "", "difficulty": ""} (DO NOT return attachment names or titles)'
    )

    parts = [
        grounded_system_prompt(),
        "Task: Extract canonical educational concepts from the provided text.",
        "Output MUST be valid JSON only and follow the schema exactly.",
        "Do NOT output filenames, attachment text labels, titles, or other metadata.",
        "Prefer canonical names like HTML, CSS, JavaScript, DOM, Flexbox, Media Queries.",
        "If no clear educational concepts are present, return empty arrays and empty strings.",
    ]
    if heuristic_payload:
        parts.append("Heuristic hint: " + str(heuristic_payload))
    parts.append("Text:")
    parts.append(text)
    parts.extend(examples)
    parts.append("Now output JSON only:")
    return "\n\n".join(parts)


def topic_reasoning_prompt(topic: str, evidence: str, relation: str) -> str:
    return _build_prompt(
        task="topic_reasoning",
        question=topic,
        context=evidence,
        instructions=(
            "Use only the topic graph evidence. "
            "If the graph does not contain enough grounded relationships, say so explicitly."
        ),
        extra={"Relation Type": relation},
    )


def _build_prompt(task: str, question: str, context: str, instructions: str, extra: dict[str, str] | None = None) -> str:
    parts = [
        grounded_system_prompt(),
        f"Task: {task}",
        instructions,
    ]
    if extra:
        for key, value in extra.items():
            if value:
                parts.append(f"{key}: {value}")
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n\n".join(parts)
