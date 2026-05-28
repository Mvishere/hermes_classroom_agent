"""CLI for the local Classroom chat assistant."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
from chat.assistant import ChatAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Classroom chat assistant")
    parser.add_argument(
        "--batch-file",
        default="",
        help="Optional path to a text file containing one question per line.",
    )
    parser.add_argument(
        "--batch-output",
        default="",
        help="Optional path to write JSON results for batch mode.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=0,
        help="Optional maximum number of batch questions to run.",
    )
    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="Print per-question latency in batch mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not config.RAG_ENABLED:
        print("Chat is disabled. Set RAG_ENABLED=1 to enable it.")
        return 1
    if not config.LLM_MODEL_PATH or not config.EMBEDDING_MODEL_PATH:
        print("Missing LLM or embedding model path in config.")
        return 1

    assistant = ChatAssistant(
        data_dir=config.DATA_DIRECTORY,
        llm_model_path=config.LLM_MODEL_PATH,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
        device=config.RAG_DEVICE,
        top_k=config.RAG_TOP_K,
        max_context_chars=config.RAG_CONTEXT_MAX_CHARS,
        max_history_turns=config.CHAT_MAX_HISTORY_TURNS,
    )
    from chat.router import Router
    router = Router(assistant, config.DATA_DIRECTORY)

    if args.batch_file:
        questions = _load_questions(Path(args.batch_file), args.batch_limit)
        results = []
        print(f"Chat batch mode: {len(questions)} question(s)")
        for index, question in enumerate(questions, start=1):
            started = time.perf_counter()
            answer = router.handle(question)
            elapsed = time.perf_counter() - started
            results.append({"question": question, "answer": answer, "seconds": round(elapsed, 2)})
            if args.show_timing:
                print(f"[{index}/{len(questions)}] {elapsed:.2f}s | Q: {question}")
            print(f"Student> {question}")
            print(f"Assistant> {answer}")
        if args.batch_output:
            Path(args.batch_output).write_text(
                json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        return 0

    print("Chat ready. Type 'exit' to quit.")
    while True:
        try:
            question = input("Student> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = router.handle(question)
        print(f"Assistant> {answer}")

    return 0


def _load_questions(path: Path, limit: int) -> list[str]:
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        question = raw_line.strip()
        if not question or question.startswith("#"):
            continue
        lines.append(question)
        if limit and len(lines) >= limit:
            break
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
