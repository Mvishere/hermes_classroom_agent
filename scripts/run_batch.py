import json
import sys
from pathlib import Path
import time

# Ensure repo root is on sys.path so local modules import correctly when running script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from chat.assistant import ChatAssistant
from chat.router import Router

QUESTIONS_PATH = Path("tests/chat_cli_stress_questions.txt")
OUTPUT_PATH = Path("tests/chat_cli_stress_results.json")


def load_questions(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        q = raw.strip()
        if not q or q.startswith("#"):
            continue
        lines.append(q)
    return lines


def main():
    if not config.RAG_ENABLED:
        print("RAG_DISABLED")
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
    router = Router(assistant, config.DATA_DIRECTORY)
    questions = load_questions(QUESTIONS_PATH)
    results = []
    for q in questions:
        start = time.perf_counter()
        route = router.route(q)
        elapsed = time.perf_counter() - start
        results.append({
            "question": q,
            "context": {
                "intent": route.intent,
                "document_type": route.document_type,
                "engine": route.engine,
                "confidence": route.confidence,
                "evidence_source": route.evidence_source,
                "matched_documents": route.matched_documents,
            },
            "answer": route.answer,
            "seconds": round(elapsed, 2),
        })
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
