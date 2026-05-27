"""CLI for the local Classroom chat assistant."""

import config
from chat.assistant import ChatAssistant


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
