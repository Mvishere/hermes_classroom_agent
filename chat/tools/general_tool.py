"""General tool: fallback to RAG-based assistant for conceptual or open questions."""
from typing import Optional

from chat.assistant import ChatAssistant


def handle(question: str, assistant: ChatAssistant) -> Optional[str]:
    # Use the assistant's RAG pipeline to answer general conceptual questions
    return assistant.answer(question)
