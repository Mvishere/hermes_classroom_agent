"""
Simple unified LLM wrapper for Hermes Classroom Agent.
Uses Qwen2.5 via Ollama backend.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import config
from rag.model_loader import load_qwen_backend
from rag.prompts import grounded_system_prompt


@dataclass
class GenerationStats:
    prompt_chars: int = 0
    inference_seconds: float = 0.0


class LLM:
    """
    Minimal stable LLM wrapper used across:
    - TopicExtractor
    - RAG chat
    - Knowledge updater
    """

    def __init__(
        self,
        model_path: str = "qwen2.5:7b",
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.backend = load_qwen_backend(
            model_path,
            backend=config.LLM_BACKEND,
            device=device,
            context_length=config.LLM_CONTEXT_LENGTH,
            quantization=config.LLM_QUANTIZATION,
            gpu_layers=config.LLM_GPU_LAYERS,
            threads=config.LLM_NUM_THREADS,
        )

        logging.info("LLM initialized: %s", model_path)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:

        system_prompt = system_prompt or grounded_system_prompt()
        full_prompt = f"{system_prompt}\n\n{prompt}"

        start = time.time()

        try:
            response = self.backend.generate(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=config.LLM_TOP_P,
            )

        except Exception as e:
            logging.exception("LLM generation failed")
            return "Error: LLM failed to generate response."

        duration = time.time() - start

        logging.info(
            "LLM response in %.2fs (chars=%d)",
            duration,
            len(prompt),
        )

        return self._clean(response)

    def stream(self, prompt: str):
        """Optional streaming support"""
        if hasattr(self.backend, "stream"):
            yield from self.backend.stream(prompt)
        else:
            yield self.generate(prompt)

    def _clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # remove accidental role leakage
        for tag in ["Question:", "Answer:", "Assistant:", "User:"]:
            if tag in text:
                text = text.split(tag)[0].strip()

        return text