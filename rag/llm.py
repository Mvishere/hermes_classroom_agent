"""Qwen2.5 local LLM facade for educational inference tasks."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import config
from rag.model_loader import load_qwen_backend
from rag.prompts import grounded_system_prompt


@dataclass(slots=True)
class GenerationStats:
    load_seconds: float = 0.0
    prompt_chars: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    inference_seconds: float = 0.0
    timed_out: bool = False


class LocalLLM:
    """Compatibility facade for the rest of the agent stack.

    The facade keeps the previous `generate(prompt)` API but routes all
    generation through a configurable Qwen2.5 backend.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        if not model_path:
            raise ValueError("LLM model path is required.")
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.top_p = config.LLM_TOP_P
        self.context_length = config.LLM_CONTEXT_LENGTH
        self.timeout_seconds = config.LLM_TIMEOUT_SECONDS
        self.backend_name = config.LLM_BACKEND
        started_at = time.perf_counter()
        self.backend = load_qwen_backend(
            model_path,
            backend=self.backend_name,
            device=device or config.LLM_DEVICE,
            context_length=self.context_length,
            quantization=config.LLM_QUANTIZATION,
            gpu_layers=config.LLM_GPU_LAYERS,
            threads=config.LLM_NUM_THREADS,
        )
        self.load_seconds = time.perf_counter() - started_at
        logging.info(
            "Qwen model loaded from %s via %s in %.2fs",
            model_path,
            self.backend_name,
            self.load_seconds,
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None, timeout_seconds: Optional[int] = None) -> str:
        stats = GenerationStats(
            load_seconds=self.load_seconds,
            prompt_chars=len(prompt),
            prompt_tokens=self._estimate_tokens(prompt),
        )
        started_at = time.perf_counter()
        effective_timeout = timeout_seconds or self.timeout_seconds
        system_prompt = system_prompt or grounded_system_prompt()
        payload = f"{system_prompt}\n\n{prompt}"

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.backend.generate,
                    payload,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                response = future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError:
            stats.timed_out = True
            logging.warning(
                "Qwen inference timed out after %ss (prompt_chars=%s, prompt_tokens~%s)",
                effective_timeout,
                stats.prompt_chars,
                stats.prompt_tokens,
            )
            return "I don't have enough information in the local course data to answer that reliably."
        except Exception:
            logging.exception("Qwen inference failed.")
            return "I don't have enough information in the local course data to answer that reliably."

        stats.inference_seconds = time.perf_counter() - started_at
        stats.response_tokens = self._estimate_tokens(response)
        logging.info(
            "Qwen inference finished in %.2fs (prompt_chars=%s, prompt_tokens~%s, response_tokens~%s)",
            stats.inference_seconds,
            stats.prompt_chars,
            stats.prompt_tokens,
            stats.response_tokens,
        )
        return self._sanitize_response(response)

    def stream(self, prompt: str, *, system_prompt: str | None = None):
        system_prompt = system_prompt or grounded_system_prompt()
        payload = f"{system_prompt}\n\n{prompt}"
        if hasattr(self.backend, "stream"):
            yield from self.backend.stream(
                payload,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            yield self.generate(prompt, system_prompt=system_prompt)

    def _estimate_tokens(self, text: str) -> int:
        try:
            tokenizer = getattr(self.backend, "tokenizer", None)
            if tokenizer is not None:
                return len(tokenizer.encode(text))
        except Exception:
            pass
        return max(1, len(text) // 4)

    def _sanitize_response(self, response: str) -> str:
        if not response:
            return response
        text = response.strip()
        for marker in ("Question:", "Student:", "Assistant:"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        return text
