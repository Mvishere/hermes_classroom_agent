"""Backend selection and loading for local Qwen2.5 inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import config


class BaseLLMBackend:
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float):
        raise NotImplementedError


class TransformersQwenBackend(BaseLLMBackend):
    def __init__(self, model_path: str, device: str, context_length: int, quantization: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = self._load_model(AutoModelForCausalLM, model_path, device, quantization)
        self.device = torch.device(device)
        self.context_length = context_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self, model_cls, model_path: str, device: str, quantization: str):
        import torch

        load_kwargs: dict[str, Any] = {"local_files_only": True}
        if device.startswith("cuda") and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        if quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs.pop("torch_dtype", None)
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                )
            except Exception:
                logging.exception("Quantized loading unavailable; falling back to standard transformers loading.")

        model = model_cls.from_pretrained(model_path, **load_kwargs)
        model.to(torch.device(device))
        model.eval()
        return model

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        messages = [
            {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self._apply_chat_template(messages)
        return self._generate_from_text(chat_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    def stream(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float):
        yield self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{messages[0]['content']}\n\nUser: {messages[1]['content']}\nAssistant:"

    def _generate_from_text(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, self.context_length - max_new_tokens - 8),
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class LlamaCppQwenBackend(BaseLLMBackend):
    def __init__(self, model_path: str, context_length: int, gpu_layers: int = 0, threads: int = 8):
        from llama_cpp import Llama

        self.llama = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_gpu_layers=gpu_layers,
            n_threads=threads,
            verbose=False,
        )

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        response = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()

    def stream(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float):
        chunks = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stream=True,
        )
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

class OllamaBackend(BaseLLMBackend):
    """Backend that talks to a locally-installed Ollama model.

    It prefers the Python `ollama` client if available, and falls back to
    calling the `ollama` CLI via subprocess when necessary.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = None
        self._use_python_client = False
        try:
            from ollama import Ollama  # type: ignore

            self._client = Ollama()
            self._use_python_client = True
        except Exception:
            # Python client unavailable; we'll use the CLI fallback at call time.
            self._client = None
            self._use_python_client = False

    def _build_prompt(self, prompt: str) -> str:
        return f"{config.LLM_SYSTEM_PROMPT}\n\n{prompt}"

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        payload = self._build_prompt(prompt)
        if self._use_python_client and self._client is not None:
            client = self._client
            # Best-effort: try common client method names and extract text.
            for method_name in ("generate", "chat", "create"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        resp = method(model=self.model_name, prompt=payload)
                    except TypeError:
                        # Some APIs expect positional args
                        resp = method(self.model_name, payload)

                    # resp may be a string or dict-like
                    if isinstance(resp, str):
                        return resp.strip()
                    if isinstance(resp, dict):
                        # try several common keys
                        for key in ("text", "content", "output", "choices"):
                            if key in resp:
                                val = resp[key]
                                if isinstance(val, list):
                                    val = val[0]
                                if isinstance(val, dict) and "content" in val:
                                    return val["content"].strip()
                                if isinstance(val, str):
                                    return val.strip()
                    # fallback to string coercion
                    return str(resp).strip()

        # CLI fallback
        import subprocess

        cmd = [
            "ollama",
            "run",
            self.model_name,
            payload,
        ]
        try:
            # Force UTF-8 decoding and replace invalid characters to avoid
            # Windows cp1252 decode errors when Ollama emits bytes outside
            # the local code page.
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Ollama CLI returned code {proc.returncode}: stderr={proc.stderr.strip()!r} stdout={proc.stdout.strip()!r}"
                )
            return proc.stdout.strip()
        except Exception as exc:
            # Surface whatever went wrong in a single error type for the caller.
            raise RuntimeError(f"Ollama CLI call failed: {exc}") from exc

    def stream(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float):
        # Ollama streaming via Python client or CLI is optional; provide simple non-streaming
        yield self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)


def load_qwen_backend(model_path: str, *, backend: str, device: str, context_length: int, quantization: str, gpu_layers: int, threads: int):
    start = __import__("time").perf_counter()
    path = Path(model_path)
    backend_name = backend.lower().strip()
    # Ollama models are referenced by name (e.g. "qwen2.5:7b") and are not local files
    if backend_name != "ollama":
        if not path.exists():
            raise FileNotFoundError(f"LLM model path not found: {model_path}")
    if backend_name == "ollama":
        loaded = OllamaBackend(model_path)
        logging.info("Loaded Qwen backend via Ollama (model=%s) in %.2fs", model_path, __import__("time").perf_counter() - start)
        return loaded

    if backend_name == "llama_cpp" or path.suffix.lower() == ".gguf":
        loaded = LlamaCppQwenBackend(str(path), context_length=context_length, gpu_layers=gpu_layers, threads=threads)
        logging.info("Loaded Qwen backend via llama.cpp in %.2fs", __import__("time").perf_counter() - start)
        return loaded

    loaded = TransformersQwenBackend(str(path), device=device, context_length=context_length, quantization=quantization)
    logging.info("Loaded Qwen backend via transformers in %.2fs", __import__("time").perf_counter() - start)
    return loaded
