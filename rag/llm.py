"""Local LLM wrapper for offline summarization."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """Loads a local causal LM and generates summaries."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ):
        if not model_path:
            raise ValueError("LLM model path is required.")
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        max_input_tokens = self._get_max_input_tokens()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        total_max_length = min(
            input_len + self.max_new_tokens,
            self._get_model_max_length(),
        )
        output_ids = self.model.generate(
            **inputs,
            max_length=total_max_length,
            do_sample=self.temperature > 0,
            **({"temperature": self.temperature} if self.temperature > 0 else {}),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _get_max_input_tokens(self) -> int:
        model_max = self._get_model_max_length()
        max_input = int(model_max) - int(self.max_new_tokens) - 1
        return max(1, max_input)

    def _get_model_max_length(self) -> int:
        model_max = getattr(self.tokenizer, "model_max_length", None)
        if not model_max or model_max > 100000:
            model_max = 2048

        config_max = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if config_max:
            model_max = min(model_max, int(config_max))
        return int(model_max)
