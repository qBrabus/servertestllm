from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModelWrapper, ModelMetadata


class QwenModel(BaseModelWrapper):
    """Wrapper around the Qwen3 VL instruction-tuned model."""

    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def __init__(self, cache_dir, hf_token=None):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="chat-completion",
            description="Qwen3 VL 30B A3B Instruct model for multimodal chat completions",
            format="chatml",
        )
        super().__init__(metadata, cache_dir, hf_token)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self) -> None:
        def _load():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                token=auth_token,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=str(self.cache_dir),
                token=auth_token,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        await asyncio.to_thread(_load)

    async def _unload(self) -> None:
        def _cleanup():
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        await asyncio.to_thread(_cleanup)

    async def infer(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            prompt = self._build_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_config = dict(
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = text[len(prompt) :].strip()
            return {"content": response}

        return await asyncio.to_thread(_run)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        history = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            history.append(f"[{role.upper()}]: {content}")
        history.append("[ASSISTANT]:")
        return "\n".join(history)
