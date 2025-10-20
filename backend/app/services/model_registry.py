from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..models.base import BaseModelWrapper
from ..models.canary import CanaryASRModel
from ..models.pyannote_model import PyannoteDiarizationModel
from ..models.qwen import QwenModel


@dataclass
class ModelStatus:
    identifier: str
    task: str
    loaded: bool
    description: str
    format: str
    params: dict = field(default_factory=dict)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, BaseModelWrapper] = {}
        self._task_index: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path("/models")
        self._hf_token: Optional[str] = None

    def configure(self, hf_token: Optional[str], cache_dir: Path) -> None:
        self._hf_token = hf_token
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._models = {
            "qwen": QwenModel(cache_dir=self._cache_dir, hf_token=self._hf_token),
            "canary": CanaryASRModel(cache_dir=self._cache_dir, hf_token=self._hf_token),
            "pyannote": PyannoteDiarizationModel(cache_dir=self._cache_dir, hf_token=self._hf_token),
        }
        self._task_index = {
            "chat-completion": "qwen",
            "speech-to-text": "canary",
            "speaker-diarization": "pyannote",
        }

    def keys(self) -> List[str]:
        return list(self._models.keys())

    async def get(self, key: str) -> BaseModelWrapper:
        async with self._lock:
            if key not in self._models:
                raise KeyError(f"Unknown model key: {key}")
            return self._models[key]

    async def get_by_task(self, task: str) -> BaseModelWrapper:
        key = self._task_index.get(task)
        if not key:
            raise KeyError(f"No model registered for task: {task}")
        return await self.get(key)

    async def ensure_loaded(self, key: str) -> None:
        model = await self.get(key)
        await model.ensure_loaded()

    async def unload(self, key: str) -> None:
        model = await self.get(key)
        await model.unload()

    async def status(self) -> Dict[str, ModelStatus]:
        result: Dict[str, ModelStatus] = {}
        async with self._lock:
            for key, model in self._models.items():
                metadata = model.metadata
                result[key] = ModelStatus(
                    identifier=metadata.identifier,
                    task=metadata.task,
                    loaded=model.is_loaded,
                    description=metadata.description,
                    format=metadata.format,
                    params=metadata.params,
                )
        return result

    async def shutdown(self) -> None:
        async with self._lock:
            for model in self._models.values():
                if model.is_loaded:
                    await model.unload()


registry = ModelRegistry()
