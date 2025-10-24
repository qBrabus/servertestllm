from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..models.base import BaseModelWrapper, ModelMetadata


@dataclass
class ModelStatus:
    identifier: str
    task: str
    loaded: bool
    description: str
    format: str
    params: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)


@dataclass
class ModelSlot:
    """Holds the factory and runtime state for a registered model."""

    metadata: ModelMetadata
    factory: Callable[[], BaseModelWrapper]
    instance: BaseModelWrapper | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._slots: Dict[str, ModelSlot] = {}
        self._task_index: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path("/models")
        self._hf_token: Optional[str] = None

    def configure(self, hf_token: Optional[str], cache_dir: Path) -> None:
        self._hf_token = hf_token
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._apply_token_to_environment()
        self._register_defaults()

    def _register_defaults(self) -> None:
        def _qwen_factory() -> BaseModelWrapper:
            from ..models.qwen import QwenModel

            return QwenModel(
                cache_dir=self._cache_dir,
                hf_token=self._hf_token,
                preferred_device_ids=[0],
            )

        def _canary_factory() -> BaseModelWrapper:
            from ..models.canary import CanaryASRModel

            return CanaryASRModel(
                cache_dir=self._cache_dir,
                hf_token=self._hf_token,
                preferred_device_ids=[0],
            )

        def _pyannote_factory() -> BaseModelWrapper:
            from ..models.pyannote_model import PyannoteDiarizationModel

            return PyannoteDiarizationModel(
                cache_dir=self._cache_dir,
                hf_token=self._hf_token,
                preferred_device_ids=[0],
            )

        self._slots = {
            "qwen": ModelSlot(
                metadata=ModelMetadata(
                    identifier="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    task="chat-completion",
                    description="Qwen3 VL 30B A3B Instruct model for multimodal chat completions",
                    format="chatml",
                ),
                factory=_qwen_factory,
            ),
            "canary": ModelSlot(
                metadata=ModelMetadata(
                    identifier="nvidia/canary-1b-v2",
                    task="speech-to-text",
                    description="NVIDIA Canary multilingual ASR model",
                    format="wav/ogg/flac",
                ),
                factory=_canary_factory,
            ),
            "pyannote": ModelSlot(
                metadata=ModelMetadata(
                    identifier="pyannote/speaker-diarization-community-1",
                    task="speaker-diarization",
                    description="Pyannote diarization community pipeline",
                    format="wav/ogg/flac",
                ),
                factory=_pyannote_factory,
            ),
        }
        self._task_index = {slot.metadata.task: key for key, slot in self._slots.items()}

    def keys(self) -> List[str]:
        return list(self._slots.keys())

    def get_hf_token(self) -> Optional[str]:
        return self._hf_token

    async def set_hf_token(self, token: Optional[str]) -> None:
        async with self._lock:
            self._hf_token = token
            self._apply_token_to_environment()
            for slot in self._slots.values():
                if slot.instance is not None:
                    slot.instance.set_hf_token(token)

    def _apply_token_to_environment(self) -> None:
        if self._hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = self._hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self._hf_token
        else:
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    async def get(self, key: str) -> BaseModelWrapper:
        async with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                raise KeyError(f"Unknown model key: {key}")
            if slot.instance is None:
                slot.instance = slot.factory()
            return slot.instance

    async def get_by_task(self, task: str) -> BaseModelWrapper:
        key = self._task_index.get(task)
        if not key:
            raise KeyError(f"No model registered for task: {task}")
        return await self.get(key)

    async def ensure_loaded(self, key: str, device_ids: Optional[List[int]] = None) -> None:
        model = await self.get(key)
        if device_ids is not None:
            preferences_changed = model.update_device_preferences(device_ids)
            if preferences_changed and model.is_loaded:
                await model.unload()
        await model.ensure_loaded()

    async def unload(self, key: str) -> None:
        async with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                raise KeyError(f"Unknown model key: {key}")
            model = slot.instance
        if model:
            await model.unload()

    async def status(self) -> Dict[str, ModelStatus]:
        result: Dict[str, ModelStatus] = {}
        async with self._lock:
            for key, slot in self._slots.items():
                model = slot.instance
                metadata = slot.metadata
                params = dict(metadata.params)
                if model:
                    params["device_ids"] = model.preferred_device_ids
                    runtime = model.runtime_status()
                else:
                    params["device_ids"] = params.get("device_ids") or []
                    cache_dir = BaseModelWrapper.compute_cache_repo_dir(
                        self._cache_dir, metadata.identifier
                    )
                    runtime = {
                        "state": "idle",
                        "progress": 0,
                        "status": "Not loaded",
                        "details": {"preferred_device_ids": []},
                        "server": None,
                        "last_error": None,
                        "downloaded": cache_dir.exists(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                result[key] = ModelStatus(
                    identifier=metadata.identifier,
                    task=metadata.task,
                    loaded=model.is_loaded if model else False,
                    description=metadata.description,
                    format=metadata.format,
                    params=params,
                    runtime=runtime,
                )
        return result

    async def shutdown(self) -> None:
        async with self._lock:
            for slot in self._slots.values():
                model = slot.instance
                if model and model.is_loaded:
                    await model.unload()


registry = ModelRegistry()
