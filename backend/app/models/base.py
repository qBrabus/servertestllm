from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class ModelMetadata:
    identifier: str
    task: str
    description: str = ""
    format: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""

    def __init__(
        self,
        metadata: ModelMetadata,
        cache_dir: Path,
        hf_token: Optional[str] = None,
        preferred_device_ids: Sequence[int] | None = None,
    ):
        self.metadata = metadata
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self._is_loaded = False
        self._lock = asyncio.Lock()
        self._preferred_device_ids = list(preferred_device_ids or [])

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def preferred_device_ids(self) -> list[int]:
        return list(self._preferred_device_ids)

    def update_device_preferences(self, device_ids: Sequence[int] | None) -> bool:
        new_ids = list(device_ids or [])
        if new_ids != self._preferred_device_ids:
            self._preferred_device_ids = new_ids
            return True
        return False

    def primary_device(self) -> Optional[int]:
        return self._preferred_device_ids[0] if self._preferred_device_ids else None

    async def ensure_loaded(self) -> None:
        if not self._is_loaded:
            async with self._lock:
                if not self._is_loaded:
                    await self.load()
                    self._is_loaded = True

    async def unload(self) -> None:
        async with self._lock:
            await self._unload()
            self._is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def _unload(self) -> None:
        ...

    @abstractmethod
    async def infer(self, *args: Any, **kwargs: Any) -> Any:
        ...
