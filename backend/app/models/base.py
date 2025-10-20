from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelMetadata:
    identifier: str
    task: str
    description: str = ""
    format: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""

    def __init__(self, metadata: ModelMetadata, cache_dir: Path, hf_token: Optional[str] = None):
        self.metadata = metadata
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self._is_loaded = False
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

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
