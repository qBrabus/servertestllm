from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
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

    _UNSET = object()

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
        self._runtime_state_lock = Lock()
        self._runtime_state: Dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "status": "Idle",
            "details": {},
            "server": None,
            "last_error": None,
            "downloaded": self.is_downloaded(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

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

    def set_hf_token(self, token: Optional[str]) -> None:
        """Update the Hugging Face token used by the wrapper."""

        self.hf_token = token

    async def ensure_loaded(self) -> None:
        if not self._is_loaded:
            async with self._lock:
                if not self._is_loaded:
                    self.update_runtime(
                        state="loading",
                        progress=5,
                        status="Preparing model load",
                        downloaded=self.is_downloaded(),
                    )
                    try:
                        await self.load()
                    except Exception as exc:
                        self.update_runtime(
                            state="error",
                            status=f"Load failed: {exc}",
                            progress=0,
                            last_error=str(exc),
                        )
                        raise
                    else:
                        self._is_loaded = True
                        self.update_runtime(
                            state="ready",
                            progress=100,
                            status="Model ready",
                            downloaded=True,
                        )

    async def unload(self) -> None:
        async with self._lock:
            await self._unload()
            self._is_loaded = False
            self.update_runtime(
                state="idle",
                progress=0,
                status="Model unloaded",
                server=None,
                downloaded=self.is_downloaded(),
            )

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def _unload(self) -> None:
        ...

    @abstractmethod
    async def infer(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @staticmethod
    def compute_cache_repo_dir(cache_dir: Path, identifier: str) -> Path:
        """Return the expected Hugging Face cache directory for a model."""

        sanitized = identifier.replace("/", "--")
        return cache_dir / f"models--{sanitized}"

    def cache_repo_dir(self) -> Path:
        return self.compute_cache_repo_dir(self.cache_dir, self.metadata.identifier)

    def is_downloaded(self) -> bool:
        repo_dir = self.cache_repo_dir()
        return repo_dir.exists()

    def update_runtime(
        self,
        *,
        state: Optional[str] = None,
        progress: Optional[float] = None,
        status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = _UNSET,
        server: Optional[Dict[str, Any] | None] = _UNSET,
        downloaded: Optional[bool] = None,
        last_error: object = _UNSET,
    ) -> None:
        """Atomically update the runtime state exposed to the dashboard."""

        with self._runtime_state_lock:
            if state is not None:
                self._runtime_state["state"] = state
            if progress is not None:
                self._runtime_state["progress"] = max(0, min(100, int(progress)))
            if status is not None:
                self._runtime_state["status"] = status
            if details is not self._UNSET:
                self._runtime_state["details"] = details or {}
            if server is not self._UNSET:
                self._runtime_state["server"] = server
            if downloaded is not None:
                self._runtime_state["downloaded"] = downloaded
            if last_error is not self._UNSET:
                self._runtime_state["last_error"] = last_error
            self._runtime_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    def runtime_status(self) -> Dict[str, Any]:
        with self._runtime_state_lock:
            snapshot = deepcopy(self._runtime_state)
        # Ensure the downloaded flag stays in sync with on-disk cache state
        snapshot["downloaded"] = snapshot.get("downloaded", False) or self.is_downloaded()
        return snapshot
