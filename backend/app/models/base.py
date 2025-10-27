from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional, Sequence

from ..config import settings


@dataclass
class ModelMetadata:
    identifier: str
    task: str
    description: str = ""
    format: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


LOGGER = logging.getLogger(__name__)


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

    async def ensure_downloaded(self) -> None:
        if self.is_downloaded():
            return
        async with self._lock:
            if self.is_downloaded():
                return
            previous_state = self.runtime_status().get("state", "idle")
            was_loaded = self._is_loaded
            self.update_runtime(
                state="loading" if not was_loaded else previous_state,
                progress=5,
                status="Downloading model artifacts",
                downloaded=False,
            )
            try:
                await self.download()
            except Exception as exc:
                self.update_runtime(
                    state=previous_state if was_loaded else "error",
                    status=f"Download failed: {exc}",
                    progress=0,
                    last_error=str(exc),
                )
                raise
            else:
                if was_loaded:
                    self.update_runtime(
                        state=previous_state,
                        status="Model ready",
                        downloaded=True,
                        progress=100,
                    )
                else:
                    self.update_runtime(
                        state="idle",
                        status="Model cached",
                        progress=100,
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
    async def download(self) -> None:
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
        return self.cache_has_artifacts(self.cache_dir, self.metadata.identifier)

    @classmethod
    def cache_has_artifacts(cls, cache_dir: Path, identifier: str) -> bool:
        repo_dir = cls.compute_cache_repo_dir(cache_dir, identifier)
        if not repo_dir.exists() or not repo_dir.is_dir():
            return False

        snapshots_dir = repo_dir / "snapshots"
        search_roots = []
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            search_roots.append(snapshots_dir)
        search_roots.append(repo_dir)

        for root in search_roots:
            try:
                next(p for p in root.rglob("*") if p.is_file())
                return True
            except StopIteration:
                continue
            except (OSError, PermissionError):  # pragma: no cover - defensive
                continue
        return False

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

    # ------------------------------------------------------------------
    # Helpers for enriched runtime information
    # ------------------------------------------------------------------

    def build_server_metadata(self, *, endpoint: str, protocol: str = "http", **extras: Any) -> Dict[str, Any]:
        """Build a descriptive payload about an exposed API endpoint.

        The returned dictionary is intended to be serialised and displayed
        in the admin UI so we include additional convenience information
        such as the computed URL and OpenAPI documentation entry point.
        """

        host = settings.api_host
        port = settings.api_port
        display_host = host
        if host in {"0.0.0.0", "::"}:
            display_host = "localhost"

        base_url = f"{protocol}://{display_host}:{port}"
        metadata: Dict[str, Any] = {
            "host": host,
            "port": port,
            "protocol": protocol,
            "endpoint": endpoint,
            "url": f"{base_url}{endpoint}",
            "docs": f"{base_url}/docs",
            "openapi": f"{base_url}/openapi.json",
        }
        metadata.update(extras)
        return metadata

    # ------------------------------------------------------------------
    # Download helpers with progress tracking
    # ------------------------------------------------------------------

    def download_snapshot(
        self,
        *,
        repo_id: str,
        auth_token: Optional[str],
        status_prefix: str,
        progress_range: tuple[int, int] = (5, 90),
        allow_patterns: Sequence[str] | None = None,
        complete_status: Optional[str] = None,
        mark_as_cached: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Download a Hugging Face snapshot while updating runtime progress."""

        from ..utils import snapshot_download_with_retry

        start_progress, end_progress = progress_range
        start_progress = max(0, min(100, start_progress))
        end_progress = max(start_progress, min(100, end_progress))

        repo_dir = self.compute_cache_repo_dir(self.cache_dir, repo_id)
        total_bytes = self._resolve_remote_size(repo_id, auth_token)

        # Initial status update
        self.update_runtime(
            status=f"{status_prefix} (0%)",
            progress=start_progress,
        )

        progress_lock = threading.Lock()
        known_total = total_bytes if total_bytes > 0 else None
        last_progress = start_progress

        def emit_progress(current_bytes: int, total: Optional[int] = None) -> None:
            nonlocal known_total, last_progress
            if total and total > 0:
                known_total = total

            total_to_use = total or known_total
            if total_to_use and total_to_use > 0:
                fraction = min(1.0, current_bytes / total_to_use)
                mapped_progress = int(start_progress + (end_progress - start_progress) * fraction)
                percent = int(fraction * 100)
                with progress_lock:
                    if mapped_progress != last_progress:
                        last_progress = mapped_progress
                        self.update_runtime(
                            status=f"{status_prefix} ({percent}%)",
                            progress=mapped_progress,
                        )
            else:
                # Fall back to displaying transferred volume when total is unknown.
                with progress_lock:
                    self.update_runtime(
                        status=f"{status_prefix} ({self._format_bytes(current_bytes)})",
                    )

        stop_event = threading.Event()
        monitor_thread: threading.Thread | None = None
        if known_total is None:
            monitor_thread = threading.Thread(
                target=self._monitor_download_progress,
                args=(repo_dir, stop_event, emit_progress),
                daemon=True,
            )
            monitor_thread.start()

        def progress_callback(progress: Any) -> None:  # pragma: no cover - callback from hub
            current = getattr(progress, "current", None)
            total = getattr(progress, "total", None)
            if current is None:
                return
            try:
                current_int = int(current)
            except (TypeError, ValueError):
                return
            total_int: Optional[int] = None
            if total is not None:
                try:
                    total_int = int(total)
                except (TypeError, ValueError):
                    total_int = None
            emit_progress(current_int, total_int)

        try:
            download_root = snapshot_download_with_retry(
                repo_id=repo_id,
                cache_dir=str(self.cache_dir),
                token=auth_token,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
                progress_callback=progress_callback,
                **kwargs,
            )
        except Exception:
            stop_event.set()
            if monitor_thread:
                monitor_thread.join(timeout=2)
            raise
        else:
            stop_event.set()
            if monitor_thread:
                monitor_thread.join(timeout=2)

        final_status = complete_status or f"{status_prefix} terminé"
        self.update_runtime(
            status=final_status,
            progress=end_progress,
            downloaded=mark_as_cached,
        )
        return Path(download_root)

    def _resolve_remote_size(self, repo_id: str, auth_token: Optional[str]) -> int:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=auth_token)
            info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
            total = 0
            for sibling in getattr(info, "siblings", []):
                size = getattr(sibling, "size", None)
                if isinstance(size, int):
                    total += size
            return total
        except Exception as exc:  # pragma: no cover - best effort logging only
            LOGGER.debug("Unable to resolve remote size for %s: %s", repo_id, exc)
            return 0

    def _monitor_download_progress(
        self,
        repo_dir: Path,
        stop_event: threading.Event,
        emit_progress: Callable[[int, Optional[int]], None],
    ) -> None:
        last_bytes = -1
        while not stop_event.is_set():
            downloaded = self._estimate_local_bytes(repo_dir)
            if downloaded != last_bytes:
                last_bytes = downloaded
                emit_progress(downloaded, None)
            stop_event.wait(0.8)

    @staticmethod
    def _estimate_local_bytes(repo_dir: Path) -> int:
        if not repo_dir.exists():
            return 0
        total = 0
        try:
            for path in repo_dir.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except OSError:  # pragma: no cover - transient file issues
                        continue
        except (OSError, PermissionError):  # pragma: no cover - defensive
            return total
        return total

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes <= 0:
            return "0 o"
        units = ["o", "Ko", "Mo", "Go", "To"]
        value = float(num_bytes)
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        if unit_index == 0:
            return f"{int(value)} {units[unit_index]}"
        return f"{value:.1f} {units[unit_index]}"
