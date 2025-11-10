from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List

import psutil


LOGGER = logging.getLogger(__name__)


@dataclass
class GPUStatus:
    id: int
    name: str
    memory_total: float
    memory_used: float
    load: float
    temperature: float | None = None


class GPUMonitor:
    """Background collector for GPU telemetry used by the admin dashboard."""

    def __init__(self) -> None:
        self._data: Dict[int, GPUStatus] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, interval: float = 5.0) -> None:
        """Start the monitoring thread if it is not already running."""

        if self._thread and self._thread.is_alive():
            LOGGER.debug("GPU monitor already running")
            return

        def _run() -> None:
            LOGGER.debug("GPU monitor thread started (interval=%ss)", interval)
            while not self._stop_event.is_set():
                try:
                    self._collect()
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.exception("Unexpected error while collecting GPU metrics")
                self._stop_event.wait(interval)
            LOGGER.debug("GPU monitor thread stopped")

        LOGGER.info("Starting GPU monitor (interval=%ss)", interval)
        self._stop_event.clear()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread and wait briefly for termination."""

        self._stop_event.set()
        if self._thread:
            LOGGER.info("Stopping GPU monitor")
            self._thread.join(timeout=1.5)
            self._thread = None

    def _collect(self) -> None:
        try:
            import torch
        except Exception:
            if self._data:
                LOGGER.warning("torch unavailable; clearing cached GPU metrics")
            self._data = {}
            return

        if not torch.cuda.is_available():
            if self._data:
                LOGGER.debug("CUDA not available; clearing GPU metrics")
            self._data = {}
            return

        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
        except Exception:
            LOGGER.warning("Unable to collect GPU information via GPUtil", exc_info=True)
            self._data = {}
            return

        if not gpus:
            LOGGER.debug("No GPUs reported by GPUtil")
            self._data = {}
            return

        for gpu in gpus:
            self._data[gpu.id] = GPUStatus(
                id=gpu.id,
                name=gpu.name,
                memory_total=gpu.memoryTotal,
                memory_used=gpu.memoryUsed,
                load=gpu.load,
                temperature=getattr(gpu, "temperature", None),
            )

    def get_status(self) -> Dict[int, GPUStatus]:
        """Return a shallow copy of the latest GPU metrics."""

        return self._data.copy()

    def system_metrics(self) -> Dict[str, float]:
        """Collect CPU and RAM usage for the host system."""

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
        }


gpu_monitor = GPUMonitor()
