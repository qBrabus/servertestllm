from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List

import psutil
import torch
import GPUtil


@dataclass
class GPUStatus:
    id: int
    name: str
    memory_total: float
    memory_used: float
    load: float
    temperature: float | None = None


class GPUMonitor:
    def __init__(self) -> None:
        self._data: Dict[int, GPUStatus] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, interval: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _run():
            while not self._stop_event.is_set():
                self._collect()
                time.sleep(interval)

        self._stop_event.clear()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _collect(self) -> None:
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
            except Exception:
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
        else:
            self._data = {}

    def get_status(self) -> Dict[int, GPUStatus]:
        return self._data.copy()

    def system_metrics(self) -> Dict[str, float]:
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
        }


gpu_monitor = GPUMonitor()
