"""Application package initialisation hooks."""

from __future__ import annotations

import multiprocessing as _mp
import os as _os


def _ensure_spawn_start_method() -> None:
    """Force the use of the ``spawn`` start method for CUDA workloads.

    PyTorch and vLLM both rely on CUDA contexts that are not compatible with the
    default ``fork`` start method that CPython enables on Linux. When the forked
    worker process touches CUDA APIs, the runtime raises
    ``RuntimeError: Cannot re-initialize CUDA in forked subprocess`` and model
    loading aborts (as observed when instantiating the Qwen engine).

    We proactively switch the global multiprocessing context *before* any CUDA
    library is imported so that all background tasks (threads created by vLLM,
    torchaudio, or NeMo) inherit the safe ``spawn`` semantics.
    """

    try:
        if _mp.get_start_method(allow_none=True) != "spawn":
            _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # The context may already be configured by the application runner; in
        # that case we simply continue with the existing configuration.
        pass

    try:
        import torch.multiprocessing as _tmp

        if _tmp.get_start_method(allow_none=True) != "spawn":
            _tmp.set_start_method("spawn", force=True)
    except Exception:
        # torch may not be installed yet, or the method is already configured.
        pass

    # Ensure vLLM follows the same rule without requiring the caller to set the
    # environment variable manually.
    _os.environ.setdefault("VLLM_WORKER_MULTIPROC_START_METHOD", "spawn")


_ensure_spawn_start_method()

__all__ = []
