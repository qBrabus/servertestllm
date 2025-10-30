from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Callable, Dict, List


def _load_module(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(module_name)
    try:
        return importlib.import_module(module_name)
    except AttributeError as exc:
        if module_name == "torchaudio" and "partially initialized" in str(exc):
            raise ImportError(
                "torchaudio ne s'est pas initialisé correctement (extension CUDA manquante)."
            ) from exc
        raise


def _normalise_cuda_value(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or None
    return str(raw)


def _probe_torch() -> Dict[str, Any]:
    torch = _load_module("torch")
    version = getattr(torch, "__version__", "unknown")
    cuda_runtime = _normalise_cuda_value(getattr(torch.version, "cuda", None))  # type: ignore[attr-defined]
    cudnn_version = None
    try:  # pragma: no cover - depends on compiled backend
        cudnn_raw = torch.backends.cudnn.version()
        if cudnn_raw is not None:
            cudnn_version = str(cudnn_raw)
    except Exception:
        cudnn_version = None
    cuda_available = bool(cuda_runtime) and torch.cuda.is_available()
    return {
        "name": "torch",
        "version": version,
        "cuda": cuda_available,
        "details": {
            "cuda_runtime": cuda_runtime,
            "cudnn": cudnn_version,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def _probe_torchvision() -> Dict[str, Any]:
    torchvision = _load_module("torchvision")
    version = getattr(torchvision, "__version__", "unknown")
    version_meta = getattr(torchvision, "version", None)
    cuda_runtime = _normalise_cuda_value(getattr(version_meta, "cuda", None)) if version_meta else None
    cuda_build = bool(cuda_runtime and cuda_runtime.lower() != "none")
    torch = _load_module("torch")
    cuda_available = cuda_build and torch.cuda.is_available()
    return {
        "name": "torchvision",
        "version": version,
        "cuda": cuda_available,
        "details": {
            "cuda_runtime": cuda_runtime,
            "cuda_build": cuda_build,
        },
    }


def _probe_torchaudio() -> Dict[str, Any]:
    torchaudio = _load_module("torchaudio")
    version = getattr(torchaudio, "__version__", "unknown")
    version_meta = getattr(torchaudio, "version", None)
    cuda_runtime = _normalise_cuda_value(getattr(version_meta, "cuda", None)) if version_meta else None
    cuda_build = bool(cuda_runtime and cuda_runtime.lower() != "none")
    cuda_available = False
    if cuda_build:
        try:
            # torchaudio 2.5 exposes this helper which verifies the CUDA extension is loaded
            from torchaudio._extension import utils as ta_utils  # type: ignore

            cuda_available = bool(getattr(ta_utils, "is_cuda_available", lambda: False)())
        except Exception:  # pragma: no cover - depends on binary distribution
            cuda_available = False
    torch_cuda = False
    try:
        torch = _load_module("torch")
        torch_cuda = torch.cuda.is_available()
    except Exception:
        torch_cuda = False
    return {
        "name": "torchaudio",
        "version": version,
        "cuda": cuda_build and (cuda_available or torch_cuda),
        "details": {
            "cuda_runtime": cuda_runtime,
            "cuda_build": cuda_build,
            "cuda_extension_available": cuda_available,
        },
    }


ProbeCallable = Callable[[], Dict[str, Any]]

_PROBES: List[ProbeCallable] = [_probe_torch, _probe_torchvision, _probe_torchaudio]


def gather_dependency_status() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for probe in _PROBES:
        probe_name = getattr(probe, "__name__", "inconnu").replace("_probe_", "")
        try:
            results.append(probe())
        except ModuleNotFoundError as missing:
            results.append(
                {
                    "name": missing.name,
                    "version": None,
                    "cuda": False,
                    "details": {},
                    "error": "Paquet non installé",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": probe_name,
                    "version": None,
                    "cuda": False,
                    "details": {},
                    "error": str(exc),
                }
            )
    return results

