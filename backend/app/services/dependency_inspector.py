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


def _format_cuda_version(raw: Any) -> str | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    major = value // 1000
    minor = (value % 1000) // 10
    patch = value % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def _probe_torchaudio() -> Dict[str, Any]:
    import re

    torchaudio = _load_module("torchaudio")
    version = getattr(torchaudio, "__version__", "unknown")
    version_meta = getattr(torchaudio, "version", None)
    cuda_runtime = _normalise_cuda_value(getattr(version_meta, "cuda", None)) if version_meta else None
    if not cuda_runtime and version_meta is not None:
        cuda_runtime = _normalise_cuda_value(getattr(version_meta, "cuda_version", None))
    cuda_build = bool(cuda_runtime and str(cuda_runtime).lower() != "none")
    cuda_extension_available = False

    # Recent torchaudio releases expose CUDA metadata via ``torchaudio.lib._torchaudio``.
    try:  # pragma: no cover - depends on optional binary extension
        from torchaudio.lib import _torchaudio as _ta  # type: ignore

        raw_version = getattr(_ta, "cuda_version", None)
        if callable(raw_version):
            raw_version = raw_version()
        formatted = _format_cuda_version(raw_version)
        if formatted:
            cuda_runtime = cuda_runtime or formatted
            cuda_build = True
            cuda_extension_available = True
    except Exception:
        cuda_extension_available = False

    # Older wheels (<2.5) exposed ``torchaudio._extension.utils``
    if not cuda_extension_available:  # pragma: no cover - depends on binary distribution
        try:
            from torchaudio._extension import utils as ta_utils  # type: ignore

            cuda_extension_available = bool(
                getattr(ta_utils, "is_cuda_available", lambda: False)()
            )
        except Exception:
            cuda_extension_available = False

    if not cuda_runtime:
        match = re.search(r"\+cu(\d+)", version)
        if match:
            digits = match.group(1)
            if len(digits) >= 2:
                major = digits[:-1]
                minor = digits[-1]
            else:
                major, minor = digits, "0"
            try:
                cuda_runtime = f"{int(major)}.{int(minor)}"
            except ValueError:
                cuda_runtime = None
        if cuda_runtime:
            cuda_build = True

    torch_cuda = False
    try:
        torch = _load_module("torch")
        torch_cuda = torch.cuda.is_available()
    except Exception:
        torch_cuda = False

    return {
        "name": "torchaudio",
        "version": version,
        "cuda": bool(cuda_build and (cuda_extension_available or torch_cuda)),
        "details": {
            "cuda_runtime": cuda_runtime,
            "cuda_build": bool(cuda_build),
            "cuda_extension_available": bool(cuda_extension_available),
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

