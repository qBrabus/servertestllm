from __future__ import annotations

import asyncio
import functools
import importlib.util
import inspect
import io
import json
import logging
import os
import re
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

from unittest import mock

from .base import BaseModelWrapper, ModelMetadata


LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - uniquement pour le typage
    from pyannote.audio import Pipeline


class PyannoteDiarizationModel(BaseModelWrapper):
    model_id = "pyannote/speaker-diarization-community-1"

    _MIN_GPU_MEMORY_BYTES = 5 * 1024 ** 3  # ~5 Go de mémoire libre requise

    @dataclass
    class _DevicePlan:
        use_gpu: bool
        device: Any
        dtype: Any | None
        free_memory: int | None = None
        total_memory: int | None = None

    def __init__(
        self,
        cache_dir: Path,
        hf_token: str | None = None,
        preferred_device_ids: list[int] | None = None,
    ):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="speaker-diarization",
            description="Pyannote diarization community pipeline",
            format="wav/ogg/flac",
        )
        super().__init__(metadata, cache_dir, hf_token, preferred_device_ids)
        self.pipeline: "Pipeline" | None = None

    async def load(self) -> None:
        self._prepare_matplotlib_environment()

        def _load():
            import importlib

            import torch

            try:
                importlib.import_module("torchaudio")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "torchaudio n'est pas installé. Installez une distribution compatible "
                    "CUDA (ex. pip install 'torchaudio==2.8.0+cu126' --index-url https://download.pytorch.org/whl/cu126)."
                ) from exc
            except Exception as exc:  # pragma: no cover - dépend de la binaire installée
                raise RuntimeError(
                    "Échec de l'initialisation de torchaudio. Vérifiez que la version correspond à "
                    "votre runtime CUDA et réinstallez le paquet."
                ) from exc

            try:
                from pyannote.audio import Pipeline
                import pyannote.audio as pyannote_audio
                from pyannote.audio.pipelines import speaker_diarization as speaker_diarization_module
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "pyannote.audio >= 4.0 doit être installé pour activer la diarisation."
                ) from exc

            if not torch.cuda.is_available():  # pragma: no cover - dépend du matériel
                raise RuntimeError("CUDA est requis pour charger le pipeline Pyannote")

            version_tokens = [int(part) for part in pyannote_audio.__version__.split(".") if part.isdigit()][:3]
            version_tuple = tuple(version_tokens + [0] * (3 - len(version_tokens)))
            if version_tuple < (4, 0, 0):
                raise RuntimeError(
                    "pyannote.audio >= 4.0.0 est requis pour ce pipeline. "
                    f"Version détectée: {pyannote_audio.__version__}."
                )

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            plan_holder: Dict[str, PyannoteDiarizationModel._DevicePlan] = {
                "plan": self._select_device_plan(torch)
            }

            self.update_runtime(
                status="Validation de l'environnement",
                progress=10,
                details={
                    "preferred_device_ids": self.preferred_device_ids,
                    "planned_device": str(plan_holder["plan"].device),
                    "use_gpu": plan_holder["plan"].use_gpu,
                    "gpu_memory": self._format_memory(
                        plan_holder["plan"].free_memory,
                        plan_holder["plan"].total_memory,
                    ),
                },
            )

            signature = inspect.signature(
                speaker_diarization_module.SpeakerDiarization.__init__
            )
            if "plda" not in signature.parameters:
                original_init = speaker_diarization_module.SpeakerDiarization.__init__
                if getattr(original_init, "__wrapped__", None) is None:

                    @functools.wraps(original_init)
                    def patched_init(self, *args, plda=None, **kwargs):  # type: ignore[override]
                        if plda is not None:
                            LOGGER.debug("Ignoring deprecated 'plda' parameter for Pyannote pipeline")
                        return original_init(self, *args, **kwargs)

                speaker_diarization_module.SpeakerDiarization.__init__ = patched_init  # type: ignore[assignment]

            repo_path = self._download_repo(auth_token)

            self.update_runtime(status="Initialisation du pipeline", progress=90, downloaded=True)

            pipeline_kwargs = {"cache_dir": str(self.cache_dir)}
            if auth_token:
                pipeline_kwargs["use_auth_token"] = auth_token
                pipeline_kwargs["token"] = auth_token

            from pyannote.audio.pipelines.utils import getter as pipeline_getter

            original_get_model = pipeline_getter.get_model

            def _resolve_local_artifact(path: Path) -> Path | None:
                """Return the first plausible checkpoint file under ``path``."""

                if path.is_file():
                    return path
                if not path.exists():
                    return None

                candidate_patterns = (
                    "*.safetensors",
                    "*.bin",
                    "*.ckpt",
                    "*.pt",
                    "*.pth",
                )
                for pattern in candidate_patterns:
                    matches = sorted(path.glob(pattern))
                    if matches:
                        return matches[0]
                for pattern in candidate_patterns:
                    matches = sorted(path.rglob(pattern))
                    if matches:
                        return matches[0]
                return None

            def _expand_reference(value: Any) -> Any:
                if isinstance(value, str) and value.startswith("$"):
                    prefix, sep, remainder = value.partition("/")
                    if prefix.lower() == "$model":
                        target = Path(repo_path)
                        if sep:
                            target = target / remainder
                        resolved = _resolve_local_artifact(target)
                        return str(resolved or target)
                if isinstance(value, dict):
                    return {key: _expand_reference(val) for key, val in value.items()}
                if isinstance(value, (list, tuple, set)):
                    expanded = [_expand_reference(item) for item in value]
                    if isinstance(value, tuple):
                        return tuple(expanded)
                    if isinstance(value, set):
                        return set(expanded)
                    return expanded
                return value

            def _normalise_local_paths(value: Any) -> Any:
                if isinstance(value, str):
                    try:
                        candidate = Path(value)
                    except (OSError, TypeError, ValueError):
                        return value
                    resolved = _resolve_local_artifact(candidate)
                    if resolved is not None:
                        return str(resolved)
                    if candidate.exists():
                        return str(candidate)
                    return value
                if isinstance(value, dict):
                    return {key: _normalise_local_paths(val) for key, val in value.items()}
                if isinstance(value, list):
                    return [_normalise_local_paths(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(_normalise_local_paths(item) for item in value)
                if isinstance(value, set):
                    return {_normalise_local_paths(item) for item in value}
                return value

            def _inject_execution_preferences(value: Any) -> Any:
                # Charger systématiquement les poids sur le CPU pour éviter les
                # allocations GPU précoces susceptibles de provoquer des
                # ``std::bad_alloc`` non interceptables. Les modules seront
                # ensuite déplacés vers l'appareil cible via
                # ``_move_pipeline_to_device``.
                map_location = "cpu"

                if isinstance(value, str):
                    return {"checkpoint": value, "map_location": map_location}

                if isinstance(value, dict):
                    updated = dict(value)
                    updated["map_location"] = map_location
                    return updated

                return value

            def patched_get_model(model: Any, use_auth_token: str | None = None):  # type: ignore[override]
                patched_model = _expand_reference(model)
                patched_model = _normalise_local_paths(patched_model)
                patched_model = _inject_execution_preferences(patched_model)
                return original_get_model(patched_model, use_auth_token=use_auth_token)

            original_speaker_get_model = getattr(speaker_diarization_module, "get_model", None)
            pipeline_getter.get_model = patched_get_model  # type: ignore[assignment]
            if original_speaker_get_model is not None:
                speaker_diarization_module.get_model = patched_get_model  # type: ignore[assignment]

            def _instantiate_pipeline() -> "Pipeline":
                """Instantiate Pyannote pipeline while forcing CPU placement."""

                def _restore_cuda_visible(value: str | None) -> None:
                    if value is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = value

                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch("torch.cuda.is_available", return_value=False)
                    )
                    stack.enter_context(
                        mock.patch("torch.cuda.device_count", return_value=0)
                    )
                    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    stack.callback(
                        lambda value=previous_cuda_visible: _restore_cuda_visible(value)
                    )

                    return Pipeline.from_pretrained(
                        self.model_id,
                        **pipeline_kwargs,
                    )

            try:
                try:
                    self.pipeline = _instantiate_pipeline()
                except Exception as exc:
                    if plan_holder["plan"].use_gpu:
                        LOGGER.warning(
                            "Échec du chargement Pyannote sur %s (%s). Nouvelle tentative sur CPU.",
                            plan_holder["plan"].device,
                            exc,
                        )
                        torch.cuda.empty_cache()
                        plan_holder["plan"] = self._DevicePlan(
                            use_gpu=False,
                            device=torch.device("cpu"),
                            dtype=None,
                        )
                        self.pipeline = _instantiate_pipeline()
                    else:
                        raise
            finally:
                pipeline_getter.get_model = original_get_model  # type: ignore[assignment]
                if original_speaker_get_model is not None:
                    speaker_diarization_module.get_model = original_speaker_get_model  # type: ignore[assignment]

            plan = plan_holder["plan"]
            if not plan.use_gpu:
                LOGGER.warning(
                    "Mémoire GPU insuffisante (%s). Pyannote fonctionnera sur CPU.",
                    self._format_memory(plan.free_memory, plan.total_memory),
                )
            if plan.use_gpu:
                torch.cuda.set_device(plan.device.index)  # type: ignore[arg-type]
            try:
                plan = self._move_pipeline_to_device(self.pipeline, plan, torch)
            except Exception as exc:  # pragma: no cover - dépend fortement du matériel
                if plan.use_gpu:
                    LOGGER.warning(
                        "Échec du transfert Pyannote sur %s (%s). Bascule sur CPU.",
                        plan.device,
                        exc,
                    )
                    torch.cuda.empty_cache()
                plan = self._DevicePlan(
                    use_gpu=False,
                    device=torch.device("cpu"),
                    dtype=None,
                )
                plan = self._move_pipeline_to_device(self.pipeline, plan, torch)

            plan_holder["plan"] = plan

            self.update_runtime(
                status="Pipeline Pyannote prêt",
                progress=98,
                server=self.build_server_metadata(
                    endpoint="/api/diarization/process",
                    type="Pyannote",
                    device=str(plan.device),
                ),
            )

        await asyncio.to_thread(_load)

    def _prepare_matplotlib_environment(self) -> None:
        """Initialise Matplotlib in a safe, headless configuration."""

        mpl_cache = self.cache_dir / "matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

        try:
            import matplotlib  # type: ignore

            try:  # pragma: no cover - depends on optional Matplotlib install
                matplotlib.use("Agg", force=True)
            except Exception:
                LOGGER.debug("Impossible de forcer le backend Matplotlib Agg", exc_info=True)

            try:
                cache_dir = Path(matplotlib.get_cachedir())
            except Exception:
                LOGGER.debug(
                    "Impossible de récupérer le cache Matplotlib", exc_info=True
                )
            else:
                version = self._detect_matplotlib_font_cache_version()
                if version is not None:
                    self._ensure_minimal_matplotlib_font_cache(cache_dir, version)

            # L'import de ``matplotlib.font_manager`` déclenche la création d'un
            # ``FontManager`` global qui scanne l'ensemble du système de
            # fichiers. Sur certaines images (typiquement celles contenant des
            # partages réseau montés automatiquement) ce scan peut consommer
            # plusieurs gigaoctets et provoquer un ``std::bad_alloc`` non
            # intercepteable côté Python. Pyannote n'a pas besoin de cette
            # fonctionnalité : nous nous contentons donc de préparer l'environnement
            # (backend, cache) sans forcer l'import du sous-module.
        except ModuleNotFoundError:
            LOGGER.debug("Matplotlib n'est pas installé; pré-initialisation ignorée")

    @staticmethod
    def _detect_matplotlib_font_cache_version() -> str | None:
        """Return the expected font cache version without importing the module."""

        try:
            spec = importlib.util.find_spec("matplotlib.font_manager")
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.debug(
                "Impossible de localiser matplotlib.font_manager", exc_info=True
            )
            return None

        if spec is None or not spec.origin:
            return None

        try:
            source = Path(spec.origin).read_text(encoding="utf-8", errors="ignore")
        except Exception:  # pragma: no cover - depends on filesystem
            LOGGER.debug(
                "Lecture du code source de matplotlib.font_manager impossible",
                exc_info=True,
            )
            return None

        match = re.search(r"FONT_MANAGER_VERSION\s*=\s*(\d+)", source)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _ensure_minimal_matplotlib_font_cache(cache_dir: Path, version: str) -> None:
        """Create a tiny font cache to prevent expensive scans on import."""

        fontlist_path = cache_dir / f"fontlist-v{version}.json"
        if fontlist_path.exists():
            return

        payload = {
            "_version": int(version),
            "_FontManager__default_weight": "normal",
            "default_size": None,
            "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
            "afmlist": [],
            "ttflist": [],
        }

        tmp_path = fontlist_path.with_suffix(".tmp")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(fontlist_path)
        except Exception:  # pragma: no cover - depends on filesystem permissions
            LOGGER.debug(
                "Impossible d'écrire le cache minimal Matplotlib", exc_info=True
            )
            tmp_path.unlink(missing_ok=True)

    def _select_device_plan(self, torch_module: Any) -> _DevicePlan:
        preferred = self.primary_device()
        device_index = preferred if preferred is not None else 0
        available = max(1, torch_module.cuda.device_count())
        if device_index >= available:
            device_index = 0

        free_bytes: int | None = None
        total_bytes: int | None = None
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(device_index)
        except Exception:  # pragma: no cover - dépend du runtime CUDA
            free_bytes = total_bytes = None

        device = torch_module.device("cuda", device_index)
        has_enough_memory = (
            free_bytes is None or free_bytes >= self._MIN_GPU_MEMORY_BYTES
        )

        dtype = None
        if has_enough_memory:
            try:
                capability: Tuple[int, int] = torch_module.cuda.get_device_capability(device_index)
            except Exception:  # pragma: no cover
                capability = (0, 0)
            major, _ = capability
            if major >= 8:
                dtype = torch_module.bfloat16
            elif major >= 7:
                dtype = torch_module.float16

        if not has_enough_memory:
            device = torch_module.device("cpu")

        return self._DevicePlan(
            use_gpu=has_enough_memory,
            device=device,
            dtype=dtype,
            free_memory=free_bytes,
            total_memory=total_bytes,
        )

    def _move_pipeline_to_device(
        self,
        pipeline: "Pipeline",
        plan: _DevicePlan,
        torch_module: Any,
    ) -> _DevicePlan:
        target_device = plan.device
        move_kwargs = {"device": target_device}
        if plan.dtype is not None:
            move_kwargs["dtype"] = plan.dtype

        errors: List[str] = []
        for name, module in self._iter_pipeline_modules(pipeline):
            if not hasattr(module, "to"):
                continue
            try:
                module.to(**move_kwargs)
            except TypeError:
                # Certaines briques n'acceptent pas ``dtype``
                module.to(device=target_device)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))

        if hasattr(pipeline, "to"):
            try:
                pipeline.to(**move_kwargs)
            except TypeError:
                pipeline.to(device=target_device)

        if not plan.use_gpu:
            return self._DevicePlan(
                use_gpu=False,
                device=torch_module.device("cpu"),
                dtype=None,
                free_memory=plan.free_memory,
                total_memory=plan.total_memory,
            )

        return plan

    def _iter_pipeline_modules(self, pipeline: "Pipeline") -> Iterable[Tuple[str, Any]]:
        segmentation = getattr(pipeline, "_segmentation", None)
        segmentation_model = getattr(segmentation, "model", None)
        if segmentation_model is not None:
            yield "segmentation", segmentation_model

        embedding_wrapper = getattr(pipeline, "_embedding", None)
        embedding_model = getattr(embedding_wrapper, "model", None)
        if embedding_model is None:
            embedding_model = getattr(embedding_wrapper, "_model", None)
        if embedding_model is not None:
            yield "embedding", embedding_model

        plda = getattr(pipeline, "_plda", None)
        if plda is not None and hasattr(plda, "to"):
            yield "plda", plda

    @staticmethod
    def _format_memory(free_bytes: int | None, total_bytes: int | None) -> str:
        if free_bytes is None or total_bytes is None:
            return "inconnue"

        def _to_gib(value: int) -> float:
            return value / float(1024**3)

        return f"{_to_gib(free_bytes):.1f} GiB libres / {_to_gib(total_bytes):.1f} GiB"

    def _download_repo(self, auth_token: str | None) -> Path:
        return self.download_snapshot(
            repo_id=self.model_id,
            auth_token=auth_token,
            status_prefix="Téléchargement Pyannote",
            progress_range=(32, 78),
            complete_status="Artefacts Pyannote disponibles",
            allow_patterns=["*.bin", "*.ckpt", "*.pt", "*.yaml", "*.json"],
            local_dir_use_symlinks=False,
        )

    async def download(self) -> None:
        def _download():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.download_snapshot(
                repo_id=self.model_id,
                auth_token=auth_token,
                status_prefix="Téléchargement Pyannote",
                progress_range=(28, 96),
                complete_status="Artefacts Pyannote en cache",
                allow_patterns=["*.bin", "*.ckpt", "*.pt", "*.yaml", "*.json"],
                local_dir_use_symlinks=False,
            )

        await asyncio.to_thread(_download)

    async def _unload(self) -> None:
        def _cleanup():
            self.pipeline = None

        await asyncio.to_thread(_cleanup)

    async def infer(self, audio_bytes: bytes, sampling_rate: int | None = None) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            import numpy as np
            import soundfile as sf
            import torch

            from app.utils.audio import ensure_mono, resample_waveform

            if self.pipeline is None:
                raise RuntimeError("Le pipeline Pyannote n'est pas initialisé")

            target_sr = sampling_rate or 16000
            if not audio_bytes:
                return {"segments": []}

            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            waveform = torch.from_numpy(audio_array)
            waveform = ensure_mono(waveform)
            if sr != target_sr:
                waveform = resample_waveform(waveform, sr, target_sr)
            waveform = waveform.squeeze(0).contiguous()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                sf.write(tmp_path, waveform.cpu().numpy().astype(np.float32), target_sr)
                diarization = self.pipeline(str(tmp_path))
            tmp_path.unlink(missing_ok=True)

            segments: List[Dict[str, Any]] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
            return {"segments": segments}

        return await asyncio.to_thread(_run)
