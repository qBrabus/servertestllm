from __future__ import annotations

import asyncio
import functools
import inspect
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .base import BaseModelWrapper, ModelMetadata


LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - uniquement pour le typage
    from pyannote.audio import Pipeline


class PyannoteDiarizationModel(BaseModelWrapper):
    model_id = "pyannote/speaker-diarization-community-1"

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
        def _load():
            import torch
            from pyannote.audio import Pipeline
            from pyannote.audio.pipelines import speaker_diarization as speaker_diarization_module

            if not torch.cuda.is_available():  # pragma: no cover - dépend du matériel
                raise RuntimeError("CUDA est requis pour charger le pipeline Pyannote")

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.update_runtime(
                status="Validation de l'environnement",
                progress=10,
                details={"preferred_device_ids": self.preferred_device_ids},
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

            def patched_get_model(model: Any, use_auth_token: str | None = None):  # type: ignore[override]
                patched_model = _expand_reference(model)
                patched_model = _normalise_local_paths(patched_model)
                return original_get_model(patched_model, use_auth_token=use_auth_token)

            original_speaker_get_model = getattr(speaker_diarization_module, "get_model", None)
            pipeline_getter.get_model = patched_get_model  # type: ignore[assignment]
            if original_speaker_get_model is not None:
                speaker_diarization_module.get_model = patched_get_model  # type: ignore[assignment]
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_id,
                    **pipeline_kwargs,
                )
            finally:
                pipeline_getter.get_model = original_get_model  # type: ignore[assignment]
                if original_speaker_get_model is not None:
                    speaker_diarization_module.get_model = original_speaker_get_model  # type: ignore[assignment]

            target_gpu = self.primary_device() or 0
            torch.cuda.set_device(target_gpu)
            try:
                self.pipeline.to(torch.device("cuda", target_gpu))
            except Exception as exc:  # pragma: no cover - remontée explicite
                raise RuntimeError(
                    f"Impossible de déplacer Pyannote sur le GPU {target_gpu}: {exc}"
                ) from exc
            self.update_runtime(
                status="Pipeline Pyannote prêt",
                progress=98,
                server=self.build_server_metadata(
                    endpoint="/api/diarization/process",
                    type="Pyannote",
                    device=f"cuda:{target_gpu}",
                ),
            )

        await asyncio.to_thread(_load)

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
            import torchaudio.functional as F

            if self.pipeline is None:
                raise RuntimeError("Le pipeline Pyannote n'est pas initialisé")

            target_sr = sampling_rate or 16000
            if not audio_bytes:
                return {"segments": []}

            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            waveform = torch.from_numpy(audio_array)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if sr != target_sr:
                waveform = F.resample(waveform, sr, target_sr)
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
