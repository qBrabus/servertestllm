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

            self.pipeline = Pipeline.from_pretrained(
                self.model_id,
                token=auth_token,
                cache_dir=str(self.cache_dir),
            )

            target_gpu = self.primary_device() or 0
            torch.cuda.set_device(target_gpu)
            try:
                self.pipeline.to(torch.device("cuda", target_gpu))
            except Exception as exc:  # pragma: no cover - remontée explicite
                raise RuntimeError(
                    f"Impossible de déplacer Pyannote sur le GPU {target_gpu}: {exc}"
                ) from exc

        await asyncio.to_thread(_load)

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
