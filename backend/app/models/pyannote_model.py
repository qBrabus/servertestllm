from __future__ import annotations

import asyncio
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import librosa
import soundfile as sf
from pyannote.audio import Pipeline

from .base import BaseModelWrapper, ModelMetadata


class PyannoteDiarizationModel(BaseModelWrapper):
    model_id = "pyannote/speaker-diarization-community-1/"

    def __init__(self, cache_dir: Path, hf_token: str | None = None):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="speaker-diarization",
            description="Pyannote diarization community pipeline",
            format="wav/ogg/flac",
        )
        super().__init__(metadata, cache_dir, hf_token)
        self.pipeline: Pipeline | None = None

    async def load(self) -> None:
        def _load():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.pipeline = Pipeline.from_pretrained(
                self.model_id,
                use_auth_token=auth_token,
                cache_dir=str(self.cache_dir),
            )

        await asyncio.to_thread(_load)

    async def _unload(self) -> None:
        def _cleanup():
            self.pipeline = None

        await asyncio.to_thread(_cleanup)

    async def infer(self, audio_bytes: bytes, sampling_rate: int | None = None) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            target_sr = sampling_rate or 16000
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, target_sr)
                diarization = self.pipeline(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
            segments: List[Dict[str, Any]] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end,
                    }
                )
            return {"segments": segments}

        return await asyncio.to_thread(_run)
