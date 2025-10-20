from __future__ import annotations

import asyncio
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel

from .base import BaseModelWrapper, ModelMetadata


class CanaryASRModel(BaseModelWrapper):
    model_id = "nvidia/canary-1b-v2"

    def __init__(self, cache_dir: Path, hf_token: str | None = None):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="speech-to-text",
            description="NVIDIA Canary multilingual ASR model",
            format="wav/ogg/flac",
        )
        super().__init__(metadata, cache_dir, hf_token)
        self.model: ASRModel | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self) -> None:
        def _load():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.model = ASRModel.from_pretrained(
                model_name=self.model_id,
                map_location=self.device,
                override_config_path=None,
                return_config=False,
                strict=False,
                cache_dir=str(self.cache_dir),
                auth_token=auth_token,
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)

        await asyncio.to_thread(_load)

    async def _unload(self) -> None:
        def _cleanup():
            if self.model is not None:
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        await asyncio.to_thread(_cleanup)

    async def infer(self, audio_bytes: bytes, sampling_rate: int | None = None) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            target_sr = sampling_rate or 16000
            audio_array, sr = sf.read(io.BytesIO(audio_bytes)) if audio_bytes else (np.array([]), target_sr)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, target_sr)
                transcript = self.model.transcribe(paths2audio_files=[tmp.name])[0]
            Path(tmp.name).unlink(missing_ok=True)
            return {"text": transcript, "sampling_rate": target_sr}

        return await asyncio.to_thread(_run)
