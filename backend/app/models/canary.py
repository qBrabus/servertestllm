from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
        self._pipeline: Any | None = None

    async def load(self) -> None:
        def _load():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=str(self.cache_dir),
                token=auth_token,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                token=auth_token,
            )
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )

        await asyncio.to_thread(_load)

    async def _unload(self) -> None:
        def _cleanup():
            self._pipeline = None
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
            if self._pipeline is None:
                raise RuntimeError("Canary ASR pipeline is not loaded")
            if audio_array.size == 0:
                return {"text": "", "sampling_rate": target_sr}
            audio_array = audio_array.astype(np.float32)
            result = self._pipeline({"array": audio_array, "sampling_rate": target_sr})
            transcript = result.get("text", "") if isinstance(result, dict) else ""
            return {"text": transcript, "sampling_rate": target_sr}

        return await asyncio.to_thread(_run)
