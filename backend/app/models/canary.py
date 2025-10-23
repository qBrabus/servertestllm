from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Any, Dict

from .base import BaseModelWrapper, ModelMetadata


class CanaryASRModel(BaseModelWrapper):
    model_id = "nvidia/canary-1b-v2"

    def __init__(
        self,
        cache_dir: Path,
        hf_token: str | None = None,
        preferred_device_ids: list[int] | None = None,
    ):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="speech-to-text",
            description="NVIDIA Canary multilingual ASR model",
            format="wav/ogg/flac",
        )
        super().__init__(metadata, cache_dir, hf_token, preferred_device_ids)
        self._pipeline: Any | None = None

    async def load(self) -> None:
        def _load():
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            primary_gpu = self.primary_device()

            model_load_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "cache_dir": str(self.cache_dir),
                "token": auth_token,
            }

            try:
                # Canary checkpoints are published as safetensors. Prefer them as they
                # are both safer and the only files available on Hugging Face Hub.
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    use_safetensors=True,
                    **model_load_kwargs,
                )
            except OSError as exc:
                # Older versions of the model wrapper expected .bin checkpoints.
                # In environments where safetensors are not available, fall back to
                # the legacy format to preserve backwards compatibility.
                if "does not appear to have a file" not in str(exc):
                    raise
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    use_safetensors=False,
                    **model_load_kwargs,
                )

            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                token=auth_token,
            )

            if torch.cuda.is_available():
                target_device_idx = primary_gpu if primary_gpu is not None else 0
                model = model.to(torch.device("cuda", target_device_idx))
                pipeline_device: int | str | torch.device = target_device_idx
            else:
                pipeline_device = -1  # HuggingFace pipeline CPU sentinel

            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=torch_dtype,
                device=pipeline_device,
            )

        await asyncio.to_thread(_load)

    async def _unload(self) -> None:
        def _cleanup():
            import torch

            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        await asyncio.to_thread(_cleanup)

    async def infer(self, audio_bytes: bytes, sampling_rate: int | None = None) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            import librosa
            import numpy as np
            import soundfile as sf

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
