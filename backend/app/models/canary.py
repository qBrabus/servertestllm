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
            from huggingface_hub import hf_hub_download
            from nemo.collections.asr.models import ASRModel

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            primary_gpu = self.primary_device()

            if torch.cuda.is_available():
                device_index = primary_gpu if primary_gpu is not None else 0
                target_device = torch.device("cuda", device_index)
            else:
                target_device = torch.device("cpu")

            nemo_path = hf_hub_download(
                repo_id=self.model_id,
                filename="canary-1b-v2.nemo",
                cache_dir=str(self.cache_dir),
                token=auth_token,
            )

            model = ASRModel.restore_from(restore_path=nemo_path, map_location=target_device)

            # Ensure the model runs on the requested device.
            model = model.to(target_device)
            model.eval()

            self._pipeline = model

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
            import tempfile

            target_sr = sampling_rate or 16000
            audio_array, sr = sf.read(io.BytesIO(audio_bytes)) if audio_bytes else (np.array([]), target_sr)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != target_sr and audio_array.size > 0:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            if self._pipeline is None:
                raise RuntimeError("Canary ASR pipeline is not loaded")
            if audio_array.size == 0:
                return {"text": "", "sampling_rate": target_sr}

            audio_array = audio_array.astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                sf.write(tmp_path, audio_array, target_sr)

            try:
                outputs = self._pipeline.transcribe(
                    paths2audio_files=[str(tmp_path)],
                    source_lang="en",
                    target_lang="en",
                    batch_size=1,
                    return_hypotheses=True,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

            transcript = ""
            if outputs:
                first = outputs[0]
                if isinstance(first, str):
                    transcript = first
                elif hasattr(first, "text"):
                    transcript = first.text or ""

            return {"text": transcript, "sampling_rate": target_sr}

        return await asyncio.to_thread(_run)
