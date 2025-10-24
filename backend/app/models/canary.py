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
            import huggingface_hub

            if not hasattr(huggingface_hub, "ModelFilter"):
                class _ModelFilter:  # pragma: no cover - compatibility shim
                    def __init__(self, *args, **kwargs) -> None:
                        self.args = args
                        self.kwargs = kwargs

                    def __call__(self, *args, **kwargs):
                        return True

                huggingface_hub.ModelFilter = _ModelFilter  # type: ignore[attr-defined]
                if hasattr(huggingface_hub, "__all__") and isinstance(
                    huggingface_hub.__all__, (list, tuple)
                ):  # pragma: no cover - defensive update
                    all_values = set(huggingface_hub.__all__)
                    all_values.add("ModelFilter")
                    huggingface_hub.__all__ = tuple(all_values)  # type: ignore[assignment]

            from nemo.collections.asr.models import ASRModel

            if not torch.cuda.is_available():  # pragma: no cover - dépend du matériel
                raise RuntimeError("CUDA est requis pour charger le modèle Canary")

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            primary_gpu = self.primary_device()
            device_index = primary_gpu if primary_gpu is not None else 0
            torch.cuda.set_device(device_index)
            target_device = torch.device("cuda", device_index)
            self.update_runtime(
                progress=15,
                status="Preparing ASR weights",
                details={"preferred_device_ids": self.preferred_device_ids},
            )

            self.update_runtime(progress=40, status="Downloading Canary checkpoint")
            repo_path = self._download_repo(auth_token)

            nemo_path = Path(repo_path) / "canary-1b-v2.nemo"
            if not nemo_path.exists():
                raise FileNotFoundError(f"Fichier Nemo introuvable dans {repo_path}")
            self.update_runtime(progress=85, status="Restoring NeMo checkpoint", downloaded=True)

            model = ASRModel.restore_from(restore_path=nemo_path, map_location=target_device)
            model = model.to(target_device)
            model.eval()

            self._pipeline = model
            self.update_runtime(
                progress=95,
                status="Canary ready",
                server={
                    "type": "NeMo ASR",
                    "endpoint": "/api/audio/transcribe",
                    "device": f"cuda:{device_index}",
                },
            )

        await asyncio.to_thread(_load)

    def _download_repo(self, auth_token: str | None) -> Path:
        from huggingface_hub import snapshot_download

        repo_path = snapshot_download(
            repo_id=self.model_id,
            cache_dir=str(self.cache_dir),
            token=auth_token,
            allow_patterns=["*.nemo"],
            local_dir_use_symlinks=False,
        )
        return Path(repo_path)

    async def download(self) -> None:
        def _download():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.update_runtime(progress=40, status="Téléchargement du modèle Canary")
            self._download_repo(auth_token)
            self.update_runtime(progress=95, status="Checkpoint Canary en cache", downloaded=True)

        await asyncio.to_thread(_download)

    async def _unload(self) -> None:
        def _cleanup():
            import torch

            self._pipeline = None
            if torch.cuda.is_available():  # pragma: no branch - best effort
                torch.cuda.empty_cache()

        await asyncio.to_thread(_cleanup)

    async def infer(self, audio_bytes: bytes, sampling_rate: int | None = None) -> Dict[str, Any]:
        await self.ensure_loaded()

        def _run() -> Dict[str, Any]:
            import numpy as np
            import soundfile as sf
            import tempfile
            import torch
            import torchaudio.functional as F

            if self._pipeline is None:
                raise RuntimeError("Le modèle Canary n'est pas chargé")

            target_sr = sampling_rate or 16000
            if not audio_bytes:
                return {"text": "", "sampling_rate": target_sr}

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
