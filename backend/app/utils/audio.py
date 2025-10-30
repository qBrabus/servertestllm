from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from scipy.signal import resample_poly


def _compute_resample_ratio(original_rate: int, target_rate: int) -> tuple[int, int]:
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Les fréquences d'échantillonnage doivent être positives")
    gcd = math.gcd(original_rate, target_rate)
    return target_rate // gcd, original_rate // gcd


def resample_waveform(waveform: torch.Tensor, original_rate: int, target_rate: int) -> torch.Tensor:
    """Resample ``waveform`` to ``target_rate`` using ``scipy.signal.resample_poly``.

    The function accepts mono or multi-channel tensors. The resampling happens on CPU
    using NumPy to avoid hard dependencies on ``torchaudio`` binary extensions.
    """

    if original_rate == target_rate:
        return waveform

    was_1d = waveform.ndim == 1
    if was_1d:
        waveform = waveform.unsqueeze(0)

    up, down = _compute_resample_ratio(original_rate, target_rate)
    waveform_np = waveform.detach().cpu().numpy()
    resampled = resample_poly(waveform_np, up, down, axis=-1)
    resampled_tensor = torch.from_numpy(np.ascontiguousarray(resampled))

    if was_1d:
        return resampled_tensor.squeeze(0)
    return resampled_tensor


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError("Le tenseur audio doit être mono ou multi-canaux")
    if waveform.shape[0] == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def normalise_audio_buffer(buffer: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(buffer), dtype=np.float32)
    if array.ndim == 0:
        array = np.expand_dims(array, 0)
    return array.astype(np.float32, copy=False)
