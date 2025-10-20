from __future__ import annotations

from typing import List

from pydantic import BaseModel


class TranscriptionSegment(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None


class TranscriptionResponse(BaseModel):
    text: str
    sampling_rate: int
    segments: List[TranscriptionSegment] | None = None


class DiarizationSegment(BaseModel):
    speaker: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    segments: List[DiarizationSegment]
