from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.audio import DiarizationResponse, DiarizationSegment
from ..services.model_registry import registry

router = APIRouter(tags=["diarization"])


@router.post("/process", response_model=DiarizationResponse)
async def diarize_audio(file: UploadFile = File(...)) -> DiarizationResponse:
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    audio_bytes = await file.read()
    model = await registry.get("pyannote")
    result = await model.infer(audio_bytes=audio_bytes)
    segments = [DiarizationSegment(**segment) for segment in result["segments"]]
    return DiarizationResponse(segments=segments)
