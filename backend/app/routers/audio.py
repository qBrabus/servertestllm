from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import settings
from ..schemas.audio import TranscriptionResponse
from ..services.model_registry import registry

router = APIRouter(tags=["audio"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    audio_bytes = await file.read()
    model = await registry.get("canary")
    result = await model.infer(audio_bytes=audio_bytes)
    return TranscriptionResponse(**result)


