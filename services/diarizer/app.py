import os
import tempfile
import shutil
from typing import List

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyannote.audio import Pipeline

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_ID = os.getenv("DIAR_MODEL_ID", "pyannote/speaker-diarization-community-1")

app = FastAPI(title="Diarization (pyannote community-1)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

PIPE: Pipeline = None

class Segment(BaseModel):
    start: float
    end: float
    speaker: str

class DiarOut(BaseModel):
    segments: List[Segment]

@app.on_event("startup")
def load_pipeline():
    global PIPE
    if HF_TOKEN is None:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN manquant pour pyannote.")
    PIPE = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if torch.cuda.is_available():
        PIPE.to(torch.device("cuda"))

@app.get("/health")
def health():
    return {"status": "ok", "loaded": PIPE is not None}

@app.post("/v1/audio/diarize", response_model=DiarOut)
def diarize(file: UploadFile = File(...)):
    if PIPE is None:
        raise HTTPException(status_code=503, detail="Pipeline non chargée")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        diar = PIPE(tmp_path)
        segs: List[Segment] = []
        # Convertir en liste simple
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segs.append(Segment(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
        # Trier par temps de début
        segs.sort(key=lambda s: (s.start, s.end))
        return DiarOut(segments=segs)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
