import os
import tempfile
import shutil
from typing import Optional, List

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nemo.collections.asr.models import EncDecMultiTaskModel
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
REPO_ID = os.getenv("ASR_REPO_ID", "nvidia/canary-1b-v2")
NEMO_FILENAME = os.getenv("ASR_NEMO_FILENAME", "canary-1b-v2.nemo")

app = FastAPI(title="ASR Canary 1B v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

MODEL: Optional[EncDecMultiTaskModel] = None

class TranscriptionOut(BaseModel):
    text: str
    segments: Optional[List[str]] = None
    sampling_rate: Optional[int] = None
    lang: Optional[str] = None

@app.on_event("startup")
def load_model():
    global MODEL
    if MODEL is not None:
        return
    if HF_TOKEN is None:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN manquant pour télécharger le checkpoint NeMo.")

    ckpt_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=NEMO_FILENAME,
        token=HF_TOKEN
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = EncDecMultiTaskModel.restore_from(ckpt_path, map_location=device)
    MODEL.eval()
    if device == "cuda":
        MODEL = MODEL.to(device)

@app.get("/health")
def health():
    loaded = MODEL is not None
    return {"status": "ok", "model_loaded": loaded}

@app.post("/v1/audio/transcriptions", response_model=TranscriptionOut)
def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # NeMo attend une liste de chemins
        texts = MODEL.transcribe(paths2audio_files=[tmp_path])
        text = texts[0] if texts else ""
        return TranscriptionOut(text=text, lang=language)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
