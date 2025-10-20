# ASR (Canary) — image GPU ok avec CUDA 12.x
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# MàJ pip toolchain
RUN pip install --no-cache-dir -U pip wheel setuptools

# IMPORTANT: pinner NeMo pour matcher torch de l'image
# pas d’aeneas ici (inutile pour inference ASR, évite des toolchains C pénibles)
RUN pip install --no-cache-dir \
    'nemo_toolkit[asr]==2.0.0' \
    soundfile librosa pydub \
    fastapi uvicorn[standard] python-multipart \
    huggingface_hub==0.25.2 requests

# App
WORKDIR /app
COPY app.py /app/app.py

# Variables (héritées via --env-file .env)
EXPOSE 6000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6000"]
