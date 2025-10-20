# Diarization (pyannote community-1)
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip wheel setuptools

# pyannote.audio 3.x + FastAPI
# On évite d'épouser la dernière mineur au hasard: 3.1.* robuste
RUN pip install --no-cache-dir \
    "pyannote.audio>=3.1,<3.3" \
    fastapi uvicorn[standard] \
    huggingface_hub==0.25.2 \
    soundfile

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 7000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
