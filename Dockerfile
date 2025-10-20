# Stage 1: Frontend build
FROM node:20-bullseye AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY frontend ./
RUN npm run build

# Stage 2: Backend
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HUGGINGFACE_HUB_CACHE=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ninja-build \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    sox \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

COPY backend /app/backend
COPY --from=frontend-builder /frontend/dist /app/frontend

ENV PYTHONPATH=/app/backend
ENV FRONTEND_DIST=/app/frontend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
