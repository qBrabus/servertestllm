# Stage 1: Frontend build
FROM node:20-bullseye AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./

# Disable npm audit/fund checks and progress spinners. These may hang or take a
# very long time when the build environment does not have unrestricted network
# access, which manifested as the Docker build being “stuck” at the npm ci step.
ENV NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_PROGRESS=false

# npm ci intermittently crashes on Node 20 with "Exit handler never called".
# Fall back to npm install for a more reliable dependency installation in CI.
RUN npm install --legacy-peer-deps
COPY frontend ./
RUN npm run build

# Stage 2: Backend
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HUGGINGFACE_HUB_CACHE=/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    cmake \
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
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        numpy==1.26.4 \
        Cython==0.29.37 \
        typing_extensions==4.12.2 && \
    python3 -m pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY --from=frontend-builder /frontend/dist /app/frontend

ENV PYTHONPATH=/app/backend
ENV FRONTEND_DIST=/app/frontend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
