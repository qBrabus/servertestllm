# 1) Build React
FROM node:20-alpine AS webbuild
WORKDIR /webui
COPY web-frontend/package.json web-frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web-frontend/ ./
RUN npm run build

# 2) Python API + static serving
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# nvidia-smi sera fourni par le runtime nvidia dans le conteneur
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic-settings httpx==0.27.2

WORKDIR /app
COPY app/ /app/app/
COPY --from=webbuild /webui/dist /app/static

ENV VLLM_BASE=http://vllm:8000 \
    ASR_BASE=http://asr-canary:6000 \
    DIAR_BASE=http://diarizer:7000

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
