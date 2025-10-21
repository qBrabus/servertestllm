# Unified Inference Gateway

This project packages a complete inference stack for large language models, speech-to-text, and speaker diarization into a single GPU-enabled container. It exposes an OpenAI-compatible API together with administrative endpoints and a full React dashboard for orchestration.

## Features

- **LLM Inference** via [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).
- **Speech-to-Text** with NVIDIA's [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2).
- **Speaker Diarization** using [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1/).
- **OpenAI-compatible REST API** (`/v1/chat/completions`, `/v1/completions`).
- **Audio APIs** for transcription (`/api/audio/transcribe`) and diarization (`/api/diarization/process`).
- **Administrative APIs** to monitor GPUs, system load, and model lifecycle management.
- **React + Material UI dashboard** for real-time monitoring, model control, audio tooling, and an interactive OpenAI playground.

## Prerequisites

- Docker with GPU support (`nvidia-docker2` or Docker 24+ with the NVIDIA runtime).
- NVIDIA GPU drivers compatible with CUDA 12.4.

## Building the Container

Use the helper script to build the unified image (default tag `unified-inference:latest`):

```bash
./build_docker.sh [custom-tag]
```

The build performs the following steps:

1. Installs frontend dependencies and generates a production build of the React dashboard.
2. Installs the Python backend dependencies (FastAPI, vLLM, Transformers, Pyannote, etc.).
3. Bundles both layers into a CUDA 12.4 runtime image with FFmpeg and audio prerequisites.

## Running the Stack

Launch the container with GPU access, expose port `8000`, and persist model weights to `./model_cache`:

```bash
./run_docker.sh [image-tag]
```

Environment variables (overridable before running the script):

- `HUGGINGFACE_TOKEN` – defaults to the provided token for gated model downloads.
- `OPENAI_KEYS` – optional comma-separated API keys required when securing the OpenAI-compatible routes.
- `HOST_PORT` – host port for the FastAPI server (default `8000`).
- `MODEL_CACHE_DIR` – host directory to persist Hugging Face caches (default `./model_cache`).

After the container starts:

- Access the **admin dashboard** at `http://<host>:<HOST_PORT>/`.
- Call the **OpenAI-compatible endpoints** at `http://<host>:<HOST_PORT>/v1/...`.
- Use the **audio APIs** at `http://<host>:<HOST_PORT>/api/audio/...` and `http://<host>:<HOST_PORT>/api/diarization/...`.

## API Overview

### OpenAI-Compatible

- `POST /v1/chat/completions`
- `POST /v1/completions`

### Audio & Diarization

- `POST /api/audio/transcribe` (multipart `file` field)
- `POST /api/diarization/process` (multipart `file` field)

### Administration

- `GET /api/admin/status`
- `POST /api/admin/models/{model_key}/load`
- `POST /api/admin/models/{model_key}/unload`

Model keys: `qwen`, `canary`, `pyannote`.

## Development

- Backend located under `backend/app` (FastAPI application entry point: `app.main:app`).
- Frontend located under `frontend` (React + Vite). Use `npm install` and `npm run dev` for local development if needed.

## Notes

- The first inference call for each model will trigger a download to `/models` inside the container (bind-mounted by the run script).
- GPU utilization metrics rely on `torch`/`GPUtil`; ensure the container runs with `--gpus all`.
- When `OPENAI_KEYS` is empty, OpenAI endpoints are accessible without authentication.
