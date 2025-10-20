# AI Stack: vLLM (Qwen3-VL-30B-A3B), ASR Canary (NeMo), Diarization (pyannote)

Multi-service stack with Docker Compose. Requires NVIDIA GPU + Container Toolkit.

## 1) Prepare folders and env
```bash
cd ai-stack
cp .env.example .env
# put your real HF token:
# HUGGINGFACE_HUB_TOKEN=hf_xxx
mkdir -p models huggingface
```

## 2) Start
```bash
docker compose up -d --build
```

Services:
- Admin UI: http://localhost:8080
- vLLM (OpenAI-compatible): http://localhost:8000/v1
- ASR (OpenAI-style): http://localhost:6000/v1/audio/transcriptions
- Diarization: http://localhost:7000/v1/audio/diarize

## 3) Test (examples)
### Chat completion
```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "qwen3-vl",
    "messages": [
      {"role":"system","content":"You are helpful"},
      {"role":"user","content":[
        {"type":"text","text":"Décris l\'image"},
        {"type":"image_url","image_url":{"url":"https://picsum.photos/seed/cat/512"}}
      ]}
    ]
  }'
```

### Transcription
```bash
curl -F "file=@/path/to/audio.wav" -F "language=fr" http://localhost:6000/v1/audio/transcriptions
```

### Diarization
```bash
curl -F "file=@/path/to/audio.wav" http://localhost:7000/v1/audio/diarize
```

## Notes
- vLLM downloads the Qwen model on first run into `./models`.
- Canary (.nemo) is pulled from HF with your token and cached in `./models/canary-1b-v2/`.
- pyannote uses `speaker-diarization-community-1` and may also use the HF token.
- Make sure your NVIDIA driver supports CUDA 12.x runtime and you have the NVIDIA Container Toolkit installed.
