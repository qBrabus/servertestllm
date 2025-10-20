import os
import subprocess
from typing import Any, Dict, List
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

VLLM_BASE = os.getenv("VLLM_BASE", "http://vllm:8000")
ASR_BASE = os.getenv("ASR_BASE", "http://asr-canary:6000")
DIAR_BASE = os.getenv("DIAR_BASE", "http://diarizer:7000")

SERVICE_TARGETS: List[Dict[str, Any]] = [
    {
        "id": "vllm",
        "name": "vLLM (Qwen3-VL-30B-A3B)",
        "base_url": VLLM_BASE,
        "health_path": "/health",
        "category": "llm",
        "public_endpoints": [
            {"label": "OpenAI Chat", "path": "/v1/chat/completions"},
            {"label": "OpenAI Completions", "path": "/v1/completions"},
        ],
    },
    {
        "id": "asr",
        "name": "ASR (NVIDIA Canary 1B v2)",
        "base_url": ASR_BASE,
        "health_path": "/health",
        "category": "speech",
        "public_endpoints": [
            {"label": "ASR", "path": "/v1/audio/transcriptions"},
        ],
    },
    {
        "id": "diar",
        "name": "Diarization (pyannote community-1)",
        "base_url": DIAR_BASE,
        "health_path": "/health",
        "category": "speech",
        "public_endpoints": [
            {"label": "Diarization", "path": "/v1/audio/diarize"},
        ],
    },
]

app = FastAPI(title="AI Stack Admin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _probe_service(client: httpx.AsyncClient, service: Dict[str, Any]) -> Dict[str, Any]:
    health_url = urljoin(service["base_url"].rstrip("/") + "/", service["health_path"].lstrip("/"))
    result: Dict[str, Any] = {
        "id": service["id"],
        "name": service["name"],
        "category": service["category"],
        "base_url": service["base_url"],
        "health_url": health_url,
        "public_endpoints": [
            {
                "label": ep["label"],
                "url": urljoin(service["base_url"].rstrip("/") + "/", ep["path"].lstrip("/")),
            }
            for ep in service.get("public_endpoints", [])
        ],
    }
    try:
        resp = await client.get(health_url)
        payload: Any
        if resp.headers.get("content-type", "").startswith("application/json"):
            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text
        else:
            payload = resp.text
        result.update(
            {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "response": payload,
            }
        )
    except Exception as exc:  # pragma: no cover - mostly network/nvidia errors
        result.update({"ok": False, "error": str(exc)})
    return result


def _query_gpus() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        return {"gpus": [], "error": "nvidia-smi introuvable dans le conteneur"}
    except subprocess.CalledProcessError as exc:
        return {"gpus": [], "error": f"nvidia-smi a échoué: {exc}"}

    gpus = []
    for raw_line in out.strip().splitlines():
        if not raw_line.strip():
            continue
        parts = [chunk.strip() for chunk in raw_line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx, name, mem_total, mem_used, util, temp = parts[:6]
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "memory_total_mb": int(mem_total),
                    "memory_used_mb": int(mem_used),
                    "utilization_pct": int(util),
                    "temperature_c": int(temp),
                }
            )
        except ValueError:
            continue
    return {"gpus": gpus}


@app.get("/api/services")
async def services() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        statuses = [await _probe_service(client, svc) for svc in SERVICE_TARGETS]
    return {"services": statuses}


@app.get("/api/gpus")
def gpus() -> Dict[str, Any]:
    return _query_gpus()


@app.get("/api/config")
def config() -> Dict[str, Any]:
    return {
        "models_root": os.getenv("MODELS_ROOT", "/models"),
        "huggingface_cache": os.getenv("HF_CACHE", "/root/.cache/huggingface"),
        "services": [
            {
                "id": svc["id"],
                "name": svc["name"],
                "category": svc["category"],
                "base_url": svc["base_url"],
                "public_endpoints": [ep for ep in svc.get("public_endpoints", [])],
            }
            for svc in SERVICE_TARGETS
        ],
    }


# Backwards compatibility with the very first prototype endpoints
@app.get("/api/health")
async def legacy_health():
    return await services()


@app.get("/api/gpu")
def legacy_gpu():
    return _query_gpus()


# Static React (mounted last so /api routes keep priority)
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
