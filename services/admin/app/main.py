import os
import subprocess
from typing import Dict, Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

VLLM_BASE = os.getenv("VLLM_BASE", "http://vllm:8000")
ASR_BASE  = os.getenv("ASR_BASE",  "http://asr-canary:6000")
DIAR_BASE = os.getenv("DIAR_BASE", "http://diarizer:7000")

app = FastAPI(title="AI Stack Admin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Static React
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")

@app.get("/api/health")
async def health():
    async with httpx.AsyncClient(timeout=3.0) as client:
        async def chk(url: str) -> Dict[str, Any]:
            try:
                r = await client.get(url)
                return {"ok": r.status_code == 200, "data": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {
            "vllm": await chk(f"{VLLM_BASE}/health"),
            "asr":  await chk(f"{ASR_BASE}/health"),
            "diar": await chk(f"{DIAR_BASE}/health"),
        }

@app.get("/api/gpu")
def gpu():
    # nécessite --gpus all au run et runtime nvidia
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], text=True)
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mem_total, mem_used, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx),
                "name": name,
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "utilization_pct": int(util),
                "temperature_c": int(temp)
            })
        return {"gpus": gpus}
    except Exception as e:
        return {"error": str(e)}
