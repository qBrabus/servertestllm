\
import os, subprocess, json, time
from typing import Dict, Any
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

VLLM_BASE = os.getenv("VLLM_BASE", "http://vllm:8000")
ASR_BASE  = os.getenv("ASR_BASE", "http://asr-canary:6000")
DIAR_BASE = os.getenv("DIAR_BASE", "http://diarizer:7000")

app = FastAPI(title="LLM/ASR/Diar Admin API")

# Serve static frontend
if os.path.isdir("/app/static"):
    app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")

@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/api/services")
def services():
    def ping(url):
        try:
            r = requests.get(url, timeout=2)
            return {"ok": r.status_code == 200, "status": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {
        "vllm": ping(f"{VLLM_BASE}/health"),
        "asr":  ping(f"{ASR_BASE}/health"),
        "diar": ping(f"{DIAR_BASE}/health"),
    }

@app.get("/api/gpus")
def gpus():
    # Fallback to nvidia-smi parsing (no NVML dependency)
    try:
        q = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            text=True
        )
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mem_total, mem_used, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx),
                "name": name,
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "utilization_pct": int(util),
                "temperature_c": int(temp),
            })
        return {"gpus": gpus}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
