from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas.admin import (
    DashboardState,
    GPUInfo,
    HuggingFaceTokenStatus,
    HuggingFaceTokenUpdate,
    ModelInfo,
    ModelLoadRequest,
    RegistryStatus,
    SystemMetrics,
)
from ..services.gpu_monitor import gpu_monitor
from ..services.model_registry import registry
from ..services.token_store import token_store

router = APIRouter(tags=["admin"])


async def _collect_model_info() -> dict[str, ModelInfo]:
    models_raw = await registry.status()
    return {
        key: ModelInfo(
            identifier=info.identifier,
            task=info.task,
            loaded=info.loaded,
            description=info.description,
            format=info.format,
            params=info.params,
        )
        for key, info in models_raw.items()
    }


@router.get("/status", response_model=DashboardState)
async def get_status() -> DashboardState:
    gpu_state = gpu_monitor.get_status()
    gpus = [
        GPUInfo(
            id=gpu_id,
            name=status.name,
            memory_total=status.memory_total,
            memory_used=status.memory_used,
            load=status.load,
            temperature=status.temperature,
        )
        for gpu_id, status in gpu_state.items()
    ]
    system = SystemMetrics(**gpu_monitor.system_metrics())
    models = await _collect_model_info()
    return DashboardState(gpus=gpus, system=system, models=models)


@router.post("/models/{model_key}/load")
async def load_model(model_key: str, payload: ModelLoadRequest | None = None) -> RegistryStatus:
    try:
        device_ids = payload.gpu_device_ids if payload else None
        await registry.ensure_loaded(model_key, device_ids=device_ids)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model")
    return RegistryStatus(models=await _collect_model_info())


@router.post("/models/{model_key}/unload")
async def unload_model(model_key: str) -> RegistryStatus:
    try:
        await registry.unload(model_key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model")
    return RegistryStatus(models=await _collect_model_info())


@router.get("/huggingface/token", response_model=HuggingFaceTokenStatus)
async def get_huggingface_token() -> HuggingFaceTokenStatus:
    current = registry.get_hf_token()
    if current is None:
        current = token_store.load()
    return HuggingFaceTokenStatus(has_token=current is not None)


@router.post("/huggingface/token", response_model=HuggingFaceTokenStatus)
async def set_huggingface_token(payload: HuggingFaceTokenUpdate) -> HuggingFaceTokenStatus:
    raw_value = payload.token.strip() if payload.token else None
    if raw_value:
        token_store.save(raw_value)
    else:
        token_store.clear()
    await registry.set_hf_token(raw_value)
    object.__setattr__(settings, "huggingface_token", raw_value)
    return HuggingFaceTokenStatus(has_token=raw_value is not None)
