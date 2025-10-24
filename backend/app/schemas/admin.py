from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total: float
    memory_used: float
    load: float
    temperature: float | None = None


class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float


class ModelInfo(BaseModel):
    identifier: str
    task: str
    loaded: bool
    description: str
    format: str
    params: Dict[str, object] | None = None


class ModelLoadRequest(BaseModel):
    gpu_device_ids: List[int] | None = None


class RegistryStatus(BaseModel):
    models: Dict[str, ModelInfo]


class DashboardState(BaseModel):
    gpus: List[GPUInfo]
    system: SystemMetrics
    models: Dict[str, ModelInfo]


class HuggingFaceTokenStatus(BaseModel):
    has_token: bool


class HuggingFaceTokenUpdate(BaseModel):
    token: str | None = None
