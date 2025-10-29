from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

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


class DependencyStatus(BaseModel):
    name: str
    version: str | None = None
    cuda: bool | None = None
    details: Dict[str, object] | None = None
    error: str | None = None


class ModelRuntimeInfo(BaseModel):
    state: Literal["idle", "loading", "ready", "error"]
    progress: int
    status: str
    details: Dict[str, object] | None = None
    server: Dict[str, object] | None = None
    downloaded: bool = False
    last_error: str | None = None
    updated_at: datetime | None = None


class ModelInfo(BaseModel):
    identifier: str
    task: str
    loaded: bool
    description: str
    format: str
    params: Dict[str, object] | None = None
    runtime: Optional[ModelRuntimeInfo] = None


class ModelLoadRequest(BaseModel):
    gpu_device_ids: List[int] | None = None


class RegistryStatus(BaseModel):
    models: Dict[str, ModelInfo]


class DashboardState(BaseModel):
    gpus: List[GPUInfo]
    system: SystemMetrics
    models: Dict[str, ModelInfo]
    dependencies: List[DependencyStatus]


class HuggingFaceTokenStatus(BaseModel):
    has_token: bool


class HuggingFaceTokenUpdate(BaseModel):
    token: str | None = None
