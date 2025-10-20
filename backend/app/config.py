from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import BaseSettings, Field, field_validator


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("info", env="LOG_LEVEL")

    # Hugging Face token for downloading gated models
    huggingface_token: str | None = Field(None, env="HUGGINGFACE_TOKEN")

    # Directory where models are cached
    model_cache_dir: Path = Field(Path("/models"), env="MODEL_CACHE_DIR")

    # Allowed CORS origins for the admin web UI
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # OpenAI compatible API key(s)
    openai_api_keys: List[str] = Field(default_factory=list, env="OPENAI_API_KEYS")

    # Control whether models should be lazily loaded
    lazy_load_models: bool = Field(True, env="LAZY_LOAD_MODELS")

    frontend_dist: Path = Field(Path("/app/frontend"), env="FRONTEND_DIST")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("openai_api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value: Any) -> List[str] | Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
