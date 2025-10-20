from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routers import admin, audio, diarization, openai
from .services.gpu_monitor import gpu_monitor
from .services.model_registry import registry

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Unified Inference Gateway", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(openai.router, prefix="/v1")
    app.include_router(audio.router, prefix="/api/audio")
    app.include_router(diarization.router, prefix="/api/diarization")
    app.include_router(admin.router, prefix="/api/admin")

    static_path = settings.frontend_dist
    if static_path.exists():
        app.mount("/", StaticFiles(directory=static_path, html=True), name="frontend")

        @app.middleware("http")
        async def spa_redirect(request: Request, call_next: Any):
            response = await call_next(request)
            if response.status_code == 404 and request.method == "GET":
                index_file = static_path / "index.html"
                if index_file.exists():
                    return FileResponse(index_file)
            return response

    @app.on_event("startup")
    async def on_startup() -> None:
        logging.basicConfig(level=settings.log_level.upper())
        registry.configure(hf_token=settings.huggingface_token, cache_dir=settings.model_cache_dir)
        gpu_monitor.start()
        if not settings.lazy_load_models:
            for key in registry.keys():
                await registry.ensure_loaded(key)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await registry.shutdown()
        gpu_monitor.stop()

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()
