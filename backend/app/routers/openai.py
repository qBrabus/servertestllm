from __future__ import annotations

import time
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import settings
from ..schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    UsageInfo,
)
from ..services.model_registry import registry

router = APIRouter(tags=["openai-compatible"])


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.openai_api_keys:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split(" ", 1)[1]
    if token not in settings.openai_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _resolve_model_key(model: str) -> str:
    aliases = {
        "qwen": "qwen",
        "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen",
    }
    return aliases.get(model, model)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    _: None = Depends(require_api_key),
) -> ChatCompletionResponse:
    model_key = _resolve_model_key(request.model)
    model = await registry.get(model_key)
    result = await model.infer(
        messages=[message.model_dump() for message in request.messages],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    message = ChatMessage(role="assistant", content=result["content"])
    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(message.content.split())
    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        model=model.metadata.identifier,
        created=int(time.time()),
        choices=[ChatCompletionChoice(index=0, message=message)],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


@router.post("/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    _: None = Depends(require_api_key),
) -> CompletionResponse:
    model_key = _resolve_model_key(request.model)
    model = await registry.get(model_key)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request.prompt},
    ]
    result = await model.infer(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    completion = {
        "text": result["content"],
        "index": 0,
        "finish_reason": "stop",
    }
    prompt_tokens = len(request.prompt.split())
    completion_tokens = len(result["content"].split())
    return CompletionResponse(
        id=str(uuid.uuid4()),
        model=model.metadata.identifier,
        created=int(time.time()),
        choices=[completion],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
