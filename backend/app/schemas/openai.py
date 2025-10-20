from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen")
    messages: List[ChatMessage]
    max_tokens: int | None = Field(default=512, ge=1, le=4096)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=0.9, ge=0, le=1)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str = Field(default="qwen")
    prompt: str
    max_tokens: int | None = Field(default=512, ge=1, le=4096)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=0.9, ge=0, le=1)


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, str]]
    usage: UsageInfo
