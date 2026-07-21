"""Pydantic request/response schemas for the gateway's public API."""
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    cheap = "cheap"
    standard = "standard"
    premium = "premium"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    service_key: str = Field(..., description="API key identifying the calling service")
    messages: list[ChatMessage]
    model: Optional[str] = Field(None, description="Explicit provider model, e.g. gpt-4o-mini")
    model_tier: Optional[ModelTier] = Field(
        None, description="Let the router pick a model for this cost/quality tier"
    )
    quality_critical: bool = Field(
        False, description="If true, router avoids downgrading to a cheaper model even under budget pressure"
    )
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class CompletionResponse(BaseModel):
    id: str
    model_used: str
    provider: str
    content: str
    usage: Usage
    cache_hit: bool = False
    downgraded_from_tier: Optional[str] = Field(
        None, description="Set when cost optimization downgraded the request from this tier due to team budget pressure"
    )
