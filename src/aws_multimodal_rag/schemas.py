"""Public API and internal domain models."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["Who is Avtandil?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    source_id: str
    title: str
    language: str | None = None
    excerpt: str
    score: float = Field(ge=0)


class UsageMetrics(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class TimingMetrics(BaseModel):
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class AskResponse(BaseModel):
    request_id: str
    answer: str
    language: str
    image_url: HttpUrl | None = None
    citations: list[Citation]
    usage: UsageMetrics
    timing: TimingMetrics
    runtime_mode: Literal["local", "aws"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    runtime_mode: Literal["local", "aws"]
    service: str = "aws-multimodal-rag"


class RetrievedItem(BaseModel):
    item_id: str
    type: Literal["text", "image"]
    title: str
    language: str | None = None
    content: str = ""
    s3_key: str | None = None
    image_url: str | None = None
    score: float = Field(default=0.0, ge=0)


class GenerationResult(BaseModel):
    answer: str
    language: str
    input_tokens: int = 0
    output_tokens: int = 0
