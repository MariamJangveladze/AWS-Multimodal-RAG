"""Runtime configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration for local and AWS runtime modes."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    runtime_mode: Literal["local", "aws"] = "local"
    aws_region: str = "eu-central-1"
    s3_bucket: str = ""
    faiss_index_key: str = "index/amo_index.faiss"
    metadata_key: str = "index/items.json"
    embedding_model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0"
    generation_model_id: str = "us.amazon.nova-pro-v1:0"
    embedding_dimension: int = Field(default=1024, ge=256)
    top_k: int = Field(default=5, ge=1, le=20)
    max_query_characters: int = Field(default=2000, ge=100, le=10000)
    api_token: str = ""
    max_image_bytes: int = Field(default=5_000_000, ge=100_000, le=20_000_000)
    presigned_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    input_cost_per_million_tokens: float = Field(default=0.0, ge=0)
    output_cost_per_million_tokens: float = Field(default=0.0, ge=0)

    def validate_aws(self) -> None:
        """Fail early when AWS mode is selected without required settings."""
        if self.runtime_mode == "aws":
            if not self.s3_bucket:
                raise ValueError("RAG_S3_BUCKET is required when RAG_RUNTIME_MODE=aws")
            if not self.api_token:
                raise ValueError("RAG_API_TOKEN is required when RAG_RUNTIME_MODE=aws")


@lru_cache
def get_settings() -> Settings:
    return Settings()
