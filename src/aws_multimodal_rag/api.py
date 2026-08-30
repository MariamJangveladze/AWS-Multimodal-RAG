"""FastAPI entry point for the multimodal RAG backend."""

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from .aws_runtime import AwsRuntime
from .config import get_settings
from .local_runtime import LocalRuntime
from .schemas import AskRequest, AskResponse, HealthResponse
from .service import RagService

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_service() -> RagService:
    settings = get_settings()
    runtime = AwsRuntime(settings) if settings.runtime_mode == "aws" else LocalRuntime()
    return RagService(runtime, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = build_service()
    yield


app = FastAPI(
    title="AWS Multimodal RAG",
    version="0.1.0",
    description="Multilingual text-and-image RAG backend powered by Amazon Bedrock.",
    lifespan=lifespan,
)


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().api_token
    if not expected:
        raise HTTPException(status_code=503, detail="Set RAG_API_TOKEN")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid API token")


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(request: Request) -> HealthResponse:
    service: RagService = request.app.state.rag_service
    return HealthResponse(runtime_mode=service.settings.runtime_mode)


@app.post(
    "/v1/ask",
    response_model=AskResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_token)],
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    service: RagService = request.app.state.rag_service
    query = payload.query.strip()
    if len(query) > service.settings.max_query_characters:
        raise HTTPException(status_code=422, detail="Query exceeds configured length limit")
    try:
        return await service.ask(query, payload.top_k)
    except Exception as exc:
        logging.getLogger("aws_multimodal_rag").exception("RAG request failed")
        raise HTTPException(
            status_code=503, detail="RAG service is temporarily unavailable"
        ) from exc
