"""Application service that owns the observable RAG request lifecycle."""

import json
import logging
from time import perf_counter
from uuid import uuid4

from .config import Settings
from .runtime import RagRuntime
from .schemas import (
    AskResponse,
    Citation,
    TimingMetrics,
    UsageMetrics,
)

logger = logging.getLogger("aws_multimodal_rag")


class RagService:
    def __init__(self, runtime: RagRuntime, settings: Settings) -> None:
        self.runtime = runtime
        self.settings = settings

    async def ask(self, query: str, top_k: int | None = None) -> AskResponse:
        request_id = f"rag_{uuid4().hex[:12]}"
        started = perf_counter()
        retrieval_started = perf_counter()
        items = await self.runtime.retrieve(query, top_k or self.settings.top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        generation_started = perf_counter()
        result = await self.runtime.generate(query, items)
        generation_ms = (perf_counter() - generation_started) * 1000
        total_ms = (perf_counter() - started) * 1000

        text_items = [item for item in items if item.type == "text"]
        image_item = next((item for item in items if item.type == "image" and item.image_url), None)
        citations = [
            Citation(
                source_id=item.item_id,
                title=item.title,
                language=item.language,
                excerpt=item.content[:240],
                score=item.score,
            )
            for item in text_items
        ]
        estimated_cost = (
            result.input_tokens * self.settings.input_cost_per_million_tokens
            + result.output_tokens * self.settings.output_cost_per_million_tokens
        ) / 1_000_000
        response = AskResponse(
            request_id=request_id,
            answer=result.answer,
            language=result.language,
            image_url=image_item.image_url if image_item else None,
            citations=citations,
            usage=UsageMetrics(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=round(estimated_cost, 8),
            ),
            timing=TimingMetrics(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            runtime_mode=self.settings.runtime_mode,
        )
        logger.info(
            json.dumps(
                {
                    "event": "rag_request_completed",
                    "request_id": request_id,
                    "runtime_mode": self.settings.runtime_mode,
                    "retrieved_items": len(items),
                    "citations": len(citations),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "estimated_cost_usd": response.usage.estimated_cost_usd,
                    "total_ms": response.timing.total_ms,
                }
            )
        )
        return response
