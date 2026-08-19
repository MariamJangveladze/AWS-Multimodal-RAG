"""Runtime contract shared by the deterministic demo and AWS implementation."""

from typing import Protocol

from .schemas import GenerationResult, RetrievedItem


class RagRuntime(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        """Return semantically relevant text and image items."""

    async def generate(self, query: str, items: list[RetrievedItem]) -> GenerationResult:
        """Generate a grounded answer from retrieved items."""
