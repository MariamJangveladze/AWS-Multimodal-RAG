"""Amazon Bedrock, S3, and FAISS runtime implementation."""

import asyncio
import base64
import json
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from .config import Settings
from .schemas import GenerationResult, RetrievedItem

SYSTEM_PROMPT = """You are Amo, a multilingual guide to The Knight in the Panther's Skin.
Answer only from the retrieved context. Respond in the language of the user's latest question.
If the evidence is insufficient, say so clearly. Do not invent stanza numbers, quotations, or URLs.
Keep the answer concise unless the user requests detail."""


class AwsRuntime:
    """Lazy-loading AWS runtime suitable for a long-lived container or Lambda."""

    def __init__(self, settings: Settings) -> None:
        settings.validate_aws()
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the AWS dependencies with: uv sync --extra aws") from exc

        retry_config = Config(retries={"max_attempts": 8, "mode": "adaptive"})
        self.settings = settings
        self.bedrock = boto3.client(
            "bedrock-runtime", region_name=settings.aws_region, config=retry_config
        )
        self.s3 = boto3.client("s3", region_name=settings.aws_region, config=retry_config)
        self._index: Any | None = None
        self._items: list[RetrievedItem] | None = None

    def _load_index(self) -> None:
        if self._index is not None and self._items is not None:
            return
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the AWS dependencies with: uv sync --extra aws") from exc

        index_path = Path(gettempdir()) / "rag_index.faiss"
        metadata_path = Path(gettempdir()) / "rag_items.json"
        self.s3.download_file(
            self.settings.s3_bucket, self.settings.faiss_index_key, str(index_path)
        )
        self.s3.download_file(
            self.settings.s3_bucket, self.settings.metadata_key, str(metadata_path)
        )
        self._index = faiss.read_index(str(index_path))
        raw_items = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._items = [RetrievedItem.model_validate(item) for item in raw_items]
        if self._index.ntotal != len(self._items):
            raise RuntimeError("FAISS vector count does not match metadata item count")

    def _embed_text(self, text: str) -> list[float]:
        body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_RETRIEVAL",
                "embeddingDimension": self.settings.embedding_dimension,
                "text": {"truncationMode": "END", "value": text},
            },
        }
        response = self.bedrock.invoke_model(
            modelId=self.settings.embedding_model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embeddings"][0]["embedding"]

    def _retrieve_sync(self, query: str, top_k: int) -> list[RetrievedItem]:
        import numpy as np

        self._load_index()
        vector = np.asarray(self._embed_text(query), dtype=np.float32).reshape(1, -1)
        distances, indices = self._index.search(vector, top_k)
        assert self._items is not None
        results: list[RetrievedItem] = []
        for distance, index_position in zip(distances[0], indices[0], strict=True):
            if index_position < 0:
                continue
            item = self._items[int(index_position)]
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            image_url = item.image_url
            if item.type == "image" and item.s3_key:
                image_url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.settings.s3_bucket, "Key": item.s3_key},
                    ExpiresIn=self.settings.presigned_url_ttl_seconds,
                )
            results.append(item.model_copy(update={"score": score, "image_url": image_url}))
        return results

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        return await asyncio.to_thread(self._retrieve_sync, query, top_k)

    def _image_block(self, item: RetrievedItem) -> dict[str, Any] | None:
        if not item.s3_key:
            return None
        response = self.s3.get_object(Bucket=self.settings.s3_bucket, Key=item.s3_key)
        content_length = int(response.get("ContentLength", 0))
        if content_length > self.settings.max_image_bytes:
            raise ValueError("Retrieved image exceeds the configured size limit")
        image_bytes = response["Body"].read(self.settings.max_image_bytes + 1)
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ValueError("Retrieved image exceeds the configured size limit")
        suffix = item.s3_key.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
        return {
            "image": {
                "format": suffix,
                "source": {"bytes": base64.b64encode(image_bytes).decode("utf-8")},
            }
        }

    def _generate_sync(self, query: str, items: list[RetrievedItem]) -> GenerationResult:
        content: list[dict[str, Any]] = []
        for item in items:
            if item.type == "text":
                content.append(
                    {
                        "text": (
                            f"SOURCE {item.item_id} | {item.title} | "
                            f"language={item.language}\n{item.content}"
                        )
                    }
                )
            else:
                block = self._image_block(item)
                if block:
                    content.append(block)
        content.append({"text": f"QUESTION\n{query}"})
        request = {
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": 500, "temperature": 0.1, "topP": 0.9},
        }
        response = self.bedrock.invoke_model(
            modelId=self.settings.generation_model_id,
            body=json.dumps(request),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        blocks = payload.get("output", {}).get("message", {}).get("content", [])
        answer = "\n".join(block["text"] for block in blocks if block.get("text")).strip()
        usage = payload.get("usage", {})
        from .local_runtime import detect_language

        return GenerationResult(
            answer=answer,
            language=detect_language(query),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
        )

    async def generate(self, query: str, items: list[RetrievedItem]) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, query, items)
