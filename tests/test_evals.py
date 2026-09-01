from pathlib import Path

import pytest

from aws_multimodal_rag.eval_runner import evaluate


@pytest.mark.asyncio
async def test_local_evaluation_guardrails_pass():
    metrics = await evaluate(Path("evals/dataset.jsonl"))
    assert metrics["cases"] == 5
    assert metrics["language_accuracy"] == 1.0
    assert metrics["citation_coverage"] == 1.0
    assert metrics["retrieval_source_accuracy"] == 1.0
    assert metrics["content_check_accuracy"] == 1.0
