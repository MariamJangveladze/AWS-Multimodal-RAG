"""Small repeatable evaluation harness for the local portfolio demo."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .local_runtime import LocalRuntime
from .service import RagService


async def evaluate(dataset_path: Path) -> dict[str, float | int]:
    service = RagService(LocalRuntime(), Settings(runtime_mode="local"))
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    language_passes = 0
    citation_passes = 0
    retrieval_passes = 0
    content_passes = 0
    for case in cases:
        response = await service.ask(case["query"])
        language_passes += response.language == case["expected_language"]
        expected_sources = set(case["expected_source_ids"])
        actual_sources = {citation.source_id for citation in response.citations}
        citation_passes += bool(response.citations) == bool(expected_sources)
        retrieval_passes += actual_sources == expected_sources
        content_passes += case["expected_phrase"].casefold() in response.answer.casefold()
    total = len(cases)
    return {
        "cases": total,
        "language_accuracy": language_passes / total,
        "citation_coverage": citation_passes / total,
        "retrieval_source_accuracy": retrieval_passes / total,
        "content_check_accuracy": content_passes / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic RAG evaluation suite")
    parser.add_argument("dataset", nargs="?", default="evals/dataset.jsonl")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(Path(args.dataset))), indent=2))


if __name__ == "__main__":
    main()
