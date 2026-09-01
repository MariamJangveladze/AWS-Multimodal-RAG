"""Deterministic, zero-cost runtime for demos, tests, and CI."""

import json
import re
from pathlib import Path

from .schemas import GenerationResult, RetrievedItem

ENGLISH_STOP_WORDS = {"a", "an", "and", "are", "is", "of", "the", "to", "what", "who"}


def detect_language(text: str) -> str:
    """Detect the three supported scripts without an external dependency."""
    if re.search(r"[\u10A0-\u10FF]", text):
        return "ka"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "en"


class LocalRuntime:
    """Lexical fixture runtime that mirrors the production response contract."""

    def __init__(self, fixture_path: Path | None = None) -> None:
        path = fixture_path or Path(__file__).parents[2] / "data" / "local_items.json"
        self._items = [RetrievedItem.model_validate(item) for item in json.loads(path.read_text())]

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        query_terms = set(re.findall(r"\w+", query.casefold())) - ENGLISH_STOP_WORDS
        ranked: list[tuple[float, RetrievedItem]] = []
        for item in self._items:
            searchable = f"{item.title} {item.content}".casefold()
            overlap = sum(1 for term in query_terms if term in searchable)
            language_bonus = 1 if item.language == detect_language(query) else 0
            score = float(overlap * 2 + language_bonus)
            if overlap:
                ranked.append((score, item.model_copy(update={"score": score})))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in ranked if score > 0][:top_k]

    async def generate(self, query: str, items: list[RetrievedItem]) -> GenerationResult:
        language = detect_language(query)
        text_items = [item for item in items if item.type == "text"]
        if not text_items:
            no_evidence = {
                "ka": "დამტკიცებულ წყაროებში საკმარისი მტკიცებულება ვერ მოიძებნა.",
                "ar": "لم يتم العثور على أدلة كافية في المصادر المعتمدة.",
                "en": "Not enough evidence was found in the approved sources.",
            }
            answer = no_evidence[language]
            return GenerationResult(
                answer=answer,
                language=language,
                input_tokens=len(query.split()),
                output_tokens=len(answer.split()),
            )
        context = text_items[0].content
        prefixes = {
            "ka": "დემო პასუხი მოძიებული კონტექსტიდან:",
            "ar": "إجابة تجريبية من السياق المسترجع:",
            "en": "Demo answer from the retrieved context:",
        }
        answer = f"{prefixes[language]} {context}"
        return GenerationResult(
            answer=answer,
            language=language,
            input_tokens=len(query.split()) + len(context.split()),
            output_tokens=len(answer.split()),
        )
