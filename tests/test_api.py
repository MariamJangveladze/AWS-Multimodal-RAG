from fastapi.testclient import TestClient

from aws_multimodal_rag.api import app
from aws_multimodal_rag.config import get_settings

AUTH = {"Authorization": "Bearer test-token"}


def test_health_reports_local_runtime(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    monkeypatch.setenv("RAG_API_TOKEN", "test-token")
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["runtime_mode"] == "local"


def test_ask_returns_grounded_contract(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    monkeypatch.setenv("RAG_API_TOKEN", "test-token")
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": "Who is Avtandil?"}, headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert "courageous commander" in payload["answer"]
    assert payload["language"] == "en"
    assert payload["citations"]
    assert payload["runtime_mode"] == "local"
    assert payload["timing"]["total_ms"] >= 0


def test_empty_query_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    monkeypatch.setenv("RAG_API_TOKEN", "test-token")
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": ""}, headers=AUTH)
    assert response.status_code == 422
