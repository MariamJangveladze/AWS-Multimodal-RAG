from fastapi.testclient import TestClient

from aws_multimodal_rag.api import app


def test_health_reports_local_runtime(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["runtime_mode"] == "local"


def test_ask_returns_grounded_contract(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": "Who is Avtandil?"})
    assert response.status_code == 200
    payload = response.json()
    assert "courageous commander" in payload["answer"]
    assert payload["language"] == "en"
    assert payload["citations"]
    assert payload["runtime_mode"] == "local"
    assert payload["timing"]["total_ms"] >= 0


def test_empty_query_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": ""})
    assert response.status_code == 422
