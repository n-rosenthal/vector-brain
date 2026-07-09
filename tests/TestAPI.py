import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(fake_org_roam_db, tmp_path, monkeypatch):
    """Indexa uma base fake e devolve um TestClient da API apontando pra ela."""
    from src import config
    from src.indexing import main as indexer_main

    orgbrain_db = tmp_path / "vector-brain.db"
    monkeypatch.setattr(config, "ORG_ROAM_DB", fake_org_roam_db)
    monkeypatch.setattr(config, "ORGBRAIN_DB", orgbrain_db)
    monkeypatch.setattr(config, "API_KEY", None)
    monkeypatch.setattr(sys, "argv", ["indexer"])
    indexer_main()

    from src.api.App import create_app
    app = create_app()
    return TestClient(app)


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["orgbrain_db_found"] is True


def test_list_nodes(api_client):
    resp = api_client.get("/nodes")
    assert resp.status_code == 200
    nodes = resp.json()
    assert len(nodes) == 3
    titles = {n["title"] for n in nodes}
    assert titles == {"Python", "Rust", "SQLite"}


def test_list_nodes_filter_by_tag(api_client):
    resp = api_client.get("/nodes", params={"tag": "programacao"})
    assert resp.status_code == 200
    nodes = resp.json()
    assert all("programacao" in n["tags"] for n in nodes)


def test_get_node_detail(api_client):
    resp = api_client.get("/nodes/node-python")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Python"
    assert len(body["chunks"]) >= 1


def test_get_node_not_found(api_client):
    resp = api_client.get("/nodes/node-inexistente")
    assert resp.status_code == 404


def test_list_tags(api_client):
    resp = api_client.get("/tags")
    assert resp.status_code == 200
    tags = {t["tag"] for t in resp.json()}
    assert "programacao" in tags
    assert "banco-de-dados" in tags


def test_search_endpoint(api_client):
    resp = api_client.get("/search", params={"q": "qualquer coisa", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert "score" in results[0]


def test_stats(api_client):
    resp = api_client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_nodes"] == 3
    assert body["total_chunks"] >= 3
    assert body["last_index_run"] is not None


def test_api_key_blocks_when_configured(api_client):
    from src import config
    # simula API key ligada DEPOIS do app já criado, mexendo direto no config
    # (o middleware lê config.API_KEY a cada request, então isso é suficiente)
    original = config.API_KEY
    config.API_KEY = "segredo123"
    try:
        resp_no_key = api_client.get("/health")
        assert resp_no_key.status_code == 401

        resp_with_key = api_client.get("/health", headers={"X-API-Key": "segredo123"})
        assert resp_with_key.status_code == 200
    finally:
        config.API_KEY = original
