import sys

import pytest


@pytest.fixture
def indexed_db(fake_org_roam_db, tmp_path, monkeypatch):
    """Roda o indexer (com embeddings falsos, via fixture autouse do
    test_indexer_smoke) e devolve o path do orgbrain.db resultante."""
    from src import config
    from src.indexing import main as indexer_main

    orgbrain_db = tmp_path / "vector-brain.db"
    monkeypatch.setattr(config, "ORG_ROAM_DB", fake_org_roam_db)
    monkeypatch.setattr(config, "ORGBRAIN_DB", orgbrain_db)
    monkeypatch.setattr(sys, "argv", ["indexer"])
    indexer_main()
    return orgbrain_db


def test_search_returns_results_from_indexed_base(indexed_db, monkeypatch):
    from src import config
    from src.searching import search as run_search

    monkeypatch.setattr(config, "ORGBRAIN_DB", indexed_db)

    results = run_search("qualquer coisa", top_k=3)

    assert len(results) == 3
    for r in results:
        assert "score" in r
        assert "title" in r
        assert r["title"] in {"Python", "Rust", "SQLite"}


def test_search_top_k_limits_results(indexed_db, monkeypatch):
    from src import config
    from src.searching import search as run_search

    monkeypatch.setattr(config, "ORGBRAIN_DB", indexed_db)

    results = run_search("teste", top_k=1)
    assert len(results) == 1


def test_search_on_empty_database_returns_empty(tmp_path, monkeypatch):
    from src import config
    from src.searching import search as run_search
    from src.indexing.Indexer import init_orgbrain_db
    import sqlite3

    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(empty_db)
    init_orgbrain_db(conn)
    conn.close()

    monkeypatch.setattr(config, "ORGBRAIN_DB", empty_db)
    results = run_search("nada aqui")
    assert results == []
