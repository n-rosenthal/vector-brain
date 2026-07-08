import sqlite3
import sys


def test_indexer_populates_database(fake_org_roam_db, tmp_path, monkeypatch):
    from src import config
    from src.indexing import main as indexer_main

    orgbrain_db = tmp_path / "vector-brain.db"
    monkeypatch.setattr(config, "ORG_ROAM_DB", fake_org_roam_db)
    monkeypatch.setattr(config, "ORGBRAIN_DB", orgbrain_db)
    monkeypatch.setattr(sys, "argv", ["indexer"])

    indexer_main()

    conn = sqlite3.connect(orgbrain_db)
    nodes = conn.execute("SELECT node_id, title FROM indexed_nodes").fetchall()
    chunks = conn.execute("SELECT node_id, text FROM chunks").fetchall()
    embeddings = conn.execute("SELECT chunk_id, dim FROM embeddings").fetchall()

    assert len(nodes) == 3
    titles = {t for _, t in nodes}
    assert titles == {"Python", "Rust", "SQLite"}
    assert len(chunks) >= 3  # pelo menos 1 chunk por node
    assert len(embeddings) == len(chunks)
    assert all(dim == 8 for _, dim in embeddings)  # FAKE_DIM


def test_indexer_skips_unchanged_nodes_on_second_run(fake_org_roam_db, tmp_path, monkeypatch):
    from src import config
    from src.indexing import main as indexer_main

    orgbrain_db = tmp_path / "vector-brain.db"
    monkeypatch.setattr(config, "ORG_ROAM_DB", fake_org_roam_db)
    monkeypatch.setattr(config, "ORGBRAIN_DB", orgbrain_db)
    monkeypatch.setattr(sys, "argv", ["indexer"])

    indexer_main()
    conn = sqlite3.connect(orgbrain_db)
    first_run_indexed_at = dict(conn.execute("SELECT node_id, indexed_at FROM indexed_nodes").fetchall())

    indexer_main()  # roda de novo, nada mudou nos arquivos
    second_run_indexed_at = dict(conn.execute("SELECT node_id, indexed_at FROM indexed_nodes").fetchall())

    # como nada mudou, indexed_at não deve ter sido atualizado
    assert first_run_indexed_at == second_run_indexed_at


def test_indexer_reprocesses_when_content_changes(fake_org_roam_db, org_files, tmp_path, monkeypatch):
    from src import config
    from src.indexing import main as indexer_main

    orgbrain_db = tmp_path / "vector-brain.db"
    monkeypatch.setattr(config, "ORG_ROAM_DB", fake_org_roam_db)
    monkeypatch.setattr(config, "ORGBRAIN_DB", orgbrain_db)
    monkeypatch.setattr(sys, "argv", ["indexer"])

    indexer_main()
    conn = sqlite3.connect(orgbrain_db)
    hash_before = conn.execute(
        "SELECT content_hash FROM indexed_nodes WHERE node_id='node-python'"
    ).fetchone()[0]

    # modifica o conteúdo do node Python
    f1 = org_files["linguagens"]
    content = f1.read_text(encoding="utf-8")
    f1.write_text(content.replace("alto nível", "alto nível e tipagem dinâmica"), encoding="utf-8")

    indexer_main()
    hash_after = conn.execute(
        "SELECT content_hash FROM indexed_nodes WHERE node_id='node-python'"
    ).fetchone()[0]

    assert hash_before != hash_after
