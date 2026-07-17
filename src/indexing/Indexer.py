"""Indexador incremental.

Uso:
    python indexer.py            # indexa/atualiza tudo que mudou
    python indexer.py --force    # reembeda tudo, ignorando hash

Lê os nodes do org-roam.db, extrai o texto de cada subtree, e só
reprocessa (chunk + embedding) os nodes cujo conteúdo mudou desde a
última execução (comparando sha256 do texto).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..extractors import connect_readonly, nodes_with_text, split_into_chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_orgbrain_db(conn: sqlite3.Connection):
    schema = config.SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()


def get_existing_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.execute("SELECT node_id, content_hash FROM indexed_nodes")
    return dict(cur.fetchall())


def upsert_node(conn: sqlite3.Connection, node, text_hash: str):
    conn.execute("""
        INSERT INTO indexed_nodes (node_id, file, title, olp, todo, tags, content_hash, mtime, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            file=excluded.file, title=excluded.title, olp=excluded.olp,
            todo=excluded.todo, tags=excluded.tags,
            content_hash=excluded.content_hash, mtime=excluded.mtime,
            indexed_at=excluded.indexed_at
    """, (
        node.node_id, node.file, node.title, json.dumps(node.olp, ensure_ascii=False),
        node.todo, json.dumps(node.tags, ensure_ascii=False), text_hash,
        0.0, datetime.now(timezone.utc).isoformat(),
    ))


def replace_chunks(conn: sqlite3.Connection, node_id: str, chunks: list[str]) -> list[int]:
    conn.execute("DELETE FROM chunks WHERE node_id = ?", (node_id,))
    ids = []
    for i, text in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks (node_id, chunk_index, text, token_count) VALUES (?, ?, ?, ?)",
            (node_id, i, text, len(text.split())),
        )
        ids.append(cur.lastrowid)
    return ids


def store_embeddings(conn: sqlite3.Connection, chunk_ids: list[int], vectors: np.ndarray, model: str):
    for chunk_id, vec in zip(chunk_ids, vectors):
        conn.execute("""INSERT INTO embeddings (

    chunk_id,

    model,

    dim,

    normalized,

    dtype,

    created_at,

    vector

)

VALUES (

    ?,

    ?,

    ?,

    1,

    'float32',

    ?,

    ?

)

ON CONFLICT(chunk_id, model)

DO UPDATE SET

    dim = excluded.dim,

    normalized = excluded.normalized,

    dtype = excluded.dtype,

    created_at = excluded.created_at,

    vector = excluded.vector)""",
            (chunk_id, model, vec.shape[0], vec.astype(np.float32).tobytes()),
        )


def embed_texts(model, texts: list[str]) -> np.ndarray:
    if config.E5_STYLE_PREFIXES:
        texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="reembeda tudo, ignora cache de hash")
    parser.add_argument("--limit", type=int, default=None,
                         help="indexa só os N primeiros nodes (pra testar antes de rodar na base inteira)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print(f"Carregando modelo {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    roam_conn = connect_readonly(config.ORG_ROAM_DB)
    brain_conn = sqlite3.connect(config.ORGBRAIN_DB)
    init_orgbrain_db(brain_conn)

    existing_hashes = {} if args.force else get_existing_hashes(brain_conn)

    print("Lendo nodes do org-roam.db e extraindo texto...")
    all_nodes_with_text = nodes_with_text(roam_conn)
    print(f"{len(all_nodes_with_text)} nodes encontrados.")
    if args.limit:
        all_nodes_with_text = all_nodes_with_text[:args.limit]
        print(f"--limit ativo: processando só os primeiros {args.limit}.")

    started_at = datetime.now(timezone.utc).isoformat()
    nodes_updated = 0
    chunks_embedded = 0

    for node, text in all_nodes_with_text:
        if not text:
            continue
        h = content_hash(text)
        if existing_hashes.get(node.node_id) == h:
            continue  # nada mudou, pula

        chunks = split_into_chunks(text)
        if not chunks:
            continue

        chunk_ids = replace_chunks(brain_conn, node.node_id, chunks)
        vectors = embed_texts(model, chunks)
        store_embeddings(brain_conn, chunk_ids, vectors, config.EMBEDDING_MODEL)
        upsert_node(brain_conn, node, h)

        nodes_updated += 1
        chunks_embedded += len(chunks)

    brain_conn.execute("""
        INSERT INTO index_runs (started_at, finished_at, nodes_seen, nodes_updated, chunks_embedded, model)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (started_at, datetime.now(timezone.utc).isoformat(), len(all_nodes_with_text),
          nodes_updated, chunks_embedded, config.EMBEDDING_MODEL))

    brain_conn.execute(
        """
        INSERT INTO metadata(key,value)
        VALUES('embedding_revision', ?)
        
        ON CONFLICT(key)
        DO UPDATE SET
        value=excluded.value
        """,
        (
            datetime.now(timezone.utc).isoformat(),
        )
    )
    
    brain_conn.commit()

    print(f"Concluído: {nodes_updated} nodes atualizados, {chunks_embedded} chunks embedados.")


if __name__ == "__main__":
    main()
