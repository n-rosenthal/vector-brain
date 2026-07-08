"""
Busca semântica sobre a base indexada.

Uso:
    python search.py "sistemas de compilação incremental"
    python search.py "zettelkasten" --top 10
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np

from .. import config


def load_all_embeddings(conn: sqlite3.Connection, model: str):
    cur = conn.execute("""
        SELECT e.chunk_id, e.vector, e.dim, c.text, c.node_id,
               n.title, n.file, n.olp, n.tags, n.todo
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN indexed_nodes n ON n.node_id = c.node_id
        WHERE e.model = ?
    """, (model,))
    rows = cur.fetchall()
    if not rows:
        return None, []

    dim = rows[0][2]
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    meta = []
    for i, (chunk_id, vec_blob, d, text, node_id, title, file, olp, tags, todo) in enumerate(rows):
        matrix[i] = np.frombuffer(vec_blob, dtype=np.float32)
        meta.append({
            "chunk_id": chunk_id, "text": text, "node_id": node_id,
            "title": title, "file": file, "olp": olp, "tags": tags, "todo": todo,
        })
    return matrix, meta


def search(query: str, top_k: int = 5):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    conn = sqlite3.connect(config.ORGBRAIN_DB)

    matrix, meta = load_all_embeddings(conn, config.EMBEDDING_MODEL)
    if matrix is None:
        print("Nenhum embedding encontrado. Rode indexer.py primeiro.")
        return []

    q_text = f"query: {query}" if config.E5_STYLE_PREFIXES else query
    q_vec = model.encode([q_text], normalize_embeddings=True)[0]

    # matrix já vem normalizada (normalize_embeddings=True na indexação),
    # então o produto escalar é a similaridade de cosseno.
    scores = matrix @ q_vec
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_idx:
        m = meta[idx]
        results.append({**m, "score": float(scores[idx])})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top)
    for r in results:
        print(f"\n[{r['score']:.3f}] {r['title']}  ({r['file']})")
        snippet = r["text"][:200].replace("\n", " ")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
