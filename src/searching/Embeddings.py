"""
Operações sobre embeddings armazenados no banco.

Este módulo centraliza todas as operações vetoriais que não envolvem
diretamente a busca semântica por texto.

As funções aqui implementadas serão reutilizadas por:

- API de embeddings
- UMAP
- PCA
- t-SNE
- clustering
- explicabilidade
- visualizações
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

import numpy as np


# ==========================================================
# Estruturas
# ==========================================================

@dataclass(slots=True)
class StoredEmbedding:
    chunk_id: int
    model: str
    dimension: int
    vector: np.ndarray


# ==========================================================
# Conversão
# ==========================================================

def _bytes_to_vector(blob: bytes) -> np.ndarray:
    """
    Converte BLOB do SQLite em ndarray float32.
    """
    return np.frombuffer(blob, dtype=np.float32)


# ==========================================================
# Codificação
# ==========================================================

def encode_text(model, text: str) -> np.ndarray:
    """
    Gera embedding normalizado de um texto.
    """

    vector = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    return vector.astype(np.float32)


# ==========================================================
# Carregamento
# ==========================================================

def embedding_by_chunk(
    conn: sqlite3.Connection,
    chunk_id: int,
) -> StoredEmbedding | None:

    row = conn.execute(
        """
        SELECT
            chunk_id,
            model,
            dim,
            vector
        FROM embeddings
        WHERE chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()

    if row is None:
        return None

    return StoredEmbedding(
        chunk_id=row[0],
        model=row[1],
        dimension=row[2],
        vector=_bytes_to_vector(row[3]),
    )


# ==========================================================
# Similaridade
# ==========================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(np.dot(a, b))


def euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(np.linalg.norm(a - b))


def similarity_between_texts(
    model,
    text_a: str,
    text_b: str,
):

    va = encode_text(model, text_a)
    vb = encode_text(model, text_b)

    return {
        "cosine_similarity": cosine_similarity(va, vb),
        "euclidean_distance": euclidean_distance(va, vb),
        "dot_product": float(np.dot(va, vb)),
    }


# ==========================================================
# Banco inteiro
# ==========================================================

def load_all_embeddings(
    conn: sqlite3.Connection,
    model_name: str,
):
    """
    Carrega todos os embeddings de um determinado modelo.

    Retorna

        matriz
        metadados
    """

    rows = conn.execute(
        """
        SELECT
            e.chunk_id,
            e.vector,
            c.node_id,
            n.title,
            n.file

        FROM embeddings e

        JOIN chunks c
            ON c.id = e.chunk_id

        JOIN indexed_nodes n
            ON n.node_id = c.node_id

        WHERE e.model = ?
        """,
        (model_name,),
    ).fetchall()

    if not rows:
        return None, []

    vectors = []
    metadata = []

    for row in rows:

        vectors.append(
            _bytes_to_vector(row[1])
        )

        metadata.append(
            {
                "chunk_id": row[0],
                "node_id": row[2],
                "title": row[3],
                "file": row[4],
            }
        )

    matrix = np.vstack(vectors)

    return matrix, metadata


# ==========================================================
# Vizinhos
# ==========================================================

def nearest_neighbors(
    conn: sqlite3.Connection,
    chunk_id: int,
    k: int = 10,
):

    row = embedding_by_chunk(conn, chunk_id)

    if row is None:
        return []

    matrix, metadata = load_all_embeddings(
        conn,
        row.model,
    )

    if matrix is None:
        return []

    scores = matrix @ row.vector

    order = np.argsort(-scores)

    result = []

    for idx in order:

        meta = metadata[idx]

        if meta["chunk_id"] == chunk_id:
            continue

        result.append(
            {
                "chunk_id": meta["chunk_id"],
                "node_id": meta["node_id"],
                "title": meta["title"],
                "file": meta["file"],
                "score": float(scores[idx]),
            }
        )

        if len(result) >= k:
            break

    return result
