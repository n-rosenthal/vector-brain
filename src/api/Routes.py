"""Rotas da API. Todas somente-leitura sobre orgbrain.db (e org-roam.db
indiretamente, via os dados já denormalizados em indexed_nodes)."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import numpy as np

from .. import config
from ..searching.Search import load_all_embeddings, search_core
from .Dependencies import get_orgbrain_conn, get_embedding_model
from .Schemas import (
    ChunkOut,
    CorpusStats,
    EmbeddingOut,
    EmbeddingQuery,
    EmbeddingQueryResult,
    FileCount,
    HealthStatus,
    IndexRunOut,
    NeighborOut,
    NodeDetail,
    NodeOut,
    SearchResultOut,
    SimilarityQuery,
    SimilarityResult,
    TagCount,
)

from ..searching.Embeddings import (
    encode_text,
    embedding_by_chunk,
    nearest_neighbors,
    similarity_between_texts,
)

router = APIRouter()


def _parse_json_list(raw) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _row_to_node_out(row: sqlite3.Row) -> NodeOut:
    return NodeOut(
        node_id=row["node_id"],
        title=row["title"] or "",
        file=row["file"],
        olp=_parse_json_list(row["olp"]),
        todo=row["todo"],
        tags=_parse_json_list(row["tags"]),
        indexed_at=row["indexed_at"],
    )


@router.get("/health", response_model=HealthStatus, tags=["meta"])
def health():
    return HealthStatus(
        status="ok",
        org_roam_db_found=config.ORG_ROAM_DB.exists(),
        orgbrain_db_found=config.ORGBRAIN_DB.exists(),
        org_roam_db_path=str(config.ORG_ROAM_DB),
        orgbrain_db_path=str(config.ORGBRAIN_DB),
    )


@router.get("/nodes", response_model=list[NodeOut], tags=["nodes"])
def list_nodes(
    tag: Optional[str] = Query(None, description="filtra nodes que tenham essa tag exata"),
    file: Optional[str] = Query(None, description="filtra por substring do caminho do arquivo"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conn: sqlite3.Connection = Depends(get_orgbrain_conn),
):
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM indexed_nodes ORDER BY title").fetchall()
    result = [_row_to_node_out(r) for r in rows]

    if tag:
        result = [n for n in result if tag in n.tags]
    if file:
        result = [n for n in result if file in n.file]

    return result[offset: offset + limit]


@router.get("/nodes/{node_id}", response_model=NodeDetail, tags=["nodes"])
def get_node(node_id: str, conn: sqlite3.Connection = Depends(get_orgbrain_conn)):
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM indexed_nodes WHERE node_id = ?", (node_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="node não encontrado na base indexada")

    chunk_rows = conn.execute(
        "SELECT chunk_index, text, token_count FROM chunks WHERE node_id = ? ORDER BY chunk_index",
        (node_id,),
    ).fetchall()
    chunks = [
        ChunkOut(chunk_index=c["chunk_index"], text=c["text"], token_count=c["token_count"])
        for c in chunk_rows
    ]
    node = _row_to_node_out(row)
    return NodeDetail(**node.model_dump(), chunks=chunks)


@router.get("/tags", response_model=list[TagCount], tags=["nodes"])
def list_tags(conn: sqlite3.Connection = Depends(get_orgbrain_conn)):
    rows = conn.execute("SELECT tags FROM indexed_nodes").fetchall()
    counter: Counter[str] = Counter()
    for (tags_json,) in rows:
        counter.update(_parse_json_list(tags_json))
    return [TagCount(tag=t, count=c) for t, c in counter.most_common()]


@router.get("/search", response_model=list[SearchResultOut], tags=["search"])
def search_endpoint(
    q: str = Query(..., min_length=1, description="query de busca semântica"),
    top_k: int = Query(5, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_orgbrain_conn),
    model=Depends(get_embedding_model),
):
    matrix, meta = load_all_embeddings(conn, config.EMBEDDING_MODEL)
    if matrix is None:
        return []

    results = search_core(q, model, matrix, meta, top_k)
    return [
        SearchResultOut(
            node_id=r["node_id"], title=r["title"], file=r["file"],
            score=r["score"], snippet=(r["text"] or "")[:300],
        )
        for r in results
    ]


@router.get("/stats", response_model=CorpusStats, tags=["meta"])
def stats(conn: sqlite3.Connection = Depends(get_orgbrain_conn)):
    total_nodes = conn.execute("SELECT COUNT(*) FROM indexed_nodes").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    tag_rows = conn.execute("SELECT tags FROM indexed_nodes").fetchall()
    tag_counter: Counter[str] = Counter()
    for (tags_json,) in tag_rows:
        tag_counter.update(_parse_json_list(tags_json))
    top_tags = [TagCount(tag=t, count=c) for t, c in tag_counter.most_common(10)]

    file_rows = conn.execute("SELECT file FROM indexed_nodes").fetchall()
    file_counter = Counter(f for (f,) in file_rows)
    top_files = [FileCount(file=f, count=c) for f, c in file_counter.most_common(10)]

    last_run_row = conn.execute("""
        SELECT started_at, finished_at, nodes_seen, nodes_updated, chunks_embedded, model
        FROM index_runs ORDER BY id DESC LIMIT 1
    """).fetchone()
    last_run = IndexRunOut(**dict(zip(
        ["started_at", "finished_at", "nodes_seen", "nodes_updated", "chunks_embedded", "model"],
        last_run_row,
    ))) if last_run_row else None

    return CorpusStats(
        total_nodes=total_nodes,
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        embedding_model=config.EMBEDDING_MODEL,
        top_tags=top_tags,
        top_files=top_files,
        last_index_run=last_run,
    )


@router.get(
    "/embeddings/{chunk_id}",
    response_model=EmbeddingOut,
    tags=["embeddings"],
)
def get_embedding(
    chunk_id: int,
    conn: sqlite3.Connection = Depends(get_orgbrain_conn),
):
    embedding = embedding_by_chunk(conn, chunk_id)

    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail="embedding não encontrado",
        )

    return EmbeddingOut(
        chunk_id=chunk_id,
        model=embedding.model,
        dimension=embedding.dimension,
        vector=embedding.vector.tolist(),
    )


@router.get(
    "/embeddings/{chunk_id}/neighbors",
    response_model=list[NeighborOut],
    tags=["embeddings"],
)
def embedding_neighbors(
    chunk_id: int,
    k: int = Query(10, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_orgbrain_conn),
):

    return nearest_neighbors(
        conn=conn,
        chunk_id=chunk_id,
        k=k,
    )


@router.post(
    "/embeddings/query",
    response_model=EmbeddingQueryResult,
    tags=["embeddings"],
)
def embed_query(
    query: EmbeddingQuery,
    model=Depends(get_embedding_model),
):

    vector = encode_text(
        model=model,
        text=query.text,
    )

    return EmbeddingQueryResult(
        model=config.EMBEDDING_MODEL,
        dimension=len(vector),
        vector=vector.tolist(),
    )

@router.post(
    "/embeddings/similarity",
    response_model=SimilarityResult,
    tags=["embeddings"],
)
def similarity_endpoint(
    query: SimilarityQuery,
    model=Depends(get_embedding_model),
):

    similarity = similarity_between_texts(
        model=model,
        text_a=query.text_a,
        text_b=query.text_b,
    )

    return SimilarityResult(
        cosine_similarity=similarity["cosine_similarity"],
        euclidean_distance=similarity["euclidean_distance"],
        dot_product=similarity["dot_product"],
    )
