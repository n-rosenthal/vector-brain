"""Modelos Pydantic pra request/response da API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class NodeOut(BaseModel):
    node_id: str
    title: str
    file: str
    olp: list[str] = Field(default_factory=list)
    todo: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    indexed_at: str


class ChunkOut(BaseModel):
    chunk_index: int
    text: str
    token_count: Optional[int] = None


class NodeDetail(NodeOut):
    chunks: list[ChunkOut] = Field(default_factory=list)


class SearchResultOut(BaseModel):
    node_id: str
    title: str
    file: str
    score: float
    snippet: str


class TagCount(BaseModel):
    tag: str
    count: int


class FileCount(BaseModel):
    file: str
    count: int


class IndexRunOut(BaseModel):
    started_at: str
    finished_at: Optional[str] = None
    nodes_seen: Optional[int] = None
    nodes_updated: Optional[int] = None
    chunks_embedded: Optional[int] = None
    model: Optional[str] = None


class CorpusStats(BaseModel):
    total_nodes: int
    total_chunks: int
    total_embeddings: int
    embedding_model: str
    top_tags: list[TagCount]
    top_files: list[FileCount]
    last_index_run: Optional[IndexRunOut] = None


class HealthStatus(BaseModel):
    status: str
    org_roam_db_found: bool
    orgbrain_db_found: bool
    org_roam_db_path: str
    orgbrain_db_path: str
