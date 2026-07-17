"""
Modelos Pydantic pra request/response da API.
"""

from __future__ import annotations

from typing import Optional, Any

from pydantic import BaseModel, Field


# ============================================================
# Nodes
# ============================================================

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


# ============================================================
# Search
# ============================================================

class SearchResultOut(BaseModel):
    node_id: str
    title: str
    file: str
    score: float
    snippet: str


# ============================================================
# Embeddings
# ============================================================

class EmbeddingOut(BaseModel):
    chunk_id: int
    model: str
    dimension: int
    vector: list[float]


class EmbeddingQuery(BaseModel):
    text: str


class EmbeddingQueryResult(BaseModel):
    model: str
    dimension: int
    vector: list[float]


class SimilarityQuery(BaseModel):
    text_a: str
    text_b: str


class SimilarityResult(BaseModel):
    cosine_similarity: float
    euclidean_distance: float
    dot_product: float


class NeighborOut(BaseModel):
    chunk_id: int
    node_id: str
    title: str
    file: str
    score: float


# ============================================================
# Metadata
# ============================================================

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


# ============================================================
# Projections
# ============================================================
class ProjectionPointOut(BaseModel):

    chunk_id: int

    node_id: str

    title: str

    file: str

    x: float

    y: float

    z: float | None = None

    attributes: dict[str, Any] = Field(default_factory=dict)


class ProjectionOut(BaseModel):

    algorithm: str

    dimensions: int

    revision: str

    generated_at: str

    total_points: int

    points: list[ProjectionPointOut]


class ProjectionStatistics(BaseModel):

    algorithm: str

    revision: str

    total_points: int

    x_min: float
    x_max: float

    y_min: float
    y_max: float

    z_min: float | None = None
    z_max: float | None = None


class ProjectionAlgorithmOut(BaseModel):

    name: str

    supports_2d: bool

    supports_3d: bool

    deterministic: bool
    
    