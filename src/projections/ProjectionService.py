"""
ProjectionService.py

Serviço responsável por gerar projeções do espaço vetorial.

Fluxo:

SQLite
    ↓
Embeddings
    ↓
Projector
    ↓
Cache
    ↓
ProjectionResult
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3

import numpy as np

from .. import config
from ..searching.Embeddings import load_all_embeddings
from .PCAProjector import PCAProjector
from .ProjectionCache import ProjectionCache


# ============================================================
# Modelos
# ============================================================

@dataclass(slots=True)
class ProjectionPoint:

    chunk_id: int

    node_id: str

    title: str

    file: str

    x: float

    y: float

    z: float | None = None


@dataclass(slots=True)
class ProjectionResult:

    algorithm: str

    dimensions: int

    revision: str

    generated_at: str

    total_points: int

    points: list[ProjectionPoint]


# ============================================================
# Serviço
# ============================================================

class ProjectionService:

    def __init__(
        self,
        conn: sqlite3.Connection,
    ):

        self.conn = conn

        self.cache = ProjectionCache(conn)

    # --------------------------------------------------------

    def _revision(self) -> str:
        """
        Lê a revisão atual dos embeddings.
        """

        row = self.conn.execute(
            """
            SELECT value

            FROM metadata

            WHERE key='embedding_revision'
            """
        ).fetchone()

        if row is None:
            return "0"

        return row[0]

    # --------------------------------------------------------

    def _projector(
        self,
        algorithm: str,
        dimensions: int,
    ):

        algorithm = algorithm.lower()

        if algorithm == "pca":

            return PCAProjector(
                dimensions=dimensions
            )

        raise ValueError(
            f"Algoritmo desconhecido: {algorithm}"
        )

    # --------------------------------------------------------

    def project(
        self,
        algorithm: str = "pca",
        dimensions: int = 2,
        parameters: dict | None = None,
    ) -> ProjectionResult:

        revision = self._revision()

        cache_key = ProjectionCache.make_key(
            revision=revision,
            algorithm=algorithm,
            dimensions=dimensions,
            parameters=parameters,
        )

        cached = self.cache.get(cache_key)

        matrix, metadata = load_all_embeddings(
            self.conn,
            config.EMBEDDING_MODEL,
        )

        if matrix is None:

            return ProjectionResult(
                algorithm=algorithm,
                dimensions=dimensions,
                revision=revision,
                generated_at=datetime.now(
                    timezone.utc
                ).isoformat(),
                total_points=0,
                points=[],
            )

        if cached is None:

            projector = self._projector(
                algorithm,
                dimensions,
            )

            projected = projector.fit_transform(
                matrix
            )

            self.cache.put(
                revision=revision,
                algorithm=algorithm,
                dimensions=dimensions,
                parameters=parameters,
                projection=projected,
                created_at=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

        else:

            projected = cached

        points: list[ProjectionPoint] = []

        for coords, meta in zip(projected, metadata):

            if dimensions == 2:

                point = ProjectionPoint(

                    chunk_id=meta["chunk_id"],

                    node_id=meta["node_id"],

                    title=meta["title"],

                    file=meta["file"],

                    x=float(coords[0]),

                    y=float(coords[1]),
                )

            else:

                point = ProjectionPoint(

                    chunk_id=meta["chunk_id"],

                    node_id=meta["node_id"],

                    title=meta["title"],

                    file=meta["file"],

                    x=float(coords[0]),

                    y=float(coords[1]),

                    z=float(coords[2]),
                )

            points.append(point)

        return ProjectionResult(

            algorithm=algorithm,

            dimensions=dimensions,

            revision=revision,

            generated_at=datetime.now(
                timezone.utc
            ).isoformat(),

            total_points=len(points),

            points=points,
        )