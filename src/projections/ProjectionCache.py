"""
ProjectionCache.py

Persistência e cache das projeções.

Cada projeção é identificada por

    revision
    algorithm
    dimensions
    parameters

A projeção é armazenada no SQLite como um ndarray serializado.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from functools import lru_cache
from typing import Any

import numpy as np


class ProjectionCache:
    """
    Cache persistente das projeções.
    """

    def __init__(self, conn: sqlite3.Connection):

        self.conn = conn

        self._create_tables()

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    def _create_tables(self):

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projection_cache (

                cache_key TEXT PRIMARY KEY,

                revision TEXT NOT NULL,

                algorithm TEXT NOT NULL,

                dimensions INTEGER NOT NULL,

                parameters TEXT NOT NULL,

                projection BLOB NOT NULL,

                created_at TEXT NOT NULL
            )
            """
        )

        self.conn.commit()

    # --------------------------------------------------------
    # Chave
    # --------------------------------------------------------

    @staticmethod
    def make_key(
        revision: str,
        algorithm: str,
        dimensions: int,
        parameters: dict[str, Any] | None,
    ) -> str:

        payload = json.dumps(
            {
                "revision": revision,
                "algorithm": algorithm,
                "dimensions": dimensions,
                "parameters": parameters or {},
            },
            sort_keys=True,
        )

        return hashlib.sha256(
            payload.encode("utf8")
        ).hexdigest()

    # --------------------------------------------------------
    # Serialização
    # --------------------------------------------------------

    @staticmethod
    def _serialize(
        projection: np.ndarray,
    ) -> bytes:

        buffer = io.BytesIO()

        np.save(
            buffer,
            projection,
            allow_pickle=False,
        )

        return buffer.getvalue()

    @staticmethod
    def _deserialize(
        blob: bytes,
    ) -> np.ndarray:

        buffer = io.BytesIO(blob)

        return np.load(
            buffer,
            allow_pickle=False,
        )

    # --------------------------------------------------------
    # Consulta
    # --------------------------------------------------------

    @lru_cache(maxsize=16)
    def get(
        self,
        cache_key: str,
    ) -> np.ndarray | None:

        row = self.conn.execute(
            """
            SELECT projection

            FROM projection_cache

            WHERE cache_key=?
            """,
            (cache_key,),
        ).fetchone()

        if row is None:
            return None

        return self._deserialize(
            row[0]
        )

    # --------------------------------------------------------
    # Inserção
    # --------------------------------------------------------

    def put(
        self,
        revision: str,
        algorithm: str,
        dimensions: int,
        parameters: dict[str, Any] | None,
        projection: np.ndarray,
        created_at: str,
    ):

        key = self.make_key(
            revision,
            algorithm,
            dimensions,
            parameters,
        )

        self.conn.execute(
            """
            INSERT OR REPLACE INTO projection_cache(

                cache_key,

                revision,

                algorithm,

                dimensions,

                parameters,

                projection,

                created_at

            )

            VALUES(

                ?,?,?,?,?,?,?

            )
            """,
            (
                key,
                revision,
                algorithm,
                dimensions,
                json.dumps(parameters or {}),
                self._serialize(projection),
                created_at,
            ),
        )

        self.conn.commit()

        self.get.cache_clear()

    # --------------------------------------------------------
    # Limpeza
    # --------------------------------------------------------

    def invalidate_revision(
        self,
        revision: str,
    ):

        self.conn.execute(
            """
            DELETE FROM projection_cache

            WHERE revision=?
            """,
            (revision,),
        )

        self.conn.commit()

        self.get.cache_clear()

    def clear(self):

        self.conn.execute(
            "DELETE FROM projection_cache"
        )

        self.conn.commit()

        self.get.cache_clear()

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    def size(self) -> int:

        row = self.conn.execute(
            """
            SELECT COUNT(*)

            FROM projection_cache
            """
        ).fetchone()

        return row[0]

    def keys(self):

        rows = self.conn.execute(
            """
            SELECT

                cache_key,

                algorithm,

                dimensions,

                revision,

                created_at

            FROM projection_cache
            """
        ).fetchall()

        return rows