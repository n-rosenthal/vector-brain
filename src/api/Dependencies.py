"""
Dependências injetáveis da API (padrão FastAPI `Depends`).

O modelo de embedding é cacheado com lru_cache: carregado uma única vez,
na primeira busca (ou no startup, se você chamar get_embedding_model()
explicitamente no lifespan da app) — evita recarregar o modelo (alguns
segundos) a cada request de /search.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from .. import config
from ..extractors import connect_readonly


def get_org_roam_conn():
    conn = connect_readonly(config.ORG_ROAM_DB)
    try:
        yield conn
    finally:
        conn.close()


def get_orgbrain_conn():
    conn = sqlite3.connect(config.ORGBRAIN_DB)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)
