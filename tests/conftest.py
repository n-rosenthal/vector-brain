"""Fixtures compartilhadas: um org-roam.db falso + arquivos .org de teste."""
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from FakeEmbeddings import FakeSentenceTransformer


@pytest.fixture(autouse=True)
def fake_sentence_transformers(monkeypatch):
    """Injeta um módulo sentence_transformers falso no sys.modules pra todos
    os testes, já que Indexer.py e Search.py fazem
    `from sentence_transformers import SentenceTransformer` dentro das
    funções (import tardio -- não dá pra mockar só a classe via monkeypatch
    de atributo, precisa interceptar o import em si)."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield


ORG_ROAM_MINIMAL_SCHEMA = """
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    file TEXT,
    level INTEGER,
    pos INTEGER,
    todo TEXT,
    title TEXT,
    olp TEXT
);
CREATE TABLE tags (
    node_id TEXT,
    tag TEXT
);
"""


@pytest.fixture
def org_files(tmp_path: Path) -> dict[str, Path]:
    """Cria 2 arquivos .org de teste e retorna os paths."""
    f1 = tmp_path / "linguagens.org"
    f1.write_text(
        "* Python\n"
        ":PROPERTIES:\n"
        ":ID: node-python\n"
        ":END:\n"
        "Python é uma linguagem de programação de alto nível, "
        "conhecida pela sintaxe legível e ecossistema de bibliotecas.\n"
        "* Rust\n"
        ":PROPERTIES:\n"
        ":ID: node-rust\n"
        ":END:\n"
        "Rust é uma linguagem de sistemas focada em segurança de memória "
        "sem garbage collector.\n",
        encoding="utf-8",
    )

    f2 = tmp_path / "bancos.org"
    f2.write_text(
        "* SQLite\n"
        ":PROPERTIES:\n"
        ":ID: node-sqlite\n"
        ":END:\n"
        "SQLite é um banco de dados relacional embutido, sem servidor, "
        "muito usado em aplicações locais.\n",
        encoding="utf-8",
    )

    return {"linguagens": f1, "bancos": f2}


@pytest.fixture
def fake_org_roam_db(tmp_path: Path, org_files) -> Path:
    """Cria um org-roam.db mínimo apontando pros arquivos de teste,
    com pos/level batendo com o conteúdo real dos arquivos."""
    db_path = tmp_path / "org-roam.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(ORG_ROAM_MINIMAL_SCHEMA)

    f1 = org_files["linguagens"]
    f2 = org_files["bancos"]
    text1 = f1.read_text(encoding="utf-8")
    text2 = f2.read_text(encoding="utf-8")

    pos_python = text1.index("* Python") + 1  # org-roam pos é 1-indexed
    pos_rust = text1.index("* Rust") + 1
    pos_sqlite = text2.index("* SQLite") + 1

    conn.executemany(
        "INSERT INTO nodes (id, file, level, pos, todo, title, olp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("node-python", str(f1), 1, pos_python, None, "Python", '()'),
            ("node-rust", str(f1), 1, pos_rust, None, "Rust", '()'),
            ("node-sqlite", str(f2), 1, pos_sqlite, None, "SQLite", '()'),
        ],
    )
    conn.executemany(
        "INSERT INTO tags (node_id, tag) VALUES (?, ?)",
        [("node-python", "programacao"), ("node-sqlite", "banco-de-dados")],
    )
    conn.commit()
    conn.close()
    return db_path
