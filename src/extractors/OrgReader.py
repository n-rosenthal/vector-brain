"""
Lê o org-roam.db (somente leitura) e extrai, para cada node, o texto bruto
do seu subtree (do headline até o próximo headline de nível <= o dele,
ou fim do arquivo).

Não reimplementamos parsing de org-mode do zero: aproveitamos que o
org-roam já sabe onde cada node começa (campo `pos`, em caracteres/bytes
a partir do início do arquivo) e qual o nível de cada um.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OrgNode:
    node_id: str
    file: str
    title: str
    level: int
    pos: int
    todo: str | None
    olp: list[str]
    tags: list[str]


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    # mode=ro garante que nunca escrevemos acidentalmente no org-roam.db
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _unquote_lisp_string(raw: str) -> str:
    """Alguns bancos org-roam gravam certos campos de texto (ex: `file`)
    já 'impressos' no formato Emacs Lisp, com aspas literais em volta e
    aspas internas escapadas: "/caminho/arquivo.org" (com as aspas fazendo
    parte da string salva no SQLite). Remove isso pra virar um path usável."""
    if isinstance(raw, str) and len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return raw


def fetch_nodes(conn: sqlite3.Connection) -> list[OrgNode]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, file, title, level, pos, todo, olp
        FROM nodes
        ORDER BY file, pos
    """)
    rows = cur.fetchall()

    # tags ficam em tabela separada
    cur.execute("SELECT node_id, tag FROM tags")
    tags_by_node: dict[str, list[str]] = {}
    for node_id, tag in cur.fetchall():
        tags_by_node.setdefault(node_id, []).append(tag)

    nodes = []
    for node_id, file, title, level, pos, todo, olp_raw in rows:
        olp = _parse_elisp_list(olp_raw)
        nodes.append(OrgNode(
            node_id=node_id,
            file=_unquote_lisp_string(file),
            title=_unquote_lisp_string(title) or "",
            level=level,
            pos=pos,
            todo=todo,
            olp=olp,
            tags=tags_by_node.get(node_id, []),
        ))
    return nodes


def _parse_elisp_list(raw) -> list[str]:
    """org-roam guarda olp/properties como texto no formato elisp: ("a" "b").
    Fazemos um parsing simples o suficiente para essa forma específica."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        # tenta json primeiro (algumas versões/serializações usam json)
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # fallback: extrai strings entre aspas
    import re
    return re.findall(r'"([^"]*)"', str(raw))


def extract_subtree_text(node: OrgNode, next_pos_same_file: int | None) -> str:
    """Lê o arquivo e retorna o texto do subtree deste node."""
    path = Path(node.file)
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8", errors="replace")

    start = node.pos - 1  # org-roam pos é 1-indexed
    end = next_pos_same_file - 1 if next_pos_same_file is not None else len(raw)
    text = raw[start:end]

    # Remove a linha do headline em si (já temos o title separado) e a
    # PROPERTIES drawer, que não agregam valor semântico para embeddings.
    lines = text.split("\n")
    cleaned = []
    in_properties = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0 and stripped.startswith("*"):
            continue  # linha do headline
        if stripped == ":PROPERTIES:":
            in_properties = True
            continue
        if stripped == ":END:" and in_properties:
            in_properties = False
            continue
        if in_properties:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def nodes_with_text(conn: sqlite3.Connection) -> list[tuple[OrgNode, str]]:
    """Retorna todos os nodes junto com seu texto de subtree extraído."""
    nodes = fetch_nodes(conn)

    # agrupa por arquivo, mantendo ordem por pos, para achar o "próximo pos"
    by_file: dict[str, list[OrgNode]] = {}
    for n in nodes:
        by_file.setdefault(n.file, []).append(n)

    result = []
    for file, file_nodes in by_file.items():
        for i, node in enumerate(file_nodes):
            next_pos = None
            for later in file_nodes[i + 1:]:
                if later.level <= node.level:
                    next_pos = later.pos
                    break
            text = extract_subtree_text(node, next_pos)
            result.append((node, text))
    return result
