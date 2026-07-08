-- orgbrain.db
-- Banco COMPLEMENTAR ao org-roam.db. Nunca escrevemos no org-roam.db,
-- apenas lemos node_id/file/mtime/hash de lá para saber o que (re)indexar.

PRAGMA journal_mode = WAL;

-- Um registro por node do org-roam que decidimos indexar.
-- Guarda o hash do CONTEÚDO (não do node inteiro) para saber se precisa
-- reprocessar/reembedar quando o arquivo mudar.
CREATE TABLE IF NOT EXISTS indexed_nodes (
    node_id       TEXT PRIMARY KEY,   -- mesmo id usado pelo org-roam (ID property)
    file          TEXT NOT NULL,
    title         TEXT,
    olp           TEXT,               -- outline path (json list), copiado do org-roam
    todo          TEXT,
    tags          TEXT,               -- json list, copiado do org-roam
    content_hash  TEXT NOT NULL,      -- sha256 do texto extraído do subtree
    mtime         REAL NOT NULL,      -- mtime do arquivo no momento da indexação
    indexed_at    TEXT NOT NULL       -- timestamp ISO da última (re)indexação
);

-- Chunks de texto de cada node. Um node pode virar 1+ chunks se o subtree
-- for grande (ex: um node de 3000 palavras vira vários chunks).
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT NOT NULL REFERENCES indexed_nodes(node_id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,   -- ordem do chunk dentro do node
    text          TEXT NOT NULL,
    token_count   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_node ON chunks(node_id);

-- Embeddings por chunk. Guardamos o nome do modelo para poder ter
-- múltiplos modelos coexistindo (ex: comparar e5-small vs bge-m3).
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id      INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,      -- float32, via numpy.tobytes()
    PRIMARY KEY (chunk_id, model)
);

-- Log simples de execuções de indexação, útil pra debugar / ver histórico.
CREATE TABLE IF NOT EXISTS index_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    nodes_seen    INTEGER,
    nodes_updated INTEGER,
    chunks_embedded INTEGER,
    model         TEXT
);
