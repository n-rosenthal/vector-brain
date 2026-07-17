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
CREATE TABLE embeddings (

    chunk_id      INTEGER NOT NULL,

    model         TEXT NOT NULL,

    dim           INTEGER NOT NULL,

    normalized    INTEGER NOT NULL DEFAULT 1,

    dtype         TEXT NOT NULL DEFAULT 'float32',

    created_at    TEXT NOT NULL,

    vector        BLOB NOT NULL,

    PRIMARY KEY(chunk_id, model),

    FOREIGN KEY(chunk_id)
        REFERENCES chunks(id)
        ON DELETE CASCADE

);

CREATE TABLE projections (

    id INTEGER PRIMARY KEY,

    algorithm TEXT NOT NULL,

    dimensions INTEGER NOT NULL,

    parameters TEXT,

    model TEXT NOT NULL,

    created_at TEXT NOT NULL

);

CREATE TABLE projection_points (

    projection_id INTEGER NOT NULL,

    chunk_id INTEGER NOT NULL,

    x REAL,

    y REAL,

    z REAL,

    PRIMARY KEY(projection_id, chunk_id),

    FOREIGN KEY(projection_id)
        REFERENCES projections(id)
        ON DELETE CASCADE,

    FOREIGN KEY(chunk_id)
        REFERENCES chunks(id)
        ON DELETE CASCADE

);

CREATE TABLE clusterings (

    id INTEGER PRIMARY KEY,

    algorithm TEXT NOT NULL,

    model TEXT NOT NULL,

    parameters TEXT,

    created_at TEXT NOT NULL

);

CREATE TABLE cluster_members (

    clustering_id INTEGER NOT NULL,

    chunk_id INTEGER NOT NULL,

    cluster INTEGER NOT NULL,

    probability REAL,

    PRIMARY KEY(clustering_id, chunk_id),

    FOREIGN KEY(clustering_id)
        REFERENCES clusterings(id)
        ON DELETE CASCADE,

    FOREIGN KEY(chunk_id)
        REFERENCES chunks(id)
        ON DELETE CASCADE

);

CREATE TABLE semantic_graphs (

    id INTEGER PRIMARY KEY,

    algorithm TEXT,

    k INTEGER,

    threshold REAL,

    model TEXT,

    created_at TEXT

);

CREATE TABLE semantic_edges (

    graph_id INTEGER,

    source_chunk INTEGER,

    target_chunk INTEGER,

    similarity REAL,

    PRIMARY KEY(
        graph_id,
        source_chunk,
        target_chunk
    ),

    FOREIGN KEY(graph_id)
        REFERENCES semantic_graphs(id)
        ON DELETE CASCADE
);

CREATE TABLE embedding_snapshots (

    id INTEGER PRIMARY KEY,

    run_id INTEGER,

    projection_id INTEGER,

    created_at TEXT
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

CREATE TABLE metadata (

    key TEXT PRIMARY KEY,

    value TEXT NOT NULL

);

CREATE INDEX idx_embeddings_model
ON embeddings(model);

CREATE INDEX idx_projection_points_projection
ON projection_points(projection_id);

CREATE INDEX idx_cluster_members_cluster
ON cluster_members(cluster);

CREATE INDEX idx_semantic_edges_source
ON semantic_edges(source_chunk);

CREATE INDEX idx_semantic_edges_target
ON semantic_edges(target_chunk);
