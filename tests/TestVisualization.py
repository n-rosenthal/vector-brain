import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, ".")

# mocka sentence_transformers antes de qualquer import que dependa dele
sys.path.insert(0, "tests")
from FakeEmbeddings import FakeSentenceTransformer  # noqa: E402
fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_module

from src import config  # noqa: E402
from src.indexing import main as indexer_main  # noqa: E402
from src.visualization import (  # noqa: E402
    load_embeddings_dataframe, plot_embedding_map_static,
    plot_embedding_map_interactive, plot_corpus_stats,
)

tmp = Path(tempfile.mkdtemp())
org_dir = tmp / "org"
org_dir.mkdir()

# alguns arquivos/nodes fake com tags variadas, pra ter algo pra visualizar
docs = {
    "python.org": [
        ("node-py-1", "Python", "programacao", "Python é uma linguagem de alto nível, dinâmica e legível."),
        ("node-py-2", "Django", "programacao", "Django é um framework web em Python, orientado a baterias inclusas."),
    ],
    "rust.org": [
        ("node-rust-1", "Rust", "programacao", "Rust é uma linguagem de sistemas com foco em segurança de memória."),
    ],
    "historia.org": [
        ("node-hist-1", "Revolução Francesa", "historia", "A Revolução Francesa começou em 1789 e mudou a Europa."),
        ("node-hist-2", "Napoleão", "historia", "Napoleão Bonaparte ascendeu ao poder após a revolução."),
    ],
    "filosofia.org": [
        ("node-fil-1", "Estoicismo", "filosofia", "O estoicismo é uma escola filosófica grega focada na virtude."),
    ],
}

roam_db = tmp / "org-roam.db"
conn = sqlite3.connect(roam_db)
conn.executescript("""
CREATE TABLE nodes (id TEXT PRIMARY KEY, file TEXT, level INTEGER, pos INTEGER, todo TEXT, title TEXT, olp TEXT);
CREATE TABLE tags (node_id TEXT, tag TEXT);
""")

for filename, nodes in docs.items():
    path = org_dir / filename
    content = ""
    positions = []
    for node_id, title, tag, body in nodes:
        positions.append((node_id, title, tag, len(content) + 1))
        content += f"* {title}\n:PROPERTIES:\n:ID: {node_id}\n:END:\n{body}\n"
    path.write_text(content, encoding="utf-8")
    for node_id, title, tag, pos in positions:
        conn.execute("INSERT INTO nodes (id, file, level, pos, todo, title, olp) VALUES (?, ?, 1, ?, NULL, ?, '()')",
                     (node_id, str(path), pos, title))
        conn.execute("INSERT INTO tags (node_id, tag) VALUES (?, ?)", (node_id, tag))
conn.commit()
conn.close()

config.ORG_ROAM_DB = roam_db
config.ORGBRAIN_DB = tmp / "orgbrain.db"

sys.argv = ["indexer"]
indexer_main()

brain_conn = sqlite3.connect(config.ORGBRAIN_DB)
df = load_embeddings_dataframe(brain_conn, config.EMBEDDING_MODEL)
print(f"\n{len(df)} nodes carregados. Tags: {df['primary_tag'].unique().tolist()}")

figures_dir = tmp / "reports" / "figures"
plot_embedding_map_static(df, out_path=figures_dir / "embedding_map.png")
plot_embedding_map_interactive(df, out_path=figures_dir / "embedding_map.html")
plot_corpus_stats(df, out_path=figures_dir / "corpus_stats.png")

print("\nArquivos gerados:")
for f in figures_dir.iterdir():
    print(" -", f, f.stat().st_size, "bytes")
