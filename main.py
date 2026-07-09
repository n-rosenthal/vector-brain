from src import config
from src.extractors import connect_readonly, nodes_with_text

conn = connect_readonly(config.ORG_ROAM_DB)
resultados = nodes_with_text(conn)
for node, text in resultados[:3]:
    print(f"--- {node.title} ({node.file}) ---")
    print(text[:300])
    print()
