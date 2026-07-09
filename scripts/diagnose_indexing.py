"""
Diagnóstico de indexação: descobre por que nodes estão vindo com texto vazio.

Uso:
    python3 -m scripts.diagnose_indexing
"""
from pathlib import Path

from src import config
from src.extractors import connect_readonly, fetch_nodes, nodes_with_text


def main():
    conn = connect_readonly(config.ORG_ROAM_DB)

    print(f"ORG_ROAM_DB = {config.ORG_ROAM_DB}\n")

    nodes = fetch_nodes(conn)
    print(f"{len(nodes)} nodes lidos da tabela `nodes`.\n")

    # 1. quantos arquivos referenciados pelos nodes existem de fato no disco?
    unique_files = {n.file for n in nodes}
    missing = [f for f in unique_files if not Path(f).exists()]
    print(f"{len(unique_files)} arquivos únicos referenciados.")
    print(f"{len(missing)} desses arquivos NÃO existem no caminho salvo no banco.\n")

    if missing:
        print("Exemplos de caminhos que não foram encontrados:")
        for f in missing[:5]:
            print(f"  - {f!r}")
        print()

    if missing and len(missing) == len(unique_files):
        print(">>> TODOS os arquivos estão \"não encontrados\". Provavelmente é um problema")
        print(">>> de path (ex: '~' literal não expandido, ou path relativo salvo pelo")
        print(">>> org-roam que não corresponde ao diretório onde você roda o Python).")
        return

    # 2. dos nodes com arquivo existente, quantos extraem texto não-vazio?
    results = nodes_with_text(conn)
    empty = [(n, t) for n, t in results if not t.strip()]
    nonempty = [(n, t) for n, t in results if t.strip()]

    print(f"{len(nonempty)} nodes com texto extraído (não vazio).")
    print(f"{len(empty)} nodes com texto vazio.\n")

    if empty:
        print("Exemplos de nodes com texto vazio (título, arquivo, pos):")
        for n, _ in empty[:5]:
            exists = Path(n.file).exists()
            print(f"  - '{n.title}' | file={n.file} (existe? {exists}) | level={n.level} pos={n.pos}")
        print()

    if nonempty:
        print("Exemplo de texto extraído com sucesso, pro primeiro node não-vazio:")
        n, t = nonempty[0]
        print(f"  Node: '{n.title}' ({n.file})")
        print(f"  Texto (300 primeiros chars): {t[:300]!r}")


if __name__ == "__main__":
    main()
