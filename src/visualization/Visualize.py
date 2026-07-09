"""
Visualização do espaço de embeddings da base indexada.

Uso:
    python -m src.visualization.Visualize
    python -m src.visualization.Visualize --method umap   # se tiver umap-learn instalado

Gera em reports/figures/:
    - embedding_map.png          (projeção 2D estática, colorida por tag)
    - embedding_map.html         (versão interativa, hover com título/arquivo)
    - corpus_stats.png           (tags mais frequentes, nodes por arquivo)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..searching.Search import load_all_embeddings

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def load_embeddings_dataframe(conn: sqlite3.Connection, model: str) -> pd.DataFrame:
    """Carrega embeddings por chunk e agrega em 1 vetor por node (média dos
    chunks), já que pra visualizar o 'mapa das notas' faz mais sentido
    1 ponto = 1 node do que 1 ponto = 1 chunk."""
    matrix, meta = load_all_embeddings(conn, model)
    if matrix is None:
        return pd.DataFrame()

    df = pd.DataFrame(meta)
    df["vector"] = list(matrix)

    def _first_tag(tags_json):
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
        except (json.JSONDecodeError, TypeError):
            tags = []
        return tags[0] if tags else "sem-tag"

    df["primary_tag"] = df["tags"].apply(_first_tag)

    grouped = df.groupby("node_id").agg(
        title=("title", "first"),
        file=("file", "first"),
        primary_tag=("primary_tag", "first"),
        vector=("vector", lambda vs: np.mean(np.vstack(vs), axis=0)),
    ).reset_index()
    return grouped


def _project_2d(vectors: np.ndarray, method: str = "pca") -> np.ndarray:
    if method == "umap":
        try:
            import umap
        except ImportError:
            print("umap-learn não instalado, caindo pra PCA. `pip install umap-learn` pra usar UMAP.")
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(vectors)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, len(vectors) - 1))
        return TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(vectors)

    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def plot_embedding_map_static(df: pd.DataFrame, method: str = "pca", out_path: Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "embedding_map.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.vstack(df["vector"].values)
    coords = _project_2d(vectors, method)
    df = df.assign(x=coords[:, 0], y=coords[:, 1])

    fig, ax = plt.subplots(figsize=(10, 8))
    tags = df["primary_tag"].unique()
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(tags), 1))
    for i, tag in enumerate(sorted(tags)):
        subset = df[df["primary_tag"] == tag]
        ax.scatter(subset["x"], subset["y"], label=tag, s=20, alpha=0.75, color=cmap(i))

    ax.set_title(f"Mapa semântico das notas ({method.upper()})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title="tag primária")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvo: {out_path}")
    return out_path


def plot_embedding_map_interactive(df: pd.DataFrame, method: str = "pca", out_path: Path | None = None) -> Path | None:
    try:
        import plotly.express as px
    except ImportError:
        print("plotly não instalado (`pip install plotly`), pulando versão interativa.")
        return None

    out_path = out_path or (FIGURES_DIR / "embedding_map.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.vstack(df["vector"].values)
    coords = _project_2d(vectors, method)
    df = df.assign(x=coords[:, 0], y=coords[:, 1])

    fig = px.scatter(
        df, x="x", y="y", color="primary_tag",
        hover_data={"title": True, "file": True, "x": False, "y": False, "primary_tag": False},
        title=f"Mapa semântico das notas ({method.upper()}) — passe o mouse pra ver o título",
    )
    fig.write_html(out_path)
    print(f"Salvo: {out_path}")
    return out_path


def plot_corpus_stats(df: pd.DataFrame, out_path: Path | None = None, top_n: int = 15) -> Path:
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "corpus_stats.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tag_counts = Counter(df["primary_tag"]).most_common(top_n)
    if tag_counts:
        labels, counts = zip(*tag_counts)
        axes[0].barh(labels[::-1], counts[::-1], color="#4C72B0")
    axes[0].set_title("Tags mais frequentes (tag primária por node)")
    axes[0].set_xlabel("nº de nodes")

    file_counts = Counter(df["file"].apply(lambda f: Path(f).name)).most_common(top_n)
    if file_counts:
        labels, counts = zip(*file_counts)
        axes[1].barh(labels[::-1], counts[::-1], color="#55A868")
    axes[1].set_title(f"Arquivos com mais nodes (top {top_n})")
    axes[1].set_xlabel("nº de nodes")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvo: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca",
                         help="algoritmo de projeção 2D (padrão: pca, mais rápido e estável)")
    args = parser.parse_args()

    conn = sqlite3.connect(config.ORGBRAIN_DB)
    df = load_embeddings_dataframe(conn, config.EMBEDDING_MODEL)

    if df.empty:
        print("Nenhum embedding encontrado. Rode a indexação primeiro (python -m src.indexing.Indexer).")
        return

    print(f"{len(df)} nodes carregados pra visualização.")
    plot_embedding_map_static(df, method=args.method)
    plot_embedding_map_interactive(df, method=args.method)
    plot_corpus_stats(df)


if __name__ == "__main__":
    main()
