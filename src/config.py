"""Configuração central do projeto ~vector-brain~."""
import os
from pathlib import Path

# Caminho para o banco do org-roam (ajuste para o seu setup).
# Padrão comum: ~/.emacs.d/org-roam.db  ou  ~/.config/emacs/org-roam.db
ORG_ROAM_DB = Path(os.environ.get(
    "ORG_ROAM_DB",
    str(Path.home() / ".emacs.d" / "org-roam.db"),
))

# Nosso banco complementar (fica ao lado, não mexe no do org-roam).
ORGBRAIN_DB = Path(os.environ.get(
    "VECTOR_BRAIN_DB",
    str(Path.home() / "org" / "roam" / "vault" / "vector-brain" / "data" / "vector-brain.db"),
))

# Modelo de embedding. Multilingual porque o conteúdo é em português.
# Alternativas: "intfloat/multilingual-e5-base" (mais qualidade, mais lento)
#               "BAAI/bge-m3" (ainda melhor, bem mais pesado)
EMBEDDING_MODEL = os.environ.get("VECTOR_BRAIN_MODEL", "intfloat/multilingual-e5-small")

# Caminho pro schema.sql, resolvido relativo a este arquivo (config.py está
# em src/, schema.sql está em src/database/). Assim não importa de onde
# você roda o comando.
SCHEMA_PATH = Path(__file__).parent / "database" / "schema.sql"

# Tamanho alvo de chunk em caracteres (aprox). Nodes menores que isso
# viram um chunk só; nodes maiores são divididos.
CHUNK_TARGET_CHARS  = 1500
CHUNK_OVERLAP_CHARS = 200

# Modelos da família e5 esperam prefixos "query: " / "passage: " nos textos.
# Se trocar de modelo para um que não use essa convenção, ajuste aqui.
E5_STYLE_PREFIXES = True
