"""
Substituto falso pra sentence_transformers.SentenceTransformer, usado nos
testes de indexação/busca pra não depender de baixar um modelo real
(~100-500MB) nem de acesso à internet durante os testes.

Gera vetores determinísticos a partir de um hash do texto, então textos
iguais (ou com prefixo query:/passage: diferente mas mesmo conteúdo)
não ficam idênticos entre si -- o suficiente pra testar o encanamento
(pipeline) sem testar a qualidade semântica real do modelo.
"""
import hashlib

import numpy as np

FAKE_DIM = 8  # sha256 = 32 bytes = 8 floats de 4 bytes


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.array([_text_to_vector(t) for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return vectors


def _text_to_vector(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # transforma os bytes do hash em floats pseudo-aleatórios mas determinísticos
    raw = np.frombuffer(h[:FAKE_DIM * 4], dtype=np.uint32).astype(np.float32)
    return (raw % 1000) / 1000.0
