"""Divide texto de um node em chunks, se necessário."""
from ..config import CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS


def split_into_chunks(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_TARGET_CHARS:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_TARGET_CHARS, n)
        # tenta cortar em quebra de parágrafo mais próxima, pra não partir frases
        if end < n:
            break_point = text.rfind("\n\n", start, end)
            if break_point > start:
                end = break_point
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP_CHARS if end - CHUNK_OVERLAP_CHARS > start else end
    return [c for c in chunks if c]
