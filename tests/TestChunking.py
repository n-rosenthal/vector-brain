from src.extractors import split_into_chunks


def test_short_text_stays_single_chunk():
    text = "Um texto curto qualquer."
    assert split_into_chunks(text) == [text]


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []
    assert split_into_chunks("   ") == []


def test_long_text_splits_into_multiple_chunks():
    # gera texto bem maior que CHUNK_TARGET_CHARS (1500)
    paragraph = "Esta é uma frase de teste para preencher espaço. " * 10
    text = "\n\n".join([paragraph] * 10)
    chunks = split_into_chunks(text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) > 0


def test_chunks_have_overlap_when_split_mid_paragraph():
    # texto sem quebras de parágrafo força corte no meio
    text = "a" * 4000
    chunks = split_into_chunks(text)
    assert len(chunks) >= 2
    # o fim de um chunk deve reaparecer no início do próximo (overlap)
    assert chunks[0][-50:] in chunks[1]
