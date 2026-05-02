def test_session_exists_returns_false_for_unknown():
    from app.services.rag import session_exists

    assert session_exists("nonexistent-session-id") is False


def test_chunk_returns_non_empty(sample_pdf):
    from app.services.extraction import extract
    from app.services.rag import _chunk

    text = extract(sample_pdf, "application/pdf")
    doc = _chunk(text)
    assert len(doc.chunks) > 0
    assert all(chunk.text.strip() for chunk in doc.chunks)
