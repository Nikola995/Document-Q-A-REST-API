def test_document_key_consistency():
    from app.services.cache import _document_key

    # same inputs must always produce the same key
    assert _document_key("session-1", "some text") == _document_key(
        "session-1", "some text"
    )


def test_document_key_session_scoped():
    from app.services.cache import _document_key

    # same text in different sessions must produce different keys
    assert _document_key("session-1", "some text") != _document_key(
        "session-2", "some text"
    )


def test_embedding_key_consistency():
    from app.services.cache import _embedding_key

    assert _embedding_key("what is this?") == _embedding_key("what is this?")
