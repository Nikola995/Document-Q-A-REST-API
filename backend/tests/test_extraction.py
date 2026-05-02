def test_pdf_extraction_returns_text(sample_pdf):
    from app.services.extraction import extract

    text = extract(sample_pdf, "application/pdf")
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_image_extraction_returns_text(sample_image):
    from app.services.extraction import extract

    text = extract(sample_image, "image/png")
    assert isinstance(text, str)
    assert len(text.strip()) > 0
