def extract(file_path: str, mime_type: str) -> list[tuple[int, str]]:
    """
    Text extraction service.

    Supports:
      - PDFs via PyMuPDF (fitz) — handles text-layer PDFs directly
      - Image files via EasyOCR (or Tesseract as fallback)

    Returns: list[tuple[int, str]]  →  [(page_number, page_text), ...]
    """
    if mime_type == "application/pdf":
        return _extract_pdf(file_path)
    else:
        return _extract_image(file_path)


def _extract_pdf(file_path: str) -> list[tuple[int, str]]:
    # TODO: implement with PyMuPDF
    raise NotImplementedError("PDF extraction not yet implemented")


def _extract_image(file_path: str) -> list[tuple[int, str]]:
    # TODO: implement with EasyOCR
    raise NotImplementedError("Image OCR not yet implemented")
