import pymupdf4llm


def extract(file_path: str, mime_type: str) -> str:
    """
    Text extraction service.

    Supports:
      - PDFs via pymupdf4llm — handles text-layer PDFs directly
      - Image files via EasyOCR

    Returns: str - The entire extracted text from the file
    """
    if mime_type == "application/pdf":
        return _extract_pdf(file_path)
    else:
        return _extract_image(file_path)


def _extract_pdf(file_path: str) -> str:
    doc = pymupdf4llm.to_markdown(file_path)
    return doc


def _extract_image(file_path: str) -> str:
    # TODO: implement with EasyOCR
    raise NotImplementedError("Image OCR not yet implemented")
