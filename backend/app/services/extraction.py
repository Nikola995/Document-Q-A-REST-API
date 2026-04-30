import pymupdf4llm
import easyocr
from pathlib import Path

def extract(file_path: Path, mime_type: str) -> str:
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


def _extract_pdf(file_path: Path) -> str:
    doc = pymupdf4llm.to_markdown(file_path)
    return doc


def _extract_image(file_path: Path) -> str:
    reader = easyocr.Reader(["en"])
    results = reader.readtext(str(file_path), detail=0, paragraph=True)
    # Create a single str from the list of str in results
    return " ".join(results)
