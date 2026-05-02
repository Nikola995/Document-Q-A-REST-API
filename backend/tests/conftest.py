import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "invoices"


@pytest.fixture
def sample_pdf() -> Path:
    return DATA_DIR / "invoice-pdfs" / "invoice_Adam Bellavance_21617.pdf"


@pytest.fixture
def sample_image() -> Path:
    return DATA_DIR / "invoice-images" / "batch1-0001.jpg"
