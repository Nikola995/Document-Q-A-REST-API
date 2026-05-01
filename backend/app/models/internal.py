from dataclasses import dataclass, field


@dataclass
class ChunkMetadata:
    document_id: str
    text: str


@dataclass
class DocumentMetadata:
    document_id: str
    filename: str


@dataclass
class FileProcessResult:
    document_id: str
    filename: str
    text: str
    already_exists: bool
