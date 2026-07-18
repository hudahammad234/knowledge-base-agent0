"""
metadata.py
Extracts document-level metadata used for filtering, citations and auditing.

Member responsible: Member 1 - Knowledge Base
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class DocumentMetadata:
    document_name: str
    file_path: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    num_pages: int
    indexed_at: str
    summary: str = ""  # filled in by the summarization bonus feature

    def to_dict(self) -> Dict:
        return asdict(self)


def extract_metadata(loaded_document, summary: str = "") -> DocumentMetadata:
    """Build a DocumentMetadata record from a LoadedDocument (see loader.py)."""
    size = os.path.getsize(loaded_document.file_path) if os.path.exists(loaded_document.file_path) else 0
    return DocumentMetadata(
        document_name=loaded_document.document_name,
        file_path=loaded_document.file_path,
        file_type=loaded_document.file_type,
        file_hash=loaded_document.file_hash,
        file_size_bytes=size,
        num_pages=len(loaded_document.pages),
        indexed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        summary=summary,
    )
