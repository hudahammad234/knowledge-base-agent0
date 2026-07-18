"""
chunker.py
Splits loaded documents into meaningful, overlapping chunks and attaches
metadata (document name, page number, chunk number) to each chunk.

Chunking strategy
------------------
Text is first split on paragraph boundaries (blank lines), then paragraphs
are greedily packed into chunks of ~CHUNK_SIZE words with CHUNK_OVERLAP words
of overlap between consecutive chunks so that context is not lost at chunk
boundaries. This keeps chunks "meaningful" (paragraph-aware) rather than
cutting text at a fixed character offset.

Member responsible: Member 1 - Knowledge Base
"""

import re
from dataclasses import dataclass
from typing import List

CHUNK_SIZE = 220       # target words per chunk
CHUNK_OVERLAP = 40      # words of overlap carried to the next chunk


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: int
    chunk_number: int
    text: str


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        # fall back to line splitting for text with no blank lines (e.g. CSV rows)
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    return paragraphs


def _pack_paragraphs(paragraphs: List[str]) -> List[str]:
    """Greedily pack paragraphs into ~CHUNK_SIZE-word windows with overlap."""
    chunks = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > CHUNK_SIZE and current_words:
            chunks.append(" ".join(current_words))
            # keep the tail as overlap for the next chunk
            overlap_words = current_words[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            current_words = overlap_words + para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def chunk_document(loaded_document) -> List[Chunk]:
    """Produce Chunk objects for every page of a LoadedDocument."""
    all_chunks: List[Chunk] = []
    chunk_counter = 0

    for raw_page in loaded_document.pages:
        paragraphs = _split_paragraphs(raw_page.text)
        if not paragraphs:
            continue
        text_chunks = _pack_paragraphs(paragraphs)
        for text in text_chunks:
            chunk_counter += 1
            all_chunks.append(
                Chunk(
                    chunk_id=f"{loaded_document.document_name}::{chunk_counter}",
                    document_name=loaded_document.document_name,
                    page_number=raw_page.page_number,
                    chunk_number=chunk_counter,
                    text=text,
                )
            )
    return all_chunks
