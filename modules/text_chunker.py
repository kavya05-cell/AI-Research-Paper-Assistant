"""Page-aware token chunking for extracted research-paper text."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split extracted page records into overlapping, token-sized chunks.

    Each page is split independently so every chunk retains an accurate
    source filename and 1-based PDF page number for later citations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for page_record in pages:
        page_text = page_record.get("text", "").strip()
        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)
        for page_chunk_index, chunk_text in enumerate(page_chunks, start=1):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **page_record.get("metadata", {}),
                        "page_chunk": page_chunk_index,
                        "total_page_chunks": len(page_chunks),
                    },
                }
            )

    return chunks
