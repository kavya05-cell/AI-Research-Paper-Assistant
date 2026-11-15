"""PDF text extraction with page-level source metadata."""

from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed or contains no extractable text."""


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Extract text from a PDF and return one record per page.

    Page numbers are 1-based so they can be used directly in source citations.
    """
    if not pdf_bytes:
        raise PDFProcessingError(f"{filename}: the uploaded file is empty.")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(
            f"{filename}: this file is not a valid or readable PDF."
        ) from exc

    try:
        if document.is_encrypted:
            raise PDFProcessingError(
                f"{filename}: password-protected PDFs are not supported."
            )

        extracted_pages: list[dict[str, Any]] = []

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                extracted_pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_index + 1,
                            "total_pages": len(document),
                        },
                    }
                )

        if not extracted_pages:
            raise PDFProcessingError(
                f"{filename}: no selectable text was found. "
                "The PDF may contain scanned images; OCR support is not "
                "implemented yet."
            )

        return extracted_pages
    finally:
        document.close()
