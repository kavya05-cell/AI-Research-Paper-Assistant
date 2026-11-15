"""FAISS vector index with cosine-similarity search and JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissVectorStore:
    """In-memory FAISS index that can be saved and loaded from disk."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.records: list[dict[str, Any]] = []

    def build(
        self,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
    ) -> None:
        """Build an inner-product index over normalized embeddings."""
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")
        if vectors.shape[0] != len(records):
            raise ValueError("Each embedding must have exactly one metadata record.")

        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.records = records

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top matching chunks, ranked by cosine similarity."""
        if self.index is None:
            raise RuntimeError("The FAISS index has not been built.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, record_index in zip(scores[0], indices[0]):
            if record_index < 0:
                continue
            results.append(
                {
                    **self.records[int(record_index)],
                    "similarity": float(score),
                }
            )
        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and its corresponding records."""
        if self.index is None:
            raise RuntimeError("Cannot save an index before building it.")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "research_papers.faiss"))

        with (target / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False, indent=2)

    def load(self, directory: str | Path) -> None:
        """Load a previously saved index and metadata records."""
        target = Path(directory)
        index_path = target / "research_papers.faiss"
        metadata_path = target / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Saved FAISS index or metadata was not found in the directory."
            )

        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as file:
            self.records = json.load(file)

        if self.index.ntotal != len(self.records):
            self.index = None
            self.records = []
            raise ValueError("FAISS index and metadata record counts do not match.")
