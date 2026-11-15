"""NumPy vector store with cosine-similarity search and JSON metadata.

This implementation avoids FAISS native DLLs, which can be blocked by
Windows Application Control policies on some managed machines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FaissVectorStore:
    """Compatibility-named vector store using exact NumPy cosine search.

    The class name is retained so the Streamlit app does not need changes.
    Vectors are normalized at build time; their dot product is cosine
    similarity. This is suitable for small and medium student-project corpora.
    """

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def build(
        self,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")
        if vectors.shape[0] != len(records):
            raise ValueError("Each embedding must have exactly one metadata record.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embeddings must not contain zero-length vectors.")

        self.vectors = vectors / norms
        self.records = records

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("The vector index has not been built.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Query vector has {query.shape[1]} dimensions; "
                f"index expects {self.vectors.shape[1]}."
            )

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("Query embedding must not be a zero vector.")

        query = query / query_norm
        scores = (self.vectors @ query.T).ravel()
        top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]

        return [
            {
                **self.records[int(index)],
                "similarity": float(scores[index]),
            }
            for index in top_indices
        ]

    def save(self, directory: str | Path) -> None:
        if self.vectors is None:
            raise RuntimeError("Cannot save an index before building it.")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "embeddings.npy", self.vectors)

        with (target / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False, indent=2)

    def load(self, directory: str | Path) -> None:
        target = Path(directory)
        vectors_path = target / "embeddings.npy"
        metadata_path = target / "metadata.json"

        if not vectors_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Saved embeddings or metadata were not found in the directory."
            )

        vectors = np.load(vectors_path).astype(np.float32)
        with metadata_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if vectors.ndim != 2 or vectors.shape[0] != len(records):
            raise ValueError(
                "Embedding matrix and metadata record counts do not match."
            )

        self.vectors = vectors
        self.records = records
