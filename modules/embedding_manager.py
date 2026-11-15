"""Sentence Transformer embedding utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and cache an embedding model.

    The first call downloads model weights if they are not already cached.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Convert text strings into L2-normalized float32 embedding vectors."""
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty text list.")

    model = load_embedding_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)
