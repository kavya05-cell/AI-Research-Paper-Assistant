"""Grounded question answering over retrieved research-paper passages.

Uses Google's official google-genai SDK directly to avoid loading optional
LangChain integrations and their native xxhash dependency chain.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def answer_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate an answer using only retrieved passages and explicit citations."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add your key to the local .env file."
        )
    if not question.strip():
        raise ValueError("Please enter a question.")
    if not retrieved_chunks:
        return (
            "I could not find relevant passages in the uploaded papers, "
            "so I cannot answer this from the available sources."
        )

    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown document")
        page = metadata.get("page", "Unknown page")
        context_blocks.append(
            f"[Passage {index} | Source: {source} | Page: {page}]\n"
            f"{chunk.get('text', '')}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are an academic research-paper assistant.

Answer the user's question using ONLY the source passages supplied below.
Do not use outside knowledge, assumptions, or facts that are not supported
by these passages.

Rules:
1. If the passages do not contain enough information, say clearly that the
   uploaded papers do not provide enough information to answer.
2. Do not invent methods, results, datasets, metrics, or conclusions.
3. Cite every substantive claim with the exact source filename and page
   number, using this format: [Source: filename.pdf, Page: 3].
4. If multiple passages support the answer, cite each relevant passage.
5. Distinguish what the paper explicitly states from any cautious inference.
6. Write a clear, concise answer. Use bullet points when useful.

USER QUESTION:
{question}

SOURCE PASSAGES:
{context}

GROUNDED ANSWER:"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    answer = response.text
    if not answer:
        raise RuntimeError("Gemini returned an empty response.")
    return answer.strip()
