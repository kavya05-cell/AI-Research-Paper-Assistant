"""Evidence-grounded comparison of multiple research papers using Gemini."""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def compare_papers(
    papers: dict[str, list[dict[str, Any]]],
    comparison_focus: str = "Compare the papers across their research problem, objectives, methodology, model or algorithm, datasets, evaluation metrics, key findings, and limitations.",
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Compare selected papers using only their retrieved source passages."""
    if len(papers) < 2:
        raise ValueError("Select at least two different research papers to compare.")
    if not comparison_focus.strip():
        raise ValueError("Enter what you want to compare.")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add your key to the local .env file.")

    paper_contexts = []
    for paper_name, chunks in papers.items():
        if not chunks:
            continue
        passages = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            page = metadata.get("page", "Unknown")
            passages.append(
                f"[{paper_name} | Page {page} | Passage {index}]\n"
                f"{chunk.get('text', '')}"
            )
        paper_contexts.append(
            f"===== PAPER: {paper_name} =====\n" + "\n\n".join(passages)
        )

    if len(paper_contexts) < 2:
        raise ValueError("At least two selected papers need retrievable text passages.")

    combined_context = "\n\n".join(paper_contexts)

    prompt = f"""You are an academic research analyst comparing scientific papers.

Use ONLY the supplied excerpts. Treat each paper independently and do not
transfer a method, dataset, result, or limitation from one paper to another.
Do not fill gaps with outside knowledge.

Comparison request:
{comparison_focus}

Instructions:
1. Begin with a short overview of what each paper investigates.
2. Provide a Markdown comparison table. Use rows for research problem,
   objective, methodology/architecture, algorithms, datasets, evaluation
   metrics, key findings, and limitations. Use "Not stated in supplied
   excerpts" where evidence is absent.
3. Follow the table with an analytical comparison describing important
   similarities and differences.
4. Cite every paper-specific factual claim using the exact paper filename
   and page number in this format: [Source: filename.pdf, Page: 4].
5. Never invent citations, page numbers, results, or missing details.
6. Clearly distinguish evidence from cautious interpretation.
7. End with a short section titled "Evidence gaps" listing information that
   could not be established from the supplied excerpts.
8. Do not claim that a paper is better unless the supplied evidence supports
   a like-for-like comparison.

SOURCE EXCERPTS:
{combined_context}

COMPARATIVE ANALYSIS:"""

    client = genai.Client(api_key=api_key)
    response = None
    retry_delays = (2, 4, 8)
    for attempt in range(len(retry_delays) + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            break
        except Exception as exc:
            error_text = str(exc).lower()
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            transient = status_code in (429, 500, 502, 503, 504) or any(
                marker in error_text
                for marker in ("503 unavailable", "service unavailable", "high demand", "resource_exhausted", "429")
            )
            if not transient or attempt == len(retry_delays):
                if transient:
                    raise RuntimeError(
                        "Gemini is temporarily overloaded or rate-limited. "
                        "Wait a minute and retry the comparison."
                    ) from exc
                raise
            time.sleep(retry_delays[attempt])

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty comparison.")
    return response.text.strip()
