"""Structured, evidence-grounded comparison of multiple research papers."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DIMENSIONS = [
    "Research problem",
    "Objective",
    "Methodology/architecture",
    "Algorithms",
    "Datasets",
    "Evaluation metrics",
    "Key findings",
    "Limitations",
]


def compare_papers(
    papers: dict[str, list[dict[str, Any]]],
    comparison_focus: str = "Compare the papers across their research problem, objectives, methodology, model or algorithm, datasets, evaluation metrics, key findings, and limitations.",
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    """Return structured comparison data; UI code renders the table."""
    if len(papers) < 2:
        raise ValueError("Select at least two different research papers to compare.")
    if not comparison_focus.strip():
        raise ValueError("Enter what you want to compare.")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add your key to the local .env file.")

    paper_names = list(papers.keys())
    contexts = []
    for paper_name, chunks in papers.items():
        if not chunks:
            continue
        passages = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            page = metadata.get("page", "Unknown")
            text = str(chunk.get("text", ""))[:2200]
            passages.append(f"[Page {page} | Passage {index}]\n{text}")
        contexts.append(f"===== PAPER: {paper_name} =====\n" + "\n\n".join(passages))

    if len(contexts) < 2:
        raise ValueError("At least two selected papers need retrievable text passages.")

    prompt = f"""You are an academic research analyst.
Compare the selected research papers using ONLY the supplied excerpts.
Do not use outside knowledge or transfer facts between papers.
Comparison focus: {comparison_focus}

Return ONLY a valid JSON object with this exact structure:
{{
  "overview": "A concise overview naming each paper and its topic.",
  "comparison": [
    {{
      "dimension": "Research problem",
      "findings": {{
        "EXACT_FILENAME_1": "Concise evidence-based statement with citation [Source: filename, Page: N].",
        "EXACT_FILENAME_2": "Concise evidence-based statement with citation [Source: filename, Page: N]."
      }}
    }}
  ],
  "similarities": ["Evidence-based similarity or difference with citations."],
  "evidence_gaps": ["Information not established by supplied excerpts."]
}}

Include these dimensions in comparison: {json.dumps(DIMENSIONS)}.
The findings object MUST contain exactly one key for every filename in this list:
{json.dumps(paper_names)}.
Use "Not stated in supplied excerpts" if evidence is missing.
Cite paper-specific claims with the exact filename and page number shown in the excerpts.
Do not invent facts, citations, page numbers, results, or limitations.
Keep overview to 2-3 sentences, each comparison cell concise (prefer under 45 words),
similarities to at most 3 items, and evidence_gaps to at most 3 items.
Do not output Markdown or code fences.

SOURCE EXCERPTS:
{"\n\n".join(contexts)}
"""

    client = genai.Client(api_key=api_key)
    response = None
    retry_delays = (2, 4, 8)
    for attempt in range(len(retry_delays) + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2200,
                    response_mime_type="application/json",
                ),
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

    raw = response.text.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.IGNORECASE)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid structured comparison data. Please retry.") from exc

    if not isinstance(result, dict) or not isinstance(result.get("comparison"), list):
        raise RuntimeError("Gemini's comparison response is missing the comparison rows.")

    # Normalize rows to a predictable shape and prevent omitted paper columns.
    normalized_rows = []
    for dimension in DIMENSIONS:
        match = next(
            (row for row in result["comparison"]
             if isinstance(row, dict) and row.get("dimension", "").lower() == dimension.lower()),
            None,
        )
        findings = match.get("findings", {}) if match else {}
        normalized_rows.append({
            "dimension": dimension,
            "findings": {
                name: str(findings.get(name, "Not stated in supplied excerpts"))
                for name in paper_names
            },
        })

    result["comparison"] = normalized_rows
    result["overview"] = str(result.get("overview", ""))
    result["similarities"] = [str(x) for x in result.get("similarities", [])][:3]
    result["evidence_gaps"] = [str(x) for x in result.get("evidence_gaps", [])][:3]
    result["paper_names"] = paper_names
    return result
