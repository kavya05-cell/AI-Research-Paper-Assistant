"""Fetch publicly accessible research-paper PDFs from direct and arXiv URLs."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import requests

from modules.pdf_processor import PDFProcessingError, extract_pdf_pages


MAX_PDF_BYTES = 30 * 1024 * 1024
REQUEST_TIMEOUT = (10, 30)
USER_AGENT = "AIResearchPaperAssistant/0.1 (academic PDF fetcher)"


class PaperFetchError(Exception):
    """Raised when a research-paper URL cannot be safely fetched as a PDF."""


def _normalize_arxiv_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host not in {"arxiv.org", "www.arxiv.org"}:
        return url

    match = re.match(r"^/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?/?$", parsed.path)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"

    old_style = re.match(r"^/(?:abs|pdf)/([a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v[0-9]+)?/?$", parsed.path)
    if old_style:
        return f"https://arxiv.org/pdf/{old_style.group(1)}.pdf"

    return url


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise PaperFetchError("Only HTTP and HTTPS paper links are supported.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise PaperFetchError("The supplied URL is invalid.")

    host = parsed.hostname.lower()
    if host in {"localhost"} or host.endswith(".localhost") or host.endswith(".local"):
        raise PaperFetchError("Local and private network URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise PaperFetchError("The paper link's hostname could not be resolved.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PaperFetchError("Links to private or non-public network addresses are blocked.")


def fetch_research_paper(url: str) -> tuple[bytes, str, str]:
    """Download a public PDF and return (bytes, final_url, filename)."""
    cleaned_url = _normalize_arxiv_url(url.strip())
    if not cleaned_url:
        raise PaperFetchError("Enter a research-paper URL.")

    current_url = cleaned_url
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}

    with requests.Session() as session:
        for _ in range(6):
            _validate_public_http_url(current_url)
            try:
                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise PaperFetchError(f"Could not reach the paper URL: {exc}") from exc

            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise PaperFetchError("The paper host returned an invalid redirect.")
                current_url = urljoin(current_url, location)
                continue

            if response.status_code != 200:
                status = response.status_code
                response.close()
                if status in {401, 403, 404, 429}:
                    raise PaperFetchError(
                        f"The paper host returned HTTP {status}. The paper may require "
                        "sign-in, subscription access, or a different public link."
                    )
                raise PaperFetchError(f"The paper host returned HTTP {status}.")

            content_type = response.headers.get("Content-Type", "").lower()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_PDF_BYTES:
                response.close()
                raise PaperFetchError("The PDF exceeds the 30 MB download limit.")

            data = bytearray()
            try:
                for block in response.iter_content(chunk_size=64 * 1024):
                    if not block:
                        continue
                    data.extend(block)
                    if len(data) > MAX_PDF_BYTES:
                        raise PaperFetchError("The PDF exceeds the 30 MB download limit.")
            finally:
                response.close()

            pdf_bytes = bytes(data)
            if not pdf_bytes.startswith(b"%PDF-"):
                if "text/html" in content_type or b"<html" in pdf_bytes[:1024].lower():
                    raise PaperFetchError(
                        "This link returned a web page, not a PDF. Try a direct PDF "
                        "link or an arXiv abstract/PDF link."
                    )
                raise PaperFetchError("The downloaded file is not a valid PDF.")

            parsed = urlparse(current_url)
            filename = parsed.path.rsplit("/", 1)[-1] or "research_paper.pdf"
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            return pdf_bytes, current_url, filename

    raise PaperFetchError("The paper URL redirected too many times.")


def extract_pages_from_url(url: str) -> tuple[list[dict], str]:
    """Fetch a PDF URL, extract its pages, and return pages plus final URL."""
    pdf_bytes, final_url, filename = fetch_research_paper(url)
    try:
        pages = extract_pdf_pages(pdf_bytes, filename)
    except PDFProcessingError as exc:
        raise PaperFetchError(str(exc)) from exc

    for page in pages:
        page["metadata"]["source_url"] = final_url
        page["metadata"]["import_method"] = "URL"
    return pages, final_url
