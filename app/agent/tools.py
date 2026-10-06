"""Outils mis à disposition de l'agent de recherche web."""

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

DEFAULT_MAX_RESULTS = 5
PDF_FILTER = "filetype:pdf"

REQUEST_TIMEOUT = 30
USER_AGENT = "rag-project/0.1"
# Balises HTML sans contenu utile pour le texte.
HTML_NOISE_TAGS = ["script", "style", "nav", "header", "footer"]


def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    pdf_only: bool = True,
) -> list[dict]:
    """Recherche des documents sur le web via DuckDuckGo.

    Retourne une liste de résultats `{"title", "url", "snippet"}`. Si `pdf_only`
    est vrai (par défaut), la recherche est filtrée sur les PDF. En l'absence de
    résultat, retourne une liste vide.
    """
    search_query = f"{query} {PDF_FILTER}" if pdf_only else query

    try:
        raw_results = DDGS().text(search_query, max_results=max_results)
    except DDGSException:
        return []

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("href") or item.get("url", ""),
            "snippet": item.get("body", ""),
        }
        for item in raw_results
    ]


def _extract_pdf(content: bytes) -> str:
    import pymupdf  # import paresseux : seulement quand on a un PDF

    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _extract_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(HTML_NOISE_TAGS):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def read_url(url: str) -> dict:
    """Télécharge un document et en extrait le texte.

    Les PDF passent par PyMuPDF, le reste par BeautifulSoup. Le type est déduit
    du `Content-Type` puis, à défaut, de l'extension `.pdf` de l'URL.

    Retourne `{"url", "content_type", "text"}`.
    """
    response = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    is_pdf = "pdf" in content_type.lower() or url.lower().endswith(".pdf")
    text = _extract_pdf(response.content) if is_pdf else _extract_html(response.content)

    return {"url": url, "content_type": content_type, "text": text}
