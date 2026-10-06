"""Outils mis à disposition de l'agent de recherche web."""

from ddgs import DDGS
from ddgs.exceptions import DDGSException

DEFAULT_MAX_RESULTS = 5
PDF_FILTER = "filetype:pdf"


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
