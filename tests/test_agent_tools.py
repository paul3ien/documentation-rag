"""Tests de l'outil de recherche web (`app.agent.tools.search_web`).

On remplace le client DuckDuckGo par un faux objet qui enregistre les appels et
renvoie une réponse fixée : aucun accès réseau n'est nécessaire.
"""

import pytest
from ddgs.exceptions import DDGSException

from app.agent import tools


class _FakeDDGS:
    """Faux `DDGS` : évaluable comme `DDGS()` et qui enregistre les appels."""

    def __init__(self):
        self.calls: list[dict] = []
        self.results: list[dict] = []
        self.error: Exception | None = None

    def __call__(self):
        return self

    def text(self, query: str, max_results: int) -> list[dict]:
        self.calls.append({"query": query, "max_results": max_results})
        if self.error is not None:
            raise self.error
        return self.results


@pytest.fixture
def ddgs(monkeypatch) -> _FakeDDGS:
    fake = _FakeDDGS()
    monkeypatch.setattr(tools, "DDGS", fake)
    return fake


def test_normalizes_results(ddgs):
    ddgs.results = [
        {"title": "T", "href": "http://x.pdf", "body": "B"},
        {"title": "U", "url": "http://y.html", "body": "C"},
    ]

    assert tools.search_web("ts5a3157") == [
        {"title": "T", "url": "http://x.pdf", "snippet": "B"},
        {"title": "U", "url": "http://y.html", "snippet": "C"},
    ]


def test_returns_empty_list_when_no_results(ddgs):
    ddgs.error = DDGSException("boom")

    assert tools.search_web("ts5a3157") == []


def test_filters_pdf_by_default(ddgs):
    tools.search_web("ts5a3157")

    assert ddgs.calls == [{"query": "ts5a3157 filetype:pdf", "max_results": 5}]


def test_can_disable_pdf_filter(ddgs):
    tools.search_web("ts5a3157", max_results=3, pdf_only=False)

    assert ddgs.calls == [{"query": "ts5a3157", "max_results": 3}]
