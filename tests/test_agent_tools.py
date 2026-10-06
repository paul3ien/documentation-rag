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


class _FakeResponse:
    """Fausse réponse HTTP minimale, suffisante pour `read_url`."""

    def __init__(self, content: bytes, content_type: str):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


def _pdf_bytes(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _serve(monkeypatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: response)


def test_read_url_extracts_pdf(monkeypatch):
    _serve(monkeypatch, _FakeResponse(_pdf_bytes("contenu du pdf"), "application/pdf"))

    result = tools.read_url("http://example.com/doc.pdf")

    assert result["url"] == "http://example.com/doc.pdf"
    assert "contenu du pdf" in result["text"]


def test_read_url_extracts_html(monkeypatch):
    html = (
        b"<html><head><style>body{color:red}</style></head>"
        b"<body><h1>Titre</h1><script>ignore()</script>"
        b"<p>Un paragraphe utile.</p></body></html>"
    )
    _serve(monkeypatch, _FakeResponse(html, "text/html; charset=utf-8"))

    result = tools.read_url("http://example.com/page")

    assert "Titre" in result["text"]
    assert "Un paragraphe utile." in result["text"]
    assert "ignore" not in result["text"]
    assert "color:red" not in result["text"]


def test_read_url_detects_pdf_by_extension(monkeypatch):
    _serve(
        monkeypatch,
        _FakeResponse(_pdf_bytes("depuis extension"), "application/octet-stream"),
    )

    result = tools.read_url("http://example.com/manual.PDF")

    assert "depuis extension" in result["text"]
