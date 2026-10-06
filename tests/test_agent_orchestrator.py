"""Tests de l'orchestration (`app.agent.orchestrator`), sans réseau ni LLM réel."""

import pytest

from app.agent import orchestrator
from app.agent.curator import CurationResult
from app.llm.ollama_client import ABSENCE_PHRASE

HITS = [{"text": "ctx", "source": "s", "score": 1.0}]


def test_is_absence_true():
    assert orchestrator.is_absence(f"Voici. {ABSENCE_PHRASE}") is True


def test_is_absence_false():
    assert orchestrator.is_absence("La résistance vaut 10 ohms.") is False


def test_reindex_rebuilds_indexes(monkeypatch):
    calls = {}

    class FakeClient:
        def collection_exists(self, name):
            return True

        def delete_collection(self, name):
            calls["deleted"] = name

    monkeypatch.setattr(orchestrator, "get_client", FakeClient)
    monkeypatch.setattr(orchestrator, "load_and_chunk", lambda d, cs, co: ["c1", "c2"])
    monkeypatch.setattr(orchestrator, "build_index", lambda chunks: calls.setdefault("index", chunks))
    monkeypatch.setattr(orchestrator, "build_bm25", lambda chunks: calls.setdefault("bm25", chunks))

    count = orchestrator.reindex()

    assert count == 2
    assert calls["deleted"] == orchestrator.QDRANT_COLLECTION
    assert calls["index"] == ["c1", "c2"]
    assert calls["bm25"] == ["c1", "c2"]


def test_answer_uses_local_when_present(monkeypatch):
    monkeypatch.setattr(orchestrator, "hybrid_search", lambda q, *a, **k: HITS)
    monkeypatch.setattr(orchestrator, "generate_answer", lambda q, ctx: "10 ohms")
    monkeypatch.setattr(
        orchestrator, "research", lambda q: (_ for _ in ()).throw(AssertionError("appelé à tort"))
    )

    result = orchestrator.answer("ron ?")

    assert result.answer == "10 ohms"
    assert result.used_web_research is False


def test_answer_web_fallback_when_absent(monkeypatch):
    monkeypatch.setattr(orchestrator, "hybrid_search", lambda q, *a, **k: HITS)
    answers = iter([ABSENCE_PHRASE, "réponse finale"])
    monkeypatch.setattr(orchestrator, "generate_answer", lambda q, ctx: next(answers))
    monkeypatch.setattr(orchestrator, "research", lambda q: "recherche brute")
    monkeypatch.setattr(
        orchestrator,
        "curate",
        lambda q, raw: CurationResult(content="# Fiche", approved=True, reason="ok"),
    )
    saved = {}
    monkeypatch.setattr(orchestrator, "save_document", lambda title, content, **k: saved.update(content=content))
    monkeypatch.setattr(orchestrator, "reindex", lambda: 3)

    result = orchestrator.answer("nouvelle question")

    assert result.used_web_research is True
    assert result.answer == "réponse finale"
    assert saved["content"] == "# Fiche"


def test_answer_web_fallback_not_approved_does_not_save(monkeypatch):
    monkeypatch.setattr(orchestrator, "hybrid_search", lambda q, *a, **k: HITS)
    monkeypatch.setattr(orchestrator, "generate_answer", lambda q, ctx: ABSENCE_PHRASE)
    monkeypatch.setattr(orchestrator, "research", lambda q: "brut")
    monkeypatch.setattr(
        orchestrator,
        "curate",
        lambda q, raw: CurationResult(content="x", approved=False, reason="ko"),
    )
    monkeypatch.setattr(orchestrator, "save_document", lambda *a, **k: pytest.fail("ne doit pas sauvegarder"))
    monkeypatch.setattr(orchestrator, "reindex", lambda: pytest.fail("ne doit pas réindexer"))

    result = orchestrator.answer("q")

    assert result.used_web_research is True
    assert result.answer == ABSENCE_PHRASE
