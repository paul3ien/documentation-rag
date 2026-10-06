"""Tests du curateur (`app.agent.curator`), avec un faux LLM (aucun réseau)."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent import curator


def test_rewrite_returns_llm_output():
    llm = FakeListChatModel(responses=["# Fiche\n\nContenu important."])

    content = curator.rewrite("Question ?", "brut", llm=llm)

    assert "Fiche" in content


def test_parse_review_ok():
    assert curator.parse_review("VERDICT: OK\nRAISON: fidèle") == (True, "fidèle")


def test_parse_review_ko():
    approved, reason = curator.parse_review("VERDICT: KO\nRAISON: hors sujet")

    assert approved is False
    assert reason == "hors sujet"


def test_parse_review_without_verdict():
    assert curator.parse_review("texte inattendu") == (False, "")


def test_review_uses_llm_verdict():
    llm = FakeListChatModel(responses=["VERDICT: OK\nRAISON: clair et exact"])

    approved, reason = curator.review("Q", "fiche", llm=llm)

    assert approved is True
    assert reason == "clair et exact"


def test_curate_rewrites_then_reviews():
    llm = FakeListChatModel(responses=["# Fiche", "VERDICT: OK\nRAISON: bon"])

    result = curator.curate("Q", "brut", llm=llm)

    assert result.content == "# Fiche"
    assert result.approved is True
    assert result.reason == "bon"
