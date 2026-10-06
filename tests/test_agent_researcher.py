"""Tests de l'agent chercheur (`app.agent.researcher`), sans réseau ni LLM réel."""

import pytest
from langchain.agents import AgentExecutor
from langchain_deepseek import ChatDeepSeek

from app.agent import researcher


def test_build_llm_uses_deepseek(monkeypatch):
    monkeypatch.setattr(researcher, "DEEPSEEK_API_KEY", "test-key")

    llm = researcher.build_llm()

    assert isinstance(llm, ChatDeepSeek)
    assert llm.model_name == researcher.DEEPSEEK_MODEL


def test_build_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(researcher, "DEEPSEEK_API_KEY", "")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        researcher.build_llm()


def test_web_search_tool_formats_results(monkeypatch):
    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda query: [{"title": "T", "url": "http://x.pdf", "snippet": "S"}],
    )

    output = researcher.web_search.invoke("ts5a3157")

    assert "T" in output
    assert "http://x.pdf" in output


def test_web_search_tool_without_results(monkeypatch):
    monkeypatch.setattr(researcher, "search_web", lambda query: [])

    assert researcher.web_search.invoke("ts5a3157") == "Aucun résultat."


def test_read_document_tool_returns_text(monkeypatch):
    monkeypatch.setattr(
        researcher,
        "read_url",
        lambda url: {"url": url, "content_type": "application/pdf", "text": "contenu"},
    )

    assert researcher.read_document.invoke("http://x.pdf") == "contenu"


def test_build_agent_constructs(monkeypatch):
    monkeypatch.setattr(researcher, "DEEPSEEK_API_KEY", "test-key")

    agent = researcher.build_agent()

    assert isinstance(agent, AgentExecutor)
