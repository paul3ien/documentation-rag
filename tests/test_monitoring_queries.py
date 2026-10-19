"""Tests du journal de requêtes (`app.monitoring.queries`)."""

import json

from app.monitoring import queries


def test_log_and_load_queries(monkeypatch, tmp_path):
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setattr(queries, "LOG_DIR", tmp_path)
    monkeypatch.setattr(queries, "QUERY_LOG_PATH", log_path)

    queries.log_query(
        "Quelle est la ron ?",
        [{"text": "ctx", "source": "s", "score": 0.4}, {"text": "c2", "source": "s2", "score": 0.7}],
        used_web_research=False,
    )
    queries.log_query("Question absente", [], used_web_research=True)

    loaded = queries.load_queries()

    assert len(loaded) == 2
    assert loaded[0]["question"] == "Quelle est la ron ?"
    assert loaded[0]["top_score"] == 0.7
    assert loaded[0]["n_sources"] == 2
    assert loaded[0]["used_web_research"] is False
    assert loaded[1]["top_score"] == 0.0
    assert loaded[1]["used_web_research"] is True
    assert "timestamp" in loaded[1]


def test_load_queries_returns_empty_when_missing(tmp_path):
    assert queries.load_queries(tmp_path / "missing.jsonl") == []


def test_log_writes_valid_jsonl(monkeypatch, tmp_path):
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setattr(queries, "LOG_DIR", tmp_path)
    monkeypatch.setattr(queries, "QUERY_LOG_PATH", log_path)

    queries.log_query("q", [], used_web_research=False)

    line = log_path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["question"] == "q"
