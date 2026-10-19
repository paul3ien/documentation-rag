"""Tests des snapshots du corpus (`app.monitoring.corpus`)."""

from app.monitoring import corpus


def test_count_documents(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "ignore.png").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("x", encoding="utf-8")

    assert corpus.count_documents(tmp_path) == 4


def test_log_and_load_snapshots(tmp_path):
    log_path = tmp_path / "corpus.jsonl"

    corpus.log_corpus_snapshot(2, 11, path=log_path)
    corpus.log_corpus_snapshot(3, 50, path=log_path)

    snapshots = corpus.load_corpus_snapshots(log_path)

    assert [s["n_docs"] for s in snapshots] == [2, 3]
    assert [s["n_chunks"] for s in snapshots] == [11, 50]
    assert "timestamp" in snapshots[0]


def test_load_snapshots_returns_empty_when_missing(tmp_path):
    assert corpus.load_corpus_snapshots(tmp_path / "missing.jsonl") == []
