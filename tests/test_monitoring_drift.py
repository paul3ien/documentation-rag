"""Tests des calculs de drift (`app.monitoring.drift`), données synthétiques."""

import numpy as np

from app.monitoring import drift


def test_retrieval_drift_detects_shift():
    queries = [{"top_score": 0.1}] * 60 + [{"top_score": 0.9}] * 20

    result = drift.retrieval_drift(queries, recent_n=20)

    assert result.enough_data is True
    assert result.drifted is True
    assert result.detail["recent_mean"] > result.detail["reference_mean"]


def test_retrieval_drift_without_enough_data():
    result = drift.retrieval_drift([{"top_score": 0.5}] * 10, recent_n=50)

    assert result.enough_data is False
    assert result.drifted is False


def test_query_drift_detects_topic_shift():
    queries = [{"question": "a"}] * 60 + [{"question": "b"}] * 20

    def fake_embed(texts):
        return [
            np.array([1.0, 0.0]) if text == "a" else np.array([0.0, 1.0])
            for text in texts
        ]

    result = drift.query_drift(queries, recent_n=20, embed=fake_embed)

    assert result.enough_data is True
    assert result.drifted is True
    assert result.detail["cosine_distance"] > 0.9


def test_query_drift_with_same_topic():
    queries = [{"question": "a"}] * 80

    def fake_embed(texts):
        return [np.array([1.0, 0.0]) for _ in texts]

    result = drift.query_drift(queries, recent_n=20, embed=fake_embed)

    assert result.drifted is False


def test_corpus_drift_detects_growth():
    snapshots = [
        {"n_docs": 2, "n_chunks": 11},
        {"n_docs": 3, "n_chunks": 50},
    ]

    result = drift.corpus_drift(snapshots)

    assert result.drifted is True
    assert result.detail["doc_delta"] == 1
    assert result.detail["chunk_delta"] == 39


def test_corpus_drift_without_enough_snapshots():
    result = drift.corpus_drift([{"n_docs": 2, "n_chunks": 11}])

    assert result.enough_data is False


def test_fallback_drift_detects_rate_change():
    queries = [{"used_web_research": False}] * 60 + [{"used_web_research": True}] * 20

    result = drift.fallback_drift(queries, recent_n=20)

    assert result.drifted is True
    assert result.detail["recent_rate"] == 1.0
    assert result.detail["reference_rate"] == 0.0


def test_performance_drift_detects_drop():
    runs = [
        {"faithfulness": 0.9},
        {"faithfulness": 0.8},
        {"faithfulness": 0.5},
    ]

    result = drift.performance_drift(runs)

    assert result.drifted is True
    assert result.detail["delta"] == -0.4


def test_performance_drift_ignores_nan():
    runs = [{"faithfulness": float("nan")}, {"faithfulness": 0.9}, {"faithfulness": 0.8}]

    result = drift.performance_drift(runs)

    assert result.enough_data is True
    assert result.detail["values"] == [0.9, 0.8]


def test_load_eval_history_from_mlflow(tmp_path):
    from mlflow.tracking import MlflowClient

    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("rag-eval")
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "faithfulness", 0.9)
    client.log_metric(run.info.run_id, "context_precision", 0.8)
    client.log_metric(run.info.run_id, "answer_relevancy", 0.7)

    history = drift.load_eval_history("rag-eval", tracking_uri=tracking_uri)

    assert len(history) == 1
    assert history[0]["faithfulness"] == 0.9
    assert history[0]["context_precision"] == 0.8
    assert history[0]["answer_relevancy"] == 0.7


def test_load_eval_history_missing_experiment(tmp_path):
    assert drift.load_eval_history("absent", tracking_uri=f"file:{tmp_path}") == []
