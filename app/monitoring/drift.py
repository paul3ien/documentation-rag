"""Calculs de drift pour le dashboard.

Cinq types :
- performance : évolution des métriques RAGAS d'un run à l'autre ;
- retrieval   : distribution des scores de récupération (test KS) ;
- query       : dérive des embeddings des questions (distance cosinus) ;
- corpus      : évolution du nombre de documents / chunks ;
- fallback    : évolution du taux de repli web.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp

RECENT_WINDOW = 50
KS_ALPHA = 0.05
PERFORMANCE_DELTA_THRESHOLD = 0.05
QUERY_DISTANCE_THRESHOLD = 0.1
FALLBACK_DELTA_THRESHOLD = 0.1


@dataclass
class DriftResult:
    """Résultat d'un calcul de drift."""

    name: str
    score: float
    drifted: bool
    enough_data: bool
    detail: dict = field(default_factory=dict)


def _split(samples: list, recent_n: int) -> tuple[list, list]:
    """Sépare une série en (référence, fenêtre récente)."""
    if len(samples) <= recent_n:
        return [], samples
    return samples[:-recent_n], samples[-recent_n:]


def performance_drift(
    runs: list[dict], metric: str = "faithfulness"
) -> DriftResult:
    """Dérive de performance = variation de `metric` entre le premier et le dernier run."""
    values = [
        run[metric]
        for run in runs
        if metric in run and run[metric] == run[metric]  # ignore NaN
    ]
    if len(values) < 2:
        return DriftResult("performance", 0.0, False, False, {"metric": metric})

    delta = float(values[-1] - values[0])
    return DriftResult(
        "performance",
        abs(delta),
        abs(delta) > PERFORMANCE_DELTA_THRESHOLD,
        True,
        {"metric": metric, "first": values[0], "last": values[-1], "delta": delta, "values": values},
    )


def load_eval_history(
    experiment_name: str = "rag-eval", tracking_uri: str | None = None
) -> list[dict]:
    """Charge l'historique des runs d'évaluation depuis MLflow.

    Retourne une liste de runs triés par date, avec les métriques RAGAS.
    """
    from mlflow.tracking import MlflowClient

    from app.config import MLFLOW_TRACKING_URI

    client = MlflowClient(tracking_uri=tracking_uri or MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    return [
        {
            "run_id": run.info.run_id,
            "timestamp": run.info.start_time,
            "faithfulness": run.data.metrics.get("faithfulness"),
            "context_precision": run.data.metrics.get("context_precision"),
            "answer_relevancy": run.data.metrics.get("answer_relevancy"),
        }
        for run in runs
    ]


def retrieval_drift(
    queries: list[dict], recent_n: int = RECENT_WINDOW
) -> DriftResult:
    """Dérive de récupération : test KS sur les `top_score` récents vs de référence."""
    scores = [q["top_score"] for q in queries if "top_score" in q]
    reference, recent = _split(scores, recent_n)
    if not reference or not recent:
        return DriftResult("retrieval", 0.0, False, False)

    statistic, p_value = ks_2samp(reference, recent)
    return DriftResult(
        "retrieval",
        float(statistic),
        bool(p_value < KS_ALPHA),
        True,
        {
            "p_value": float(p_value),
            "reference_mean": float(np.mean(reference)),
            "recent_mean": float(np.mean(recent)),
            "reference": [float(x) for x in reference],
            "recent": [float(x) for x in recent],
        },
    )


def _embed(texts: list[str]) -> list[np.ndarray]:
    from app.retrieval.vector_store import get_embedder

    return [np.asarray(v, dtype=float) for v in get_embedder().embed(texts)]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def query_drift(
    queries: list[dict], recent_n: int = RECENT_WINDOW, embed=None
) -> DriftResult:
    """Dérive des questions : distance cosinus entre les centroïdes d'embeddings."""
    questions = [q["question"] for q in queries if q.get("question")]
    reference, recent = _split(questions, recent_n)
    if not reference or not recent:
        return DriftResult("query", 0.0, False, False)

    embed = embed or _embed
    centroid_reference = np.mean(embed(reference), axis=0)
    centroid_recent = np.mean(embed(recent), axis=0)
    distance = 1.0 - _cosine(centroid_reference, centroid_recent)

    return DriftResult(
        "query",
        float(distance),
        bool(distance > QUERY_DISTANCE_THRESHOLD),
        True,
        {"cosine_distance": float(distance), "n_reference": len(reference), "n_recent": len(recent)},
    )


def corpus_drift(snapshots: list[dict]) -> DriftResult:
    """Dérive du corpus : variation du nombre de documents / chunks."""
    if len(snapshots) < 2:
        return DriftResult("corpus", 0.0, False, False)

    first, last = snapshots[0], snapshots[-1]
    doc_delta = last["n_docs"] - first["n_docs"]
    chunk_delta = last["n_chunks"] - first["n_chunks"]
    growth = chunk_delta / first["n_chunks"] if first["n_chunks"] else 0.0

    return DriftResult(
        "corpus",
        abs(float(growth)),
        bool(chunk_delta != 0),
        True,
        {"doc_delta": doc_delta, "chunk_delta": chunk_delta, "growth": float(growth)},
    )


def fallback_drift(
    queries: list[dict], recent_n: int = RECENT_WINDOW
) -> DriftResult:
    """Dérive du taux de repli web (fenêtre récente vs référence)."""
    flags = [bool(q.get("used_web_research", False)) for q in queries]
    reference, recent = _split(flags, recent_n)
    if not reference or not recent:
        return DriftResult("fallback", 0.0, False, False)

    reference_rate = float(np.mean(reference))
    recent_rate = float(np.mean(recent))
    delta = recent_rate - reference_rate

    return DriftResult(
        "fallback",
        abs(delta),
        bool(abs(delta) > FALLBACK_DELTA_THRESHOLD),
        True,
        {"reference_rate": reference_rate, "recent_rate": recent_rate, "delta": delta},
    )


def compute_all(
    queries: list[dict],
    snapshots: list[dict] | None = None,
    runs: list[dict] | None = None,
    embed=None,
) -> dict[str, DriftResult]:
    """Calcule les drifts disponibles à partir des données fournies."""
    results = {
        "retrieval": retrieval_drift(queries),
        "query": query_drift(queries, embed=embed),
        "fallback": fallback_drift(queries),
        "corpus": corpus_drift(snapshots or []),
    }
    if runs is not None:
        results["performance"] = performance_drift(runs)
    return results
