"""Dashboard de monitoring du RAG (Streamlit).

Lancer :

    streamlit run dashboard/main.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.agent.tools import CORPUS_DIR
from app.monitoring.corpus import count_documents, load_corpus_snapshots
from app.monitoring.drift import (
    QUERY_DISTANCE_THRESHOLD,
    corpus_drift,
    fallback_drift,
    load_eval_history,
    performance_drift,
    query_drift,
    retrieval_drift,
)
from app.monitoring.queries import load_queries
from app.retrieval.bm25_index import load_bm25

RAGAS_METRICS = ["faithfulness", "context_precision", "answer_relevancy"]

st.set_page_config(page_title="RAG monitoring", layout="wide")
st.title("RAG monitoring")

queries = load_queries()
snapshots = load_corpus_snapshots()
runs = load_eval_history()

fallback_rate = (
    sum(bool(q.get("used_web_research")) for q in queries) / len(queries)
    if queries
    else 0.0
)
latest = snapshots[-1] if snapshots else {"n_docs": 0, "n_chunks": 0}
if not snapshots:
    try:
        _, _chunks = load_bm25()
    except FileNotFoundError:
        _chunks = []
    latest = {"n_docs": count_documents(CORPUS_DIR), "n_chunks": len(_chunks)}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Requêtes", len(queries))
col2.metric("Taux de repli web", f"{fallback_rate:.0%}")
col3.metric("Documents", latest["n_docs"])
col4.metric("Chunks", latest["n_chunks"])


def _status(result) -> None:
    if not result.enough_data:
        st.info("Pas assez de données pour mesurer la dérive.")
    elif result.drifted:
        st.warning(f"Dérive détectée (score {result.score:.3f}).")
    else:
        st.success(f"Stable (score {result.score:.3f}).")


# --- Performance drift -------------------------------------------------------
st.header("Performance drift (RAGAS)")
if not runs:
    st.info("Aucun run d'évaluation dans MLflow (`python -m eval.ragas_eval`).")
else:
    df = pd.DataFrame(runs)
    df["run"] = range(1, len(df) + 1)
    long = df.melt(
        id_vars="run", value_vars=RAGAS_METRICS, var_name="métrique", value_name="score"
    )
    fig = px.line(long, x="run", y="score", color="métrique", markers=True)
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig)

    result = performance_drift(runs)
    if not result.enough_data:
        st.info("Pas assez de runs pour mesurer une dérive.")
    else:
        state = "dérive détectée" if result.drifted else "stable"
        st.write(
            f"Dérive : **{result.detail['delta']:+.3f}** "
            f"({result.detail['first']:.3f} → {result.detail['last']:.3f}) — {state}"
        )


# --- Autres drifts -----------------------------------------------------------
st.header("Drifts (journal de requêtes et corpus)")

drifts = {
    "retrieval": retrieval_drift(queries),
    "query": query_drift(queries),
    "fallback": fallback_drift(queries),
    "corpus": corpus_drift(snapshots),
}

summary = pd.DataFrame(
    [
        {
            "drift": name,
            "score": round(result.score, 3),
            "drifted": result.drifted,
            "assez de données": result.enough_data,
        }
        for name, result in drifts.items()
    ]
)
st.dataframe(summary, hide_index=True)

left, right = st.columns(2)

with left:
    st.subheader("Retrieval drift")
    result = drifts["retrieval"]
    _status(result)
    if result.enough_data:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=result.detail["reference"], name="référence", opacity=0.65)
        )
        fig.add_trace(
            go.Histogram(x=result.detail["recent"], name="récent", opacity=0.65)
        )
        fig.update_layout(barmode="overlay", xaxis_title="top_score")
        st.plotly_chart(fig)

    st.subheader("Query drift (embeddings)")
    result = drifts["query"]
    _status(result)
    if result.enough_data:
        fig = go.Figure(
            go.Bar(x=["distance cosinus"], y=[result.detail["cosine_distance"]])
        )
        fig.add_hline(
            y=QUERY_DISTANCE_THRESHOLD, line_dash="dash", annotation_text="seuil"
        )
        st.plotly_chart(fig)

with right:
    st.subheader("Fallback drift")
    result = drifts["fallback"]
    _status(result)
    if result.enough_data:
        fig = go.Figure(
            go.Bar(
                x=["référence", "récent"],
                y=[result.detail["reference_rate"], result.detail["recent_rate"]],
            )
        )
        fig.update_layout(yaxis_title="taux de repli web", yaxis_range=[0, 1])
        st.plotly_chart(fig)

    st.subheader("Corpus drift")
    result = drifts["corpus"]
    _status(result)
    if snapshots:
        df = pd.DataFrame(snapshots)
        df["snapshot"] = range(1, len(df) + 1)
        fig = px.line(df, x="snapshot", y=["n_docs", "n_chunks"], markers=True)
        st.plotly_chart(fig)
