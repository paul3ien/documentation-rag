"""Dashboard de monitoring du RAG (Streamlit).

Lancer :

    streamlit run dashboard/main.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.agent.tools import CORPUS_DIR
from app.monitoring.corpus import count_documents, load_corpus_snapshots
from app.monitoring.drift import load_eval_history, performance_drift
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
        state = "⚠️ dérive détectée" if result.drifted else "stable"
        st.write(
            f"Dérive : **{result.detail['delta']:+.3f}** "
            f"({result.detail['first']:.3f} → {result.detail['last']:.3f}) — {state}"
        )
