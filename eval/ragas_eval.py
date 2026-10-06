"""Évaluation du RAG avec RAGAS, suivi des scores dans MLflow."""

import json

import mlflow
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from app.config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    MLFLOW_TRACKING_URI,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from app.llm.ollama_client import generate_answer
from app.retrieval.hybrid import hybrid_search

QA_PATH = str(BASE_DIR / "eval" / "datasets" / "sample_qa.json")
OUTPUT_PATH = str(BASE_DIR / "eval" / "results.json")

METRICS = [faithfulness, context_precision, answer_relevancy]


def build_eval_dataset(qa_path: str = QA_PATH, top_k: int = 5) -> Dataset:
    """Construit le dataset RAGAS.

    Pour chaque question du jeu de test, on récupère le contexte via la
    recherche hybride puis on génère une réponse avec le LLM.
    """
    with open(qa_path, encoding="utf-8") as f:
        qa_items = json.load(f)

    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in qa_items:
        hits = hybrid_search(item["question"], top_k=top_k)
        contexts = [hit["text"] for hit in hits] or ["(aucun contexte trouvé)"]

        records["question"].append(item["question"])
        records["answer"].append(generate_answer(item["question"], contexts))
        records["contexts"].append(contexts)
        records["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(records)


def run_evaluation(qa_path: str = QA_PATH, output_path: str = OUTPUT_PATH) -> dict:
    """Évalue le RAG avec RAGAS et journalise les scores dans MLflow."""
    print("Construction du dataset d'évaluation...", flush=True)
    dataset = build_eval_dataset(qa_path)
    print(f"{len(dataset)} questions à évaluer.", flush=True)

    judge_llm = LangchainLLMWrapper(
        ChatOllama(base_url=OLLAMA_HOST, model=JUDGE_MODEL, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    )

    print("Évaluation RAGAS en cours (peut prendre plusieurs minutes)...", flush=True)
    results = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=600, max_retries=2, max_workers=1),
    )

    scores = {
        name: float(value)
        for name, value in results.to_pandas().mean(numeric_only=True).items()
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("rag-eval")
    with mlflow.start_run():
        mlflow.log_param("llm_model", OLLAMA_MODEL)
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("qa_dataset", qa_path)
        mlflow.log_metrics(scores)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print("Scores :", scores, flush=True)
    return scores


if __name__ == "__main__":
    run_evaluation()
