"""API HTTP exposant le RAG : recherche hybride + génération Ollama."""

from fastapi import FastAPI
from pydantic import BaseModel

from app.llm.ollama_client import generate_answer
from app.retrieval.hybrid import hybrid_search

app = FastAPI(title="rag-project", version="0.1.0")


class Query(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    source: str
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=Answer)
def query(q: Query) -> Answer:
    """Répond à une question à partir des documents indexés."""
    hits = hybrid_search(q.question)
    answer = generate_answer(q.question, [hit["text"] for hit in hits])
    return Answer(answer=answer, sources=[Source(**hit) for hit in hits])
