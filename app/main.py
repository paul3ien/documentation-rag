"""API HTTP exposant le RAG : recherche hybride, génération et repli web."""

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.orchestrator import answer as answer_question
from app.config import DEEPSEEK_API_KEY

app = FastAPI(title="rag-project", version="0.1.0")

# Le repli web n'est proposé que si une clé DeepSeek est configurée.
WEB_FALLBACK_ENABLED = bool(DEEPSEEK_API_KEY)


class Query(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    source: str
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    used_web_research: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=Answer)
def query(q: Query) -> Answer:
    """Répond à une question, avec repli web si l'information est absente."""
    result = answer_question(q.question, allow_web=WEB_FALLBACK_ENABLED)
    return Answer(
        answer=result.answer,
        sources=[Source(**source) for source in result.sources],
        used_web_research=result.used_web_research,
    )
