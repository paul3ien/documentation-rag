"""Orchestration multi-agent.

Flux : RAG local → si l'information est absente, repli web (agent chercheur puis
curateur), écriture dans le corpus local, ré-indexation, puis réponse via le RAG.
"""

from dataclasses import dataclass

from app.agent.curator import curate
from app.agent.researcher import research
from app.agent.tools import CORPUS_DIR, save_document
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, QDRANT_COLLECTION
from app.ingestion.loaders import load_and_chunk
from app.llm.ollama_client import ABSENCE_PHRASE, generate_answer
from app.retrieval.bm25_index import build_bm25
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import build_index, get_client


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    used_web_research: bool = False


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def is_absence(answer: str) -> bool:
    """Vrai si la réponse locale signale que l'information est absente."""
    return _normalize(ABSENCE_PHRASE) in _normalize(answer)


def reindex(docs_dir=None) -> int:
    """Reconstruit les index (Qdrant + BM25) à partir du corpus local."""
    client = get_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    chunks = load_and_chunk(str(docs_dir or CORPUS_DIR), CHUNK_SIZE, CHUNK_OVERLAP)
    build_index(chunks)
    build_bm25(chunks)
    return len(chunks)


def answer(question: str, *, allow_web: bool = True) -> Answer:
    """Répond via le RAG local, avec repli web si l'information est absente.

    En repli web : l'agent chercheur rassemble l'information, le curateur la
    reformule et la valide, puis la fiche est écrite dans le corpus local et les
    index sont reconstruits.
    """
    hits = hybrid_search(question)
    local_answer = generate_answer(question, [hit["text"] for hit in hits])

    if not (allow_web and is_absence(local_answer)):
        return Answer(answer=local_answer, sources=hits, used_web_research=False)

    raw_material = research(question)
    curation = curate(question, raw_material)

    if curation.approved:
        save_document(question, curation.content)
        reindex()
        hits = hybrid_search(question)
        local_answer = generate_answer(question, [hit["text"] for hit in hits])

    return Answer(answer=local_answer, sources=hits, used_web_research=True)
