"""Index lexical BM25 (rank-bm25) persisté sur disque."""

import pickle
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.config import BM25_INDEX_PATH
from app.ingestion.loaders import Chunk


def _tokenize(text: str) -> list[str]:
    """Tokenisation volontairement simple : minuscules + découpage sur les espaces."""
    return text.lower().split()


def build_bm25(chunks: list[Chunk], path: str = BM25_INDEX_PATH) -> BM25Okapi:
    """Construit l'index BM25 des chunks et le persiste (index + chunks)."""
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    load_bm25.cache_clear()
    return bm25


@lru_cache(maxsize=1)
def load_bm25(path: str = BM25_INDEX_PATH) -> tuple[BM25Okapi, list[Chunk]]:
    """Recharge l'index BM25 et les chunks associés (mis en cache mémoire)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def search_bm25(query: str, limit: int, path: str = BM25_INDEX_PATH) -> list[tuple[int, float]]:
    """Retourne les `limit` chunks les mieux classés par BM25.

    Le résultat est une liste de paires (id_du_chunk, score), triée par score
    décroissant.
    """
    bm25, _ = load_bm25(path)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]
    return [(i, float(scores[i])) for i in ranked]
