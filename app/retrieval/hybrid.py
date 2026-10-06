"""Recherche hybride : fusion par rang (RRF) des résultats vecteurs et BM25."""

from app.config import RRF_K, TOP_K_BM25, TOP_K_FINAL, TOP_K_VECTOR
from app.retrieval.bm25_index import load_bm25, search_bm25
from app.retrieval.vector_store import search_vector


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fusionne plusieurs classements d'ids par Reciprocal Rank Fusion.

    Chaque classement est une liste d'ids ordonnée du meilleur au moins bon.
    Retourne une liste de paires (id, score) triée par score décroissant.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])


def hybrid_search(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """Combine la recherche vectorielle et BM25, puis renvoie le top_k final.

    Chaque résultat est un dict {"text", "source", "score"}.
    """
    vector_ids = [doc_id for doc_id, _ in search_vector(query, TOP_K_VECTOR)]
    bm25_ids = [doc_id for doc_id, _ in search_bm25(query, TOP_K_BM25)]

    fused = reciprocal_rank_fusion([vector_ids, bm25_ids])[:top_k]

    # Les chunks sont stockés à côté de l'index BM25 : ils servent à résoudre
    # les ids fusionnés en contenu citable.
    _, chunks = load_bm25()
    return [
        {
            "text": chunks[doc_id].text,
            "source": chunks[doc_id].source,
            "score": score,
        }
        for doc_id, score in fused
    ]
