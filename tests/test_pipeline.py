"""Test d'intégration : ingestion -> indexation -> recherche hybride.

Ce test télécharge le modèle d'embeddings (au premier lancement) et écrit un
index Qdrant local dans un dossier temporaire.
"""

from pathlib import Path

import pytest

from app.ingestion.loaders import load_and_chunk
from app.retrieval.bm25_index import build_bm25
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import build_index


@pytest.fixture(scope="module")
def indexed_corpus(tmp_path_factory) -> list:
    docs: Path = tmp_path_factory.mktemp("docs")
    (docs / "chip.md").write_text(
        "Le TS5A3157 a une résistance ON-state de 10 ohms. "
        "Sa tension d'alimentation va de 1.65V à 5.5V.",
        encoding="utf-8",
    )
    (docs / "crepes.md").write_text(
        "La recette des crêpes nécessite de la farine, des œufs et du lait.",
        encoding="utf-8",
    )

    chunks = load_and_chunk(str(docs), chunk_size=50, overlap=10)
    build_index(chunks)
    build_bm25(chunks)
    return chunks


def test_hybrid_search_returns_relevant_chunk(indexed_corpus):
    hits = hybrid_search("Quelle est la résistance ON-state du TS5A3157 ?", top_k=2)

    assert hits, "la recherche devrait renvoyer au moins un résultat"
    assert all({"text", "source", "score"} <= set(hit) for hit in hits)
    assert any("10 ohms" in hit["text"] for hit in hits)
