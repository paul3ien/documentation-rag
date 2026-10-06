"""Indexe les documents de `data/raw_docs` dans Qdrant (vecteurs) et BM25."""

from app.config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion.loaders import load_and_chunk
from app.retrieval.bm25_index import build_bm25
from app.retrieval.vector_store import build_index

DOCS_DIR = str(BASE_DIR / "data" / "raw_docs")


def main(docs_dir: str = DOCS_DIR) -> None:
    """Charge, découpe puis indexe tous les documents de `docs_dir`."""
    print(f"Ingestion des documents de {docs_dir}...")
    chunks = load_and_chunk(docs_dir, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        print("Aucun document à indexer.")
        return

    build_index(chunks)
    build_bm25(chunks)
    print(f"{len(chunks)} chunks indexés.")


if __name__ == "__main__":
    main()
