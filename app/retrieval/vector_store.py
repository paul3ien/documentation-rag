"""Index vectoriel Qdrant (mode local) propulsé par les embeddings fastembed."""

from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_PATH
from app.ingestion.loaders import Chunk


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    """Retourne l'embedder fastembed (instancié une seule fois, à la demande)."""
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Retourne le client Qdrant en mode local (base persistée sur disque)."""
    return QdrantClient(path=QDRANT_PATH)


def build_index(chunks: list[Chunk]) -> None:
    """Crée la collection si nécessaire puis y indexe les chunks."""
    client = get_client()

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    vectors = list(get_embedder().embed([c.text for c in chunks]))
    points = [
        PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={"text": chunk.text, "source": chunk.source},
        )
        for i, (vector, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(QDRANT_COLLECTION, points=points)


def search_vector(query: str, limit: int) -> list[tuple[int, float]]:
    """Recherche les `limit` chunks les plus proches du texte `query`.

    Retourne une liste de paires (id_du_chunk, score de similarité).
    """
    vector = list(get_embedder().embed([query]))[0].tolist()
    result = get_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
    )
    return [(int(hit.id), float(hit.score)) for hit in result.points]
