"""Configuration centralisée du projet.

Les valeurs par défaut proviennent de `params.yaml`. Chacune peut être
surchargée par une variable d'environnement (ou un fichier `.env` chargé
automatiquement).

Ordre de priorité : variable d'environnement > params.yaml > valeur par défaut.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_params() -> dict:
    params_path = BASE_DIR / "params.yaml"
    if not params_path.exists():
        return {}
    return yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}


_PARAMS = _load_params()


def _setting(env: str, key: str, default, cast=str):
    """Résout une valeur : env > params.yaml > défaut."""
    value = os.environ.get(env, _PARAMS.get(key, default))
    return cast(value)


# --- Ollama (LLM local) ---
OLLAMA_HOST = _setting("OLLAMA_HOST", "ollama_host", "http://localhost:11434")
OLLAMA_MODEL = _setting("OLLAMA_MODEL", "ollama_model", "qwen2.5:3b-instruct-q4_K_M")
JUDGE_MODEL = _setting("JUDGE_MODEL", "judge_model", OLLAMA_MODEL)

# --- DeepSeek (agent de recherche web) ---
# La clé vient uniquement de l'environnement / .env (jamais de params.yaml,
# qui est versionné).
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = _setting("DEEPSEEK_MODEL", "deepseek_model", "deepseek-chat")

# --- Embeddings (fastembed : léger, sans torch) ---
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "embedding_model", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = _setting("EMBEDDING_DIM", "embedding_dim", 384, int)

# --- Qdrant (mode local, pas de serveur permanent) ---
QDRANT_PATH = _setting("QDRANT_PATH", "qdrant_path", str(BASE_DIR / "storage" / "qdrant_db"))
QDRANT_COLLECTION = _setting("QDRANT_COLLECTION", "qdrant_collection", "tech_docs")

# --- Index BM25 ---
BM25_INDEX_PATH = _setting("BM25_INDEX_PATH", "bm25_path", str(BASE_DIR / "storage" / "bm25_index.pkl"))

# --- Ingestion / chunking ---
CHUNK_SIZE = _setting("CHUNK_SIZE", "chunk_size", 200, int)
CHUNK_OVERLAP = _setting("CHUNK_OVERLAP", "overlap", 50, int)

# --- Recherche hybride ---
TOP_K_VECTOR = _setting("TOP_K_VECTOR", "top_k_vector", 10, int)
TOP_K_BM25 = _setting("TOP_K_BM25", "top_k_bm25", 10, int)
TOP_K_FINAL = _setting("TOP_K_FINAL", "top_k_final", 3, int)
RRF_K = _setting("RRF_K", "rrf_k", 60, int)

# --- MLflow ---
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
MLFLOW_TRACKING_URI = _setting("MLFLOW_TRACKING_URI", "mlflow_tracking_uri", "file:./mlruns")
