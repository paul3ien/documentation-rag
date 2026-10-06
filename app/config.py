"""Configuration centralisée du projet.

Toutes les valeurs sont surchargeables via des variables d'environnement
(ou un fichier .env chargé automatiquement).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama (LLM local) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OLLAMA_MODEL)

# --- Embeddings (fastembed : léger, sans torch) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# --- Qdrant (mode local, pas de serveur permanent) ---
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "storage" / "qdrant_db"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tech_docs")

# --- Index BM25 ---
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(BASE_DIR / "storage" / "bm25_index.pkl"))

# --- Ingestion / chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Recherche hybride ---
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))
RRF_K = int(os.getenv("RRF_K", "60"))

# --- MLflow ---
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
