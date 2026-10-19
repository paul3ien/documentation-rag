"""Snapshots du corpus (nombre de documents et de chunks) pour le corpus drift."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR
from app.ingestion.loaders import SUPPORTED_EXTENSIONS

LOG_DIR = BASE_DIR / "logs"
CORPUS_LOG_PATH = LOG_DIR / "corpus.jsonl"


def count_documents(docs_dir) -> int:
    """Compte les documents supportés présents dans `docs_dir`."""
    root = Path(docs_dir)
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def log_corpus_snapshot(
    n_docs: int, n_chunks: int, path: Path | str | None = None
) -> None:
    """Enregistre un instantané du corpus (une ligne JSON)."""
    log_path = Path(path) if path is not None else CORPUS_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_docs": n_docs,
        "n_chunks": n_chunks,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_corpus_snapshots(path: Path | str | None = None) -> list[dict]:
    """Charge tous les instantanés du corpus."""
    log_path = Path(path) if path is not None else CORPUS_LOG_PATH
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
