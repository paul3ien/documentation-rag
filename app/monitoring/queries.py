"""Journalisation des requêtes pour l'analyse de drift."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"


def log_query(question: str, results: list[dict], used_web_research: bool) -> None:
    """Ajoute une requête au journal (une ligne JSON par requête)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "top_score": max((r.get("score", 0.0) for r in results), default=0.0),
        "n_sources": len(results),
        "used_web_research": used_web_research,
    }
    with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_queries(path: Path | str | None = None) -> list[dict]:
    """Charge et renvoie toutes les requêtes journalisées."""
    log_path = Path(path) if path is not None else QUERY_LOG_PATH
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
