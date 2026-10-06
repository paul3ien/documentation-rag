"""Chargement des documents bruts et découpage en chunks."""

from dataclasses import dataclass, field
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

# Types de fichiers pris en charge pour l'ingestion.
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".py"}


@dataclass
class Chunk:
    """Un fragment de document, prêt à être indexé."""

    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier.

    Les PDF passent par PyMuPDF (import paresseux : on ne charge la lib
    que lorsqu'on rencontre réellement un PDF).
    """
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)

    return path.read_text(encoding="utf-8", errors="ignore")


def load_and_chunk(
    docs_dir: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Parcourt `docs_dir` et découpe chaque document en chunks.

    Le découpage se fait mot à mot, avec un recouvrement (`overlap`) entre
    chunks consécutifs pour ne pas couper le contexte.
    """
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size doit être strictement supérieur à overlap")

    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = extract_text(path)
        if not text.strip():
            continue

        words = text.split()
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i : i + chunk_size])
            chunks.append(
                Chunk(
                    id=f"{path.stem}_{i}",
                    text=chunk_text,
                    source=str(path),
                    metadata={"filename": path.name},
                )
            )
            # Une fois la fin du document atteinte, inutile de continuer :
            # le pas suivant ne produirait qu'un chunk redondant (overlap).
            if i + chunk_size >= len(words):
                break

    return chunks
