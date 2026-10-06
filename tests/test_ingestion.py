from pathlib import Path

import pymupdf
import pytest

from app.ingestion.loaders import load_and_chunk


def test_chunking_does_not_duplicate_the_tail(tmp_path: Path):
    words = [f"mot{i}" for i in range(25)]
    (tmp_path / "doc.md").write_text(" ".join(words), encoding="utf-8")

    chunks = load_and_chunk(str(tmp_path), chunk_size=10, overlap=2)

    assert [c.id for c in chunks] == ["doc_0", "doc_8", "doc_16"]
    assert chunks[0].text.split()[0] == "mot0"
    assert chunks[-1].text.split()[-1] == "mot24"


def test_empty_and_unsupported_files_are_ignored(tmp_path: Path):
    (tmp_path / "vide.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not text")

    assert load_and_chunk(str(tmp_path)) == []


def test_pdf_is_extracted(tmp_path: Path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "contenu du pdf")
    pdf.save(tmp_path / "manual.pdf")
    pdf.close()

    chunks = load_and_chunk(str(tmp_path))

    assert len(chunks) == 1
    assert "contenu du pdf" in chunks[0].text
    assert chunks[0].metadata["filename"] == "manual.pdf"


def test_chunk_size_must_exceed_overlap(tmp_path: Path):
    with pytest.raises(ValueError):
        load_and_chunk(str(tmp_path), chunk_size=5, overlap=5)
