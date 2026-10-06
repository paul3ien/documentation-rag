# rag-project (v2)

**v2** d'une architecture RAG existante.

Le but : répondre à des questions sur de la **documentation technique** en
s'appuyant uniquement sur les documents fournis, sans que le modèle n'ajoute
d'informations de sa propre initiative.

## Fonctionnement

1. **Ingestion** — les documents (`md`, `txt`, `pdf`, `py`) sont extraits puis découpés en chunks.
2. **Indexation** — chaque chunk est encodé en vecteur (Qdrant local + `fastembed`) et indexé pour la recherche lexicale (`rank-bm25`).
3. **Recherche hybride** — les résultats vecteurs et BM25 sont fusionnés par Reciprocal Rank Fusion (RRF).
4. **Génération** — un LLM local (Ollama) répond strictement à partir du contexte récupéré.

## Stack

- **API** : FastAPI / Uvicorn
- **Vector store** : Qdrant (mode local) + embeddings `fastembed`
- **Recherche lexicale** : `rank-bm25`
- **LLM** : Ollama (local)
- **Évaluation** : RAGAS + MLflow
- **MLOps** : DVC

## Structure

```
.
├── app/
│   ├── config.py          # configuration centralisée
│   ├── ingestion/         # extraction + découpage des documents
│   ├── retrieval/         # index vecteurs, BM25, recherche hybride
│   └── llm/               # client Ollama + prompt
├── scripts/               # scripts d'ingestion
├── eval/                  # évaluation RAGAS
├── tests/                 # tests (pytest)
├── data/raw_docs/         # documents sources (non versionnés)
└── storage/               # index générés (non versionnés)
```

## Installation

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Prérequis : [Ollama](https://ollama.com) installé et un modèle disponible, p. ex.
`ollama pull qwen2.5:3b-instruct-q4_K_M`.

## Utilisation

```sh
# 1. Indexer les documents déposés dans data/raw_docs/
python scripts/ingest.py

# 2. Lancer l'API
uvicorn app.main:app --reload
```

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

## Avancement

- [x] Configuration centralisée
- [x] Ingestion / découpage en chunks
- [x] Index vectoriel (Qdrant + fastembed)
- [x] Index BM25
- [x] Recherche hybride (RRF)
- [x] Client LLM (Ollama)
- [x] API FastAPI
- [ ] Évaluation RAGAS + MLflow
- [ ] DVC / paramètres
