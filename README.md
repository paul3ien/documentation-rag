# rag-project (v2)

**v2** d'une architecture RAG existante.

Le but : répondre à des questions sur de la **documentation technique** en
s'appuyant uniquement sur les documents fournis, sans que le modèle n'ajoute
d'informations de sa propre initiative. Si un document n'est pas disponible
localement, un **agent** va le chercher sur le web et l'intègre au corpus.

## Fonctionnement

1. **Ingestion** — les documents (`md`, `txt`, `pdf`, `py`) sont extraits puis découpés en chunks.
2. **Indexation** — chaque chunk est encodé en vecteur (Qdrant local + `fastembed`) et indexé pour la recherche lexicale (`rank-bm25`).
3. **Recherche hybride** — les résultats vecteurs et BM25 sont fusionnés par Reciprocal Rank Fusion (RRF).
4. **Génération** — un LLM local (Ollama) répond strictement à partir du contexte récupéré.
5. **Repli web** — si un document n'est pas trouvé en local, un agent (LangChain + DeepSeek) le cherche sur le web, le reformule, l'ajoute au corpus puis le ré-indexe (voir « Agent de recherche web »).

## Stack

- **API** : FastAPI / Uvicorn
- **Vector store** : Qdrant (mode local) + embeddings `fastembed`
- **Recherche lexicale** : `rank-bm25`
- **LLM** : Ollama (local)
- **Agent web** : LangChain + DeepSeek (`langchain-deepseek`), recherche DuckDuckGo (`ddgs`)
- **Évaluation** : RAGAS + MLflow
- **Monitoring** : journalisation JSONL + calculs de drift
- **Dashboard** : Streamlit + Plotly
- **MLOps** : DVC

## Structure

```
.
├── app/
│   ├── config.py          # configuration centralisée
│   ├── ingestion/         # extraction + découpage des documents
│   ├── retrieval/         # index vecteurs, BM25, recherche hybride
│   ├── llm/               # client Ollama + prompt
│   ├── agent/             # agent de recherche web (LangChain + DeepSeek)
│   └── monitoring/        # journalisation + calculs de drift
├── dashboard/             # dashboard Streamlit (KPI + drifts)
├── scripts/               # scripts d'ingestion
├── eval/                  # évaluation RAGAS
├── tests/                 # tests (pytest)
├── params.yaml            # paramètres du pipeline
├── dvc.yaml               # pipeline DVC (stages ingest, eval)
├── data/raw_docs/         # documents sources (non versionnés)
├── logs/                  # journaux de monitoring (non versionnés)
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
`ollama pull qwen2.5:3b-instruct-q4_K_M`. L'agent web nécessite en plus une clé
DeepSeek (voir ci-dessous).

## Utilisation

```sh
# 1. Indexer les documents déposés dans data/raw_docs/
python scripts/ingest.py

# 2. Lancer l'API
uvicorn app.main:app --reload
```

## Docker

Par défaut, les conteneurs utilisent **Ollama installé sur l'hôte** (plus léger,
pas d'image Ollama de plusieurs Go). Sur l'hôte :

```sh
ollama serve
ollama pull qwen2.5:3b-instruct-q4_K_M
```

Puis :

```sh
docker compose up --build

# première indexation des documents
docker compose run --rm api python scripts/ingest.py
```

- API : http://localhost:8001/docs
- Dashboard : http://localhost:8502

Le conteneur joint l'hôte via `host.docker.internal:11434`. Si la connexion
échoue, fais écouter Ollama sur toutes les interfaces :

```sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Tout en Docker

Pour embarquer Ollama (et tirer le modèle automatiquement) :

```sh
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
```

⚠️ Cette variante télécharge l'image Ollama (plusieurs Go) : prévoir de l'espace
disque (Docker Desktop → Resources → Disk).

Les dossiers `data/raw_docs`, `storage`, `logs`, `eval` et `mlruns` sont montés
depuis l'hôte, donc index, journaux, résultats d'évaluation et dashboard
persistent. L'API et le dashboard partagent la **même image** (construite une
seule fois). La clé DeepSeek est lue depuis `.env` (`DEEPSEEK_API_KEY`) et n'est
jamais copiée dans l'image.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

## Évaluation

Un jeu de questions/réponses de référence est évalué avec RAGAS
(faithfulness, context precision, answer relevancy) et les scores sont suivis
dans MLflow :

```sh
python -m eval.ragas_eval
```

### Résultats

Dernier run sur `eval/datasets/sample_qa.json` (juge `qwen2.5:3b-instruct-q4_K_M`) :

| Métrique | Score |
|----------|-------|
| Faithfulness | 0.92 |
| Context precision | 0.82 |
| Answer relevancy | 0.89 |

Les scores sont écrits dans `eval/results.json` et journalisés dans MLflow
(`mlflow ui` pour les consulter).

L'évaluation de la version précédente est conservée dans MLflow sous
l'expérience `rag-eval-v1` (4 runs), pour garder une trace : faithfulness
`NaN`, context precision `0.67`, answer relevancy `0.64`.

## Pipeline (DVC)

Le pipeline est décrit dans `dvc.yaml` (stages `ingest` puis `eval`) et ses
paramètres dans `params.yaml`, surchargeables par variables d'environnement.

```sh
dvc repro ingest   # (re)génère les index

dvc repro          # pipeline complet (ingest + éval ; nécessite Ollama)
```

## Agent de recherche web

Quand une question ne trouve pas de réponse dans le corpus local, un agent
**LangChain + DeepSeek** prend le relais :

1. **Recherche** — DuckDuckGo (`ddgs`) cherche des documents sur le web (PDF).
2. **Lecture** — le document est téléchargé puis son texte extrait (PyMuPDF).
3. **Reformulation** — DeepSeek en extrait les éléments importants et le réécrit
   de façon **claire et concise**, puis **vérifie** que la synthèse est fidèle et
   exploitable (auto-contrôle).
4. **Enregistrement** — la synthèse est écrite en Markdown dans `data/raw_docs/`.
5. **Ré-indexation** — la pipeline d'ingestion existante l'intègre au corpus
   (Qdrant + BM25) ; le RAG classique peut alors la retrouver.

### Configuration

L'agent utilise l'API DeepSeek. La clé n'est **jamais** dans le code : elle est
lue depuis un fichier `.env` (non versionné) ou la variable d'environnement
`DEEPSEEK_API_KEY`.

## Monitoring & dashboard

Les requêtes et l'évolution du corpus sont journalisés dans `logs/` (non
versionné) :

- `logs/queries.jsonl` — question, `top_score`, nombre de sources, repli web ;
- `logs/corpus.jsonl` — instantanés `n_docs` / `n_chunks`.

Cinq types de drift sont suivis :

| Drift | Calcul |
|-------|--------|
| Performance | variation des métriques RAGAS entre runs (MLflow) |
| Retrieval | test KS sur les scores de récupération récents vs référence |
| Query | distance cosinus entre centroïdes d'embeddings des questions |
| Corpus | évolution du nombre de documents / chunks |
| Fallback | évolution du taux de repli web |

Dashboard :

```sh
streamlit run dashboard/main.py
```

## Avancement

Base (v2) :

- [x] Configuration centralisée
- [x] Ingestion / découpage en chunks
- [x] Index vectoriel (Qdrant + fastembed)
- [x] Index BM25
- [x] Recherche hybride (RRF)
- [x] Client LLM (Ollama)
- [x] API FastAPI
- [x] Évaluation RAGAS + MLflow
- [x] DVC / paramètres

Agent de recherche web :

- [x] Configuration DeepSeek (`.env`)
- [x] Outils de l'agent (recherche web, lecture, sauvegarde)
- [x] Agent LangChain + DeepSeek
- [x] Orchestration multi-agent (routeur → chercheur → curateur)
- [x] Intégration au RAG (repli web + ré-indexation)

Monitoring & dashboard :

- [x] Journalisation des requêtes
- [x] Snapshots du corpus
- [x] Calcul des 5 types de drift
- [x] Dashboard Streamlit (KPI + drifts)
