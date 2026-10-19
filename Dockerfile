# syntax=docker/dockerfile:1

# --- Build stage -------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Installe le package `app` dans le venv (non éditable).
COPY pyproject.toml params.yaml ./
COPY app ./app
RUN pip install --no-build-isolation --no-deps .

# Pré-télécharge le modèle d'embeddings pour un démarrage hors ligne.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN python -c "from app.config import EMBEDDING_MODEL; from fastembed import TextEmbedding; TextEmbedding(model_name=EMBEDDING_MODEL)"

# --- Runtime stage -----------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed_cache /opt/fastembed_cache

COPY pyproject.toml ./
COPY app ./app
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY eval ./eval
COPY params.yaml dvc.yaml ./
COPY data ./data

# Dossiers persistés via des volumes (voir docker-compose.yml).
RUN mkdir -p storage logs mlruns data/raw_docs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
