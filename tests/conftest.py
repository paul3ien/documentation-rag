"""Configuration pytest.

On isole les artefacts générés (index Qdrant, index BM25) dans un dossier
temporaire **avant** le premier import de `app`, pour ne jamais toucher aux
données réelles du projet pendant les tests.

Le cache du modèle d'embeddings est laissé à l'emplacement par défaut de
`fastembed`, afin d'être réutilisé d'un run à l'autre.
"""

import os
import tempfile

_TEST_STORAGE = tempfile.mkdtemp(prefix="rag-tests-")

os.environ["QDRANT_PATH"] = os.path.join(_TEST_STORAGE, "qdrant_db")
os.environ["BM25_INDEX_PATH"] = os.path.join(_TEST_STORAGE, "bm25_index.pkl")
