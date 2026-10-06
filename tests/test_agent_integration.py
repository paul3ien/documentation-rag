"""Tests d'intégration de l'agent : ils frappent réellement le réseau.

Ils sont **désactivés par défaut** (marqueur `network`, exclu via `addopts`).
Pour les lancer :

    pytest -m network

On vérifie uniquement la **structure** des résultats (jamais leur contenu, qui
change au fil du temps).
"""

import pytest

from app.agent import tools


@pytest.mark.network
def test_search_web_real_contract():
    results = tools.search_web("TS5A3157 datasheet", max_results=3)

    assert results, "DuckDuckGo devrait renvoyer au moins un résultat"
    assert {"title", "url", "snippet"} <= set(results[0])


@pytest.mark.network
def test_read_url_real_document():
    results = tools.search_web("TS5A3157 datasheet pdf", max_results=3)
    if not results:
        pytest.skip("aucun résultat de recherche")

    document = tools.read_url(results[0]["url"])

    assert document["url"] == results[0]["url"]
    assert document["text"].strip(), "le document devrait contenir du texte"
