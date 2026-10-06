from fastapi.testclient import TestClient

from app import main
from app.agent.orchestrator import Answer as OrchestratorAnswer


def _fake_answer(monkeypatch, *, used_web_research: bool) -> None:
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda question, **kwargs: OrchestratorAnswer(
            answer="10 ohms",
            sources=[{"text": "ron = 10 ohms", "source": "ts5a3157.md", "score": 0.5}],
            used_web_research=used_web_research,
        ),
    )


def test_health():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(monkeypatch):
    _fake_answer(monkeypatch, used_web_research=False)

    client = TestClient(main.app)
    response = client.post("/query", json={"question": "Quelle est la ron ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "10 ohms"
    assert body["sources"] == [
        {"text": "ron = 10 ohms", "source": "ts5a3157.md", "score": 0.5}
    ]
    assert body["used_web_research"] is False


def test_query_reports_web_research(monkeypatch):
    _fake_answer(monkeypatch, used_web_research=True)

    client = TestClient(main.app)
    response = client.post("/query", json={"question": "Une question absente du corpus"})

    assert response.status_code == 200
    assert response.json()["used_web_research"] is True
