from fastapi.testclient import TestClient

from app import main


def test_health():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        main,
        "hybrid_search",
        lambda question: [
            {"text": "ron = 10 ohms", "source": "ts5a3157.md", "score": 0.5}
        ],
    )
    monkeypatch.setattr(main, "generate_answer", lambda question, context: "10 ohms")

    client = TestClient(main.app)
    response = client.post("/query", json={"question": "Quelle est la ron ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "10 ohms"
    assert body["sources"] == [
        {"text": "ron = 10 ohms", "source": "ts5a3157.md", "score": 0.5}
    ]
