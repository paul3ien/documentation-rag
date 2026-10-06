import json

from eval import ragas_eval
from eval.ragas_eval import build_eval_dataset


def _write_qa(tmp_path, items):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return str(qa_path)


def test_build_eval_dataset(monkeypatch, tmp_path):
    qa_path = _write_qa(
        tmp_path,
        [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ],
    )
    monkeypatch.setattr(
        ragas_eval,
        "hybrid_search",
        lambda question, top_k=5: [{"text": f"ctx-{question}", "source": "s", "score": 0.1}],
    )
    monkeypatch.setattr(ragas_eval, "generate_answer", lambda question, contexts: f"rep-{question}")

    dataset = build_eval_dataset(qa_path)

    assert len(dataset) == 2
    assert dataset[0]["question"] == "Q1"
    assert dataset[0]["answer"] == "rep-Q1"
    assert dataset[0]["contexts"] == ["ctx-Q1"]
    assert dataset[0]["ground_truth"] == "A1"


def test_build_eval_dataset_without_context(monkeypatch, tmp_path):
    qa_path = _write_qa(tmp_path, [{"question": "Q1", "ground_truth": "A1"}])
    monkeypatch.setattr(ragas_eval, "hybrid_search", lambda question, top_k=5: [])
    monkeypatch.setattr(ragas_eval, "generate_answer", lambda question, contexts: "rep")

    dataset = build_eval_dataset(qa_path)

    assert dataset[0]["contexts"] == ["(aucun contexte trouvé)"]
