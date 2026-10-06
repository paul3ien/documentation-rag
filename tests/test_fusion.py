from app.retrieval.hybrid import reciprocal_rank_fusion


def test_document_present_in_both_rankings_wins():
    fused = reciprocal_rank_fusion([[10, 20], [20, 30]], k=60)

    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == 20
    assert set(ids) == {10, 20, 30}


def test_scores_are_sorted_descending():
    fused = reciprocal_rank_fusion([[1, 2, 3]], k=60)

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_empty_rankings_give_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []
