"""Duplicate document identities must not become admitted retrieval scores."""
import math

import pytest

from retrieval.metrics import evaluate, map_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from retrieval.runner import BenchRunner


METRICS = [recall_at_k, mrr_at_k, map_at_k, ndcg_at_k]


def judgments(metric):
    return {"q": {"a": 1}} if metric is ndcg_at_k else {"q": {"a"}}


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("repeated", ["a", "unjudged"])
def test_repeated_doc_ids_are_rejected_even_when_not_relevant(metric, repeated):
    with pytest.raises(ValueError, match="duplicate document"):
        metric({"q": [repeated, repeated, "a"]}, judgments(metric), 2)


@pytest.mark.parametrize("metric", METRICS)
def test_cutoff_uses_original_positions_and_ignores_tail(metric):
    run = {"q": ["unjudged", "a", "a"]}
    assert metric(run, judgments(metric), 1) == 0.0
    expected = 1.0 / math.log2(3) if metric is ndcg_at_k else 1.0 if metric is recall_at_k else 0.5
    assert metric(run, judgments(metric), 2) == pytest.approx(expected)
    with pytest.raises(ValueError, match="duplicate document"):
        metric(run, judgments(metric), 3)


def test_valid_rankings_keep_existing_binary_and_graded_scores():
    scores = evaluate({"q": ["unjudged", "a", "b"]}, {"q": {"a": 3, "b": 1}}, ks=(2, 3))
    assert scores["recall@2"] == 0.5
    assert scores["mrr@2"] == 0.5
    assert scores["map@2"] == 0.25
    assert scores["ndcg@2"] == pytest.approx((3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert scores["recall@3"] == 1.0
    assert scores["mrr@3"] == 0.5
    assert scores["map@3"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert scores["ndcg@3"] == pytest.approx((3 / math.log2(3) + 0.5) / (3 + 1 / math.log2(3)))


def test_existing_nonrelevant_and_negative_grade_handling_is_preserved():
    scores = evaluate(
        {"q": ["negative", "positive", "unjudged"]},
        {"q": {"negative": -1, "positive": 2, "zero": 0}},
        ks=(3,),
    )
    assert scores["recall@3"] == 1.0
    assert scores["mrr@3"] == 0.5
    assert scores["map@3"] == 0.5
    assert scores["ndcg@3"] == pytest.approx((-1 + 2 / math.log2(3)) / 1.5)


def test_document_identity_is_scoped_to_each_query():
    scores = evaluate({"q": ["a"], "r": ["a"]}, {"q": {"a": 1}, "r": {"a": 1}}, ks=(1,))
    assert set(scores.values()) == {1.0}


def test_evaluate_returns_no_partial_metrics_for_duplicate_ranking():
    with pytest.raises(ValueError, match="duplicate document"):
        evaluate({"q": ["a", "a"]}, {"q": {"a": 1}}, ks=(1, 2))


def test_runner_records_duplicate_reranker_output_as_failed_without_metrics(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BENCH_ENDPOINTS", "https://rerank.test")
    monkeypatch.setattr(
        "retrieval.runner.CrossEncoderReranker.rerank",
        lambda self, query, candidates: [("a", 0.9), ("a", 0.8)],
    )
    receipt = BenchRunner().run_rerank(
        {"a": "document"}, {"q": "query"}, {"q": {"a": 1}},
        {"q": ["a"]}, "https://rerank.test", "test-model", "test-revision", 1,
    )
    assert receipt.status == "FAILED"
    assert receipt.metrics is None
    assert receipt.result_hash is None
    assert "duplicate document" in receipt.detail
