"""Only bound finite reranker scores may produce measured retrieval receipts."""
from unittest.mock import Mock

import pytest

from retrieval import fusion
from retrieval.runner import BenchRunner


ENDPOINT = "https://reranker.example.test/rerank"
CORPUS = {"a": "first document", "b": "second document"}
QUERIES = {"q": "query"}
QRELS = {"q": {"b": 1}}


def run(runner=None, queries=None):
    return (runner or BenchRunner()).run_rerank(
        CORPUS, queries or QUERIES, QRELS, {"q": ["a", "b"], "r": ["a", "b"]},
        ENDPOINT, "model", "immutable-revision", 2,
    )


@pytest.mark.parametrize("entry", [
    {"index": -1, "score": 0.9},
    {"index": True, "score": 0.9},
    {"index": False, "score": 0.9},
    {"index": 0.0, "score": 0.9},
    {"index": 2, "score": 0.9},
    {"score": 0.9},
    {"index": 1},
    {"index": 1, "score": True},
    {"index": 1, "score": "0.9"},
    {"index": 1, "score": None},
    {"index": 1, "score": float("nan")},
    {"index": 1, "score": float("inf")},
    {"index": 1, "score": -float("inf")},
    {"index": 1, "score": 10 ** 400},
    {"index": 1, "relevance_score": None, "score": 0.9},
    {"index": 1, "relevance_score": float("nan"), "score": 0.9},
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_unbound_or_invalid_scores_never_produce_measured_receipts(monkeypatch, entry, wrapped):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    results = [entry]
    post = Mock(return_value={"results": results} if wrapped else results)
    monkeypatch.setattr(fusion, "_post_json", post)
    receipt = run()
    assert receipt.status == "FAILED"
    assert receipt.metrics is None
    assert receipt.result_hash is None
    post.assert_called_once()


@pytest.mark.parametrize("body", [{}, {"results": None}, {"results": {}}, None, 1, "results"])
def test_response_requires_an_explicit_results_list(monkeypatch, body):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    monkeypatch.setattr(fusion, "_post_json", Mock(return_value=body))
    receipt = run()
    assert receipt.status == "FAILED"
    assert receipt.metrics is None
    assert receipt.result_hash is None


def test_adapter_rejects_duplicate_candidate_indices(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    monkeypatch.setattr(fusion, "_post_json", Mock(return_value=[
        {"index": 1, "score": 0.9}, {"index": 1, "score": 0.8},
    ]))
    with pytest.raises(ValueError, match="indices"):
        fusion.CrossEncoderReranker(ENDPOINT, "model").rerank("query", list(CORPUS.items()))


@pytest.mark.parametrize("wrapped", [False, True])
def test_valid_indices_raw_scores_and_existing_precedence_are_preserved(monkeypatch, wrapped):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    results = [{"index": 1, "score": -2.5},
               {"index": 0, "relevance_score": 8, "score": 100, "text": "metadata"}]
    post = Mock(return_value={"results": results} if wrapped else results)
    monkeypatch.setattr(fusion, "_post_json", post)
    assert fusion.CrossEncoderReranker(ENDPOINT, "model").rerank(
        "query", list(CORPUS.items()),
    ) == [("a", 8), ("b", -2.5)]
    assert post.call_args.args[1]["texts"] == list(CORPUS.values())


@pytest.mark.parametrize("results", [[], [{"index": 1, "score": 0.9}]])
def test_valid_partial_rankings_remain_supported(monkeypatch, results):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    monkeypatch.setattr(fusion, "_post_json", Mock(return_value=results))
    receipt = run()
    assert receipt.status == "MEASURED"
    assert receipt.metrics is not None
    assert receipt.result_hash is not None


def test_invalid_later_query_discards_partial_run_and_next_run_can_succeed(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    valid = [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]
    monkeypatch.setattr(fusion, "_post_json", Mock(side_effect=[valid, [{"index": -1, "score": 1}], valid]))
    runner = BenchRunner()
    failed = run(runner, {"q": "query", "r": "another query"})
    assert failed.status == "FAILED"
    assert failed.metrics is None
    assert failed.result_hash is None
    measured = run(runner)
    assert measured.status == "MEASURED"
    assert measured.metrics["ndcg@1"] == 1.0
    assert measured.result_hash is not None
