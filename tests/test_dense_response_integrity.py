"""Malformed embedding evidence must not become a measured retrieval score."""
import math
from unittest.mock import Mock

import pytest

from retrieval import fusion
from retrieval.runner import BenchRunner


ENDPOINT = "https://embedding.example.test/v1/embeddings"


def response(*vectors):
    return {"data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]}


@pytest.mark.parametrize("body", [
    response([1., 0.]),
    response([1., 0.], [0., 1.], [1., 1.]),
    {"data": [{"index": 0, "embedding": [1., 0.]}, {"index": 0, "embedding": [0., 1.]}]},
    {"data": [{"index": -1, "embedding": [1., 0.]}, {"index": 0, "embedding": [0., 1.]}]},
    {"data": [{"index": 0, "embedding": [1., 0.]}, {"index": 2, "embedding": [0., 1.]}]},
    {"data": [{"index": False, "embedding": [1., 0.]}, {"index": 1, "embedding": [0., 1.]}]},
    {"data": [{"index": 0., "embedding": [1., 0.]}, {"index": 1, "embedding": [0., 1.]}]},
    response([float("nan"), 0.], [0., 1.]),
    response([float("inf"), 0.], [0., 1.]),
    response([], []),
    response([True, 0.], [0., 1.]),
    response(["1", 0.], [0., 1.]),
    response([1., 0.], [0.]),
])
@pytest.mark.parametrize("lane", ["dense", "hybrid"])
def test_invalid_document_embeddings_never_produce_measured_receipts(monkeypatch, body, lane):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    post = Mock(side_effect=[body, response([1., 0.])])
    monkeypatch.setattr(fusion, "_post_json", post)
    run = BenchRunner()
    call = run.run_dense if lane == "dense" else run.run_hybrid
    receipt = call({"a": "first", "b": "second"}, {"q": "query"},
                   {"q": {"a": 1}}, ENDPOINT, "model", "revision")
    assert receipt.status == "FAILED"
    assert receipt.metrics is None
    assert receipt.result_hash is None
    assert post.call_count == 1  # Refuse the malformed corpus before querying.


@pytest.mark.parametrize("query", [response([1., 0., 0.]), response([1.]), response()])
def test_query_embeddings_must_match_index_dimension_and_count(monkeypatch, query):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    monkeypatch.setattr(fusion, "_post_json", Mock(side_effect=[response([1., 0.]), query]))
    receipt = BenchRunner().run_dense({"a": "first"}, {"q": "query"}, {"q": {"a": 1}},
                                      ENDPOINT, "model", "revision")
    assert receipt.status == "FAILED"
    assert receipt.metrics is None
    assert receipt.result_hash is None


def test_response_indices_preserve_document_identity_even_when_out_of_order(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    vectors = response([0., 1.], [1., 0.])
    vectors["data"].reverse()
    monkeypatch.setattr(fusion, "_post_json", Mock(side_effect=[vectors, response([1., 0.])]))
    retriever = fusion.DenseRetriever(ENDPOINT, "model")
    retriever.index(["a", "b"], ["first", "second"])
    assert retriever.search("query") == [("b", 1.0), ("a", 0.0)]


def test_index_count_mismatch_is_rejected_before_request(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    post = Mock(return_value=response([1., 0.]))
    monkeypatch.setattr(fusion, "_post_json", post)
    with pytest.raises(ValueError, match="equal length"):
        fusion.DenseRetriever(ENDPOINT, "model").index(["a", "b"], ["first"])
    post.assert_not_called()


def test_failed_reindex_preserves_previously_valid_index(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, ENDPOINT)
    monkeypatch.setattr(fusion, "_post_json", Mock(side_effect=[response([1., 0.]), response()]))
    retriever = fusion.DenseRetriever(ENDPOINT, "model")
    retriever.index(["a"], ["first"])
    with pytest.raises(ValueError):
        retriever.index(["b"], ["second"])
    assert retriever.doc_ids == ["a"]
    assert retriever.doc_vecs == [[1., 0.]]


@pytest.mark.parametrize("scale", [1., 1e308, 1e-300])
def test_cosine_is_finite_and_scale_invariant_for_valid_vectors(scale):
    a, b = [scale, scale], [scale, 0.]
    score = fusion.DenseRetriever._cos(a, b)
    assert math.isfinite(score)
    assert score == pytest.approx(1 / math.sqrt(2))


def test_zero_vector_keeps_existing_zero_similarity():
    assert fusion.DenseRetriever._cos([0., 0.], [1., 0.]) == 0.0
