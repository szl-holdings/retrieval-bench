"""Missing benchmark inputs and duplicate ranks must not become measurements."""
import pytest
from unittest.mock import Mock, patch

from retrieval.metrics import evaluate, map_at_k, ndcg_at_k
from retrieval.runner import BenchRunner


CORPUS = {"d": "text"}
QUERIES = {"q": "text"}
QRELS = {"q": {"d": 1}}


@pytest.mark.parametrize("missing", ["corpus", "queries", "qrels"])
@pytest.mark.parametrize("lane", ["sparse", "dense", "hybrid", "rerank", "multivector"])
def test_missing_inputs_block_before_producer_work(missing, lane):
    inputs = {"corpus": CORPUS, "queries": QUERIES, "qrels": QRELS}
    inputs[missing] = {}
    runner = BenchRunner()
    producer = Mock(side_effect=AssertionError("producer must not be constructed"))
    with patch("retrieval.runner.BM25Index", producer), \
         patch("retrieval.runner.DenseRetriever", producer), \
         patch("retrieval.runner.CrossEncoderReranker", producer), \
         patch("retrieval.runner.LateInteractionIndex", producer):
        if lane == "sparse":
            out = runner.run_sparse(**inputs)
        elif lane == "dense":
            out = runner.run_dense(**inputs, endpoint="https://fixture.invalid", model="m", model_revision="rev")
        elif lane == "hybrid":
            out = runner.run_hybrid(**inputs, dense_endpoint="https://fixture.invalid", model="m", model_revision="rev")
        elif lane == "rerank":
            out = runner.run_rerank(**inputs, first_stage_run={}, rerank_endpoint="https://fixture.invalid",
                                    model="m", model_revision="rev", pool_size=1)
        else:
            out = runner.run_multivector(**inputs, encoder=Mock())
    assert out.status == "BLOCKED"
    assert out.lane == {"sparse": "sparse_bm25", "hybrid": "hybrid"}.get(lane, lane)
    assert missing in out.detail
    assert out.metrics is None and out.result_hash is None
    assert runner.receipts == [out]
    producer.assert_not_called()


@pytest.mark.parametrize("metric", [evaluate, map_at_k, ndcg_at_k])
def test_duplicate_rank_cannot_inflate_metric(metric):
    qrels = {"q": {"d": 1}} if metric != map_at_k else {"q": {"d"}}
    with pytest.raises(ValueError, match="duplicate"):
        if metric == evaluate:
            metric({"q": ["d", "d"]}, qrels, ks=(2,))
        else:
            metric({"q": ["d", "d"]}, qrels, 2)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate({"q": ["d"]}, QRELS, ks=(k,))


def test_zero_relevance_and_no_hits_remain_valid_zero_scores():
    assert set(evaluate({"q": ["d"]}, {"q": {"d": 0}}, ks=(1,)).values()) == {0.0}
    assert set(evaluate({"q": []}, QRELS, ks=(1,)).values()) == {0.0}
    # A nonempty judgment cohort with an unjudged query is preserved.
    assert set(evaluate({"q": ["d"]}, {"q": {}}, ks=(1,)).values()) == {0.0}
    out = BenchRunner().run_sparse(CORPUS, QUERIES, {"q": {"d": 0}})
    assert out.status == "MEASURED"
    assert set(out.metrics.values()) == {0.0}


def test_valid_ranked_metrics_are_unchanged():
    out = evaluate({"q": ["d", "other"]}, QRELS, ks=(2,))
    assert set(out.values()) == {1.0}
