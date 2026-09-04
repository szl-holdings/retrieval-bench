import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.multivector import (LateInteractionIndex, maxsim,
                                   EncoderUnavailable)
from retrieval.runner import BenchRunner

# Deterministic fixture vectors (2D for hand-verifiable math).
def enc(text):
    table = {
        "q":      [[1.0, 0.0], [0.0, 1.0]],
        "doc_a":  [[1.0, 0.0], [0.0, 0.9]],
        "doc_b":  [[0.0, 1.0], [-1.0, 0.0]],
    }
    return table[text]

CORPUS = {"a": "doc_a", "b": "doc_b"}
QUERIES = {"q": "q"}
QRELS = {"q": {"a": 2}}


def test_maxsim_prefers_aligned_document():
    q = enc("q")
    assert maxsim(q, enc("doc_a")) > maxsim(q, enc("doc_b"))

def test_maxsim_empty_inputs_zero():
    assert maxsim([], [[1.0]]) == 0.0
    assert maxsim([[1.0]], []) == 0.0

def test_maxsim_hand_computed_value():
    q = [[1.0, 0.0], [0.0, 1.0]]
    d = [[1.0, 0.0], [0.0, 0.9]]
    assert abs(maxsim(q, d) - 2.0) < 1e-9

def test_index_requires_encoder():
    idx = LateInteractionIndex(None)
    assert not idx.is_configured()
    try:
        idx.index(["a"], ["doc_a"])
        assert False, "should raise"
    except EncoderUnavailable:
        pass

def test_index_and_search_with_encoder():
    idx = LateInteractionIndex(enc)
    idx.index(["a", "b"], ["doc_a", "doc_b"])
    results = idx.search("q", top_k=2)
    assert results[0][0] == "a"
    assert len(results) == 2

def test_search_without_encoder_blocked():
    idx = LateInteractionIndex(None)
    try:
        idx.search("q")
        assert False, "should raise"
    except EncoderUnavailable:
        pass

def test_runner_multivector_blocked_without_encoder():
    r = BenchRunner().run_multivector(CORPUS, QUERIES, QRELS, encoder=None)
    assert r.status == "BLOCKED"
    assert r.config["family"] == "multivector"

def test_runner_multivector_measured_with_encoder():
    r = BenchRunner().run_multivector(CORPUS, QUERIES, QRELS, encoder=enc,
                                      model_revision="fixture-encoder-v1")
    assert r.status == "MEASURED"
    assert r.metrics["ndcg@10"] == 1.0

def test_compare_rejects_cross_family():
    runner = BenchRunner()
    sparse = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    mv = runner.run_multivector(CORPUS, QUERIES, QRELS, encoder=enc,
                                model_revision="fixture-encoder-v1")
    assert sparse.status == "MEASURED" and mv.status == "MEASURED"
    result = runner.compare(sparse.run_id, mv.run_id)
    assert result["status"] == "INVALID"
    assert "family" in result["reason"]


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"ALL {len(fns)} MULTIVECTOR TESTS PASSED")
