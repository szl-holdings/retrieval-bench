import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.bm25 import BM25Index, tokenize
from retrieval.metrics import recall_at_k, mrr_at_k, ndcg_at_k, evaluate
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.runner import BenchRunner

CORPUS = {
    "d1": "the cat sat on the mat",
    "d2": "dogs are loyal animals and great pets",
    "d3": "python is a programming language used for data science",
    "d4": "the mat was on the floor near the cat",
    "d5": "machine learning models require training data",
    "d6": "cats and dogs are common household pets",
}
QUERIES = {"q1": "cat mat", "q2": "programming language data", "q3": "pets animals"}
QRELS = {"q1": {"d1": 2, "d4": 1}, "q2": {"d3": 2, "d5": 1}, "q3": {"d2": 2, "d6": 1}}


def test_tokenize():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]

def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.index(list(CORPUS.keys()), list(CORPUS.values()))
    top = idx.search("cat mat", top_k=1)[0][0]
    assert top in ("d1", "d4")

def test_bm25_empty_index():
    idx = BM25Index()
    assert idx.search("anything", top_k=5) == []

def test_recall_perfect():
    run = {"q1": ["d1", "d4", "d2"], "q2": ["d3", "d5"], "q3": ["d2", "d6"]}
    binary = {"q1": {"d1", "d4"}, "q2": {"d3", "d5"}, "q3": {"d2", "d6"}}
    assert recall_at_k(run, binary, 2) == 1.0

def test_mrr_first_hit():
    run = {"q1": ["d1", "d4"]}
    binary = {"q1": {"d1"}}
    assert mrr_at_k(run, binary, 5) == 1.0

def test_mrr_second_hit():
    run = {"q1": ["dX", "d1"]}
    binary = {"q1": {"d1"}}
    assert mrr_at_k(run, binary, 5) == 0.5

def test_ndcg_perfect_ordering():
    run = {"q1": ["d1", "d4"]}
    graded = {"q1": {"d1": 2, "d4": 1}}
    assert abs(ndcg_at_k(run, graded, 2) - 1.0) < 1e-9

def test_rrf_fuses_rankings():
    list_a = [("d1", 10.0), ("d2", 5.0)]
    list_b = [("d2", 0.9), ("d1", 0.8)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [d for d, _ in fused]
    assert set(ids) == {"d1", "d2"}

def test_runner_sparse_measured():
    r = BenchRunner().run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    assert r.status == "MEASURED"
    assert r.metrics["ndcg@10"] == 1.0

def test_receipt_binds_query_set():
    runner = BenchRunner()
    a = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    changed = dict(QUERIES)
    changed["q1"] = "dog loyalty"
    b = runner.run_sparse(CORPUS, changed, QRELS, top_k=5)
    assert a.dataset_hash == b.dataset_hash
    assert a.qrels_hash == b.qrels_hash
    assert a.query_hash != b.query_hash

def test_runner_dense_blocked_without_endpoint():
    r = BenchRunner().run_dense(CORPUS, QUERIES, QRELS, None, "m", "rev")
    assert r.status == "BLOCKED"

def test_runner_hybrid_blocked_without_endpoint():
    r = BenchRunner().run_hybrid(CORPUS, QUERIES, QRELS, None, "m", "rev")
    assert r.status == "BLOCKED"

def test_runner_rerank_blocked_without_endpoint():
    r = BenchRunner().run_rerank(CORPUS, QUERIES, QRELS, {}, None, "m", "rev", 100)
    assert r.status == "BLOCKED"

def test_compare_rejects_pool_mismatch():
    runner = BenchRunner()
    a = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    b = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    a.config["pool_size"], b.config["pool_size"] = 100, 25
    assert runner.compare(a.run_id, b.run_id)["status"] == "INVALID"

def test_compare_rejects_query_mismatch():
    runner = BenchRunner()
    a = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    changed = dict(QUERIES)
    changed["q1"] = "dog loyalty"
    b = runner.run_sparse(CORPUS, changed, QRELS, top_k=5)
    result = runner.compare(a.run_id, b.run_id)
    assert result["status"] == "INVALID"
    assert "query set" in result["reason"]

def test_compare_valid_pair():
    runner = BenchRunner()
    a = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    b = runner.run_sparse(CORPUS, QUERIES, QRELS, top_k=5)
    result = runner.compare(a.run_id, b.run_id)
    assert result["status"] == "VALID"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"ALL {len(fns)} RETRIEVAL TESTS PASSED")
