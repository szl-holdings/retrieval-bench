"""Admitted sparse BM25 seed receipt must verify on the hash chain."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from verifier import verify  # noqa: E402

from retrieval.runner import BenchRunner


def test_seed_receipt_chain_has_one_measured():
    paths = sorted((ROOT / "receipts").glob("*.json"))
    errors, measured = verify([str(p) for p in paths])
    assert errors == []
    assert any(p.endswith("001-sparse-bm25-seed-v1.json") for p in measured)


def test_fixture_replays_to_same_metric_hashes():
    fix = ROOT / "fixtures" / "sparse-bm25-seed-v1"
    corpus = json.loads((fix / "corpus.json").read_text())
    queries = json.loads((fix / "queries.json").read_text())
    qrels = json.loads((fix / "qrels.json").read_text())
    out = BenchRunner().run_sparse(corpus, queries, qrels)
    receipt = json.loads((ROOT / "receipts" / "001-sparse-bm25-seed-v1.json").read_text())
    assert out.status == "MEASURED"
    assert out.dataset_hash == receipt["metrics"]["dataset_hash"]
    assert out.query_hash == receipt["metrics"]["query_hash"]
    assert out.qrels_hash == receipt["metrics"]["qrels_hash"]
    assert out.result_hash == receipt["metrics"]["result_hash"]
