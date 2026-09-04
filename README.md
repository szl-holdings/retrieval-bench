# Retrieval Benchmark Plane

A tested benchmark lane for comparing retrieval and reranking stacks —
sparse (BM25), dense, hybrid (RRF), cross-encoder rerank, and multivector
late interaction — with a five-state honesty contract and SHA-256 evidence
receipts.

Doctrine v11: no fabricated scores. Lanes without a configured endpoint
return BLOCKED with a reason. Comparisons across mismatched datasets,
qrels, candidate pool sizes, or retrieval families return INVALID.

## Verified behavior

- BM25 lane runs on any supplied corpus/queries/graded qrels and returns
  MEASURED with computed nDCG@k, MRR@k, MAP@k, Recall@k (k = 1, 5, 10).
- Dense / hybrid / rerank lanes return BLOCKED when no endpoint is set;
  the identical code path returns MEASURED once a real OpenAI-compatible
  /v1/embeddings or TEI-style /rerank endpoint is provided.
- Multivector lane (v0.2): ColBERT-style MaxSim over token-level vectors.
  Without a token encoder it returns BLOCKED; with one, it MEASURES. It is
  never compared against single-vector lanes — cross-family is INVALID.
- Fairness gate rejects cross-pool-size, cross-dataset, and cross-family
  comparisons.
- 23 unit tests pass (tests/test_retrieval.py + tests/test_multivector.py).

## Five-state run vocabulary

| State | Meaning |
|---|---|
| MEASURED | Run completed with verifiable, computed metrics |
| BLOCKED | Missing model, dataset, endpoint, or credentials |
| INVALID | Fairness/integrity check failed |
| FAILED | Execution started but errored |
| PROMOTED | Passed effectiveness + operational gates |

## Pipeline

1. Sparse — pure-Python Okapi BM25 (k1, b tunable), zero dependencies.
2. Dense — adapter for any OpenAI-compatible /v1/embeddings endpoint,
   cosine similarity in pure Python.
3. Hybrid — Reciprocal Rank Fusion over sparse + dense lists (rrf_k).
4. Rerank — cross-encoder adapter (TEI-style /rerank) over a fixed
   first-stage candidate pool; pool size recorded in the receipt.
5. Multivector — late-interaction MaxSim over token-level vectors via a
   caller-supplied encoder (e.g. a ColBERT endpoint on the owner node).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /healthz | Liveness + receipt count |
| POST | /v1/run/sparse | Run BM25 lane |
| POST | /v1/run/dense | Run dense lane |
| POST | /v1/run/hybrid | Run RRF hybrid lane |
| POST | /v1/run/rerank | Rerank a first-stage candidate pool |
| POST | /v1/compare | Fairness-gated comparison |
| GET | /v1/receipts | List evidence receipts |
| GET | /v1/receipts/{run_id} | Fetch one receipt |
| GET | /metrics | Prometheus exposition |

## Quick start

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8081
python tests/test_retrieval.py && python tests/test_multivector.py
```

## Files

- retrieval/bm25.py — pure-Python BM25 index + search
- retrieval/metrics.py — recall/MRR/MAP/nDCG
- retrieval/fusion.py — dense adapter, RRF fusion, cross-encoder adapter
- retrieval/multivector.py — late-interaction MaxSim index (v0.2)
- retrieval/runner.py — five-state runner + fairness gate + receipts
- api/main.py — FastAPI app wiring the single-vector lanes behind HTTP
- tests/ — 23 unit tests
- configs/ — reference configs
- .github/workflows/ci.yml — CI on push/PR (both suites, Python 3.11/3.12)

Doctrine v11. Apache-2.0. Λ = Conjecture 1 (advisory).
