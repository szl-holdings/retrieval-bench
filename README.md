# Retrieval Benchmark Plane

A tested benchmark lane for comparing retrieval and reranking stacks —
sparse (BM25), dense, hybrid (RRF), and cross-encoder rerank — with a
five-state honesty contract and SHA-256 evidence receipts.

Doctrine v11: no fabricated scores. Lanes without a configured endpoint
return BLOCKED with a reason. Comparisons across mismatched datasets,
qrels, or candidate pool sizes return INVALID.

## Verified behavior

- BM25 lane runs on any supplied corpus/queries/graded qrels and returns
  MEASURED with computed nDCG@k, MRR@k, MAP@k, Recall@k (k = 1, 5, 10).
- Dense / hybrid / rerank lanes return BLOCKED when no endpoint is set;
  the identical code path returns MEASURED once a real OpenAI-compatible
  /v1/embeddings or TEI-style /rerank endpoint is provided.
- Fairness gate rejects cross-pool-size and cross-dataset comparisons.
- 14 unit tests pass (python tests/test_retrieval.py).

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
python tests/test_retrieval.py   # no server needed for unit tests
```

## Files

- retrieval/bm25.py — pure-Python BM25 index + search
- retrieval/metrics.py — recall/MRR/MAP/nDCG
- retrieval/fusion.py — dense adapter, RRF fusion, cross-encoder adapter
- retrieval/runner.py — five-state runner + fairness gate + receipts
- api/main.py — FastAPI app wiring all lanes behind HTTP
- tests/test_retrieval.py — unit tests
- configs/ — CI workflow, Prometheus scrape config
