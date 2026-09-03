"""
Retrieval Benchmark Plane — FastAPI backend.

Endpoints:
  GET  /healthz
  POST /v1/run/sparse
  POST /v1/run/dense
  POST /v1/run/hybrid
  POST /v1/run/rerank
  POST /v1/compare
  GET  /v1/receipts
  GET  /v1/receipts/{run_id}
  GET  /metrics

Run: uvicorn api.main:app --host 0.0.0.0 --port 8081
"""
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.runner import BenchRunner

app = FastAPI(title="Retrieval Benchmark Plane", version="0.1.0")
RUNNER = BenchRunner()
FIRST_STAGE_RUNS: Dict[str, Dict[str, List[str]]] = {}


class CorpusQuerySet(BaseModel):
    corpus: Dict[str, str] = Field(..., description="doc_id -> text")
    queries: Dict[str, str] = Field(..., description="query_id -> text")
    qrels: Dict[str, Dict[str, int]] = Field(..., description="query_id -> {doc_id: grade}")


class SparseRequest(CorpusQuerySet):
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10


class DenseRequest(CorpusQuerySet):
    endpoint: Optional[str] = None
    model: str
    model_revision: str
    top_k: int = 10


class HybridRequest(CorpusQuerySet):
    dense_endpoint: Optional[str] = None
    model: str
    model_revision: str
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60
    top_k: int = 10


class RerankRequest(CorpusQuerySet):
    rerank_endpoint: Optional[str] = None
    model: str
    model_revision: str
    pool_size: int = Field(100, ge=1, le=1000)
    first_stage_run_id: str


class CompareRequest(BaseModel):
    run_id_a: str
    run_id_b: str


def _to_jsonable(receipt):
    return receipt.to_dict()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "n_receipts": len(RUNNER.receipts)}


@app.post("/v1/run/sparse")
def run_sparse(req: SparseRequest):
    r = RUNNER.run_sparse(req.corpus, req.queries, req.qrels, req.k1, req.b, req.top_k)
    if r.status == "MEASURED":
        # stash the ranked lists so a later rerank run can use them as first stage
        from retrieval.bm25 import BM25Index
        idx = BM25Index(k1=req.k1, b=req.b)
        idx.index(list(req.corpus.keys()), list(req.corpus.values()))
        FIRST_STAGE_RUNS[r.run_id] = {
            qid: [d for d, _ in idx.search(q, req.top_k)] for qid, q in req.queries.items()
        }
    return _to_jsonable(r)


@app.post("/v1/run/dense")
def run_dense(req: DenseRequest):
    r = RUNNER.run_dense(req.corpus, req.queries, req.qrels, req.endpoint,
                         req.model, req.model_revision, req.top_k)
    return _to_jsonable(r)


@app.post("/v1/run/hybrid")
def run_hybrid(req: HybridRequest):
    r = RUNNER.run_hybrid(req.corpus, req.queries, req.qrels, req.dense_endpoint,
                          req.model, req.model_revision, req.k1, req.b, req.rrf_k, req.top_k)
    return _to_jsonable(r)


@app.post("/v1/run/rerank")
def run_rerank(req: RerankRequest):
    first_stage = FIRST_STAGE_RUNS.get(req.first_stage_run_id)
    if first_stage is None:
        raise HTTPException(status_code=400,
                            detail="unknown first_stage_run_id; run /v1/run/sparse first")
    r = RUNNER.run_rerank(req.corpus, req.queries, req.qrels, first_stage,
                          req.rerank_endpoint, req.model, req.model_revision, req.pool_size)
    return _to_jsonable(r)


@app.post("/v1/compare")
def compare(req: CompareRequest):
    return RUNNER.compare(req.run_id_a, req.run_id_b)


@app.get("/v1/receipts")
def list_receipts():
    return {"receipts": [r.to_dict() for r in RUNNER.receipts]}


@app.get("/v1/receipts/{run_id}")
def get_receipt(run_id: str):
    r = next((x for x in RUNNER.receipts if x.run_id == run_id), None)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return r.to_dict()


@app.get("/metrics")
def metrics():
    by_status = {}
    for r in RUNNER.receipts:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    lines = [
        "# HELP retrieval_runs_total Benchmark runs by status",
        "# TYPE retrieval_runs_total gauge",
    ]
    for status, n in sorted(by_status.items()):
        lines.append(f'retrieval_runs_total{{status="{status}"}} {n}')
    lines.append(f"retrieval_receipts_total {len(RUNNER.receipts)}")
    return "\n".join(lines) + "\n"
