"""
Retrieval benchmark runner with five-state run vocabulary:

- MEASURED: completed with verifiable results
- BLOCKED: missing model, dataset, endpoint, or credentials
- INVALID: fairness/integrity check failed (e.g. mismatched candidate pool sizes)
- FAILED: execution started but errored
- PROMOTED: passed effectiveness + operational gates (set only by compare/promote)

Receipts bind dataset hash, query-set hash, qrels hash, model revision, config,
and result hash. Query provenance is part of the evidence contract because changing
query text can change retrieval metrics even when corpus and qrels are unchanged.

v0.2 adds the multivector (late-interaction) lane, kept strictly separate
from single-vector comparisons.
"""
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from .bm25 import BM25Index
from .fusion import DenseRetriever, CrossEncoderReranker, reciprocal_rank_fusion, EndpointUnavailable
from .multivector import LateInteractionIndex, EncoderUnavailable
from .metrics import evaluate


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@dataclass
class RunReceipt:
    run_id: str
    lane: str
    status: str
    dataset_hash: str
    query_hash: str
    qrels_hash: str
    model_revision: str
    config: dict
    metrics: Optional[dict]
    result_hash: Optional[str]
    timestamp: float
    detail: str = ""

    def to_dict(self):
        return self.__dict__.copy()


class BenchRunner:
    def __init__(self):
        self.receipts: List[RunReceipt] = []

    def _record(self, lane, status, corpus, queries, qrels, model_rev, config, metrics=None, detail=""):
        result_hash = _hash(metrics) if metrics is not None else None
        r = RunReceipt(
            run_id=str(uuid.uuid4()),
            lane=lane,
            status=status,
            dataset_hash=_hash(corpus),
            query_hash=_hash(queries),
            qrels_hash=_hash(qrels),
            model_revision=model_rev,
            config=config,
            metrics=metrics,
            result_hash=result_hash,
            timestamp=time.time(),
            detail=detail,
        )
        self.receipts.append(r)
        return r

    def run_sparse(self, corpus: Dict[str, str], queries: Dict[str, str],
                   qrels: Dict[str, Dict[str, int]], k1=1.5, b=0.75, top_k=10):
        try:
            idx = BM25Index(k1=k1, b=b)
            idx.index(list(corpus.keys()), list(corpus.values()))
            run = {qid: [d for d, _ in idx.search(q, top_k)] for qid, q in queries.items()}
            metrics = evaluate(run, qrels, ks=(1, 5, 10))
            return self._record("sparse_bm25", "MEASURED", corpus, queries, qrels,
                                f"bm25-k1={k1}-b={b}", {"top_k": top_k}, metrics)
        except Exception as e:
            return self._record("sparse_bm25", "FAILED", corpus, queries, qrels,
                                f"bm25-k1={k1}-b={b}", {"top_k": top_k}, None, repr(e))

    def run_dense(self, corpus: Dict[str, str], queries: Dict[str, str],
                  qrels: Dict[str, Dict[str, int]], endpoint: Optional[str],
                  model: str, model_revision: str, top_k=10):
        retriever = DenseRetriever(endpoint, model)
        config = {"top_k": top_k, "endpoint_set": endpoint is not None}
        if not retriever.is_configured():
            return self._record("dense", "BLOCKED", corpus, queries, qrels, model_revision, config,
                                None, "no embedding endpoint configured")
        try:
            retriever.index(list(corpus.keys()), list(corpus.values()))
            run = {qid: [d for d, _ in retriever.search(q, top_k)] for qid, q in queries.items()}
            metrics = evaluate(run, qrels, ks=(1, 5, 10))
            return self._record("dense", "MEASURED", corpus, queries, qrels, model_revision, config, metrics)
        except EndpointUnavailable as e:
            return self._record("dense", "BLOCKED", corpus, queries, qrels, model_revision, config, None, str(e))
        except Exception as e:
            return self._record("dense", "FAILED", corpus, queries, qrels, model_revision, config, None, repr(e))

    def run_hybrid(self, corpus, queries, qrels, dense_endpoint, model, model_revision,
                   k1=1.5, b=0.75, rrf_k=60, top_k=10):
        sparse_receipt = self.run_sparse(corpus, queries, qrels, k1, b, top_k)
        if sparse_receipt.status != "MEASURED":
            return self._record("hybrid", "FAILED", corpus, queries, qrels, model_revision,
                                {"stage": "sparse"}, None, "sparse stage failed")

        dense = DenseRetriever(dense_endpoint, model)
        config = {"rrf_k": rrf_k, "top_k": top_k, "endpoint_set": dense_endpoint is not None}
        if not dense.is_configured():
            return self._record("hybrid", "BLOCKED", corpus, queries, qrels, model_revision, config,
                                None, "dense endpoint unavailable — hybrid fusion impossible")
        try:
            idx = BM25Index(k1=k1, b=b)
            idx.index(list(corpus.keys()), list(corpus.values()))
            dense.index(list(corpus.keys()), list(corpus.values()))
            run = {}
            for qid, q in queries.items():
                s_list = idx.search(q, top_k)
                d_list = dense.search(q, top_k)
                fused = reciprocal_rank_fusion([s_list, d_list], k=rrf_k)
                run[qid] = [d for d, _ in fused[:top_k]]
            metrics = evaluate(run, qrels, ks=(1, 5, 10))
            return self._record("hybrid_rrf", "MEASURED", corpus, queries, qrels, model_revision, config, metrics)
        except EndpointUnavailable as e:
            return self._record("hybrid_rrf", "BLOCKED", corpus, queries, qrels, model_revision, config, None, str(e))
        except Exception as e:
            return self._record("hybrid_rrf", "FAILED", corpus, queries, qrels, model_revision, config, None, repr(e))

    def run_rerank(self, corpus, queries, qrels, first_stage_run: Dict[str, List[str]],
                   rerank_endpoint: Optional[str], model: str, model_revision: str,
                   pool_size: int):
        """Fairness contract: pool_size is recorded in config; comparisons across
        different pool sizes must be marked INVALID by the comparator."""
        reranker = CrossEncoderReranker(rerank_endpoint, model)
        config = {"pool_size": pool_size, "endpoint_set": rerank_endpoint is not None}
        if not reranker.is_configured():
            return self._record("rerank", "BLOCKED", corpus, queries, qrels, model_revision, config,
                                None, "no rerank endpoint configured")
        try:
            run = {}
            for qid, q in queries.items():
                pool = first_stage_run.get(qid, [])[:pool_size]
                candidates = [(d, corpus[d]) for d in pool if d in corpus]
                ranked = reranker.rerank(q, candidates)
                run[qid] = [d for d, _ in ranked]
            metrics = evaluate(run, qrels, ks=(1, 5, 10))
            return self._record("rerank", "MEASURED", corpus, queries, qrels, model_revision, config, metrics)
        except EndpointUnavailable as e:
            return self._record("rerank", "BLOCKED", corpus, queries, qrels, model_revision, config, None, str(e))
        except Exception as e:
            return self._record("rerank", "FAILED", corpus, queries, qrels, model_revision, config, None, repr(e))

    def run_multivector(self, corpus: Dict[str, str], queries: Dict[str, str],
                        qrels: Dict[str, Dict[str, int]], encoder=None,
                        model_revision: str = "unconfigured", top_k: int = 10):
        """Late-interaction lane (ColBERT-style MaxSim over token vectors).

        Kept strictly separate from single-vector lanes: index cost and memory
        differ by an order of magnitude, so cross-family comparisons are INVALID.
        Without a token-level encoder the lane is BLOCKED, never simulated.
        """
        idx = LateInteractionIndex(encoder)
        config = {"top_k": top_k, "encoder_set": encoder is not None,
                  "family": "multivector"}
        if not idx.is_configured():
            return self._record("multivector", "BLOCKED", corpus, queries, qrels, model_revision,
                                config, None, "no token-level encoder configured")
        try:
            idx.index(list(corpus.keys()), list(corpus.values()))
            run = {qid: [d for d, _ in idx.search(q, top_k)] for qid, q in queries.items()}
            metrics = evaluate(run, qrels, ks=(1, 5, 10))
            return self._record("multivector", "MEASURED", corpus, queries, qrels, model_revision,
                                config, metrics)
        except EncoderUnavailable as e:
            return self._record("multivector", "BLOCKED", corpus, queries, qrels, model_revision,
                                config, None, str(e))
        except Exception as e:
            return self._record("multivector", "FAILED", corpus, queries, qrels, model_revision,
                                config, None, repr(e))

    def compare(self, receipt_a_id: str, receipt_b_id: str) -> dict:
        """Fairness gate: two runs may only be compared if dataset, query set,
        qrels, family, and candidate pool sizes match. Otherwise the comparison
        itself is INVALID."""
        a = next((r for r in self.receipts if r.run_id == receipt_a_id), None)
        b = next((r for r in self.receipts if r.run_id == receipt_b_id), None)
        if a is None or b is None:
            return {"status": "INVALID", "reason": "unknown run id"}
        if a.status != "MEASURED" or b.status != "MEASURED":
            return {"status": "INVALID", "reason": "both runs must be MEASURED to compare"}
        if a.dataset_hash != b.dataset_hash or a.query_hash != b.query_hash or a.qrels_hash != b.qrels_hash:
            return {"status": "INVALID", "reason": "dataset, query set, or qrels mismatch"}
        fa, fb = a.config.get("family", "single-vector"), b.config.get("family", "single-vector")
        if fa != fb:
            return {"status": "INVALID",
                    "reason": f"retrieval family mismatch: {fa} vs {fb} — cross-family comparison is not evidence"}
        pa, pb = a.config.get("pool_size"), b.config.get("pool_size")
        if pa is not None and pb is not None and pa != pb:
            return {"status": "INVALID", "reason": f"candidate pool size mismatch: {pa} vs {pb}"}
        winner = a if (a.metrics or {}).get("ndcg@10", 0) >= (b.metrics or {}).get("ndcg@10", 0) else b
        return {"status": "VALID", "winner_run_id": winner.run_id,
                "winner_lane": winner.lane, "winner_ndcg@10": winner.metrics.get("ndcg@10")}
