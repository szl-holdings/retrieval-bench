"""
Dense retrieval adapter, hybrid RRF fusion, and cross-encoder rerank adapter.

Dense and rerank stages call real endpoints (OpenAI-compatible embeddings
API and a cross-encoder scoring API). If no endpoint is configured the lane
returns BLOCKED — never a fabricated embedding or score.
"""
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple


class EndpointUnavailable(Exception):
    pass


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise EndpointUnavailable(str(e))


class DenseRetriever:
    """Embeds corpus + queries through an OpenAI-compatible /v1/embeddings
    endpoint and ranks by cosine similarity computed in pure Python."""

    def __init__(self, endpoint: Optional[str], model: str, timeout: float = 30.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def is_configured(self) -> bool:
        return self.endpoint is not None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.endpoint:
            raise EndpointUnavailable("no embedding endpoint configured")
        url = self.endpoint.rstrip("/") + "/v1/embeddings"
        body = _post_json(url, {"model": self.model, "input": texts}, self.timeout)
        return [d["embedding"] for d in sorted(body["data"], key=lambda x: x["index"])]

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def index(self, doc_ids: List[str], texts: List[str]):
        self.doc_ids = list(doc_ids)
        self.doc_vecs = self._embed(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        qv = self._embed([query])[0]
        scored = [(did, self._cos(qv, dv)) for did, dv in zip(self.doc_ids, self.doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Standard RRF: score(d) = sum over lists of 1 / (k + rank_in_list)."""
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (docid, _score) in enumerate(ranked, start=1):
            fused[docid] = fused.get(docid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class CrossEncoderReranker:
    """Calls a rerank endpoint (TEI-style /rerank or compatible) to rescore
    a fixed candidate pool. The pool size is part of the fairness contract:
    comparing rerankers over different pool sizes is INVALID."""

    def __init__(self, endpoint: Optional[str], model: str, timeout: float = 60.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def is_configured(self) -> bool:
        return self.endpoint is not None

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """candidates: [(doc_id, doc_text)] -> [(doc_id, score)] sorted desc."""
        if not self.endpoint:
            raise EndpointUnavailable("no rerank endpoint configured")
        url = self.endpoint.rstrip("/") + "/rerank"
        body = _post_json(url, {
            "model": self.model,
            "query": query,
            "texts": [t for _, t in candidates],
        }, self.timeout)
        results = body.get("results", body if isinstance(body, list) else [])
        scored = [(candidates[r["index"]][0], r.get("relevance_score", r.get("score", 0.0)))
                  for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
