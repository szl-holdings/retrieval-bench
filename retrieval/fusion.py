"""
Dense retrieval adapter, hybrid RRF fusion, and cross-encoder rerank adapter.

Dense and rerank stages call real endpoints (OpenAI-compatible embeddings
API and a cross-encoder scoring API). If no endpoint is configured the lane
returns BLOCKED — never a fabricated embedding or score.
"""
import json
import math
import os
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit


class EndpointUnavailable(Exception):
    pass


ENDPOINT_ALLOWLIST_ENV = "RETRIEVAL_BENCH_ENDPOINTS"


def allowed_endpoints() -> Tuple[str, ...]:
    """Operator-configured embedding/rerank endpoints, e.g. ``https://embed.internal:8443/v1/embeddings``.

    Comma-separated absolute http(s) URLs. Empty means no remote endpoint is admitted and every
    dense/rerank lane returns BLOCKED — the plane never issues a request to a URL a caller chose
    (CodeQL py/full-ssrf).
    """
    raw = os.environ.get(ENDPOINT_ALLOWLIST_ENV, "")
    endpoints = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = urlsplit(item)
        if parts.scheme in ("http", "https") and parts.hostname and not parts.username and not parts.password:
            endpoints.append(item)
    return tuple(endpoints)


def admit_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Return the operator-configured endpoint that exactly equals the request's, else None.

    A request can only *select* among configured endpoints; the string handed to the HTTP
    client is the operator's own constant, never the caller's bytes.
    """
    if not endpoint:
        return None
    wanted = str(endpoint).strip()
    for configured in allowed_endpoints():
        if configured == wanted:
            return configured
    return None


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The configured URL is the whole admission boundary, not only a first hop.
        raise EndpointUnavailable("endpoint redirects are not allowed; configure the final URL")


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict | list:
    admitted = admit_endpoint(url)
    if admitted is None:
        raise EndpointUnavailable(
            f"endpoint origin is not on the operator allowlist ({ENDPOINT_ALLOWLIST_ENV})")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(admitted, data=data, headers={"Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(_RejectRedirects())
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise EndpointUnavailable(str(e))


class DenseRetriever:
    """Embeds corpus + queries through an OpenAI-compatible /v1/embeddings
    endpoint and ranks by cosine similarity computed in pure Python."""

    def __init__(self, endpoint: Optional[str], model: str, timeout: float = 30.0):
        # Only an operator-allowlisted origin is ever stored; anything else is None -> BLOCKED.
        self.endpoint = admit_endpoint(endpoint)
        self.model = model
        self.timeout = timeout

    def is_configured(self) -> bool:
        return self.endpoint is not None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.endpoint:
            raise EndpointUnavailable("no embedding endpoint configured")
        body = _post_json(self.endpoint, {"model": self.model, "input": texts}, self.timeout)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding response must contain one vector per input")
        vectors = {}
        dimension = None
        for entry in data:
            index = entry.get("index") if isinstance(entry, dict) else None
            if type(index) is not int or not 0 <= index < len(texts) or index in vectors:
                raise ValueError("embedding indices must identify each input exactly once")
            vector = self._vector(entry.get("embedding"))
            if dimension is not None and len(vector) != dimension:
                raise ValueError("embedding dimensions must match")
            dimension = len(vector)
            vectors[index] = vector
        return [vectors[index] for index in range(len(texts))]

    @staticmethod
    def _vector(raw) -> List[float]:
        if not isinstance(raw, list) or not raw or any(type(x) not in (int, float) for x in raw):
            raise ValueError("embedding must be a nonempty vector of finite numbers")
        try:
            vector = [float(x) for x in raw]
        except (OverflowError, ValueError):
            raise ValueError("embedding values must be finite numbers") from None
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("embedding values must be finite numbers")
        return vector

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        a, b = DenseRetriever._vector(a), DenseRetriever._vector(b)
        if len(a) != len(b):
            raise ValueError("embedding dimensions must match")
        scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
        if not scale_a or not scale_b:
            return 0.0
        # Independent scaling avoids overflow/underflow in finite vectors.
        a, b = [x / scale_a for x in a], [x / scale_b for x in b]
        dot = math.fsum(x * y for x, y in zip(a, b))
        na = math.sqrt(math.fsum(x * x for x in a))
        nb = math.sqrt(math.fsum(y * y for y in b))
        return dot / (na * nb)

    def index(self, doc_ids: List[str], texts: List[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have equal length")
        vectors = self._embed(texts)
        self.doc_ids = list(doc_ids)
        self.doc_vecs = vectors

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
        self.endpoint = admit_endpoint(endpoint)  # allowlisted origin or None -> BLOCKED
        self.model = model
        self.timeout = timeout

    def is_configured(self) -> bool:
        return self.endpoint is not None

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """candidates: [(doc_id, doc_text)] -> [(doc_id, score)] sorted desc."""
        if not self.endpoint:
            raise EndpointUnavailable("no rerank endpoint configured")
        body = _post_json(self.endpoint, {
            "model": self.model,
            "query": query,
            "texts": [t for _, t in candidates],
        }, self.timeout)
        results = body if isinstance(body, list) else body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ValueError("rerank response must contain a results list")
        scored = []
        seen = set()
        for entry in results:
            index = entry.get("index") if isinstance(entry, dict) else None
            if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
                raise ValueError("rerank indices must identify distinct candidates in the submitted pool")
            score = entry.get("relevance_score", entry.get("score"))
            try:
                valid_score = type(score) in (int, float) and math.isfinite(score)
            except OverflowError:
                valid_score = False
            if not valid_score:
                raise ValueError("rerank scores must be finite numbers")
            seen.add(index)
            scored.append((candidates[index][0], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
