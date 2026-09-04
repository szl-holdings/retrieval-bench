"""
Multi-vector / late-interaction retrieval lane (ColBERT-style scoring).

Deliberately separate from the single-vector dense lane: a document here is
a MATRIX of token-level vectors scored via MaxSim, not one vector with
cosine similarity. Memory footprint and index cost differ by an order of
magnitude, so multivector runs are never mixed into single-vector
comparisons — the fairness gate rejects it.

The lane is honest about its dependency: token-level encoders are supplied
by the caller (e.g. a ColBERT endpoint on the owner node). With no encoder
configured the lane returns BLOCKED. The MaxSim math itself is pure Python
and fully testable with deterministic fixture vectors.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

TokenVectors = List[List[float]]           # one document/query: [n_tokens, dim]
TokenEncoder = Callable[[str], TokenVectors]


class EncoderUnavailable(Exception):
    pass


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def maxsim(query_vecs: TokenVectors, doc_vecs: TokenVectors) -> float:
    """ColBERT late interaction: sum over query tokens of the max cosine
    similarity against any document token. Empty inputs score 0.0."""
    if not query_vecs or not doc_vecs:
        return 0.0
    total = 0.0
    for qv in query_vecs:
        total += max(_cos(qv, dv) for dv in doc_vecs)
    return total


class LateInteractionIndex:
    def __init__(self, encoder: Optional[TokenEncoder]):
        self.encoder = encoder
        self.doc_ids: List[str] = []
        self.doc_vecs: List[TokenVectors] = []

    def is_configured(self) -> bool:
        return self.encoder is not None

    def index(self, doc_ids: List[str], texts: List[str]) -> int:
        if self.encoder is None:
            raise EncoderUnavailable("no token-level encoder configured")
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have equal length")
        for did, text in zip(doc_ids, texts):
            self.doc_ids.append(did)
            self.doc_vecs.append(self.encoder(text))
        return len(self.doc_ids)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.encoder is None:
            raise EncoderUnavailable("no token-level encoder configured")
        qv = self.encoder(query)
        scored = [(did, maxsim(qv, dv)) for did, dv in zip(self.doc_ids, self.doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
