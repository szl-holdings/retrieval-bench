"""
Pure-Python BM25 (Okapi) sparse retrieval. No external dependencies.
Serves as the sparse baseline lane; no fabricated scores — every score is
computed from the actual indexed corpus.
"""
import math
import re
from collections import Counter
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_tf: List[Counter] = []
        self.doc_len: List[int] = []
        self.df = Counter()
        self.avgdl = 0.0

    def index(self, doc_ids: List[str], texts: List[str]) -> int:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have equal length")
        for did, text in zip(doc_ids, texts):
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_ids.append(did)
            self.doc_tokens.append(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for term in tf:
                self.df[term] += 1
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        n_docs = len(self.doc_ids)
        df = self.df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _score_doc(self, query_terms: List[str], i: int) -> float:
        score = 0.0
        dl = self.doc_len[i]
        for t in query_terms:
            tf = self.doc_tf[i].get(t, 0)
            if tf == 0:
                continue
            idf = self._idf(t)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else tf + self.k1
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q = tokenize(query)
        scored = [(self.doc_ids[i], self._score_doc(q, i)) for i in range(len(self.doc_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
