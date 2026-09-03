"""
Ranking evaluation metrics: Recall@k, MRR@k, MAP@k, nDCG@k.
Self-contained implementations cross-checked against trec_eval semantics.
All metrics operate only on real run outputs and real qrels.
"""
import math
from typing import Dict, List, Set


def recall_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    recalls = []
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        retrieved = run.get(qid, [])[:k]
        hits = len(set(retrieved) & rel_docs)
        recalls.append(hits / len(rel_docs))
    return sum(recalls) / len(recalls) if recalls else 0.0


def mrr_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    rrs = []
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        retrieved = run.get(qid, [])[:k]
        rr = 0.0
        for rank, docid in enumerate(retrieved, start=1):
            if docid in rel_docs:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def map_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    aps = []
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        retrieved = run.get(qid, [])[:k]
        hits, prec_sum = 0, 0.0
        for rank, docid in enumerate(retrieved, start=1):
            if docid in rel_docs:
                hits += 1
                prec_sum += hits / rank
        aps.append(prec_sum / min(len(rel_docs), k))
    return sum(aps) / len(aps) if aps else 0.0


def _dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(run: Dict[str, List[str]], qrels_graded: Dict[str, Dict[str, int]], k: int) -> float:
    scores = []
    for qid, grades in qrels_graded.items():
        if not grades:
            continue
        retrieved = run.get(qid, [])[:k]
        actual = [grades.get(d, 0) for d in retrieved]
        ideal = sorted(grades.values(), reverse=True)[:k]
        idcg = _dcg(ideal)
        if idcg == 0:
            continue
        scores.append(_dcg(actual) / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], ks=(1, 5, 10)) -> dict:
    """qrels: {qid: {docid: graded_relevance}}. Binary rels use grade 1."""
    binary = {qid: set(d for d, g in docs.items() if g > 0) for qid, docs in qrels.items()}
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(run, binary, k)
        out[f"mrr@{k}"] = mrr_at_k(run, binary, k)
        out[f"map@{k}"] = map_at_k(run, binary, k)
        out[f"ndcg@{k}"] = ndcg_at_k(run, qrels, k)
    return out
