"""Information-retrieval metrics for evaluating the retriever in isolation.

Why evaluate retrieval separately from generation: when a RAG answer is wrong
there are only two possible causes -- the right context was never retrieved, or
it was retrieved and the model failed to use it. You cannot fix what you cannot
localize, and these metrics localize it. If recall@k is high but answers are
bad, the problem is the prompt or the model. If recall@k is low, no amount of
prompt engineering will save you.

All functions take:
    retrieved: ranked list of chunk ids, best first
    relevant:  set of chunk ids that are actually relevant (the gold labels)

Conventions:
  * All metrics return 0.0 when there are no relevant documents, rather than
    raising or returning NaN, so aggregate means stay well-defined.
  * "@k" always truncates the retrieved list to the first k items.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def _truncate(retrieved: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(retrieved[:k])


def hit_rate_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0 if ANY relevant document appears in the top k, else 0.0.

    The most forgiving metric. For single-gold eval sets this is identical to
    recall@k, which is why we report both only when gold sets can be multi-item.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(_truncate(retrieved, k)) & relevant else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant documents that appear in the top k.

    This is the headline metric for RAG retrieval: it answers "did we put the
    evidence in front of the model?"
    """
    if not relevant:
        return 0.0
    found = set(_truncate(retrieved, k)) & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Matters for RAG because every irrelevant chunk you stuff into the prompt
    costs tokens, adds latency, and measurably increases distraction-driven
    hallucination. High recall with low precision is a real cost, not a win.
    """
    if not relevant:
        return 0.0
    top = _truncate(retrieved, k)
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """1 / rank of the FIRST relevant document (0.0 if none in top k).

    Averaged over queries this is MRR. It is rank-sensitive in a way recall is
    not: recall@5 treats position 1 and position 5 identically, MRR does not.
    Use it when you care about putting the best evidence first, which you do
    when the prompt budget is tight.
    """
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(_truncate(retrieved, k), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Set[str],
    k: int,
    *,
    gains: Dict[str, float] | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    DCG applies a logarithmic positional discount, then we divide by the ideal
    DCG so the result is comparable across queries with different numbers of
    relevant documents.

    Supports graded relevance via ``gains`` (chunk id -> gain). With binary
    labels every relevant chunk has gain 1.0. Graded relevance is where nDCG
    earns its keep over MRR/recall -- it can distinguish "perfect chunk" from
    "related but partial chunk".
    """
    if not relevant:
        return 0.0

    gain_of = gains or {doc_id: 1.0 for doc_id in relevant}

    dcg = 0.0
    for rank, doc_id in enumerate(_truncate(retrieved, k), start=1):
        gain = gain_of.get(doc_id, 0.0) if doc_id in relevant else 0.0
        if gain:
            # +1 so rank 1 has discount log2(2) = 1.0 (no penalty).
            dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted(
        (gain_of.get(d, 1.0) for d in relevant), reverse=True
    )[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))

    return dcg / idcg if idcg > 0 else 0.0


DEFAULT_K_VALUES = (1, 3, 5, 10)


def evaluate_query(
    retrieved: Sequence[str],
    relevant: Set[str],
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> Dict[str, float]:
    """Compute the full metric suite for a single query."""
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"hit@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        out[f"mrr@{k}"] = reciprocal_rank(retrieved, relevant, k)
    return out


def aggregate(
    per_query: Sequence[Dict[str, float]]
) -> Dict[str, float]:
    """Macro-average per-query metric dicts into corpus-level numbers.

    Macro-averaging (mean of per-query scores) rather than micro-averaging
    (pooling all hits) because each eval question should count equally
    regardless of how many gold chunks it has.
    """
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: sum(q.get(key, 0.0) for q in per_query) / n for key in keys}


__all__ = (
    "hit_rate_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "evaluate_query",
    "aggregate",
    "DEFAULT_K_VALUES",
)
