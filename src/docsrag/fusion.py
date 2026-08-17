"""Rank fusion for hybrid retrieval.

The core problem: BM25 returns unbounded relevance scores (0 to ~30+, corpus
dependent) while cosine similarity returns [-1, 1]. You cannot add or average
them directly. Whichever scale is larger silently dominates, and the mix
changes as the corpus grows. This is the most common bug in hand-built hybrid
retrievers.

Two principled options are implemented here:

1. Reciprocal Rank Fusion (RRF): discard scores entirely, use only rank.
       score(d) = sum over rankers r of  w_r / (k + rank_r(d))
   Robust, scale-free, needs no tuning. From Cormack et al., 2009. The constant
   k (60 by convention) damps the influence of the very top ranks so a single
   ranker cannot dictate the fused result.

2. Normalized score fusion: min-max normalize each ranker to [0, 1] and take a
   weighted sum. Retains score magnitude (RRF throws it away), but is sensitive
   to outliers and to how many candidates you retrieve.

Interview note: RRF's weakness is exactly that it is scale-free. If one ranker
is confidently correct with a huge score margin, RRF cannot express that; it
only sees rank 1. Normalized fusion can. In practice RRF still usually wins
because score distributions are unstable across queries.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Ranking = Sequence[Tuple[int, float]]


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: One ranked (doc_id, score) list per retriever, already sorted
            best-first. Scores are ignored; only position matters.
        k: Rank damping constant. Smaller k sharpens the advantage of top ranks.
        weights: Optional per-retriever weight. Defaults to uniform.
        top_k: Truncate the fused output.

    Returns:
        Fused (doc_id, rrf_score) sorted best-first. Ties are broken by
        ascending doc_id so results are deterministic and tests are stable.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} != rankings length {len(rankings)}"
        )

    fused: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0.0:
            continue
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k] if top_k is not None else ordered


def _min_max_normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores to [0, 1]. A flat list maps to all-1.0, not all-0.0.

    Mapping a constant list to 1.0 preserves the signal that every one of those
    documents was retrieved at all, which we would otherwise throw away.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0.0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def normalized_score_fusion(
    rankings: Sequence[Ranking],
    *,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """Fuse ranked lists by min-max normalizing each then weighted-summing.

    Documents missing from a ranker contribute 0.0 for that ranker, which acts
    as an implicit penalty for not being retrieved.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")

    total_weight = sum(weights) or 1.0
    fused: Dict[int, float] = {}

    for ranking, weight in zip(rankings, weights):
        if not ranking or weight == 0.0:
            continue
        normalized = _min_max_normalize([s for _, s in ranking])
        for (doc_id, _), norm in zip(ranking, normalized):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * norm

    scaled = {d: s / total_weight for d, s in fused.items()}
    ordered = sorted(scaled.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k] if top_k is not None else ordered


def fuse(
    rankings: Sequence[Ranking],
    *,
    method: str = "rrf",
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """Dispatch to a fusion strategy by name (used by the eval config sweep)."""
    if method == "rrf":
        return reciprocal_rank_fusion(rankings, k=k, weights=weights, top_k=top_k)
    if method in ("normalized", "norm", "weighted"):
        return normalized_score_fusion(rankings, weights=weights, top_k=top_k)
    raise ValueError(f"unknown fusion method: {method!r}")


__all__ = (
    "reciprocal_rank_fusion",
    "normalized_score_fusion",
    "fuse",
)
