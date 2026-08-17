"""Retrieval evaluation and configuration sweeps.

This module is the reason the project is defensible. It answers, with numbers:

  * Does hybrid retrieval actually beat dense-only on THIS corpus?
  * Does reranking earn its latency?
  * What does chunk size do to recall?
  * How much recall do I lose if I only put 3 chunks in the prompt instead of 10?

Retrieval evaluation requires no LLM and no API key, so a full sweep runs in
seconds and costs nothing. That is what makes it practical to actually iterate
rather than guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..config import RerankConfig, RetrievalConfig, Settings
from ..index import RagIndex
from ..metrics import DEFAULT_K_VALUES, aggregate, evaluate_query
from ..pipeline import RagPipeline
from .dataset import EvalExample, describe_eval_set, filter_verified


@dataclass
class RetrievalReport:
    """Scored result of one configuration against one eval set."""

    label: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    n_queries: int
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    per_query: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "config": self.config,
            "metrics": self.metrics,
            "n_queries": self.n_queries,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_p95": self.latency_ms_p95,
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a scipy/numpy dependency here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0, int(round((pct / 100.0) * len(ordered) + 0.5)) - 1)
    )
    return ordered[index]


def evaluate_retrieval(
    pipeline: RagPipeline,
    examples: Sequence[EvalExample],
    *,
    label: str = "default",
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    keep_per_query: bool = True,
) -> RetrievalReport:
    """Score a pipeline's retrieval against a labelled eval set.

    Important detail: we request the largest k in ``k_values`` from the retriever
    and then evaluate every smaller k by truncation. Retrieving once instead of
    once per k keeps sweeps fast and, more importantly, guarantees the metrics
    are all computed over the SAME ranking.
    """
    k_list = sorted(set(k_values))
    max_k = max(k_list)

    per_query_metrics: List[Dict[str, float]] = []
    per_query_detail: List[Dict[str, Any]] = []
    latencies: List[float] = []

    original_top_k = pipeline.settings.retrieval.top_k
    original_rerank_top_k = pipeline.settings.rerank.top_k

    # Widen the output so every k in k_list is measurable from one retrieval.
    pipeline.settings = pipeline.settings.with_overrides(
        retrieval=RetrievalConfig(
            **{
                **pipeline.settings.retrieval.__dict__,
                "top_k": max(max_k, original_top_k),
                "candidate_k": max(
                    pipeline.settings.retrieval.candidate_k, max_k
                ),
            }
        ),
        rerank=RerankConfig(
            **{
                **pipeline.settings.rerank.__dict__,
                "top_k": max(max_k, original_rerank_top_k),
            }
        ),
    )

    try:
        for example in examples:
            if not example.gold_chunk_ids:
                continue

            start = time.perf_counter()
            retrieved = pipeline.retrieve(example.question)
            latencies.append((time.perf_counter() - start) * 1000.0)

            retrieved_ids = [r.chunk.id for r in retrieved]
            metrics = evaluate_query(
                retrieved_ids, example.relevant, k_values=k_list
            )
            per_query_metrics.append(metrics)

            if keep_per_query:
                first_hit = next(
                    (
                        i + 1
                        for i, cid in enumerate(retrieved_ids)
                        if cid in example.relevant
                    ),
                    None,
                )
                per_query_detail.append(
                    {
                        "id": example.id,
                        "question": example.question,
                        "gold_chunk_ids": example.gold_chunk_ids,
                        "retrieved_ids": retrieved_ids[:max_k],
                        "first_hit_rank": first_hit,
                        "recall@5": metrics.get("recall@5", 0.0),
                        "verified": example.verified,
                    }
                )
    finally:
        # Always restore, even if a query raises, so a failed sweep step does
        # not silently corrupt later steps.
        pipeline.settings = pipeline.settings.with_overrides(
            retrieval=RetrievalConfig(
                **{
                    **pipeline.settings.retrieval.__dict__,
                    "top_k": original_top_k,
                }
            ),
            rerank=RerankConfig(
                **{
                    **pipeline.settings.rerank.__dict__,
                    "top_k": original_rerank_top_k,
                }
            ),
        )

    return RetrievalReport(
        label=label,
        config=pipeline.settings.describe(),
        metrics=aggregate(per_query_metrics),
        n_queries=len(per_query_metrics),
        latency_ms_mean=(
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        latency_ms_p95=_percentile(latencies, 95.0),
        per_query=per_query_detail,
    )


def default_sweep_configs(base: Settings) -> List[tuple[str, Settings]]:
    """The ablation ladder that produces the results table for the README.

    Each step changes exactly ONE thing relative to a meaningful baseline, so
    every row attributes its delta to a single design decision. A sweep where
    two variables move at once tells you nothing about either.
    """
    def with_retrieval(**overrides: Any) -> Settings:
        """A first-stage-only config: reranking is forced OFF.

        The baseline rows must isolate the first stage. If the caller enabled
        reranking on ``base`` (e.g. via ``--rerank``), inheriting it here would
        apply the cross-encoder to every row, collapsing the comparison and
        charging every baseline the reranker's latency.
        """
        return base.with_overrides(
            retrieval=RetrievalConfig(
                **{**base.retrieval.__dict__, **overrides}
            ),
            rerank=RerankConfig(**{**base.rerank.__dict__, "enabled": False}),
        )

    configs: List[tuple[str, Settings]] = [
        ("bm25 only", with_retrieval(mode="bm25")),
        ("dense only", with_retrieval(mode="dense")),
        ("hybrid (RRF)", with_retrieval(mode="hybrid", fusion="rrf")),
        (
            "hybrid (normalized)",
            with_retrieval(mode="hybrid", fusion="normalized"),
        ),
    ]

    if base.rerank.enabled:
        reranked = base.with_overrides(
            retrieval=RetrievalConfig(
                **{**base.retrieval.__dict__, "mode": "hybrid", "fusion": "rrf"}
            ),
            rerank=RerankConfig(**{**base.rerank.__dict__, "enabled": True}),
        )
        configs.append(("hybrid + rerank", reranked))

    return configs


def sweep(
    index: RagIndex,
    examples: Sequence[EvalExample],
    *,
    base: Settings,
    configs: Sequence[tuple[str, Settings]] | None = None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> List[RetrievalReport]:
    """Evaluate several configurations against the same index and eval set."""
    configs = configs or default_sweep_configs(base)
    reports: List[RetrievalReport] = []

    for label, settings in configs:
        # Rebuilding the index would re-embed everything; instead we rebind the
        # existing index to new retrieval settings. Valid because chunking and
        # embeddings are identical across these rows -- only search changes.
        index.retrieval = settings.retrieval
        pipeline = RagPipeline(index, settings=settings)
        print(f"  evaluating: {label}", flush=True)
        reports.append(
            evaluate_retrieval(
                pipeline, examples, label=label, k_values=k_values
            )
        )

    return reports


HEADLINE_METRICS = (
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
)


def to_markdown_table(
    reports: Sequence[RetrievalReport],
    *,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> str:
    """Render reports as a Markdown table, ready to paste into the README."""
    if not reports:
        return "_No results._"

    header = (
        "| Configuration | "
        + " | ".join(metrics)
        + " | p95 latency (ms) |"
    )
    divider = "|---" * (len(metrics) + 2) + "|"

    rows: List[str] = []
    best = {
        metric: max((r.metrics.get(metric, 0.0) for r in reports), default=0.0)
        for metric in metrics
    }

    for report in reports:
        cells: List[str] = []
        for metric in metrics:
            value = report.metrics.get(metric, 0.0)
            # Bold the winner per column so the table is readable at a glance.
            cells.append(
                f"**{value:.3f}**"
                if value >= best[metric] > 0.0
                else f"{value:.3f}"
            )
        rows.append(
            f"| {report.label} | "
            + " | ".join(cells)
            + f" | {report.latency_ms_p95:.1f} |"
        )

    return "\n".join([header, divider, *rows])


def save_reports(
    reports: Sequence[RetrievalReport],
    directory: str | Path,
    *,
    examples: Sequence[EvalExample] | None = None,
) -> Path:
    """Persist full JSON results plus a Markdown summary."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reports": [r.to_dict() for r in reports],
    }
    if examples is not None:
        payload["eval_set"] = describe_eval_set(examples)

    (out_dir / "retrieval_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = ["# Retrieval evaluation results", ""]
    if examples is not None:
        stats = describe_eval_set(examples)
        markdown.append(
            f"Eval set: {stats['total']} questions "
            f"({stats['verified']} human-verified)."
        )
        markdown.append("")
    markdown.append(to_markdown_table(reports))
    markdown.append("")

    verified = filter_verified(examples) if examples else []
    if examples and verified and len(verified) < len(examples):
        markdown.append(
            f"> Note: {len(examples) - len(verified)} questions are "
            "LLM-generated and not yet human-verified. Synthetic questions "
            "share vocabulary with their source chunk, which inflates lexical "
            "retrieval scores. Treat cross-configuration deltas as more "
            "trustworthy than absolute values."
        )
        markdown.append("")

    summary_path = out_dir / "retrieval_results.md"
    summary_path.write_text("\n".join(markdown), encoding="utf-8")
    return summary_path


__all__ = (
    "RetrievalReport",
    "evaluate_retrieval",
    "sweep",
    "default_sweep_configs",
    "to_markdown_table",
    "save_reports",
    "HEADLINE_METRICS",
)
