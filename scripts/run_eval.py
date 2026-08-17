#!/usr/bin/env python3
"""Run the evaluation sweep and write the results table.

This is the script that produces the numbers you put in your README and on your
resume.

Usage:
    python scripts/run_eval.py                       # retrieval ablation only (free)
    python scripts/run_eval.py --rerank              # include the rerank row
    python scripts/run_eval.py --verified-only       # human-verified subset only
    python scripts/run_eval.py --faithfulness 25     # also grade 25 answers

Retrieval evaluation needs no API key and runs in seconds. Faithfulness grading
costs two LLM calls per question, so it is opt-in and capped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.config import RerankConfig, Settings  # noqa: E402
from docsrag.evaluation.dataset import (  # noqa: E402
    describe_eval_set,
    filter_verified,
    load_eval_set,
)
from docsrag.evaluation.retrieval import (  # noqa: E402
    default_sweep_configs,
    save_reports,
    sweep,
    to_markdown_table,
)
from docsrag.index import RagIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--evalset", type=Path, default=Path("eval/evalset.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="add a hybrid+rerank row to the sweep",
    )
    parser.add_argument(
        "--rerank-provider",
        default=None,
        choices=("cross-encoder", "llm"),
        help="llm needs no model download",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only human-verified examples",
    )
    parser.add_argument(
        "--faithfulness",
        type=int,
        default=0,
        metavar="N",
        help="grade N answers with an LLM judge (0 disables)",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    index_dir = args.index or settings.paths.index_dir
    results_dir = args.out or settings.paths.results_dir

    if args.rerank:
        settings = settings.with_overrides(
            rerank=RerankConfig(
                **{
                    **settings.rerank.__dict__,
                    "enabled": True,
                    "provider": args.rerank_provider or settings.rerank.provider,
                }
            )
        )

    print(f"Loading index from {index_dir} ...")
    index = RagIndex.load(index_dir, settings=settings)
    print(f"  {len(index)} chunks")

    examples = load_eval_set(args.evalset)
    stats = describe_eval_set(examples)
    print(
        f"Eval set: {stats['total']} questions "
        f"({stats['verified']} verified)"
    )

    if args.verified_only:
        examples = filter_verified(examples)
        if not examples:
            raise SystemExit(
                "no verified examples. Set verified=true on reviewed questions "
                "in the eval set first, or drop --verified-only."
            )
        print(f"  restricted to {len(examples)} verified questions")

    print("\nRunning retrieval sweep ...")
    reports = sweep(
        index,
        examples,
        base=settings,
        configs=default_sweep_configs(settings),
    )

    print("\n" + to_markdown_table(reports) + "\n")
    summary_path = save_reports(reports, results_dir, examples=examples)
    print(f"Results written to {summary_path}")

    if args.faithfulness > 0:
        from docsrag.evaluation.faithfulness import (
            evaluate_faithfulness,
            save_judgements,
            summarize_judgements,
        )
        from docsrag.pipeline import RagPipeline

        # Grade the best-performing configuration by recall@5, since that is the
        # configuration you would actually ship.
        best = max(reports, key=lambda r: r.metrics.get("recall@5", 0.0))
        print(f"\nGrading answers for best config: {best.label}")

        pipeline = RagPipeline(index, settings=settings)
        judgements = evaluate_faithfulness(
            pipeline, examples, limit=args.faithfulness, sleep_seconds=0.3
        )
        summary = summarize_judgements(judgements)
        judge_path = save_judgements(judgements, results_dir)

        print("\nAnswer quality:")
        for key, value in summary.items():
            print(f"  {key}: {value:.3f}")
        print(f"Written to {judge_path}")

    print(
        "\nPaste the table above into your README under a 'Results' heading, "
        "and state the eval-set size and verification status next to it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
