#!/usr/bin/env python3
"""Ask the RAG system a question from the command line.

Usage:
    python scripts/ask.py "How do I center a div with flexbox?"
    python scripts/ask.py "..." --mode dense --top-k 8
    python scripts/ask.py "..." --retrieval-only     # no LLM call, no API key

--retrieval-only is the debugging tool you will actually use most: it shows what
the retriever found and each component score, so you can tell whether a bad
answer is a retrieval failure or a generation failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.config import RerankConfig, RetrievalConfig, Settings  # noqa: E402
from docsrag.pipeline import RagPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument(
        "--mode", default=None, choices=("dense", "bm25", "hybrid")
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--rerank-provider", default=None, choices=("cross-encoder", "llm")
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="show retrieved passages without calling the LLM",
    )
    args = parser.parse_args()

    settings = Settings.from_env()

    if args.mode or args.top_k:
        settings = settings.with_overrides(
            retrieval=RetrievalConfig(
                **{
                    **settings.retrieval.__dict__,
                    "mode": args.mode or settings.retrieval.mode,
                    "top_k": args.top_k or settings.retrieval.top_k,
                }
            )
        )
    if args.rerank:
        settings = settings.with_overrides(
            rerank=RerankConfig(
                **{
                    **settings.rerank.__dict__,
                    "enabled": True,
                    "provider": args.rerank_provider or settings.rerank.provider,
                    "top_k": args.top_k or settings.rerank.top_k,
                }
            )
        )

    index_dir = args.index or settings.paths.index_dir
    pipeline = RagPipeline.from_index_dir(index_dir, settings=settings)

    if args.retrieval_only:
        retrieved = pipeline.retrieve(args.question)
        if not retrieved:
            print("No passages retrieved above the score threshold.")
            return 0
        print(f"\nTop {len(retrieved)} passages for: {args.question}\n")
        for position, item in enumerate(retrieved, start=1):
            print(f"[{position}] score={item.score:.4f}", end="")
            print(
                f"  dense={item.dense_score:.4f}  bm25={item.bm25_score:.4f}",
                end="",
            )
            if item.rerank_score is not None:
                print(f"  rerank={item.rerank_score:.3f}", end="")
            print(f"\n    {item.chunk.breadcrumb}")
            print(f"    source: {item.chunk.source}")
            snippet = item.chunk.text[:220].replace("\n", " ")
            print(f"    {snippet}...\n")
        return 0

    result = pipeline.answer(args.question)

    print("\n" + "=" * 70)
    print(result.answer.text)
    print("=" * 70)

    if result.answer.citations:
        print("\nSources:")
        for citation in result.answer.citations:
            label = citation.breadcrumb or citation.title
            print(f"  [{citation.marker}] {label}")
            print(f"      {citation.source}  (score {citation.score:.4f})")
    elif not result.answer.refused:
        print("\nWarning: the answer contained no valid citations.")

    timings = result.timings_ms
    print(
        f"\nLatency: retrieve {timings.get('retrieve', 0):.0f} ms | "
        f"generate {timings.get('generate', 0):.0f} ms | "
        f"total {timings.get('total', 0):.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
