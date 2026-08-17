#!/usr/bin/env python3
"""Build the retrieval index from a corpus directory.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --corpus data/corpus --out artifacts/index
    python scripts/build_index.py --provider hash        # offline smoke test
    python scripts/build_index.py --max-tokens 256       # chunking ablation

The index is written as inert data (JSONL + .npy + JSON), never pickle, so
loading it cannot execute code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.chunking import Chunk, chunk_markdown  # noqa: E402
from docsrag.config import ChunkConfig, EmbeddingConfig, Settings  # noqa: E402
from docsrag.index import RagIndex  # noqa: E402

TEXT_EXTENSIONS = {".md", ".markdown", ".mdx", ".txt", ".rst"}


def load_corpus(corpus_dir: Path) -> List[tuple[Path, str]]:
    """Read every supported text file under ``corpus_dir``."""
    if not corpus_dir.is_dir():
        raise SystemExit(
            f"corpus directory not found: {corpus_dir}\n"
            "Fetch one first:  python scripts/fetch_corpus.py"
        )

    documents: List[tuple[Path, str]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip rather than crash: real corpora contain surprises.
            print(f"  skipped (encoding): {path}", file=sys.stderr)
            continue
        if text.strip():
            documents.append((path, text))
    return documents


def chunk_corpus(
    documents: List[tuple[Path, str]],
    corpus_dir: Path,
    config: ChunkConfig,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path, text in documents:
        # Store a repo-relative source so artifacts are portable across machines.
        try:
            source = str(path.relative_to(corpus_dir))
        except ValueError:
            source = str(path)
        chunks.extend(
            chunk_markdown(
                text,
                source=source,
                max_tokens=config.max_tokens,
                overlap_tokens=config.overlap_tokens,
                min_tokens=config.min_tokens,
            )
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--provider",
        default=None,
        help="embedding provider: sentence-transformers | openai | hash",
    )
    parser.add_argument("--model", default=None, help="embedding model name")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=("numpy", "faiss"),
        help="dense index backend (numpy is exhaustive cosine and is the default)",
    )
    args = parser.parse_args()

    settings = Settings.from_env()

    if args.max_tokens or args.overlap_tokens:
        settings = settings.with_overrides(
            chunk=ChunkConfig(
                max_tokens=args.max_tokens or settings.chunk.max_tokens,
                overlap_tokens=(
                    args.overlap_tokens
                    if args.overlap_tokens is not None
                    else settings.chunk.overlap_tokens
                ),
                min_tokens=settings.chunk.min_tokens,
            )
        )

    if args.provider or args.model:
        settings = settings.with_overrides(
            embedding=EmbeddingConfig(
                **{
                    **settings.embedding.__dict__,
                    "provider": args.provider or settings.embedding.provider,
                    "model": args.model or settings.embedding.model,
                }
            )
        )

    corpus_dir = args.corpus or settings.paths.corpus_dir
    out_dir = args.out or settings.paths.index_dir

    print(f"Loading corpus from {corpus_dir} ...")
    documents = load_corpus(corpus_dir)
    if not documents:
        raise SystemExit(f"no readable documents found under {corpus_dir}")
    print(f"  {len(documents)} documents")

    print(
        f"Chunking (max_tokens={settings.chunk.max_tokens}, "
        f"overlap={settings.chunk.overlap_tokens}) ..."
    )
    chunks = chunk_corpus(documents, corpus_dir, settings.chunk)
    if not chunks:
        raise SystemExit("chunking produced no chunks")

    token_counts = [c.token_count for c in chunks]
    print(
        f"  {len(chunks)} chunks | "
        f"tokens min={min(token_counts)} "
        f"mean={sum(token_counts) // len(token_counts)} "
        f"max={max(token_counts)}"
    )

    print(f"Embedding with provider={settings.embedding.provider} ...")
    started = time.perf_counter()
    index = RagIndex.build(
        chunks, settings=settings, backend=args.backend, progress=True
    )
    elapsed = time.perf_counter() - started
    print(f"  embedded in {elapsed:.1f}s ({len(chunks) / max(elapsed, 1e-9):.0f} chunks/s)")

    index.save(out_dir)
    print(f"\nIndex written to {out_dir}")
    print(f"  chunks: {len(index)}  dimension: {index.vector_store.dimension}")
    print("\nNext:  python scripts/ask.py 'your question here'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
