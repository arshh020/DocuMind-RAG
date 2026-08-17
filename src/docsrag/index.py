"""The retrieval index: chunk store + BM25 + dense vectors + hybrid search.

This is the object the eval harness pokes at, so it deliberately exposes
retrieval WITHOUT generation. Being able to evaluate retrieval in isolation is
the whole reason the project has credible numbers.

Artifact layout on disk (all inert data, no pickle):
    artifacts/index/
        chunks.jsonl      one chunk per line
        vectors.npy       float32 (n_chunks, dim), L2-normalized
        vector_ids.json   row -> chunk id, parallel to vectors.npy
        manifest.json     build metadata for provenance
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .config import RetrievalConfig, Settings
from .embeddings import Embedder, get_embedder
from .fusion import fuse
from .tokenize import tokenize
from .vectorstore import VectorStore, build_vector_store


@dataclass
class RetrievedChunk:
    """A chunk plus the retrieval evidence that surfaced it.

    Keeping the component scores (not just the fused score) makes retrieval
    debuggable: you can see immediately whether a hit came from lexical match,
    semantic match, or both.
    """

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None

    def to_passage(self) -> Dict[str, Any]:
        """Shape expected by the generation layer."""
        return {
            "id": self.chunk.id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "title": self.chunk.title,
            "breadcrumb": self.chunk.breadcrumb,
            "score": self.score,
        }


class RagIndex:
    """Hybrid retrieval index over a chunked corpus."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vector_store: VectorStore,
        *,
        embedder: Embedder,
        retrieval: RetrievalConfig | None = None,
        manifest: Dict[str, Any] | None = None,
    ) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.vector_store = vector_store
        self.embedder = embedder
        self.retrieval = retrieval or RetrievalConfig()
        self.manifest = manifest or {}

        self._by_id = {chunk.id: chunk for chunk in self.chunks}
        # BM25 is built over embed_text (breadcrumb + body) so heading terms are
        # lexically searchable too. A query for "map syntax" should match the
        # Syntax section of the map page even if the body never says "map".
        self._bm25 = BM25(
            [tokenize(chunk.embed_text) for chunk in self.chunks],
            k1=self.retrieval.bm25_k1,
            b=self.retrieval.bm25_b,
        )
        # BM25 works in row space; the vector store works in id space. This maps
        # between them.
        self._row_to_id = [chunk.id for chunk in self.chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    # ---------------------------------------------------------------- retrieval

    def _dense_search(self, query: str, k: int) -> List[tuple[str, float]]:
        query_vector = self.embedder.encode([query])[0]
        return self.vector_store.search(query_vector, k)

    def _bm25_search(self, query: str, k: int) -> List[tuple[str, float]]:
        hits = self._bm25.search(query, k)
        return [(self._row_to_id[row], score) for row, score in hits]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        mode: str | None = None,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks for a query using the configured strategy.

        Returns at most ``top_k`` results, best first. Component scores are
        preserved on each result for debugging.
        """
        config = self.retrieval
        mode = (mode or config.mode).lower()
        top_k = top_k or config.top_k
        candidate_k = candidate_k or max(config.candidate_k, top_k)

        dense_hits: List[tuple[str, float]] = []
        bm25_hits: List[tuple[str, float]] = []

        if mode in ("dense", "hybrid"):
            dense_hits = self._dense_search(query, candidate_k)
        if mode in ("bm25", "lexical", "hybrid"):
            bm25_hits = self._bm25_search(query, candidate_k)

        dense_scores = dict(dense_hits)
        bm25_scores = dict(bm25_hits)

        if mode == "dense":
            ranked = dense_hits
        elif mode in ("bm25", "lexical"):
            ranked = bm25_hits
        elif mode == "hybrid":
            # fuse() operates on integer ids, so map chunk ids to indices and
            # back. Doing it here keeps fusion.py generic and unit-testable.
            id_to_int: Dict[str, int] = {}
            int_to_id: Dict[int, str] = {}

            def encode_ranking(
                hits: Sequence[tuple[str, float]]
            ) -> List[tuple[int, float]]:
                encoded: List[tuple[int, float]] = []
                for chunk_id, score in hits:
                    if chunk_id not in id_to_int:
                        index = len(id_to_int)
                        id_to_int[chunk_id] = index
                        int_to_id[index] = chunk_id
                    encoded.append((id_to_int[chunk_id], score))
                return encoded

            fused = fuse(
                [encode_ranking(dense_hits), encode_ranking(bm25_hits)],
                method=config.fusion,
                k=config.rrf_k,
                weights=[config.dense_weight, config.bm25_weight],
            )
            ranked = [(int_to_id[i], score) for i, score in fused]
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        results: List[RetrievedChunk] = []
        for chunk_id, score in ranked:
            if score < config.min_score:
                continue
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    dense_score=float(dense_scores.get(chunk_id, 0.0)),
                    bm25_score=float(bm25_scores.get(chunk_id, 0.0)),
                )
            )
            if len(results) >= top_k:
                break

        return results

    # -------------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        self.vector_store.save(path)

        manifest = dict(self.manifest)
        manifest.update(
            {
                "chunk_count": len(self.chunks),
                "vector_dimension": self.vector_store.dimension,
                "built_at": manifest.get("built_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> "RagIndex":
        settings = settings or Settings.from_env()
        path = Path(directory)

        chunks: List[Chunk] = []
        with (path / "chunks.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    chunks.append(Chunk.from_dict(json.loads(line)))

        vector_store = VectorStore.load(path)

        manifest_path = path / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )

        return cls(
            chunks,
            vector_store,
            embedder=embedder or get_embedder(settings.embedding),
            retrieval=settings.retrieval,
            manifest=manifest,
        )

    # ------------------------------------------------------------------- build

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        *,
        settings: Settings,
        embedder: Embedder | None = None,
        backend: str = "numpy",
        progress: bool = True,
    ) -> "RagIndex":
        """Embed chunks and construct a searchable index."""
        embedder = embedder or get_embedder(settings.embedding)

        if not chunks:
            empty = np.zeros((0, settings.embedding.dimension), dtype=np.float32)
            return cls(
                [],
                build_vector_store(empty, [], backend=backend),
                embedder=embedder,
                retrieval=settings.retrieval,
            )

        texts = [chunk.embed_text for chunk in chunks]
        batch = max(1, settings.embedding.batch_size)
        blocks: List[np.ndarray] = []

        for start in range(0, len(texts), batch):
            blocks.append(embedder.encode(texts[start : start + batch]))
            if progress:
                done = min(start + batch, len(texts))
                print(f"  embedded {done}/{len(texts)} chunks", flush=True)

        vectors = np.vstack(blocks).astype(np.float32)
        store = build_vector_store(
            vectors, [chunk.id for chunk in chunks], backend=backend
        )

        return cls(
            chunks,
            store,
            embedder=embedder,
            retrieval=settings.retrieval,
            manifest={
                "embedding_provider": settings.embedding.provider,
                "embedding_model": settings.embedding.model,
                "embedding_dimension": int(vectors.shape[1]),
                "chunk_count": len(chunks),
                "chunk_max_tokens": settings.chunk.max_tokens,
                "chunk_overlap_tokens": settings.chunk.overlap_tokens,
                "vector_backend": backend,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )


__all__ = ("RagIndex", "RetrievedChunk")
