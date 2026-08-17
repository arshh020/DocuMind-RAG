"""Dense vector index with a numpy backend and an optional FAISS backend.

Why numpy is the DEFAULT and not a fallback:

At our corpus scale (tens of thousands of chunks, 384 dimensions) an exhaustive
cosine search is a single matrix-vector product. For 20,000 chunks that is a
20000x384 float32 matmul, roughly 7.7M multiply-adds, which numpy does in about
a millisecond. FAISS IndexFlatL2 does exactly the same exhaustive scan -- it is
not an approximate index -- so it buys essentially nothing here beyond a
dependency and a pickle-deserialization security caveat.

FAISS earns its place when you need an APPROXIMATE index (IVF, HNSW, PQ) because
the corpus no longer fits a linear scan inside your latency budget. That is
roughly the million-vector range, not the ten-thousand range. The backend is
pluggable so that switch is a config change, and the eval harness can prove the
recall cost of approximation when you make it.

Being able to defend that reasoning is worth more in an interview than having
imported FAISS.

Persistence uses .npy plus JSON, not pickle. Deliberate: the tutorial approach
requires allow_dangerous_deserialization=True because it pickles the docstore,
and pickle executes arbitrary code on load. Our artifacts are inert data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .embeddings import l2_normalize


class VectorStore:
    """Exhaustive cosine-similarity index over normalized vectors.

    Because vectors are L2-normalized at write time, cosine similarity reduces
    to a plain inner product, so search is one matmul and one top-k selection.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        ids: Sequence[str],
        *,
        normalized: bool = True,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids length {len(ids)} != vector count {vectors.shape[0]}"
            )

        self.vectors = vectors if normalized else l2_normalize(vectors)
        self.ids: List[str] = list(ids)
        self._id_to_row = {chunk_id: row for row, chunk_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return self.vectors.shape[0]

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if len(self) else 0

    def row_of(self, chunk_id: str) -> int | None:
        return self._id_to_row.get(chunk_id)

    def search(
        self, query_vector: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Return the top-k ``(chunk_id, cosine_similarity)`` pairs."""
        if len(self) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm

        scores = self.vectors @ query
        k = max(1, min(k, scores.shape[0]))

        # argpartition is O(n); a full argsort would be O(n log n) and we only
        # need the top k.
        top_unsorted = np.argpartition(-scores, k - 1)[:k]
        top = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in top]

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "vector_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        path = Path(directory)
        vectors = np.load(path / "vectors.npy")
        ids = json.loads((path / "vector_ids.json").read_text(encoding="utf-8"))
        return cls(vectors, ids, normalized=True)


class FaissVectorStore(VectorStore):
    """FAISS-backed drop-in replacement, used when APPROXIMATE search is needed.

    Uses IndexFlatIP over normalized vectors, which is mathematically identical
    to cosine similarity. Swap to IndexIVFFlat or IndexHNSWFlat here when the
    corpus outgrows a linear scan, then re-run the eval sweep to quantify the
    recall you traded away for latency.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        ids: Sequence[str],
        *,
        normalized: bool = True,
    ) -> None:
        super().__init__(vectors, ids, normalized=normalized)
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "faiss-cpu is not installed. Use the default numpy backend, or\n"
                "  pip install 'docsrag[faiss]'"
            ) from exc

        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(self.vectors)

    def search(
        self, query_vector: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        if len(self) == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        k = max(1, min(k, len(self)))
        scores, indices = self._index.search(query, k)
        return [
            (self.ids[int(row)], float(score))
            for row, score in zip(indices[0], scores[0])
            if row >= 0
        ]


def build_vector_store(
    vectors: np.ndarray,
    ids: Sequence[str],
    *,
    backend: str = "numpy",
) -> VectorStore:
    backend = backend.lower()
    if backend in ("numpy", "np", "flat"):
        return VectorStore(vectors, ids)
    if backend == "faiss":
        return FaissVectorStore(vectors, ids)
    raise ValueError(f"unknown vector store backend: {backend!r}")


__all__ = ("VectorStore", "FaissVectorStore", "build_vector_store")
