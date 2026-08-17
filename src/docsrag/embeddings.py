"""Pluggable embedding providers.

Three backends, one interface:

* ``sentence-transformers`` -- the real local default (all-MiniLM-L6-v2, 384-d).
* ``openai`` -- any OpenAI-compatible embeddings endpoint, called over plain
  urllib so the core package needs no HTTP dependency.
* ``hash`` -- deterministic feature hashing. No model, no network, no download.
  This exists so the ENTIRE pipeline (index, retrieve, evaluate) runs in CI and
  in tests offline. It is not semantically competitive and must never be used
  for real answers; it is a plumbing test double.

Design note that matters for correctness: every backend L2-normalizes its output
when ``normalize=True``. all-MiniLM-L6-v2 is trained with cosine similarity, so
unnormalized vectors searched by inner product or L2 give subtly wrong rankings.
This is the exact bug present in most tutorial RAG code, and it produces no error
message -- only worse results.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import List, Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig
from .tokenize import tokenize


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Replace zero norms with 1.0 so all-zero rows stay all-zero instead of NaN.
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Embedder(Protocol):
    """Minimal embedding interface used by the rest of the system."""

    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension)."""
        ...


class HashEmbedder:
    """Deterministic feature-hashing embedder. Offline test double.

    Maps each token to a bucket via SHA-1 and accumulates a signed count, i.e.
    the classic hashing trick. Signs are derived from a second hash bit so that
    unrelated tokens colliding in the same bucket tend to cancel rather than
    reinforce.

    Properties we actually rely on: deterministic across processes, cheap, and
    it gives non-trivial lexical overlap signal -- enough to exercise the full
    retrieval and evaluation path without a model download.
    """

    def __init__(self, dimension: int = 384, *, normalize: bool = True) -> None:
        self.dimension = int(dimension)
        self.normalize = normalize

    @staticmethod
    def _bucket_and_sign(token: str, dimension: int) -> tuple[int, float]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket, sign = self._bucket_and_sign(token, self.dimension)
                out[row, bucket] += sign
        return l2_normalize(out) if self.normalize else out


class SentenceTransformerEmbedder:
    """Local embedding via sentence-transformers.

    Imported lazily so that merely importing this module does not pull in torch
    (a multi-second import and a ~2 GB install).
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. Either\n"
                "  pip install 'docsrag[local]'\n"
                "or set EMBEDDING_PROVIDER=hash for an offline smoke test."
            ) from exc

        self._model = SentenceTransformer(config.model)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._config = config

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAICompatibleEmbedder:
    """Embeddings via any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Uses urllib rather than requests/httpx to keep the core dependency-free.
    Useful when you do not want a 2 GB local install, or when the deployment
    target has no GPU and little disk.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self.dimension = config.dimension
        self._api_key = os.getenv(config.api_key_env, "")
        if not self._api_key:
            raise ValueError(
                f"{config.api_key_env} is not set but EMBEDDING_PROVIDER=openai"
            )
        base = config.api_base or "https://api.openai.com/v1"
        self._url = base.rstrip("/") + "/embeddings"

    def _post(self, batch: Sequence[str]) -> List[List[float]]:
        payload = json.dumps(
            {"model": self._config.model, "input": list(batch)}
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"embedding request failed ({exc.code}): {detail}"
            ) from exc

        # Sort by index: the API does not guarantee response order.
        items = sorted(body["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors: List[List[float]] = []
        batch_size = max(1, self._config.batch_size)
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._post(texts[start : start + batch_size]))

        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1]
        return l2_normalize(matrix) if self._config.normalize else matrix


def get_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory. Keeps provider selection in exactly one place."""
    provider = config.provider.lower()
    if provider in ("hash", "offline", "test"):
        return HashEmbedder(config.dimension, normalize=config.normalize)
    if provider in ("sentence-transformers", "st", "local", "huggingface"):
        return SentenceTransformerEmbedder(config)
    if provider in ("openai", "api", "openai-compatible"):
        return OpenAICompatibleEmbedder(config)
    raise ValueError(f"unknown embedding provider: {config.provider!r}")


__all__ = (
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAICompatibleEmbedder",
    "get_embedder",
    "l2_normalize",
)
