"""Reranking stage.

Why rerank at all: bi-encoder retrieval embeds the query and the document
INDEPENDENTLY, so the model never sees them together. It cannot reason about
whether this specific passage answers this specific question; it can only place
both near each other in a fixed vector space. A cross-encoder scores the
(query, passage) PAIR jointly with full attention across both, which is far more
accurate but far too slow to run over a whole corpus.

The standard resolution is a two-stage funnel:
  stage 1 (cheap, high recall):  retrieve ~30 candidates by hybrid search
  stage 2 (expensive, high precision): rerank those 30, keep the best 5

This is why RetrievalConfig.candidate_k is much larger than top_k. Reranking can
only reorder what stage 1 found, so stage-1 recall is the ceiling on final
quality -- a useful thing to be able to state precisely.

Two providers:
  * ``cross-encoder``: ms-marco-MiniLM-L-6-v2, the standard choice. Needs a
    ~90 MB model download.
  * ``llm``: reuses the Groq chat client to score passages. No download, which
    matters on a deadline, and surprisingly effective, but costs an API call and
    adds latency.
"""

from __future__ import annotations

import json
import re
from typing import List, Protocol, Sequence, Tuple

from .config import RerankConfig


class Reranker(Protocol):
    def rerank(
        self, query: str, passages: Sequence[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """Return ``(passage_index, score)`` sorted best-first."""
        ...


class NoopReranker:
    """Identity reranker: preserves retrieval order.

    Exists so the pipeline has no conditional branches around reranking, and so
    the eval sweep can treat 'no rerank' as just another configuration.
    """

    def rerank(
        self, query: str, passages: Sequence[str], top_k: int
    ) -> List[Tuple[int, float]]:
        # Descending pseudo-scores keep the output contract identical to a real
        # reranker (sorted best-first with meaningful ordering).
        return [(i, 1.0 - i * 1e-6) for i in range(min(top_k, len(passages)))]


class CrossEncoderReranker:
    """Cross-encoder reranking via sentence-transformers."""

    def __init__(self, config: RerankConfig) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking.\n"
                "  pip install 'docsrag[local]'\n"
                "or set RERANK_PROVIDER=llm to rerank with the Groq model instead."
            ) from exc
        self._model = CrossEncoder(config.model)
        self._config = config

    def rerank(
        self, query: str, passages: Sequence[str], top_k: int
    ) -> List[Tuple[int, float]]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(
            pairs, batch_size=self._config.batch_size, show_progress_bar=False
        )
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(scores)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]


_RERANK_SYSTEM_PROMPT = (
    "You are a search-relevance grader. For each numbered passage, output an "
    "integer relevance score from 0 to 10 for how well it answers the user's "
    "question. 10 means the passage directly and completely answers it. 0 means "
    "it is unrelated. Judge only relevance to the question, not writing "
    "quality. Respond with a JSON object mapping passage number to score, for "
    'example {"1": 8, "2": 0, "3": 3}. Output only the JSON object.'
)


class LLMReranker:
    """Rerank by asking the LLM to grade each passage's relevance.

    Needs no model download, which is why it is the recommended option when you
    are time-constrained. Truncates passages before sending them: relevance is
    almost always decidable from the opening of a passage, and this keeps the
    prompt (and cost, and latency) bounded.
    """

    def __init__(self, config: RerankConfig, client: "object" = None) -> None:
        from .generate import GroqClient  # local import avoids a cycle

        self._config = config
        self._client = client or GroqClient.from_env()

    def rerank(
        self, query: str, passages: Sequence[str], top_k: int
    ) -> List[Tuple[int, float]]:
        if not passages:
            return []

        listing = "\n\n".join(
            f"[{i + 1}] {passage[:700]}" for i, passage in enumerate(passages)
        )
        user = f"Question: {query}\n\nPassages:\n\n{listing}"

        try:
            raw = self._client.chat(
                system=_RERANK_SYSTEM_PROMPT,
                user=user,
                temperature=0.0,
                max_tokens=600,
            )
            scores = self._parse_scores(raw, len(passages))
        except Exception:
            # Reranking is an enhancement, never a hard dependency. If grading
            # fails we degrade to retrieval order rather than failing the query.
            return NoopReranker().rerank(query, passages, top_k)

        ranked = sorted(
            ((i, score) for i, score in enumerate(scores)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]

    @staticmethod
    def _parse_scores(raw: str, count: int) -> List[float]:
        """Extract scores, tolerating prose or code fences around the JSON."""
        scores = [0.0] * count
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                for key, value in parsed.items():
                    index = int(str(key).strip()) - 1
                    if 0 <= index < count:
                        scores[index] = float(value)
                return scores
            except (ValueError, TypeError):
                pass

        # Fallback: scrape "number: score" pairs out of free text.
        for key, value in re.findall(r"(\d+)\s*[:=]\s*(\d+(?:\.\d+)?)", raw):
            index = int(key) - 1
            if 0 <= index < count:
                scores[index] = float(value)
        return scores


def get_reranker(config: RerankConfig) -> Reranker:
    if not config.enabled:
        return NoopReranker()
    provider = config.provider.lower()
    if provider in ("cross-encoder", "crossencoder", "ce"):
        return CrossEncoderReranker(config)
    if provider == "llm":
        return LLMReranker(config)
    if provider in ("none", "noop"):
        return NoopReranker()
    raise ValueError(f"unknown rerank provider: {config.provider!r}")


__all__ = (
    "Reranker",
    "NoopReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "get_reranker",
)
