"""End-to-end RAG pipeline: retrieve, rerank, generate.

This is a thin orchestration layer on purpose. All the interesting logic lives in
testable modules (chunking, bm25, fusion, rerank, generate) so that this class
stays readable and so the eval harness can bypass generation entirely when it
only wants to measure retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import Settings
from .generate import Answer, GroqClient, generate_answer
from .index import RagIndex, RetrievedChunk
from .rerank import Reranker, get_reranker


@dataclass
class PipelineResult:
    """Everything a caller (API, UI, or eval) could want about one query."""

    question: str
    answer: Answer
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "retrieved": [
                {
                    "id": r.chunk.id,
                    "source": r.chunk.source,
                    "title": r.chunk.title,
                    "breadcrumb": r.chunk.breadcrumb,
                    "score": r.score,
                    "dense_score": r.dense_score,
                    "bm25_score": r.bm25_score,
                    "rerank_score": r.rerank_score,
                    "text": r.chunk.text[:500],
                }
                for r in self.retrieved
            ],
            "timings_ms": self.timings_ms,
        }


class RagPipeline:
    """Compose an index, a reranker, and an LLM into a question-answering system."""

    def __init__(
        self,
        index: RagIndex,
        *,
        settings: Settings,
        reranker: Reranker | None = None,
        llm: GroqClient | None = None,
    ) -> None:
        self.index = index
        self.settings = settings
        self.reranker = reranker if reranker is not None else get_reranker(settings.rerank)
        self._llm = llm

    @property
    def llm(self) -> GroqClient:
        """Lazily constructed so retrieval-only evaluation needs no API key."""
        if self._llm is None:
            self._llm = GroqClient.from_env(self.settings.generation)
        return self._llm

    @classmethod
    def from_index_dir(
        cls,
        directory: str | Path | None = None,
        *,
        settings: Settings | None = None,
    ) -> "RagPipeline":
        settings = settings or Settings.from_env()
        directory = directory or settings.paths.index_dir
        index = RagIndex.load(directory, settings=settings)
        return cls(index, settings=settings)

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Retrieve then rerank. No generation, so this needs no LLM key.

        Note the widths: we pull ``candidate_k`` from retrieval, hand all of them
        to the reranker, and only then narrow to ``top_k``. Narrowing before
        reranking would make the reranker pointless.
        """
        config = self.settings.retrieval
        rerank_config = self.settings.rerank

        final_k = rerank_config.top_k if rerank_config.enabled else config.top_k

        candidates = self.index.retrieve(
            question,
            top_k=config.candidate_k,
            candidate_k=config.candidate_k,
        )
        if not candidates:
            return []

        if not rerank_config.enabled:
            return candidates[:final_k]

        passages = [c.chunk.embed_text for c in candidates]
        ranked = self.reranker.rerank(question, passages, final_k)

        out: List[RetrievedChunk] = []
        for index_position, score in ranked:
            if not 0 <= index_position < len(candidates):
                continue
            candidate = candidates[index_position]
            candidate.rerank_score = float(score)
            out.append(candidate)
        return out

    def answer(self, question: str) -> PipelineResult:
        """Full pipeline for one question, with stage timings."""
        timings: Dict[str, float] = {}

        start = time.perf_counter()
        retrieved = self.retrieve(question)
        timings["retrieve"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        if not retrieved:
            # Refusal without an LLM call at all.
            answer = generate_answer(question, [], client=None)  # type: ignore[arg-type]
        else:
            answer = generate_answer(
                question,
                [r.to_passage() for r in retrieved],
                client=self.llm,
                config=self.settings.generation,
            )
        timings["generate"] = (time.perf_counter() - start) * 1000.0
        timings["total"] = timings["retrieve"] + timings["generate"]

        return PipelineResult(
            question=question,
            answer=answer,
            retrieved=retrieved,
            timings_ms=timings,
        )


__all__ = ("RagPipeline", "PipelineResult")
