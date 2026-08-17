"""FastAPI service exposing the RAG pipeline.

Run locally:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    GET  /health         liveness plus index stats
    POST /ask            full pipeline: retrieve, rerank, generate, cite
    POST /retrieve       retrieval only, no LLM call and no API key needed
    GET  /config         effective configuration (secrets redacted)

Design notes worth knowing for interviews:

* The index loads ONCE at startup via the lifespan handler, not per request.
  Loading per request would re-read vectors from disk and re-tokenize the whole
  corpus for BM25 on every call.
* /retrieve exists so the retrieval layer is observable in production without
  spending tokens. Being able to answer "was that a retrieval failure or a
  generation failure?" in prod is the difference between debugging and guessing.
* Errors are mapped to real status codes: 503 when the index is missing (an
  operational problem) versus 502 when the upstream LLM fails (a dependency
  problem). A single 500 for everything makes on-call miserable.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docsrag.config import RerankConfig, RetrievalConfig, Settings
from docsrag.pipeline import RagPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("docsrag.api")

state: Dict[str, Any] = {"pipeline": None, "settings": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at process start."""
    settings = Settings.from_env()
    state["settings"] = settings
    try:
        started = time.perf_counter()
        state["pipeline"] = RagPipeline.from_index_dir(settings=settings)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "index loaded: %d chunks in %.0f ms",
            len(state["pipeline"].index),
            elapsed,
        )
    except Exception as exc:
        # Start anyway so /health can report WHY the service is unhealthy.
        # Crashing at boot with no endpoint gives an opaque restart loop.
        state["error"] = str(exc)
        logger.error("failed to load index: %s", exc)
    yield
    state.clear()


app = FastAPI(
    title="DocsRAG API",
    description="Retrieval-augmented question answering over technical documentation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: Optional[str] = Field(
        None, description="dense | bm25 | hybrid; defaults to configured mode"
    )
    top_k: Optional[int] = Field(None, ge=1, le=20)
    rerank: Optional[bool] = None


class CitationOut(BaseModel):
    marker: int
    source: str
    title: str = ""
    breadcrumb: str = ""
    score: float = 0.0


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationOut] = []
    grounded: bool = True
    refused: bool = False
    timings_ms: Dict[str, float] = {}


class PassageOut(BaseModel):
    id: str
    source: str
    breadcrumb: str = ""
    score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None
    text: str = ""


class RetrieveResponse(BaseModel):
    question: str
    passages: List[PassageOut] = []
    latency_ms: float = 0.0


def get_pipeline() -> RagPipeline:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Index not loaded: "
                f"{state.get('error') or 'unknown error'}. "
                "Build it with: python scripts/build_index.py"
            ),
        )
    return pipeline


def apply_overrides(pipeline: RagPipeline, request: AskRequest) -> Settings:
    """Per-request configuration overrides, useful for live A/B comparison."""
    settings = pipeline.settings
    if request.mode or request.top_k is not None:
        settings = settings.with_overrides(
            retrieval=RetrievalConfig(
                **{
                    **settings.retrieval.__dict__,
                    "mode": request.mode or settings.retrieval.mode,
                    "top_k": request.top_k or settings.retrieval.top_k,
                }
            )
        )
    if request.rerank is not None:
        settings = settings.with_overrides(
            rerank=RerankConfig(
                **{**settings.rerank.__dict__, "enabled": request.rerank}
            )
        )
    return settings


@app.get("/health")
def health() -> Dict[str, Any]:
    pipeline = state.get("pipeline")
    if pipeline is None:
        return {"status": "degraded", "error": state.get("error")}
    return {
        "status": "ok",
        "chunks": len(pipeline.index),
        "dimension": pipeline.index.vector_store.dimension,
        "manifest": pipeline.index.manifest,
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    settings = state.get("settings")
    if settings is None:
        raise HTTPException(status_code=503, detail="settings unavailable")
    # describe() never includes API keys.
    return settings.describe()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: AskRequest) -> RetrieveResponse:
    pipeline = get_pipeline()
    original = pipeline.settings
    pipeline.settings = apply_overrides(pipeline, request)
    try:
        started = time.perf_counter()
        retrieved = pipeline.retrieve(request.question)
        latency = (time.perf_counter() - started) * 1000
    finally:
        pipeline.settings = original

    return RetrieveResponse(
        question=request.question,
        passages=[
            PassageOut(
                id=item.chunk.id,
                source=item.chunk.source,
                breadcrumb=item.chunk.breadcrumb,
                score=item.score,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                rerank_score=item.rerank_score,
                text=item.chunk.text[:800],
            )
            for item in retrieved
        ],
        latency_ms=latency,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    original = pipeline.settings
    pipeline.settings = apply_overrides(pipeline, request)
    try:
        result = pipeline.answer(request.question)
    except RuntimeError as exc:
        # Missing/invalid API key or upstream LLM failure.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        pipeline.settings = original

    logger.info(
        "ask question=%r grounded=%s refused=%s total_ms=%.0f",
        request.question[:80],
        result.answer.grounded,
        result.answer.refused,
        result.timings_ms.get("total", 0.0),
    )

    return AskResponse(
        question=result.question,
        answer=result.answer.text,
        citations=[
            CitationOut(
                marker=c.marker,
                source=c.source,
                title=c.title,
                breadcrumb=c.breadcrumb,
                score=c.score,
            )
            for c in result.answer.citations
        ],
        grounded=result.answer.grounded,
        refused=result.answer.refused,
        timings_ms=result.timings_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
