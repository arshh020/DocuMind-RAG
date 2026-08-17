"""Streamlit UI for DocsRAG.

Run:
    streamlit run app/streamlit_app.py

The thing this UI does that most tutorial RAG demos do not: it renders the
retrieved passages with their source paths and per-component scores next to every
answer. That turns the demo from "trust me" into "here is the evidence", and it
is the single most persuasive thing you can put in front of an interviewer
sharing your screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from docsrag.config import RerankConfig, RetrievalConfig, Settings
from docsrag.pipeline import RagPipeline

st.set_page_config(
    page_title="DocuMind", page_icon="", layout="wide"
)


@st.cache_resource(show_spinner="Loading index...")
def load_pipeline(index_dir: str) -> RagPipeline:
    """Cached so the index is loaded once per process, not once per keystroke."""
    settings = Settings.from_env()
    return RagPipeline.from_index_dir(index_dir, settings=settings)


def main() -> None:
    st.title("DocuMind")
    st.caption(
        "Retrieval-augmented question answering over technical documentation, "
        "with hybrid retrieval and inspectable evidence."
    )

    settings = Settings.from_env()

    with st.sidebar:
        st.header("Retrieval settings")
        mode = st.selectbox(
            "Mode",
            options=("hybrid", "dense", "bm25"),
            index=("hybrid", "dense", "bm25").index(settings.retrieval.mode)
            if settings.retrieval.mode in ("hybrid", "dense", "bm25")
            else 0,
            help=(
                "dense = embedding similarity (semantic). "
                "bm25 = lexical keyword match, good for exact API names. "
                "hybrid = both, fused with reciprocal rank fusion."
            ),
        )
        fusion = st.selectbox(
            "Fusion",
            options=("rrf", "normalized"),
            disabled=mode != "hybrid",
            help="RRF uses ranks only, so it is robust to incomparable score scales.",
        )
        candidate_k = st.slider(
            "Candidates retrieved (stage 1)", 5, 60, settings.retrieval.candidate_k
        )
        top_k = st.slider(
            "Passages sent to the LLM (stage 2)", 1, 15, settings.retrieval.top_k
        )
        rerank = st.checkbox(
            "Rerank candidates",
            value=settings.rerank.enabled,
            help="Cross-encoder or LLM reranking of stage-1 candidates.",
        )
        rerank_provider = st.selectbox(
            "Reranker",
            options=("cross-encoder", "llm"),
            disabled=not rerank,
            help="'llm' needs no local model download.",
        )
        retrieval_only = st.checkbox(
            "Retrieval only (no LLM call)",
            value=False,
            help="Inspect what the retriever found without spending tokens.",
        )

        st.divider()
        st.caption(
            f"Embeddings: `{settings.embedding.provider}` / "
            f"`{settings.embedding.model}`"
        )
        st.caption(f"Generator: `{settings.generation.model}`")

    try:
        pipeline = load_pipeline(str(settings.paths.index_dir))
    except Exception as exc:
        st.error(
            f"Could not load the index: {exc}\n\n"
            "Build it first:\n"
            "```\npython scripts/fetch_corpus.py\npython scripts/build_index.py\n```"
        )
        return

    with st.sidebar:
        st.success(f"{len(pipeline.index):,} chunks indexed")

    # Apply the sidebar settings to this run.
    pipeline.settings = settings.with_overrides(
        retrieval=RetrievalConfig(
            **{
                **settings.retrieval.__dict__,
                "mode": mode,
                "fusion": fusion,
                "candidate_k": max(candidate_k, top_k),
                "top_k": top_k,
            }
        ),
        rerank=RerankConfig(
            **{
                **settings.rerank.__dict__,
                "enabled": rerank,
                "provider": rerank_provider,
                "top_k": top_k,
            }
        ),
    )
    if rerank:
        # Reranker choice changed, so rebuild it rather than reuse a stale one.
        from docsrag.rerank import get_reranker

        pipeline.reranker = get_reranker(pipeline.settings.rerank)

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the difference between == and === in JavaScript?",
    )

    if not question:
        st.info(
            "Ask a question about the indexed documentation. Try toggling "
            "between `dense`, `bm25`, and `hybrid` on the same question to see "
            "how retrieval mode changes the evidence."
        )
        return

    if retrieval_only:
        with st.spinner("Retrieving..."):
            retrieved = pipeline.retrieve(question)
        if not retrieved:
            st.warning("No passages passed the score threshold.")
            return
        st.subheader(f"{len(retrieved)} passages retrieved")
        render_passages(retrieved)
        return

    with st.spinner("Retrieving and generating..."):
        try:
            result = pipeline.answer(question)
        except RuntimeError as exc:
            st.error(
                f"{exc}\n\nSet `GROQ_API_KEY` in your `.env`, or tick "
                "'Retrieval only' to use the app without an LLM."
            )
            return

    answer_column, evidence_column = st.columns([3, 2], gap="large")

    with answer_column:
        st.subheader("Answer")
        if result.answer.refused:
            st.warning(result.answer.text)
        else:
            st.markdown(result.answer.text)

        if result.answer.citations:
            st.markdown("**Sources**")
            for citation in result.answer.citations:
                label = citation.breadcrumb or citation.title or citation.source
                st.markdown(f"`[{citation.marker}]` {label}  \n<small>{citation.source}</small>", unsafe_allow_html=True)
        elif not result.answer.refused:
            st.warning(
                "This answer contained no valid citations, so it is not "
                "verifiably grounded in the retrieved context. Treat it with "
                "suspicion."
            )

        timings = result.timings_ms
        metric_columns = st.columns(3)
        metric_columns[0].metric("Retrieve", f"{timings.get('retrieve', 0):.0f} ms")
        metric_columns[1].metric("Generate", f"{timings.get('generate', 0):.0f} ms")
        metric_columns[2].metric("Total", f"{timings.get('total', 0):.0f} ms")

    with evidence_column:
        st.subheader("Retrieved evidence")
        render_passages(result.retrieved)


def render_passages(retrieved) -> None:
    """Render passages with source paths and component scores."""
    for position, item in enumerate(retrieved, start=1):
        header = f"[{position}] {item.chunk.breadcrumb or item.chunk.title}"
        with st.expander(header, expanded=position == 1):
            score_bits = [
                f"**fused** `{item.score:.4f}`",
                f"dense `{item.dense_score:.4f}`",
                f"bm25 `{item.bm25_score:.4f}`",
            ]
            if item.rerank_score is not None:
                score_bits.append(f"rerank `{item.rerank_score:.3f}`")
            st.markdown(" | ".join(score_bits))
            st.caption(f"source: {item.chunk.source}  |  chunk id: {item.chunk.id}")
            st.markdown(item.chunk.text)


if __name__ == "__main__":
    main()
