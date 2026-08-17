"""DocsRAG: a measured retrieval-augmented generation system over technical docs.

The point of this package is not that it does RAG -- everything does RAG. The
point is that every retrieval design choice in it is measured against an
evaluation set instead of assumed.

Quick start::

    from docsrag import Settings, RagPipeline

    pipeline = RagPipeline.from_index_dir(settings=Settings.from_env())
    result = pipeline.answer("How does Array.prototype.map handle holes?")
    print(result.answer.text)
    for citation in result.answer.citations:
        print(citation.marker, citation.source)
"""

from .bm25 import BM25
from .chunking import Chunk, chunk_markdown, parse_sections
from .config import (
    ChunkConfig,
    EmbeddingConfig,
    GenerationConfig,
    RerankConfig,
    RetrievalConfig,
    Settings,
)
from .embeddings import get_embedder, l2_normalize
from .fusion import normalized_score_fusion, reciprocal_rank_fusion
from .generate import Answer, Citation, GroqClient, generate_answer
from .index import RagIndex, RetrievedChunk
from .pipeline import PipelineResult, RagPipeline
from .rerank import get_reranker
from .vectorstore import VectorStore, build_vector_store

__version__ = "0.1.0"

__all__ = (
    "__version__",
    "BM25",
    "Chunk",
    "chunk_markdown",
    "parse_sections",
    "Settings",
    "ChunkConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "RerankConfig",
    "GenerationConfig",
    "get_embedder",
    "l2_normalize",
    "reciprocal_rank_fusion",
    "normalized_score_fusion",
    "Answer",
    "Citation",
    "GroqClient",
    "generate_answer",
    "RagIndex",
    "RetrievedChunk",
    "RagPipeline",
    "PipelineResult",
    "get_reranker",
    "VectorStore",
    "build_vector_store",
)
