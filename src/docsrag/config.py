"""Configuration for every stage of the pipeline.

Design choices worth defending in an interview:

* **Plain dataclasses plus os.getenv**, not a settings framework. Zero extra
  dependencies, and the full configuration surface is readable in one file.
* **Every knob is a config field, not a literal buried in a function.** This is
  what makes the ablation sweep possible: the eval harness constructs variant
  Settings objects and re-runs. If chunk size were hard-coded inside the chunker,
  "does chunk size matter?" would be unanswerable without editing code.
* **describe() never returns secrets** and is stamped into every eval result,
  so a results file always records the exact configuration that produced it.
  Benchmark numbers without provenance are just decoration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
FUSION_STRATEGIES = ("rrf", "normalized")


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> int:
    """Minimal .env loader.

    Hand-rolled rather than depending on python-dotenv: it is about fifteen lines
    and removes a dependency from the install path.

    Supports KEY=value, export KEY=value, # comments, and single or double
    quoted values. Returns the number of variables set.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return 0

    count = 0
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


@dataclass
class ChunkConfig:
    """Chunking parameters.

    max_tokens=512 is a starting point, not a truth. It is chosen to match the
    512-token input limit of all-MiniLM-L6-v2: text beyond that limit is
    silently truncated by the model, so embedding larger chunks means embedding
    text the model never reads. Measure alternatives with the eval sweep rather
    than copying a number from a tutorial.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 16

    @classmethod
    def from_env(cls) -> "ChunkConfig":
        return cls(
            max_tokens=_env_int("CHUNK_MAX_TOKENS", 512),
            overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 64),
            min_tokens=_env_int("CHUNK_MIN_TOKENS", 16),
        )


@dataclass
class EmbeddingConfig:
    """Embedding provider settings.

    provider is one of:
      * sentence-transformers -- local model, no API cost, needs torch.
      * openai -- any OpenAI-compatible /embeddings endpoint.
      * hash -- deterministic hashing embedder. A TEST DOUBLE ONLY. It makes the
        whole pipeline runnable offline for tests and smoke checks. Its
        similarity scores are meaningless; never use it for real answers.
    """

    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    normalize: bool = True
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            provider=_env("EMBEDDING_PROVIDER", "sentence-transformers"),
            model=_env(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            dimension=_env_int("EMBEDDING_DIMENSION", 384),
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 64),
            normalize=_env_bool("EMBEDDING_NORMALIZE", True),
            api_base=_env("EMBEDDING_API_BASE", "https://api.openai.com/v1"),
            api_key=_env("EMBEDDING_API_KEY", ""),
        )


@dataclass
class RetrievalConfig:
    """Retrieval and fusion parameters.

    candidate_k versus top_k is the two-stage funnel. Stage 1 retrieves
    candidate_k cheaply; stage 2 narrows to top_k for the prompt. Stage-1 recall
    is a hard ceiling on final answer quality: a chunk that is not in the
    candidate set can never be cited, no matter how good the reranker or LLM is.
    """

    candidate_k: int = 30
    top_k: int = 5
    mode: str = "hybrid"
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"mode must be one of {RETRIEVAL_MODES}, got {self.mode!r}"
            )
        if self.fusion not in FUSION_STRATEGIES:
            raise ValueError(
                f"fusion must be one of {FUSION_STRATEGIES}, got {self.fusion!r}"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.candidate_k < self.top_k:
            self.candidate_k = self.top_k

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            candidate_k=_env_int("RETRIEVAL_CANDIDATE_K", 30),
            top_k=_env_int("RETRIEVAL_TOP_K", 5),
            mode=_env("RETRIEVAL_MODE", "hybrid"),
            fusion=_env("RETRIEVAL_FUSION", "rrf"),
            rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            dense_weight=_env_float("RETRIEVAL_DENSE_WEIGHT", 1.0),
            bm25_weight=_env_float("RETRIEVAL_BM25_WEIGHT", 1.0),
            bm25_k1=_env_float("RETRIEVAL_BM25_K1", 1.5),
            bm25_b=_env_float("RETRIEVAL_BM25_B", 0.75),
            min_score=_env_float("RETRIEVAL_MIN_SCORE", 0.0),
        )


@dataclass
class RerankConfig:
    """Reranking parameters.

    Disabled by default so the baseline is honest: you should be able to show
    what the system does WITHOUT reranking before claiming reranking helped.

    provider='llm' needs no model download, which matters when you are on a
    deadline or deploying to a container with a small image budget.
    """

    enabled: bool = False
    provider: str = "cross-encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 32

    @classmethod
    def from_env(cls) -> "RerankConfig":
        return cls(
            enabled=_env_bool("RERANK_ENABLED", False),
            provider=_env("RERANK_PROVIDER", "cross-encoder"),
            model=_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=_env_int("RERANK_TOP_K", 5),
            batch_size=_env_int("RERANK_BATCH_SIZE", 32),
        )


@dataclass
class GenerationConfig:
    """LLM generation parameters.

    temperature=0.0 is deliberate. For grounded question answering over
    documentation there is no upside to sampling: you want the answer the
    evidence supports, reproducibly. A nonzero temperature on a factual
    reference assistant is a bug, not a style choice, and it also makes
    evaluation noisy because the same question yields different answers.
    """

    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_seconds: float = 60.0
    api_key: str = ""
    reasoning_effort: str = "low"

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        return cls(
            provider=_env("LLM_PROVIDER", "groq"),
            model=_env("LLM_MODEL", "openai/gpt-oss-20b"),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            max_tokens=_env_int("LLM_MAX_TOKENS", 800),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
            api_key=_env("GROQ_API_KEY", ""),
            reasoning_effort=_env("LLM_REASONING_EFFORT", "low"),
        )


@dataclass
class Paths:
    """Filesystem layout. All relative to the repository root by default."""

    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("artifacts/index")
    eval_dir: Path = Path("eval")
    results_dir: Path = Path("artifacts/results")

    def __post_init__(self) -> None:
        self.corpus_dir = Path(self.corpus_dir)
        self.index_dir = Path(self.index_dir)
        self.eval_dir = Path(self.eval_dir)
        self.results_dir = Path(self.results_dir)

    @classmethod
    def from_env(cls) -> "Paths":
        return cls(
            corpus_dir=Path(_env("CORPUS_DIR", "data/corpus")),
            index_dir=Path(_env("INDEX_DIR", "artifacts/index")),
            eval_dir=Path(_env("EVAL_DIR", "eval")),
            results_dir=Path(_env("RESULTS_DIR", "artifacts/results")),
        )


@dataclass
class Settings:
    """Top-level configuration container."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: Paths = field(default_factory=Paths)

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> "Settings":
        """Load configuration from the environment, reading .env first."""
        load_dotenv(dotenv_path)
        return cls(
            chunk=ChunkConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            rerank=RerankConfig.from_env(),
            generation=GenerationConfig.from_env(),
            paths=Paths.from_env(),
        )

    def with_overrides(
        self,
        *,
        chunk: Optional[ChunkConfig] = None,
        embedding: Optional[EmbeddingConfig] = None,
        retrieval: Optional[RetrievalConfig] = None,
        rerank: Optional[RerankConfig] = None,
        generation: Optional[GenerationConfig] = None,
        paths: Optional[Paths] = None,
    ) -> "Settings":
        """Return a NEW Settings with selected sections replaced.

        Immutable by design. The eval sweep builds one Settings per
        configuration; if overrides mutated in place, row 4 of the results table
        would silently inherit row 3's settings and the whole comparison would be
        invalid.
        """
        return replace(
            self,
            chunk=chunk or self.chunk,
            embedding=embedding or self.embedding,
            retrieval=retrieval or self.retrieval,
            rerank=rerank or self.rerank,
            generation=generation or self.generation,
            paths=paths or self.paths,
        )

    def describe(self) -> Dict[str, Any]:
        """Serializable configuration snapshot with secrets removed.

        Stamped into every eval result file so results are always attributable
        to an exact configuration.
        """
        return {
            "chunk": dict(self.chunk.__dict__),
            "embedding": {
                key: value
                for key, value in self.embedding.__dict__.items()
                if key != "api_key"
            },
            "retrieval": dict(self.retrieval.__dict__),
            "rerank": dict(self.rerank.__dict__),
            "generation": {
                key: value
                for key, value in self.generation.__dict__.items()
                if key != "api_key"
            },
        }


__all__ = (
    "Settings",
    "ChunkConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "RerankConfig",
    "GenerationConfig",
    "Paths",
    "load_dotenv",
    "RETRIEVAL_MODES",
    "FUSION_STRATEGIES",
)
