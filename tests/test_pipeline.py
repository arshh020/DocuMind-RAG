"""End-to-end tests for the index, vector store, generation, and pipeline.

These run fully offline. The whole system is exercised with ``HashEmbedder`` (a
deterministic hashing embedder) and a fake LLM client, so there is no network
call, no API key, and no model download. That is a deliberate design property:
if your RAG system can only be tested against a live paid API, it will not be
tested.

HashEmbedder produces meaningless similarity scores and must never be used for
real answers. It is a test double for the plumbing only.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from docsrag.chunking import chunk_markdown
from docsrag.config import (
    EmbeddingConfig,
    RerankConfig,
    RetrievalConfig,
    Settings,
)
from docsrag.embeddings import HashEmbedder, l2_normalize
from docsrag.generate import (
    build_context_block,
    extract_citation_markers,
    generate_answer,
)
from docsrag.index import RagIndex
from docsrag.pipeline import RagPipeline
from docsrag.vectorstore import VectorStore

DOCUMENTS = {
    "flexbox.md": """# CSS Flexbox

## Centering

To center a child both horizontally and vertically, set display to flex on the
parent, then use justify-content center and align-items center.

## Flex direction

The flex-direction property sets the main axis, either row or column.
""",
    "promises.md": """# JavaScript Promises

## Async await

The await keyword pauses execution of an async function until the promise
settles, then resumes with the resolved value.

## Error handling

Use try and catch around await, or attach a catch handler to the promise chain.
""",
    "arrays.md": """# Array methods

## Map

The map method returns a new array built from the return value of the callback
applied to each element.

## Filter

The filter method returns a new array containing only elements for which the
callback returned a truthy value.
""",
}


def build_test_settings(**retrieval_overrides) -> Settings:
    base = Settings.from_env()
    return base.with_overrides(
        embedding=EmbeddingConfig(
            **{
                **base.embedding.__dict__,
                "provider": "hash",
                "dimension": 128,
            }
        ),
        retrieval=RetrievalConfig(
            **{
                **base.retrieval.__dict__,
                "candidate_k": 10,
                "top_k": 3,
                "mode": "hybrid",
                **retrieval_overrides,
            }
        ),
        rerank=RerankConfig(**{**base.rerank.__dict__, "enabled": False}),
    )


def build_test_index(settings: Settings | None = None) -> RagIndex:
    settings = settings or build_test_settings()
    chunks = []
    for name, text in DOCUMENTS.items():
        chunks.extend(chunk_markdown(text, source=name, max_tokens=128))
    return RagIndex.build(
        chunks, settings=settings, embedder=HashEmbedder(dimension=128)
    )


class FakeLLM:
    """Stand-in for GroqClient that records prompts and returns canned text."""

    def __init__(self, reply: str = "Flexbox centers content. [1]"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, system: str, user: str, **kwargs) -> str:
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.reply


class TestVectorStore(unittest.TestCase):
    def test_normalized_inner_product_equals_cosine(self):
        """Cosine similarity done correctly: normalize, then inner product.

        Getting this wrong (L2 distance on unnormalized vectors, then calling it
        cosine) is a real and common bug. Here we pin the identity.
        """
        rng = np.random.default_rng(0)
        a = rng.normal(size=(1, 64)).astype(np.float32)
        b = rng.normal(size=(1, 64)).astype(np.float32)

        manual = float(
            np.dot(a[0], b[0])
            / (np.linalg.norm(a[0]) * np.linalg.norm(b[0]))
        )
        via_normalization = float(np.dot(l2_normalize(a)[0], l2_normalize(b)[0]))
        self.assertAlmostEqual(manual, via_normalization, places=5)

    def test_l2_normalize_produces_unit_vectors(self):
        rng = np.random.default_rng(1)
        vectors = l2_normalize(rng.normal(size=(5, 32)).astype(np.float32))
        norms = np.linalg.norm(vectors, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_l2_normalize_handles_zero_vector(self):
        """A zero vector must not produce NaN and poison every later search."""
        result = l2_normalize(np.zeros((1, 8), dtype=np.float32))
        self.assertFalse(np.isnan(result).any())

    def test_identical_vector_scores_one(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = VectorStore(vectors, ids=["a", "b"])
        results = store.search(np.array([1.0, 0.0], dtype=np.float32), k=1)
        self.assertEqual(results[0][0], "a")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_search_returns_descending_scores(self):
        rng = np.random.default_rng(2)
        store = VectorStore(
            rng.normal(size=(20, 16)).astype(np.float32),
            ids=[str(i) for i in range(20)],
        )
        results = store.search(rng.normal(size=16).astype(np.float32), k=5)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_larger_than_corpus_is_safe(self):
        store = VectorStore(
            np.eye(3, dtype=np.float32), ids=["a", "b", "c"]
        )
        results = store.search(np.array([1, 0, 0], dtype=np.float32), k=50)
        self.assertEqual(len(results), 3)

    def test_save_and_load_round_trip(self):
        vectors = np.eye(4, dtype=np.float32)
        store = VectorStore(vectors, ids=["a", "b", "c", "d"])
        with tempfile.TemporaryDirectory() as temp:
            store.save(Path(temp))
            restored = VectorStore.load(Path(temp))
        self.assertEqual(restored.ids, store.ids)
        self.assertEqual(restored.dimension, store.dimension)


class TestRagIndex(unittest.TestCase):
    def setUp(self):
        self.settings = build_test_settings()
        self.index = build_test_index(self.settings)

    def test_index_contains_chunks(self):
        self.assertGreater(len(self.index), 0)

    def test_dense_retrieval_returns_results(self):
        results = self.index.retrieve("how do I center a div", mode="dense")
        self.assertTrue(results)

    def test_bm25_retrieval_finds_exact_terms(self):
        """Lexical retrieval is what saves you on exact API names."""
        results = self.index.retrieve("justify-content", mode="bm25", top_k=3)
        self.assertTrue(results)
        joined = " ".join(r.chunk.text.lower() for r in results)
        self.assertIn("justify-content", joined)

    def test_bm25_finds_filter_method(self):
        results = self.index.retrieve("filter method truthy", mode="bm25", top_k=3)
        self.assertTrue(results)
        self.assertIn(
            "filter", " ".join(r.chunk.text.lower() for r in results)
        )

    def test_hybrid_retrieval_returns_results(self):
        results = self.index.retrieve("await keyword", mode="hybrid", top_k=3)
        self.assertTrue(results)

    def test_retrieval_respects_top_k(self):
        results = self.index.retrieve("array", mode="bm25", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_component_scores_are_recorded(self):
        """Debuggability: we keep dense and lexical scores separately."""
        results = self.index.retrieve("map method callback", mode="hybrid", top_k=3)
        self.assertTrue(results)
        self.assertTrue(
            any(r.dense_score != 0.0 or r.bm25_score != 0.0 for r in results)
        )

    def test_retrieved_ids_exist_in_index(self):
        results = self.index.retrieve("promise", mode="hybrid", top_k=5)
        known = {c.id for c in self.index.chunks}
        for item in results:
            self.assertIn(item.chunk.id, known)

    def test_to_passage_shape(self):
        results = self.index.retrieve("flexbox", mode="bm25", top_k=1)
        self.assertTrue(results)
        passage = results[0].to_passage()
        for key in ("id", "text", "source", "breadcrumb", "score"):
            self.assertIn(key, passage)

    def test_save_and_load_preserves_retrieval(self):
        """Artifacts must survive a round trip, and contain no pickle."""
        query = "center a child element"
        before = [r.chunk.id for r in self.index.retrieve(query, top_k=3)]

        with tempfile.TemporaryDirectory() as temp:
            self.index.save(temp)
            files = {p.name for p in Path(temp).iterdir()}
            self.assertIn("chunks.jsonl", files)
            self.assertIn("vectors.npy", files)
            self.assertIn("manifest.json", files)
            for name in files:
                self.assertFalse(
                    name.endswith((".pkl", ".pickle")),
                    "index artifacts must not use pickle",
                )
            restored = RagIndex.load(temp, settings=self.settings)

        self.assertEqual(len(restored), len(self.index))
        self.assertEqual(
            before, [r.chunk.id for r in restored.retrieve(query, top_k=3)]
        )

    def test_manifest_records_provenance(self):
        manifest = self.index.manifest
        self.assertIn("embedding_provider", manifest)
        self.assertIn("chunk_count", manifest)


class TestGeneration(unittest.TestCase):
    def test_context_block_numbers_passages(self):
        passages = [
            {"text": "First fact.", "source": "a.md", "breadcrumb": "A"},
            {"text": "Second fact.", "source": "b.md", "breadcrumb": "B"},
        ]
        block = build_context_block(passages)
        self.assertIn("[1]", block)
        self.assertIn("[2]", block)
        self.assertIn("a.md", block)

    def test_extract_citation_markers(self):
        self.assertEqual(
            extract_citation_markers("Facts [1] and more [3]."), [1, 3]
        )

    def test_extract_citation_markers_deduplicates(self):
        self.assertEqual(extract_citation_markers("[2] then [2] again"), [2])

    def test_extract_citation_markers_none_present(self):
        self.assertEqual(extract_citation_markers("No citations here."), [])

    def test_refuses_without_calling_llm_when_no_passages(self):
        """Zero retrieved passages must short-circuit before spending a token."""
        llm = FakeLLM()
        answer = generate_answer("anything?", [], client=llm)
        self.assertTrue(answer.refused)
        self.assertEqual(llm.calls, [], "LLM must not be called with no context")

    def test_refuses_on_insufficient_context_sentinel(self):
        llm = FakeLLM(reply="INSUFFICIENT_CONTEXT")
        answer = generate_answer(
            "unanswerable?",
            [{"text": "unrelated", "source": "x.md"}],
            client=llm,
        )
        self.assertTrue(answer.refused)

    def test_valid_citation_is_resolved_to_source(self):
        llm = FakeLLM(reply="Set display to flex. [1]")
        answer = generate_answer(
            "how to center?",
            [{"text": "use flex", "source": "flexbox.md", "breadcrumb": "CSS"}],
            client=llm,
        )
        self.assertFalse(answer.refused)
        self.assertTrue(answer.grounded)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].source, "flexbox.md")

    def test_out_of_range_citation_is_dropped(self):
        """A [7] when only 1 passage exists is a hallucinated reference.

        Rendering it would show the user a source that was never retrieved.
        """
        llm = FakeLLM(reply="Claim with a bogus marker. [7]")
        answer = generate_answer(
            "question?",
            [{"text": "only passage", "source": "a.md"}],
            client=llm,
        )
        self.assertEqual(answer.citations, [])
        self.assertFalse(answer.grounded)

    def test_answer_without_citations_is_not_grounded(self):
        llm = FakeLLM(reply="A confident claim with no citation at all.")
        answer = generate_answer(
            "question?", [{"text": "ctx", "source": "a.md"}], client=llm
        )
        self.assertFalse(answer.grounded)

    def test_prompt_includes_retrieved_context(self):
        llm = FakeLLM()
        generate_answer(
            "how to center?",
            [{"text": "UNIQUE_MARKER_TEXT", "source": "a.md"}],
            client=llm,
        )
        self.assertIn("UNIQUE_MARKER_TEXT", llm.calls[0]["user"])

    def test_answer_serializes(self):
        llm = FakeLLM(reply="Answer. [1]")
        answer = generate_answer(
            "q?", [{"text": "ctx", "source": "a.md"}], client=llm
        )
        payload = answer.to_dict()
        self.assertIn("text", payload)
        self.assertIn("citations", payload)


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.settings = build_test_settings()
        self.index = build_test_index(self.settings)

    def test_retrieve_only_needs_no_llm(self):
        pipeline = RagPipeline(self.index, settings=self.settings)
        self.assertTrue(pipeline.retrieve("how do I center a div"))

    def test_full_answer_flow(self):
        pipeline = RagPipeline(
            self.index,
            settings=self.settings,
            llm=FakeLLM(reply="Use flexbox centering. [1]"),
        )
        result = pipeline.answer("how do I center a div?")
        self.assertFalse(result.answer.refused)
        self.assertTrue(result.retrieved)
        self.assertIn("total", result.timings_ms)

    def test_result_serializes_for_api(self):
        pipeline = RagPipeline(
            self.index, settings=self.settings, llm=FakeLLM(reply="Answer. [1]")
        )
        payload = pipeline.answer("what does map do?").to_dict()
        self.assertIn("answer", payload)
        self.assertIn("retrieved", payload)

    def test_top_k_bounds_passages_sent_to_llm(self):
        settings = build_test_settings(top_k=2)
        pipeline = RagPipeline(
            self.index, settings=settings, llm=FakeLLM(reply="A. [1]")
        )
        self.assertLessEqual(len(pipeline.answer("array methods").retrieved), 2)

    def test_all_retrieval_modes_work_end_to_end(self):
        for mode in ("dense", "bm25", "hybrid"):
            settings = build_test_settings(mode=mode)
            self.index.retrieval = settings.retrieval
            pipeline = RagPipeline(
                self.index, settings=settings, llm=FakeLLM(reply="A. [1]")
            )
            with self.subTest(mode=mode):
                self.assertTrue(
                    pipeline.retrieve("await keyword async function"),
                    f"mode {mode} returned nothing",
                )


class TestEvaluationHarness(unittest.TestCase):
    def test_retrieval_evaluation_produces_metrics(self):
        """The eval harness must run offline, so you can iterate for free."""
        from docsrag.evaluation.dataset import EvalExample
        from docsrag.evaluation.retrieval import evaluate_retrieval

        settings = build_test_settings()
        index = build_test_index(settings)

        # Build gold labels from the index itself so the test is self-contained.
        target = next(
            c for c in index.chunks if "justify-content" in c.text.lower()
        )
        examples = [
            EvalExample(
                id="q1",
                question="how do I center with justify-content and align-items",
                gold_chunk_ids=[target.id],
                verified=True,
            )
        ]

        pipeline = RagPipeline(index, settings=settings)
        report = evaluate_retrieval(pipeline, examples, label="test")

        self.assertEqual(report.n_queries, 1)
        self.assertIn("recall@5", report.metrics)
        self.assertEqual(
            report.metrics["recall@5"], 1.0, "gold chunk should be found"
        )

    def test_sweep_compares_configurations(self):
        from docsrag.evaluation.dataset import EvalExample
        from docsrag.evaluation.retrieval import sweep, to_markdown_table

        settings = build_test_settings()
        index = build_test_index(settings)
        target = next(c for c in index.chunks if "filter" in c.text.lower())
        examples = [
            EvalExample(
                id="q1",
                question="filter method truthy callback",
                gold_chunk_ids=[target.id],
            )
        ]

        reports = sweep(index, examples, base=settings)
        self.assertGreaterEqual(len(reports), 3)
        table = to_markdown_table(reports)
        self.assertIn("recall@5", table)
        self.assertIn("|", table)

    def test_eval_set_round_trip(self):
        from docsrag.evaluation.dataset import (
            EvalExample,
            load_eval_set,
            save_eval_set,
        )

        examples = [
            EvalExample(id="q1", question="a?", gold_chunk_ids=["x"], verified=True),
            EvalExample(id="q2", question="b?", gold_chunk_ids=["y"]),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evalset.jsonl"
            self.assertEqual(save_eval_set(examples, path), 2)
            restored = load_eval_set(path)
        self.assertEqual([e.id for e in restored], ["q1", "q2"])
        self.assertTrue(restored[0].verified)
        self.assertFalse(restored[1].verified)


if __name__ == "__main__":
    unittest.main(verbosity=2)
