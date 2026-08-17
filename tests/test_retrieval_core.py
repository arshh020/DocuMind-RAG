"""Tests for the retrieval primitives: tokenizer, BM25, fusion, metrics.

Written with stdlib unittest so they run with either:
    python -m unittest discover -s tests -v
    pytest tests -v

These are the tests that matter most, because BM25 scoring, rank fusion, and IR
metrics are exactly the places where a subtle bug silently produces
plausible-but-wrong numbers. A retrieval bug does not crash; it just quietly
makes your evaluation meaningless.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.bm25 import BM25
from docsrag.fusion import (
    _min_max_normalize,
    normalized_score_fusion,
    reciprocal_rank_fusion,
)
from docsrag.metrics import (
    aggregate,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from docsrag.tokenize import approx_token_count, tokenize


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_removes_stopwords(self):
        tokens = tokenize("the quick brown fox is a animal")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertIn("quick", tokens)

    def test_splits_camel_case_identifiers(self):
        """The whole point: 'add event listener' must match 'addEventListener'."""
        tokens = tokenize("addEventListener")
        self.assertIn("addeventlistener", tokens)
        self.assertIn("add", tokens)
        self.assertIn("event", tokens)
        self.assertIn("listener", tokens)

    def test_splits_snake_case(self):
        tokens = tokenize("get_user_by_id")
        self.assertIn("get_user_by_id", tokens)
        self.assertIn("user", tokens)

    def test_query_matches_identifier_after_split(self):
        query_tokens = set(tokenize("add event listener"))
        doc_tokens = set(tokenize("element.addEventListener(type, handler)"))
        self.assertTrue(query_tokens & doc_tokens)

    def test_splits_alpha_digit_boundary(self):
        tokens = tokenize("utf8")
        self.assertIn("utf", tokens)
        self.assertIn("8", tokens)

    def test_empty_and_punctuation_only(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("...!!!"), [])

    def test_approx_token_count_is_positive_for_text(self):
        self.assertGreater(approx_token_count("hello world foo bar"), 0)
        self.assertEqual(approx_token_count(""), 0)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are common pets",
            "quantum entanglement in particle physics",
        ]
        self.bm25 = BM25.from_texts(self.documents)

    def test_ranks_exact_match_first(self):
        results = self.bm25.search("quantum entanglement", k=2)
        self.assertTrue(results)
        self.assertEqual(results[0][0], 3)

    def test_returns_nothing_for_out_of_vocabulary_query(self):
        self.assertEqual(self.bm25.search("zzzzz nonexistent", k=5), [])

    def test_excludes_zero_score_documents(self):
        """Documents sharing no query term must not appear at all.

        If they did, a hybrid ranking would be padded with irrelevant documents
        that dilute fusion.
        """
        results = self.bm25.search("quantum", k=10)
        self.assertEqual(len(results), 1)

    def test_scores_are_descending(self):
        results = self.bm25.search("cat dog sat", k=4)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_idf_is_strictly_positive(self):
        """Guards the classic Okapi pathology.

        With the textbook IDF formula, a term appearing in more than half the
        documents gets a NEGATIVE weight, so matching it actively hurts a
        document. We use the +1 smoothed variant to keep IDF positive.

        Note "sat" appears in half the corpus and "on" would appear in half too,
        which is exactly the region where the unsmoothed formula goes negative.
        """
        for term in ["sat", "quantum", "cat", "dog"]:
            self.assertGreater(self.bm25.get_idf(term), 0.0, f"idf({term})")

    def test_stopwords_are_not_indexed(self):
        """"the" carries no retrieval signal, so it never reaches the index.

        Consequence worth knowing: get_idf returns 0.0 for any out-of-vocabulary
        term, stopword or typo alike, so a stopword-only query scores nothing
        rather than matching every document.
        """
        self.assertEqual(self.bm25.get_idf("the"), 0.0)
        self.assertEqual(self.bm25.search("the on and are", k=5), [])

    def test_common_term_has_lower_idf_than_rare_term(self):
        self.assertLess(self.bm25.get_idf("sat"), self.bm25.get_idf("quantum"))

    def test_top_k_limits_results(self):
        self.assertLessEqual(len(self.bm25.search("sat on", k=1)), 1)

    def test_empty_corpus_is_safe(self):
        self.assertEqual(BM25.from_texts([]).search("anything", k=5), [])

    def test_vocab_size_reported(self):
        self.assertGreater(self.bm25.vocab_size, 0)

    def test_score_all_returns_one_score_per_document(self):
        scores = self.bm25.score_all("cat")
        self.assertEqual(len(scores), len(self.documents))


class TestFusion(unittest.TestCase):
    def test_rrf_rewards_agreement_across_rankings(self):
        """A document ranked 2nd by both beats one ranked 1st by only one.

        This is the entire value proposition of hybrid retrieval: consensus
        between independent signals is stronger evidence than a single
        confident vote.
        """
        dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
        lexical = [(4, 5.0), (2, 4.0), (5, 3.0)]
        fused = reciprocal_rank_fusion([dense, lexical], k=60)
        ordering = [doc_id for doc_id, _ in fused]
        scores = dict(fused)

        # Doc 2 is 2nd in both lists; docs 1 and 4 are 1st in exactly one.
        self.assertEqual(ordering[0], 2)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[2], scores[4])

    def test_rrf_agreement_margin_is_narrow_by_design(self):
        """Documents in both lists win, but only just, and that is intended.

        Here doc 3 is 1st in one list and 3rd in the other (1/61 + 1/63), while
        doc 2 is 2nd in both (2/62). Those totals differ by under 0.1%, so RRF
        treats "one strong vote plus one weak vote" as roughly equal to "two
        medium votes". Worth knowing before trusting a leaderboard built on tiny
        RRF gaps: at this margin the ordering is effectively a coin flip, which
        is a large part of why the eval harness reports many queries rather than
        reasoning about single examples.
        """
        dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
        lexical = [(3, 5.0), (2, 4.0), (9, 3.0)]
        scores = dict(reciprocal_rank_fusion([dense, lexical], k=60))

        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[3], scores[1])
        self.assertLess(abs(scores[3] - scores[2]) / scores[2], 0.01)

    def test_rrf_ignores_score_magnitudes(self):
        """RRF depends only on rank, which is why it survives incomparable scales.

        Cosine similarity lives in [-1, 1]; BM25 is unbounded. Adding them
        directly lets BM25 dominate purely because its numbers are bigger.
        """
        small = [(1, 0.001), (2, 0.0005)]
        large = [(1, 900.0), (2, 450.0)]
        self.assertEqual(
            [d for d, _ in reciprocal_rank_fusion([small])],
            [d for d, _ in reciprocal_rank_fusion([large])],
        )

    def test_rrf_handles_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_rrf_single_ranking_preserves_order(self):
        single = [(7, 0.9), (3, 0.5), (5, 0.1)]
        fused = reciprocal_rank_fusion([single])
        self.assertEqual([d for d, _ in fused], [7, 3, 5])

    def test_rrf_deterministic_tie_break(self):
        """Ties must break deterministically or evaluation becomes irreproducible."""
        first = reciprocal_rank_fusion([[(5, 1.0)], [(3, 1.0)]])
        second = reciprocal_rank_fusion([[(5, 1.0)], [(3, 1.0)]])
        self.assertEqual(first, second)

    def test_rrf_weights_shift_ranking(self):
        dense = [(1, 0.9), (2, 0.8)]
        lexical = [(2, 9.0), (1, 1.0)]
        dense_heavy = reciprocal_rank_fusion([dense, lexical], weights=[5.0, 1.0])
        lexical_heavy = reciprocal_rank_fusion([dense, lexical], weights=[1.0, 5.0])
        self.assertEqual(dense_heavy[0][0], 1)
        self.assertEqual(lexical_heavy[0][0], 2)

    def test_min_max_normalize_maps_to_unit_range(self):
        normalized = _min_max_normalize([1.0, 3.0, 5.0])
        self.assertAlmostEqual(min(normalized), 0.0)
        self.assertAlmostEqual(max(normalized), 1.0)

    def test_min_max_normalize_flat_input(self):
        """Identical scores must not divide by zero."""
        self.assertEqual(_min_max_normalize([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0])

    def test_normalized_fusion_combines_both_signals(self):
        fused = normalized_score_fusion([[(1, 0.9), (2, 0.1)], [(2, 10.0), (1, 1.0)]])
        self.assertEqual(len(fused), 2)
        self.assertEqual({d for d, _ in fused}, {1, 2})


class TestMetrics(unittest.TestCase):
    def test_hit_rate(self):
        self.assertEqual(hit_rate_at_k(["a", "b", "c"], {"c"}, 3), 1.0)
        self.assertEqual(hit_rate_at_k(["a", "b", "c"], {"c"}, 2), 0.0)

    def test_recall_partial_credit(self):
        self.assertAlmostEqual(
            recall_at_k(["a", "x", "b"], {"a", "b", "z"}, 3), 2 / 3
        )

    def test_recall_is_one_when_all_found(self):
        self.assertEqual(recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_precision(self):
        self.assertAlmostEqual(precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4), 0.5)

    def test_reciprocal_rank_positions(self):
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"a"}, 3), 1.0)
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"b"}, 3), 0.5)
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"z"}, 3), 0.0)

    def test_reciprocal_rank_respects_the_cutoff(self):
        """A hit beyond k has not been shown to the user, so it scores 0."""
        self.assertEqual(reciprocal_rank(["x", "y", "a"], {"a"}, 2), 0.0)
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], {"a"}, 3), 1 / 3)

    def test_ndcg_rewards_higher_placement(self):
        better = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        worse = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        self.assertGreater(better, worse)

    def test_ndcg_perfect_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_ndcg_zero_when_nothing_relevant_retrieved(self):
        self.assertEqual(ndcg_at_k(["x", "y"], {"a"}, 2), 0.0)

    def test_empty_relevant_set_is_zero_not_crash(self):
        self.assertEqual(recall_at_k(["a"], set(), 1), 0.0)
        self.assertEqual(ndcg_at_k(["a"], set(), 1), 0.0)

    def test_empty_retrieved_list(self):
        self.assertEqual(recall_at_k([], {"a"}, 5), 0.0)
        self.assertEqual(reciprocal_rank([], {"a"}, 5), 0.0)

    def test_evaluate_query_emits_expected_keys(self):
        metrics = evaluate_query(["a", "b", "c"], {"b"}, k_values=(1, 3))
        for key in (
            "recall@1",
            "recall@3",
            "precision@1",
            "hit@1",
            "hit@3",
            "ndcg@3",
            "mrr@3",
        ):
            self.assertIn(key, metrics)

    def test_evaluate_query_values_are_correct(self):
        """Pin the actual numbers, not just the key names.

        The gold document sits at rank 2, so: nothing is found at k=1, and at
        k=3 recall is 1.0, precision is 1/3, and MRR is 1/2.
        """
        metrics = evaluate_query(["a", "b", "c"], {"b"}, k_values=(1, 3))
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertAlmostEqual(metrics["precision@3"], 1 / 3)
        self.assertAlmostEqual(metrics["mrr@3"], 0.5)

    def test_aggregate_macro_averages(self):
        aggregated = aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
        self.assertAlmostEqual(aggregated["recall@5"], 0.5)

    def test_aggregate_empty_input(self):
        self.assertEqual(aggregate([]), {})

    def test_metrics_bounded_zero_to_one(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        for value in evaluate_query(retrieved, relevant).values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
