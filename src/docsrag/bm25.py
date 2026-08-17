"""BM25-Okapi lexical retriever, implemented from scratch.

Why hand-rolled instead of ``rank_bm25``:

1. It is ~80 lines and removes a dependency.
2. You can explain every term in an interview, which you cannot do with a
   black-box import.
3. It lets us use a sparse CSC-style layout so scoring touches only the
   postings lists for query terms, not the whole corpus.

The scoring function (Robertson/Sparck-Jones BM25):

    score(D, Q) = sum_{t in Q} IDF(t) * ( f(t,D) * (k1 + 1) )
                                       / ( f(t,D) + k1 * (1 - b + b * |D|/avgdl) )

    IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

where ``f(t,D)`` is term frequency in the document, ``n(t)`` is document
frequency, ``N`` is corpus size, ``|D|`` is document length and ``avgdl`` is
mean document length.

Parameter intuition (common interview question):
  * ``k1`` controls term-frequency saturation. Higher = repeated terms keep
    adding score. 1.2-2.0 is the standard range; we default to 1.5.
  * ``b`` controls length normalization. b=0 ignores document length entirely,
    b=1 fully normalizes. 0.75 is the near-universal default.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .tokenize import tokenize


class BM25:
    """An in-memory BM25-Okapi index over a fixed document collection.

    Memory layout: for each term we store parallel arrays of ``(doc_ids,
    term_freqs)``. This is an inverted index, so query cost is proportional to
    the number of postings for the query terms rather than to corpus size.
    """

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")

        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(corpus_tokens)

        self.doc_lengths = np.array(
            [len(doc) for doc in corpus_tokens], dtype=np.float64
        )
        # Guard against an empty corpus so avgdl is never NaN.
        self.avg_doc_length = (
            float(self.doc_lengths.mean()) if self.n_docs else 0.0
        )

        # Inverted index: term -> (np.array[doc_id], np.array[term_freq])
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: Dict[str, float] = {}
        self._build(corpus_tokens)

    def _build(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        raw: Dict[str, List[Tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(corpus_tokens):
            for term, freq in Counter(tokens).items():
                raw.setdefault(term, []).append((doc_id, freq))

        for term, postings in raw.items():
            doc_ids = np.fromiter(
                (d for d, _ in postings), dtype=np.int64, count=len(postings)
            )
            freqs = np.fromiter(
                (f for _, f in postings), dtype=np.float64, count=len(postings)
            )
            self._postings[term] = (doc_ids, freqs)

            # Document frequency -> IDF. The +1 inside the log keeps IDF
            # strictly positive, which avoids the classic BM25 pathology where
            # terms appearing in >half the corpus get a NEGATIVE weight and
            # actively push relevant documents down the ranking.
            n_t = len(postings)
            self._idf[term] = math.log(
                1.0 + (self.n_docs - n_t + 0.5) / (n_t + 0.5)
            )

    @classmethod
    def from_texts(cls, texts: Sequence[str], **kwargs) -> "BM25":
        return cls([tokenize(t) for t in texts], **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    def get_idf(self, term: str) -> float:
        """Exposed for debugging/inspection: which query terms actually matter."""
        return self._idf.get(term, 0.0)

    def score_all(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Return a dense score vector of shape ``(n_docs,)``.

        Documents that share no term with the query keep a score of exactly 0.0.
        """
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0 or self.avg_doc_length == 0.0:
            return scores

        # Length-normalization denominator component, precomputed per document.
        length_norm = self.k1 * (
            1.0 - self.b + self.b * (self.doc_lengths / self.avg_doc_length)
        )

        # A query term repeated twice should count twice, so iterate the
        # multiset rather than a set.
        for term, q_freq in Counter(query_tokens).items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, freqs = posting
            idf = self._idf[term]
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + length_norm[doc_ids]
            scores[doc_ids] += q_freq * idf * (numerator / denominator)

        return scores

    def search(
        self, query: str, k: int = 10, *, tokens: Sequence[str] | None = None
    ) -> List[Tuple[int, float]]:
        """Return the top-``k`` ``(doc_id, score)`` pairs, highest score first.

        Documents with a zero score are excluded: a BM25 hit with no shared
        term is not a hit at all, and returning them would silently pad the
        candidate list with noise before fusion.
        """
        query_tokens = list(tokens) if tokens is not None else tokenize(query)
        if not query_tokens:
            return []

        scores = self.score_all(query_tokens)
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []

        k = max(1, min(k, nonzero.size))
        # argpartition is O(n) vs O(n log n) for a full sort -- matters once the
        # corpus is large and k is small.
        top_unsorted = nonzero[
            np.argpartition(-scores[nonzero], k - 1)[:k]
        ]
        top = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [(int(i), float(scores[i])) for i in top]


__all__ = ("BM25",)
