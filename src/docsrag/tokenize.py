"""Tokenization for lexical (BM25) retrieval.

Why this exists instead of `str.split()`:

Technical documentation is full of identifiers like ``addEventListener``,
``Array.prototype.map``, ``max-width`` and ``snake_case_name``. A naive
whitespace tokenizer makes these opaque single tokens, so a user query for
"add event listener" scores 0.0 against a document containing
``addEventListener``. That is a real and very common lexical-retrieval failure
on code-adjacent corpora.

Strategy: emit BOTH the whole normalized identifier AND its sub-parts. The
whole token gives exact-match precision; the parts give recall. BM25's IDF
weighting then handles the rest -- rare whole identifiers stay highly
discriminative while common parts like "get" or "set" get down-weighted
automatically.

No third-party dependencies: keeps install time low and makes this module
trivially unit-testable.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# Split on anything that is not alphanumeric. We deliberately keep digits
# because versions/status codes ("h2", "utf8", "404") carry real signal in docs.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

# Identifier boundary: everything EXCEPT the characters that commonly appear
# inside a single technical identifier (. _ - /). Splitting on this first lets us
# keep "array.prototype.map", "get_user_by_id", and "max-width" intact as whole
# searchable terms before we also break them into parts. Splitting on
# _NON_ALNUM first would destroy those identifiers permanently, losing the
# exact-match precision that makes lexical retrieval worth having.
_IDENTIFIER_SPLIT = re.compile(r"[^A-Za-z0-9._\-/]+")

# camelCase / PascalCase boundary: lower->upper, or acronym followed by word
# ("HTMLElement" -> "HTML", "Element").
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Letter/digit boundaries: "utf8" -> "utf", "8"; "h2" -> "h", "2".
_ALPHA_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")

# A small, deliberately conservative stopword list. We keep it short because
# BM25's IDF already suppresses ubiquitous terms; over-aggressive stopword
# removal hurts phrase-ish queries ("what is this" -> empty query).
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its
    of on or that the their there these this to was were what when where which
    who why will with you your
    """.split()
)


def _split_identifier(token: str) -> List[str]:
    """Break a single identifier into meaningful sub-parts.

    ``addEventListener`` -> ``["add", "event", "listener"]``
    ``HTMLElement``      -> ``["html", "element"]``
    ``utf8``             -> ``["utf", "8"]``
    """
    parts = _CAMEL_BOUNDARY.split(token)
    out: List[str] = []
    for part in parts:
        out.extend(p for p in _ALPHA_DIGIT.split(part) if p)
    return [p.lower() for p in out]


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    min_length: int = 1,
) -> List[str]:
    """Tokenize text for lexical retrieval.

    Emits the normalized whole token plus its sub-parts when splitting actually
    produces something new. Order is preserved and duplicates are kept, because
    BM25 is a bag-of-words model that uses term frequency.
    """
    if not text:
        return []

    tokens: List[str] = []
    for candidate in _IDENTIFIER_SPLIT.split(text):
        # Trim separators that are punctuation here rather than part of the
        # identifier, e.g. a trailing period at the end of a sentence.
        candidate = candidate.strip("._-/")
        if not candidate:
            continue

        # Ordered, de-duplicated per source word. De-duplicating WITHIN a word
        # matters: for an ordinary word the whole token and its only atom are
        # identical, and emitting both would double its term frequency and
        # quietly corrupt BM25 scoring.
        emitted: List[str] = [candidate.lower()]

        for atom in _NON_ALNUM.split(candidate):
            if not atom:
                continue
            atom_lower = atom.lower()
            if atom_lower not in emitted:
                emitted.append(atom_lower)
            parts = _split_identifier(atom)
            if len(parts) > 1:
                for part in parts:
                    if part not in emitted:
                        emitted.append(part)

        for tok in emitted:
            if len(tok) < min_length:
                continue
            if remove_stopwords and tok in STOPWORDS:
                continue
            tokens.append(tok)

    return tokens


def tokenize_many(
    texts: Iterable[str], **kwargs
) -> List[List[str]]:
    """Vectorized convenience wrapper around :func:`tokenize`."""
    return [tokenize(t, **kwargs) for t in texts]


def approx_token_count(text: str) -> int:
    """Cheap proxy for LLM token count, used for chunk sizing.

    We intentionally avoid a real BPE tokenizer here so the package has no
    heavyweight dependency. The ~4-characters-per-token heuristic is accurate
    enough for chunk *budgeting* (we only need consistency, not exactness), and
    we take the max against word count so code-dense text is not underestimated.

    If you want exact counts, install ``tiktoken`` and swap this function --
    every caller goes through it.
    """
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4
    word_estimate = len(text.split())
    return max(char_estimate, word_estimate)


__all__: Sequence[str] = (
    "tokenize",
    "tokenize_many",
    "approx_token_count",
    "STOPWORDS",
)
