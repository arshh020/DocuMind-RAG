# Architecture and design decisions

This document exists because the code is the easy part. What an interviewer
actually probes is *why* each piece is the way it is, and what you gave up to
get it. Every section below states the decision, the reasoning, and the
trade-off accepted.

## Data flow

```
corpus/*.md
   | strip_frontmatter -> parse_sections (heading-aware, fence-safe)
   v
chunk_markdown  ->  Chunk{id, text, embed_text, breadcrumb, token_count}
   |                        |
   | embed_text             | embed_text
   v                        v
BM25 inverted index    Embedder -> L2-normalized float32 matrix
   |                        |
   +----------+-------------+
              v
      RagIndex.retrieve(query)
        dense top-30  +  bm25 top-30
              v
         RRF fusion (rank-based)
              v
        optional reranker (cross-encoder or LLM)
              v
           top 5 passages
              v
  generate_answer: numbered context, temperature 0, cite-or-refuse
              v
     Answer{text, citations[], timings{}}
```

The important structural property: **retrieval is separable from generation.**
`RagIndex.retrieve()` is a pure function of the index and the query, which is
what makes the evaluation harness possible. If retrieval only existed inside a
chain that also called an LLM, measuring it would cost money and be
non-deterministic, and in practice nobody would measure it at all.

## Decision 1: heading-aware chunking instead of fixed-size splitting

**Decision.** Parse Markdown into heading-scoped sections, track a breadcrumb
stack (h1 > h2 > h3), never split inside a fenced code block, and prepend the
breadcrumb to the text that gets embedded.

**Reasoning.** A fixed 500-character window produces chunks like *"It returns a
new array and does not mutate the original."* What is "it"? The heading that
said `Array.prototype.map()` was 400 characters earlier and landed in a
different chunk. The embedding of an orphaned pronoun is close to meaningless,
and no amount of retrieval tuning downstream recovers information the chunker
threw away.

Documentation already tells you where the semantic boundaries are. Headings are
free, perfectly reliable structure, and character-count splitting discards them.

Two details that matter more than they look:

- `text` (what the user sees) and `embed_text` (breadcrumb + body, what gets
  embedded) are stored separately. The retriever sees contextualized text; the
  citation shows clean text.
- BM25 is built over `embed_text` too, so heading words are lexically
  searchable. A query for "map syntax" matches the Syntax section of the map
  page even when the body never repeats the word "map".

**Trade-off.** Chunk sizes become uneven, because sections are uneven. A tiny
"See also" section and a long "Description" section are not comparable units,
so `_merge_tiny` merges undersized chunks forward and a hard-split fallback
breaks up oversized ones. Uniform-size chunks would be simpler to reason about;
they would just retrieve worse.

**Also handled.** A paragraph larger than the whole budget cannot be fixed by
paragraph-level packing. Without a fallback the chunker emits one enormous
chunk, and two things then break silently: the embedding model truncates at its
context limit so the tail is never indexed at all, and one retrieved chunk can
consume the generator's whole context budget. `_hard_split` splits such prose at
sentence boundaries, then at word windows. Code fences are exempt: a truncated
code example is worse than an oversized chunk.

## Decision 2: an identifier-aware tokenizer

**Decision.** Emit the whole identifier *and* its parts:
`get_user_by_id` produces `get_user_by_id`, `get`, `user`, `id`;
`addEventListener` produces `addeventlistener`, `add`, `event`, `listener`.

**Reasoning.** On a code corpus this is the difference between a lexical
retriever that works and one that does not. A user who types
`Array.prototype.map` must match that exact string, so the whole identifier has
to survive tokenization. A user who types "add event listener" as three words
must also match `addEventListener`, so the parts have to be indexed too.
Standard whitespace-and-punctuation tokenizers give you one or the other, never
both.

**Trade-off.** Index size grows, and duplicate terms would distort term
frequencies, so emission is de-duplicated *within* each source word: for an
ordinary word like "hello" the whole token and its only atom are identical, and
emitting both would double its term frequency and quietly corrupt BM25 scoring.
Across separate occurrences, counts are preserved, because that is real signal.

## Decision 3: smoothed BM25 IDF

**Decision.** Use `idf(t) = log(1 + (N - n_t + 0.5) / (n_t + 0.5))` rather than
the textbook `log((N - n_t + 0.5) / (n_t + 0.5))`.

**Reasoning.** In the unsmoothed form, a term appearing in more than about half
the documents gets a **negative** IDF. Matching such a term then actively
*lowers* a document's score, which is indefensible: a document containing your
query word should never rank below one that does not. On a single-topic corpus
(all CSS documentation, say) plenty of terms cross that threshold. The `1 +`
keeps IDF strictly positive while preserving the ordering rare terms above
common ones.

**Trade-off.** Scores are no longer comparable with published BM25 numbers from
implementations that use the unsmoothed form. Since the scores are only ever used
for ranking within one query, and fusion is rank-based anyway, that costs
nothing here. There is a unit test asserting IDF positivity, because this is
exactly the kind of bug that never announces itself.

**Related.** Stopwords are dropped at index time, so `get_idf("the")` returns
0.0 as an out-of-vocabulary term, and a stopword-only query scores nothing
rather than matching every document in the corpus.

## Decision 4: hybrid retrieval with Reciprocal Rank Fusion

**Decision.** Run BM25 and dense retrieval independently, take 30 candidates
from each, and fuse with RRF: `score(d) = sum over lists of 1 / (k + rank(d))`,
with `k = 60`.

**Reasoning.** The two retrievers fail in different, complementary ways.

- Dense embeddings capture paraphrase: "how do I make text not overflow" finds a
  section that never uses the word "overflow". They are bad at exact tokens: a
  384-dimension MiniLM vector does not reliably distinguish `flex-basis` from
  `flex-grow`, and it cannot spell an error code.
- BM25 nails exact identifiers, error codes, version numbers, and rare terms.
  It is helpless against vocabulary mismatch: query "center a div" against a
  document that says "align an element" scores zero.

Fusing them recovers both. **Why rank-based rather than score-based:** cosine
similarity lives in [-1, 1] while BM25 is unbounded and corpus-dependent. Adding
them directly lets BM25 dominate purely because its numbers are bigger, and any
weight you pick is tuned to one corpus. RRF only reads ranks, so it is immune to
incomparable scales. `k = 60` is the value from the original Cormack et al.
paper; it damps the influence of the very top ranks so a single confident
retriever cannot monopolize the fused list. A normalized-score fusion strategy is
also implemented so the sweep can measure whether that claim holds on this
corpus rather than asserting it.

**Trade-off.** RRF discards score magnitude, which is real information: a
retriever that is *certain* about its top hit is treated the same as one that is
barely ahead. It also means two documents can end up nearly tied, and the eval
harness has a test pinning exactly that: one strong vote plus one weak vote
comes out within 0.1% of two medium votes. At that margin the ordering is
effectively a coin flip, which is precisely why the harness reports averages
over many queries instead of reasoning from single examples.

**Candidate pool.** `candidate_k = 30` feeding `top_k = 5` exists so the
reranker has something to rerank. Retrieving 5 and reranking 5 only reorders a
list you were already going to send; the recall ceiling is set by the candidate
pool, and no reranker can retrieve a document the candidate stage missed.

## Decision 5: normalize once, then use inner product

**Decision.** L2-normalize every vector at index time and at query time, then
rank by dot product.

**Reasoning.** For unit vectors, inner product **is** cosine similarity, so this
is exact rather than an approximation, and it turns search into a single matrix
multiply that numpy hands to BLAS. It also removes a whole class of bug: the
mismatch where an index is built for L2 distance but queried as though scores
were cosine similarity, which silently degrades ranking without ever raising an
error. Normalizing at write time makes the invariant impossible to violate
later. Zero vectors are handled explicitly so a degenerate input cannot produce
NaN scores that poison a whole ranking.

**Trade-off.** Vector magnitude is discarded. For text embeddings that is
standard practice and magnitude carries little meaning, but it is a choice, not
a law.

## Decision 6: an index format with no pickle in it

**Decision.** Persist as `chunks.jsonl` + `vectors.npy` + `vector_ids.json` +
`manifest.json`. No pickle anywhere.

**Reasoning.** Unpickling executes arbitrary code. The common FAISS + LangChain
pattern requires `allow_dangerous_deserialization=True`, and the flag is named
that for a reason: loading an index file becomes remote code execution if that
file can ever be replaced, which on a deployed service means a compromised
volume, a poisoned build artifact, or a teammate's download folder. The
alternative costs nothing: JSONL and `.npy` are inert data, diffable, and
readable by anything.

The manifest records embedding provider, model, dimension, chunk count, chunking
parameters, backend, and build timestamp. Without that provenance you cannot
answer "which model built this index?", and querying an index with a different
embedding model than it was built with produces confident nonsense with no error
message.

**Trade-off.** JSONL is larger and slower to load than a binary blob. At this
corpus size it is milliseconds, and there is a unit test asserting no `.pkl` or
`.pickle` file appears in a saved index.

## Decision 7: reranking implemented but OFF by default

**Decision.** Ship a cross-encoder reranker and an LLM reranker, default
`RERANK_ENABLED=false`, and include a rerank row in the eval sweep.

**Reasoning.** A cross-encoder scores the query and passage *together*, so it
sees interactions a single dot product between independently-computed vectors
cannot. It usually improves ranking quality. It also adds real latency: 30
candidates through a cross-encoder is 30 forward passes, tens to hundreds of
milliseconds on CPU, and the LLM variant adds a network round trip.

So it is a measured option rather than a default. Turn it on if *your* sweep on
*your* corpus shows it earns the latency. That sentence is the entire point of
the project.

**Trade-off.** More code paths and more configuration surface. Mitigated by a
`NoopReranker` implementing the same protocol, so the pipeline has exactly one
code path regardless of configuration.

## Decision 8: a citation-or-refuse contract

**Decision.** Number the retrieved passages, instruct the model to cite `[1]`,
`[2]` inline, emit `INSUFFICIENT_CONTEXT` when the passages do not answer the
question, and validate every citation marker against the passage count,
discarding out-of-range markers.

**Reasoning.** Retrieval quality means nothing if generation invents anyway. The
tutorial pattern retrieves context, passes it to a chain, and throws the context
away, so the user cannot check anything and the developer cannot tell a retrieval
failure from a generation failure. Three concrete safeguards here:

1. **Temperature 0.** This is extractive QA over provided context. Sampling
   creativity buys nothing but hallucinations.
2. **The LLM is never called when retrieval returns nothing.** Zero passages
   short-circuits to a refusal. Calling the model with an empty context block is
   an invitation to answer from parametric memory, which is exactly the failure
   mode a RAG system exists to prevent. There is a test asserting the model is
   not called in that case.
3. **Markers are validated, not trusted.** A `[7]` when five passages were
   supplied is a hallucinated citation and gets dropped, because a fake citation
   is worse than no citation: it looks verified.

**Trade-off.** The system refuses more often than a system that always answers,
which feels worse in a casual demo and is correct for a documentation assistant.

## Decision 9: the evaluation harness is the actual product

**Decision.** Retrieval evaluation runs offline with no API key, reports
recall@k / precision@k / hit-rate@k / MRR / nDCG plus p95 latency, and sweeps
several configurations in one command. Faithfulness grading is separate and
opt-in.

**Reasoning.** Retrieval metrics need no LLM, so they must not require one:
something free and instant gets run on every change, and something that costs
money gets run once and rots. The sweep retrieves once at the largest k and
truncates for smaller k, so every column of the table comes from identical
retrievals. It restores mutated settings in a `finally` block, because an eval
harness that leaks state into later rows produces numbers that are wrong in a
way nobody notices.

Both families are needed: retrieval metrics answer "did we find the right
passage?", faithfulness answers "did we then use it honestly?". Neither implies
the other, and the two failure modes have completely different fixes.

**Honest limits.** With one gold chunk per question, recall@k equals hit-rate@k
and precision@5 is capped at 0.2 by construction. LLM-generated questions are
biased toward what the source chunk happens to say, which is why `verified` and
`--verified-only` exist. An LLM judge shares failure modes with the generator it
grades. State these when you present the numbers; being able to name the
weaknesses of your own evaluation is a senior signal.

## Decision 10: configuration through a typed dataclass tree

**Decision.** One `Settings` tree of frozen dataclasses, built from environment
variables with defaults, threaded explicitly through every call.

**Reasoning.** Retrieval behavior is defined by about twenty parameters. Reading
`os.environ` at each use site makes them invisible and untestable. A `Settings`
object means a test can construct an exact configuration with no environment
mutation, and the eval sweep can rebind one field per row while keeping
everything else fixed, which is what makes an ablation an ablation rather than a
collection of anecdotes.

**Trade-off.** More ceremony than reading globals, and the settings object has to
be passed around. Worth it the first time you need to answer "what exactly was
the configuration when this number was produced?"

## What I would do next, with more time

An honest roadmap is a better interview answer than pretending the system is
finished.

1. **Query rewriting**, measured. Conversational follow-ups ("what about in
   Safari?") retrieve badly because the query is not self-contained. Rewrite
   against history and confirm with the harness that it actually helps.
2. **Section-aware gold labels.** Adjacent chunks from the same section are often
   equally correct answers, so single-chunk labels understate real recall. Graded
   relevance would make nDCG more meaningful.
3. **Retrieval caching.** Identical queries recompute embeddings and BM25 scores
   every time; an LRU cache keyed on the normalized query is nearly free.
4. **Incremental indexing.** Rebuilding everything to change one document is
   acceptable at this size and wrong at scale. Chunk ids are already content
   hashes, so an upsert path is a natural extension.
5. **A human-labelled evaluation slice.** Fifty questions written by a person,
   without seeing the corpus first, would be the strongest quality evidence in
   the project.
