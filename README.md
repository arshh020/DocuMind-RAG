# DocsRAG

A question-answering system over technical documentation that **measures its own
retrieval quality** instead of assuming it works.

Hybrid retrieval (BM25 + dense vectors, fused with Reciprocal Rank Fusion),
optional cross-encoder reranking, grounded answers with inline citations, and an
evaluation harness that scores every configuration on a labelled question set.

```
"Why does fetch not reject on a 404?"

  -> BM25 (lexical)    ->  top 30 candidates  --.
                                                >-- RRF fusion -> rerank -> top 5
  -> Dense (cosine)    ->  top 30 candidates  --'                            |
                                                                             v
                                              LLM, temperature 0, cite-or-refuse
                                                                             |
                                                                             v
                                             Answer + [1][2] source citations
```

## Why this project looks different from the usual RAG demo

Most RAG portfolio projects are a 60-line script: split text every 500
characters, embed, retrieve top-3, ask an LLM. They cannot answer the only
question that matters in an interview: _how do you know your retrieval is any
good?_

This one is built around that question.

| Concern           | Typical tutorial project                             | This project                                                                                 |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Chunking          | fixed 500 chars, splits code blocks mid-example      | heading-aware, fence-safe, breadcrumb-prefixed, hard-split fallback for oversized paragraphs |
| Retrieval         | dense only                                           | BM25 + dense, fused with RRF, per-component scores retained for debugging                    |
| Tuning            | vibes                                                | `make eval` sweeps 4-5 configurations over a labelled set and prints a table                 |
| Quality metrics   | none                                                 | recall@k, precision@k, MRR, nDCG, hit rate, p95 latency                                      |
| Answer quality    | trust the LLM                                        | LLM-judge faithfulness scoring + a citation-or-refuse contract                               |
| Citations         | discarded                                            | passages are numbered, markers validated, out-of-range markers dropped                       |
| Index format      | `pickle` with `allow_dangerous_deserialization=True` | JSONL + `.npy` + JSON, zero pickle, loading cannot execute code                              |
| Tests             | none                                                 | 104 unit tests, all offline, no API key required                                             |
| Core dependencies | LangChain + FAISS + more                             | **numpy** (frameworks are optional extras)                                                   |

The chunker, tokenizer, BM25 scorer, RRF fusion and every metric are implemented
directly, roughly 1,100 lines of readable Python. That is deliberate: you cannot
defend `k1=1.5, b=0.75` in an interview if a framework chose it for you.

## Quickstart: full pipeline in about 60 seconds, no API key, no downloads

```bash
git clone <your-repo-url> && cd docsrag
python -m pip install -e .          # numpy only
make test                            # 104 tests
make smoke                           # build index -> query it -> run the eval sweep
```

`make smoke` uses `--provider hash`, a deterministic stand-in for a real
embedding model. It exists so the pipeline, the tests and CI can run with no
network and no cost. **Hash-embedding numbers are not real quality numbers** and
must never be reported as results.

## Real run

```bash
python -m pip install -e ".[local,api,ui]"   # sentence-transformers + torch
cp .env.example .env                          # add GROQ_API_KEY for generation

python scripts/fetch_corpus.py --corpus mdn-javascript   # or mdn-css, mdn-http, peps
make index                                    # embed with all-MiniLM-L6-v2

python scripts/ask.py "How do I center a div with flexbox?"
python scripts/ask.py "..." --retrieval-only  # inspect retrieval, skip the LLM

python scripts/make_evalset.py --n 60         # LLM-drafted questions, then verify by hand
make eval                                     # the results table below
```

## Results

**Eval set:** 42 questions, every one human-reviewed (3 of 30 LLM-generated
questions were discarded during curation: one had a drifted premise after
paraphrasing, one was unanswerable without its source passage, one was a
near-duplicate of another item). Corpus: 2,778 chunks from MDN HTTP docs (375
fetched pages + 3 hand-authored seed docs). At n=42, one question is worth 2.4
points of any metric — treat gaps under ~5 points as noise.

| Configuration       | recall@1  | recall@3  | recall@5  | recall@10 | mrr@10    | ndcg@10   | p95 latency (ms) |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------------- |
| bm25 only           | 0.667     | 0.738     | 0.810     | 0.881     | 0.729     | 0.765     | 0.3              |
| dense only          | 0.464     | 0.774     | 0.821     | 0.869     | 0.626     | 0.680     | 17.6             |
| hybrid (RRF)        | 0.595     | 0.810     | 0.893     | 0.952     | 0.722     | 0.780     | 14.2             |
| hybrid (normalized) | 0.714     | 0.821     | 0.869     | 0.940     | 0.790     | 0.823     | 20.4             |
| hybrid + rerank     | **0.738** | **0.893** | **0.964** | **0.988** | **0.836** | **0.871** | 1631.8           |

Only the `hybrid + rerank` row runs a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top-30 hybrid candidates; the
first four rows measure retrieval alone.

### Findings

1. **Hybrid retrieval beats either method alone from recall@5 onward**
   (0.893 RRF vs. 0.810 BM25, 0.821 dense). Dense and lexical search fail on
   different questions — the eval set deliberately includes both fully
   paraphrased questions with no shared vocabulary and exact identifier
   lookups (e.g. `localHostOrDomainIs`). Fusion recovers whichever stage found
   the answer.

2. **Reranking is the single largest quality lever, and by far the most
   expensive.** Over the best non-reranked config (hybrid RRF), the
   cross-encoder adds +14.3 points recall@1 and +7.1 recall@5 for roughly
   115x the latency (14.2 ms -> 1.63 s p95, CPU-only). It ships disabled by
   default: 1.6 s is not an interactive latency budget for this project.

3. **Given a good enough candidate pool, the choice of fusion strategy stops
   mattering as much.** Hybrid RRF and hybrid normalized both narrow to the
   reranker's ordering once one is applied, since the reranker scores the
   same top-30 pool regardless of how that pool was ranked going in. The
   first stage's job under a reranker is candidate recall, not final ranking.

## How the evaluation works

1. **Questions.** `scripts/make_evalset.py` drafts questions from real chunks
   with an LLM, then discards low-quality ones (too short, self-referential,
   answer leaked into the question).
2. **Gold labels.** The chunk a question was generated from is its gold chunk.
   Chunk ids are SHA-256 content hashes, so they stay stable across rebuilds and
   your labels do not silently rot.
3. **Verification.** LLM-generated labels are noisy. Read them and set
   `verified: true`; `--verified-only` restricts scoring to that subset. Say
   which number you are quoting.
4. **Sweep.** `evaluation/retrieval.py` re-runs the same questions against each
   configuration, retrieving once at the largest k and truncating for smaller k,
   so the comparison is apples to apples.
5. **Faithfulness.** `--faithfulness N` samples N answers and asks a judge model
   whether every claim is supported by the retrieved context. This catches the
   failure retrieval metrics cannot: correct passages, invented answer.

A metric caveat worth understanding before you quote anything: with one gold
chunk per question, recall@k and hit-rate@k are the same measurement, and
precision@5 is capped at 0.2. That is a property of the labelling scheme, not a
weakness in the system. nDCG and MRR are the informative columns because they
reward _where_ in the ranking the answer landed.

## Project layout

```
src/docsrag/
  tokenize.py      identifier-aware tokenizer (get_user_by_id survives whole)
  chunking.py      heading-aware, fence-safe Markdown chunking
  bm25.py          Okapi BM25 with smoothed strictly-positive IDF
  embeddings.py    sentence-transformers | OpenAI-compatible | hash stub
  vectorstore.py   numpy exhaustive cosine, optional FAISS backend
  fusion.py        RRF and normalized-score fusion
  rerank.py        cross-encoder | LLM | no-op rerankers
  index.py         the retrieval index: chunks + BM25 + vectors, no pickle
  generate.py      grounded generation, citation markers, refusal contract
  pipeline.py      retrieve -> rerank -> answer, with per-stage timings
  evaluation/      dataset, retrieval sweep, LLM-judge faithfulness
scripts/           fetch_corpus, build_index, make_evalset, run_eval, ask
app/               FastAPI service + Streamlit UI
tests/             104 offline unit tests
```

## Deployment

```bash
make docker-up     # API on :8000/docs, UI on :8501
```

The index is mounted as a volume, not baked into the image, because it is data:
rebuilding it needs the corpus and an embedding model, and a stale index inside
an image is a bug waiting to happen. `/health` reports index status; `/retrieve`
returns 503 until an index is present.

Hosting that works with this layout, both genuinely free with no card required:
Streamlit Community Cloud (easiest) or Hugging Face Spaces (Docker-native).

## Known limitations

Stating these is a feature. Every one of them is a real engineering trade-off,
and naming the limits of your own system is what separates a candidate who built
something from one who followed a tutorial.

- **Exhaustive dense search.** numpy scans every vector. Correct and fast to
  ~100k chunks; beyond that switch on the FAISS backend.
- **In-process index.** Loaded into memory per worker. Fine for one machine, not
  for horizontal scaling.
- **Rebuild, not incremental update.** Changing one document rebuilds the whole
  index. Acceptable at this corpus size, wrong at production scale.
- **Approximate token counting.** `max(chars/4, word_count)` rather than a real
  tokenizer, to keep the core dependency-free. Swap in `tiktoken` if exact
  budgets matter.
- **Single-vector retrieval.** No query rewriting, HyDE, or multi-hop
  decomposition; multi-part questions retrieve for the whole query at once.
- **Judge model bias.** LLM-as-judge faithfulness correlates with human
  judgement but is not ground truth, and it shares failure modes with the
  generator.

## Resume bullet

Fill the blanks from your own `make eval` output:

> Built a hybrid-retrieval RAG service over \_**\_ documentation pages (\_\_**
> chunks): BM25 + dense embeddings fused with Reciprocal Rank Fusion, optional
> cross-encoder reranking, and grounded answers with inline citations. Built an
> evaluation harness (recall@k, MRR, nDCG, p95 latency, LLM-judge faithfulness)
> over \_**\_ labelled questions and used it to select the retrieval
> configuration, improving recall@5 from \_\_** (dense-only baseline) to \_\_\_\_.
> Python, numpy, FastAPI, Streamlit, Docker; 104 unit tests, no pickle in the
> index format.

See `ARCHITECTURE.md` for the reasoning behind each decision and `RUNBOOK.md`
for the hour-by-hour build plan.

## License

MIT for the code. Fetched corpora keep their own licenses (MDN content is CC-BY-SA 2.5+, Python PEPs are public domain) and are gitignored for that reason.
