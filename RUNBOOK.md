# RUNBOOK: from clone to deployed in 6 hours

The code is already written. Your six hours are for **running it, producing real
numbers, deploying it, and being able to defend every line.** That last part is
what gets you the offer, and it is the part you cannot outsource.

Read this rule first, because it is the one that can sink you:

> **Never put a number in the README that you did not produce on your machine.**
> The results table ships empty. Fabricated benchmarks are the fastest way to
> fail an interview: the follow-up is always "walk me through how you measured
> that", and there is no recovery from not knowing.

## Hour 0:00-0:45 - Get it running and green

```bash
cd docsrag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[api,ui,dev]"
make test          # expect 104 passing, no network, no API key
make smoke         # index -> query -> eval sweep, all offline
```

`make smoke` uses the `hash` embedding provider, a deterministic stand-in for a
real model. It proves the plumbing works. **Its numbers are meaningless** and
never leave your terminal.

While that runs, create a Groq API key (free tier, no card): console.groq.com,
then `cp .env.example .env` and paste it in. Generation is the only part that
needs a key at all; retrieval and the entire retrieval evaluation run without
one.

Checkpoint: 104 tests green, `artifacts/results/retrieval_results.md` exists.

## Hour 0:45-1:30 - Real corpus, real embeddings

```bash
python -m pip install -e ".[local]"                    # sentence-transformers + torch
python scripts/fetch_corpus.py --corpus mdn-css        # ~10 MB sparse clone
make index                                             # downloads MiniLM once (~90 MB)
```

Start with **one** corpus (`mdn-css`, `mdn-javascript`, `mdn-http`, or `peps`).
One corpus of a few hundred pages is plenty; a bigger corpus costs indexing time
and buys nothing you can show.

Sanity-check retrieval by hand before trusting any metric:

```bash
python scripts/ask.py "how do I center a div" --retrieval-only
python scripts/ask.py "what does flex-basis do" --retrieval-only --mode bm25
python scripts/ask.py "what does flex-basis do" --retrieval-only --mode dense
```

Read the passages. If the top hit is obviously wrong, fix that now: reduce
`CHUNK_MAX_TOKENS` to 256, or check that `manifest.json` records the model you
think you used. Debugging retrieval later, through generation, is much harder.

Then answer one question end to end:

```bash
python scripts/ask.py "how do I center a div with flexbox"
```

Checkpoint: `manifest.json` shows your real model and chunk count; one grounded,
cited answer printed.

## Hour 1:30-2:45 - Build the eval set (this is the project)

```bash
python scripts/make_evalset.py --n 60
```

Then **read the file** at `eval/evalset.jsonl`. This is the highest-value hour in
the whole build, and it is the step everyone else skips.

For each example: does the question make sense on its own, and does the gold
chunk really answer it? Delete the bad ones, fix wording where the question
leaked its answer, and set `"verified": true` on the ones you personally checked.
Aim for 40-60 verified examples. Note the hand-written seed set already in
`eval/evalset.jsonl` as a format reference.

Why it matters: LLM-generated questions are biased toward the phrasing of the
chunk they came from, which inflates retrieval scores. A verified subset is the
difference between "I generated an eval set" and "I built an eval set I trust",
and an interviewer will ask which one you have.

Now produce the real numbers:

```bash
make eval                                   # all examples
python scripts/run_eval.py --verified-only  # the number to quote
```

Paste the generated table into the README's empty Results table, and write the
eval-set size and verified count next to it.

Checkpoint: a results table filled with numbers you produced, and you can say
which retrieval configuration won and by how much.

## Hour 2:45-3:30 - Run one honest ablation

You now have a measuring instrument, so use it. Pick **one** question, answer it
with data, and write the answer in the README:

```bash
# Does chunk size matter on this corpus?
python scripts/build_index.py --max-tokens 256 --out artifacts/index-256
python scripts/run_eval.py --index artifacts/index-256 --out artifacts/results-256

# Does reranking earn its latency?
python scripts/run_eval.py --rerank --rerank-provider llm
```

A one-line finding such as "hybrid RRF beat dense-only by X points of recall@5 on
52 verified questions, and reranking added Y ms of p95 latency for Z points" is
worth more than any additional feature you could build in the same 45 minutes.
It is also the sentence that goes on your resume.

Optional, if the key is working:

```bash
python scripts/run_eval.py --faithfulness 25   # 2 LLM calls per question
```

Checkpoint: one measured finding, written down in your own words.

## Hour 3:30-4:15 - Serve it

```bash
make api    # http://localhost:8000/docs
make ui     # http://localhost:8501
```

Check `/health` (it reports whether the index is loaded), then `/ask` and
`/retrieve` from the Swagger page. **This is the first time FastAPI and Streamlit
actually run**, so budget for small fixes here: an import path, a Pydantic field,
a missing extra. That is why this slot exists before deployment rather than after.

In the UI, confirm the passage panel shows the retrieved chunks and their
component scores. That panel is the single most impressive thing to show a
reviewer, because it proves the system is inspectable rather than a black box.

Checkpoint: both surfaces answer a question locally, with citations.

## Hour 4:15-5:15 - Deploy

Pick the option that matches how you want to demo it.

**Streamlit Community Cloud** (easiest, free, no Docker): push to GitHub, connect
the repo at share.streamlit.io, set `app/streamlit_app.py` as the entrypoint, add
`GROQ_API_KEY` in Secrets. Constraint: it builds the index at startup unless you
commit `artifacts/index/`, so use a small corpus and `EMBEDDING_PROVIDER` you can
afford at boot.

**Hugging Face Spaces, Docker SDK** (free, closest to real infrastructure): push
the repo with the `Dockerfile`, set the key as a Space secret. Persistent storage
for `artifacts/` is a paid add-on, so commit the index or rebuild on boot.

Both of the above are free. Skip Fly.io and Render here — their free tiers
either expired or now require a card on file, which breaks the "free"
premise of this runbook. If you want a real persistent-volume container
host later, revisit that decision then, with a card and a budget in hand.

Before you call it deployed:

```bash
make docker-up     # verify the image works locally FIRST
```

Also set `CORS_ORIGINS` to your UI origin instead of `*`, and confirm the key is
a platform secret and not committed. `.env` is gitignored; check `git log -p` if
you are unsure it was never added.

Checkpoint: a public URL that answers a question with citations.

## Hour 5:15-6:00 - Make it defensible

No new features in this hour. Instead:

1. Fill in every blank in the README: results table, eval-set size, verified
   count, and the resume bullet at the bottom.
2. Add a screenshot of the UI showing an answer with its retrieved passages.
3. Read `ARCHITECTURE.md` end to end. Ten decisions, each with a reason and a
   trade-off. These are your interview answers.
4. Rehearse the five questions below out loud. Actually out loud.

### The five questions you will be asked

1. **"How do you know your retrieval is good?"** Name the eval set size, the
   verified subset, the metrics, and the winning configuration. This is the
   question the whole project exists to answer.
2. **"Why hybrid instead of just embeddings?"** Complementary failure modes:
   dense handles paraphrase and fails on exact identifiers; BM25 is the reverse.
   Then give your measured delta.
3. **"Why RRF instead of adding the scores?"** Cosine is bounded, BM25 is not;
   summing lets BM25 dominate by scale alone. RRF uses ranks only. Mention that
   you also implemented normalized-score fusion so the sweep could check it.
4. **"How did you choose your chunk size?"** 512 tokens with 64 overlap as a
   starting point, heading-aware boundaries, then the ablation you actually ran.
   Never "it is what the tutorial used".
5. **"What happens when the answer is not in the corpus?"** The refusal path:
   `INSUFFICIENT_CONTEXT`, no LLM call at all on zero passages, and citation
   markers validated against the passage count so hallucinated citations are
   dropped.

Also be ready for: **"What is broken about it?"** Answer from the Known
Limitations section. Candidates who cannot criticize their own project read as
though they do not understand it.

## If you have less than 6 hours

**Three hours:** hours 0:00-1:30 (running, real index) plus 1:30-2:15 (a 25-question
verified eval set and `make eval`). Skip deployment; a repo with real numbers
beats a deployed demo with none.

**Ninety minutes:** `make test`, `make smoke`, one corpus, `make index`, and 15
hand-verified eval questions. Push it. You will still be able to answer question
1 above, which is more than most candidates can.

## Troubleshooting

- **`ModuleNotFoundError: docsrag`** - install with `pip install -e .`, or run
  scripts from the repo root (they add `src/` to the path themselves).
- **Torch download too slow** - work with `--provider hash` for plumbing, or set
  `EMBEDDING_PROVIDER=openai` with an OpenAI-compatible key.
- **Every retrieval looks wrong** - check `manifest.json`. Querying an index with
  a different embedding model than built it produces confident nonsense and no
  error.
- **Answers ignore the context** - inspect with `--retrieval-only` first. If the
  passages are right and the answer is wrong, it is generation; if the passages
  are wrong, no prompt engineering will save you.
- **Eval scores suspiciously high** - your questions probably echo the chunk
  wording. Rewrite a few in your own words and re-run; that gap is worth
  mentioning in the interview.
- **Rebuilt the index and eval scores collapsed** - chunk ids are content
  hashes, so changing chunking parameters changes the ids your gold labels point
  at. Regenerate or remap the eval set when you change chunking.
