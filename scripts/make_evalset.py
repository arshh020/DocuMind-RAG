#!/usr/bin/env python3
"""Generate a labelled evaluation set from the index.

Method: sample chunks, ask the LLM to write a question answerable ONLY from that
chunk, and record (question -> chunk id) as a gold label. This is standard
practice and it is the only way to get 100 labelled questions inside a few hours.

Read this before you quote the numbers anywhere:

Synthetic eval sets have a KNOWN, measurable bias. The generator sees the chunk
while writing the question, so the question tends to reuse the chunk's exact
vocabulary. That inflates lexical (BM25) retrieval scores relative to how the
system performs on real user questions, which are shorter, vaguer, and use
different words than the docs do.

Mitigations built into this script:
  * ``--paraphrase`` asks for a second, deliberately reworded question per chunk
    that avoids the chunk's distinctive terms.
  * Every generated example is written with ``verified: false``. You then read a
    sample and flip the good ones to true. Report both numbers.
  * Questions that are trivially answerable from the chunk title are rejected.

Because cross-configuration DELTAS are much more robust to this bias than
absolute values, the ablation table remains meaningful even before verification.
Say exactly that in an interview and you will be ahead of most candidates.

Usage:
    python scripts/make_evalset.py --n 100
    python scripts/make_evalset.py --n 60 --paraphrase --out eval/evalset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.config import Settings  # noqa: E402
from docsrag.evaluation.dataset import (  # noqa: E402
    EvalExample,
    describe_eval_set,
    save_eval_set,
)
from docsrag.generate import GroqClient  # noqa: E402
from docsrag.index import RagIndex  # noqa: E402

GENERATE_SYSTEM_PROMPT = """You write evaluation questions for a documentation \
search system.

Given a documentation passage, write ONE question that:
  - is fully answerable using only this passage
  - a real developer would plausibly type into a search box
  - is specific enough that only this passage answers it well
  - does NOT mention the passage, the document, or "according to the text"
  - is a single sentence, under 25 words

Also write a one-sentence gold answer drawn only from the passage.

Respond with ONLY a JSON object:
{"question": "...", "answer": "..."}
"""

PARAPHRASE_SYSTEM_PROMPT = """Reword the given question so that it means the \
same thing but avoids the distinctive technical vocabulary of the original. Use \
the words a developer might use when they do NOT yet know the correct \
terminology. Keep it one sentence.

Respond with ONLY a JSON object: {"question": "..."}
"""


def parse_json_object(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in: {raw[:200]!r}")
    return json.loads(match.group(0))


def chat_with_backoff(client, *, max_retries=4, base_delay=8.0, **kwargs) -> str:
    for attempt in range(max_retries + 1):
        try:
            return client.chat(**kwargs)
        except RuntimeError as exc:
            if "429" not in str(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            print(f"    rate limited, waiting {delay:.0f}s ...", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def is_low_quality(question: str, chunk_title: str) -> bool:
    """Reject questions that do not test retrieval.

    A question that merely restates the chunk title measures nothing: any
    retriever finds it. Filtering these keeps the eval set discriminative.
    """
    stripped = question.strip()
    if len(stripped) < 15 or not stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    if lowered.startswith(("what is this", "what does this", "what is the passage")):
        return True
    title = chunk_title.lower().strip()
    # Title fully contained AND question barely longer than the title.
    if title and title in lowered and len(lowered) < len(title) + 20:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="questions to generate")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/evalset.jsonl"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=80,
        help="skip chunks shorter than this; tiny chunks make degenerate questions",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="also emit a vocabulary-shifted variant of each question",
    )
    parser.add_argument("--delay", type=float, default=2.5)
    args = parser.parse_args()

    settings = Settings.from_env()
    index_dir = args.index or settings.paths.index_dir

    print(f"Loading index from {index_dir} ...")
    index = RagIndex.load(index_dir, settings=settings)
    print(f"  {len(index)} chunks")

    eligible = [c for c in index.chunks if c.token_count >= args.min_tokens]
    if not eligible:
        raise SystemExit(
            "no chunks long enough; lower --min-tokens or rebuild the index"
        )

    random.seed(args.seed)
    sample_size = min(args.n, len(eligible))
    sampled = random.sample(eligible, sample_size)
    print(f"  sampling {sample_size} of {len(eligible)} eligible chunks")

    client = GroqClient.from_env(settings.generation)
    examples: List[EvalExample] = []
    rejected = 0

    for position, chunk in enumerate(sampled, start=1):
        print(f"  [{position}/{sample_size}] {chunk.breadcrumb[:60]}", flush=True)
        try:
            payload = parse_json_object(
                chat_with_backoff(
                    client,
                    system=GENERATE_SYSTEM_PROMPT,
                    user=f"Passage:\n\n{chunk.text[:2500]}",
                    temperature=0.3,
                    max_tokens=600,
                )
            )
            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()

            if is_low_quality(question, chunk.title):
                rejected += 1
                continue

            base_id = f"q{len(examples) + 1:04d}"
            examples.append(
                EvalExample(
                    id=base_id,
                    question=question,
                    gold_chunk_ids=[chunk.id],
                    gold_answer=answer,
                    source=chunk.source,
                    verified=False,
                    tags=["generated"],
                )
            )

            if args.paraphrase:
                variant = parse_json_object(
                    chat_with_backoff(
                        client,
                        system=PARAPHRASE_SYSTEM_PROMPT,
                        user=f"Question: {question}",
                        temperature=0.5,
                        max_tokens=300,
                    )
                )
                reworded = str(variant.get("question", "")).strip()
                if reworded and not is_low_quality(reworded, chunk.title):
                    examples.append(
                        EvalExample(
                            id=f"{base_id}p",
                            question=reworded,
                            gold_chunk_ids=[chunk.id],
                            gold_answer=answer,
                            source=chunk.source,
                            verified=False,
                            tags=["generated", "paraphrase"],
                        )
                    )
        except Exception as exc:
            print(f"    skipped: {str(exc)[:160]}", file=sys.stderr)
            rejected += 1

        time.sleep(args.delay)

    written = save_eval_set(examples, args.out)
    stats = describe_eval_set(examples)

    print(f"\nWrote {written} examples to {args.out}")
    print(f"  rejected/failed: {rejected}")
    print(f"  tags: {stats['tags']}")
    print(
        "\nIMPORTANT: every example has verified=false.\n"
        "Spend 20-30 minutes reading a sample, fix or delete bad ones, and set\n"
        'verified to true on the good ones. Then report BOTH the full-set and\n'
        "verified-subset numbers. That distinction is what makes the evaluation\n"
        "credible instead of decorative."
    )
    print("\nNext:  python scripts/run_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
