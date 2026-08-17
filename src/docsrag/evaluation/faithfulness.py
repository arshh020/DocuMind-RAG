"""LLM-as-judge grading for answer quality.

Retrieval metrics tell you whether the evidence reached the model. They say
nothing about whether the model used it. Two failure modes remain:

* **Unfaithfulness / hallucination**: the answer asserts things the context does
  not support. This is the one that matters most for a reference assistant.
* **Irrelevance**: the answer is faithful but does not address the question.

We grade both with a judge model, using a deliberately strict rubric and
temperature 0. Known limitations, which you should state plainly rather than
pretend away:

  1. The judge is the same family of model as the generator, so shared blind
     spots are possible. Mitigate by using a stronger judge model than the
     generator.
  2. LLM judges correlate well with human judgement but are not ground truth.
     Hand-check a sample and report the agreement rate.
  3. Judging is not free. This is why retrieval evaluation is separate and runs
     first: you fix retrieval cheaply before you spend tokens on grading.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..generate import GroqClient
from ..pipeline import RagPipeline
from .dataset import EvalExample

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of retrieval-augmented \
answers. You will receive a QUESTION, the CONTEXT passages that were retrieved, \
and the ANSWER that was generated.

Grade three things:

1. faithfulness (0-5): Is every factual claim in the ANSWER supported by the \
CONTEXT? 5 = fully supported. 0 = mostly fabricated. Any claim not present in \
the CONTEXT lowers this score, even if it happens to be true in general.
2. relevance (0-5): Does the ANSWER actually address the QUESTION? 5 = directly \
and completely. 0 = off topic.
3. citation_validity (0-5): Are the bracketed citation markers used correctly \
and attached to the claims they support? 5 = all correct. 0 = absent or wrong.

Also list any unsupported claims you found.

A correct refusal (the answer says the information is not available) when the \
CONTEXT genuinely lacks the answer should score faithfulness 5 and relevance 5.

Respond with ONLY a JSON object:
{"faithfulness": <int>, "relevance": <int>, "citation_validity": <int>, \
"unsupported_claims": ["..."], "reasoning": "one sentence"}
"""


@dataclass
class JudgeResult:
    example_id: str
    question: str
    faithfulness: float = 0.0
    relevance: float = 0.0
    citation_validity: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""
    refused: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Extract the JSON verdict, tolerating code fences or surrounding prose."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge response: {raw[:200]}")
    return json.loads(match.group(0))


def judge_answer(
    question: str,
    context: str,
    answer: str,
    *,
    client: GroqClient,
) -> Dict[str, Any]:
    """Grade one answer. Raises on unparseable judge output."""
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Grade the answer now."
    )
    raw = client.chat(
        system=JUDGE_SYSTEM_PROMPT,
        user=user,
        temperature=0.0,
        max_tokens=500,
    )
    return _parse_judge_response(raw)


def evaluate_faithfulness(
    pipeline: RagPipeline,
    examples: Sequence[EvalExample],
    *,
    judge_model: str = "openai/gpt-oss-20b",
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> List[JudgeResult]:
    """Run the full pipeline over examples and grade each answer.

    Args:
        judge_model: Deliberately separate from the generator model so you can
            judge with a stronger model than you generate with.
        limit: Grade only the first N examples. Faithfulness grading costs two
            LLM calls per question (answer + judge), so cap it while iterating.
        sleep_seconds: Simple throttle for rate limits.
    """
    from ..generate import build_context_block

    judge = GroqClient.from_env()
    judge.model = judge_model

    subset = list(examples)[:limit] if limit else list(examples)
    results: List[JudgeResult] = []

    for position, example in enumerate(subset, start=1):
        print(f"  judging {position}/{len(subset)}: {example.id}", flush=True)
        try:
            outcome = pipeline.answer(example.question)
            context = build_context_block(
                [r.to_passage() for r in outcome.retrieved]
            )
            verdict = judge_answer(
                example.question,
                context or "(no context retrieved)",
                outcome.answer.text,
                client=judge,
            )
            results.append(
                JudgeResult(
                    example_id=example.id,
                    question=example.question,
                    faithfulness=float(verdict.get("faithfulness", 0)),
                    relevance=float(verdict.get("relevance", 0)),
                    citation_validity=float(
                        verdict.get("citation_validity", 0)
                    ),
                    unsupported_claims=list(
                        verdict.get("unsupported_claims") or []
                    ),
                    reasoning=str(verdict.get("reasoning", "")),
                    refused=outcome.answer.refused,
                )
            )
        except Exception as exc:
            # One bad question must not abort a long grading run.
            results.append(
                JudgeResult(
                    example_id=example.id,
                    question=example.question,
                    error=str(exc)[:300],
                )
            )

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return results


def summarize_judgements(results: Sequence[JudgeResult]) -> Dict[str, float]:
    """Aggregate judge scores into headline numbers."""
    scored = [r for r in results if not r.error]
    if not scored:
        return {}

    count = len(scored)
    faithfulness = sum(r.faithfulness for r in scored) / count
    relevance = sum(r.relevance for r in scored) / count
    citations = sum(r.citation_validity for r in scored) / count

    # The headline safety number: share of answers with any unsupported claim,
    # excluding correct refusals.
    non_refusals = [r for r in scored if not r.refused]
    unsupported_rate = (
        sum(1 for r in non_refusals if r.unsupported_claims) / len(non_refusals)
        if non_refusals
        else 0.0
    )

    return {
        "faithfulness_mean": faithfulness,
        "relevance_mean": relevance,
        "citation_validity_mean": citations,
        "unsupported_claim_rate": unsupported_rate,
        "refusal_rate": sum(1 for r in scored if r.refused) / count,
        "graded": float(count),
        "errors": float(len(results) - count),
    }


def save_judgements(
    results: Sequence[JudgeResult], directory: str | Path
) -> Path:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_judgements(results)
    (out_dir / "faithfulness_results.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "summary": summary,
                "results": [r.to_dict() for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = ["# Answer quality (LLM-as-judge)", ""]
    if summary:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Faithfulness (0-5) | {summary['faithfulness_mean']:.2f} |",
            f"| Relevance (0-5) | {summary['relevance_mean']:.2f} |",
            f"| Citation validity (0-5) | {summary['citation_validity_mean']:.2f} |",
            f"| Unsupported-claim rate | {summary['unsupported_claim_rate']:.1%} |",
            f"| Refusal rate | {summary['refusal_rate']:.1%} |",
            f"| Questions graded | {int(summary['graded'])} |",
            "",
            "> Judge model scores are not ground truth. Hand-check a sample and "
            "report the agreement rate before quoting these numbers anywhere "
            "that matters.",
        ]
    else:
        lines.append("_No successful judgements._")

    path = out_dir / "faithfulness_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = (
    "JudgeResult",
    "judge_answer",
    "evaluate_faithfulness",
    "summarize_judgements",
    "save_judgements",
    "JUDGE_SYSTEM_PROMPT",
)
