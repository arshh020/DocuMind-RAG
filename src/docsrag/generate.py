"""Grounded answer generation with enforced citations and explicit refusal.

Three things here that the tutorial version of this project does not do:

1. **The prompt lives in version control.** Pulling a prompt from a remote hub at
   request time means a network round-trip on every query, an untracked prompt,
   and a hard dependency on someone else's uptime inside your hot path.

2. **Citations are mandatory and machine-checkable.** The model must tag every
   claim with [1], [2], ... and we parse those markers back out, so the UI can
   show exactly which chunk supported which sentence. For a reference assistant,
   an uncited answer is not a feature, it is a defect.

3. **Refusal is a real code path, not a hope.** If retrieval returns nothing
   above threshold we never call the LLM at all. "I do not know" produced
   deterministically by control flow is worth more than "I do not know" that the
   model may or may not decide to say.

The Groq client uses urllib against the OpenAI-compatible endpoint, so the core
package has zero HTTP dependencies.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .config import GenerationConfig

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
USER_AGENT = "docsrag/0.1 (+https://github.com/docsrag)"

SYSTEM_PROMPT = """You are a technical documentation assistant. You answer \
strictly from the numbered context passages provided.

Rules:
1. Use ONLY information present in the context. Never rely on prior knowledge.
2. Cite every factual claim with the passage number in square brackets, like \
[1] or [2][3]. Every sentence containing a fact must carry at least one citation.
3. If the context does not contain enough information to answer, reply with \
exactly: INSUFFICIENT_CONTEXT
4. Do not invent function names, parameters, syntax, or behaviour that is not \
shown in the context.
5. Prefer quoting exact syntax and identifiers from the context over paraphrasing \
them.
6. Be concise. Include a short code example only when the context contains one.
"""

REFUSAL_MESSAGE = (
    "I could not find information about that in the indexed documentation. "
    "Try rephrasing, or check whether this topic is part of the corpus."
)


class GroqClient:
    """Minimal OpenAI-compatible chat client for Groq."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        *,
        timeout: float = 60.0,
        endpoint: str = GROQ_ENDPOINT,
        reasoning_effort: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self._api_key = api_key
        self.model = model
        self._timeout = timeout
        self._endpoint = endpoint
        self._reasoning_effort = reasoning_effort

    @classmethod
    def from_env(
        cls, config: GenerationConfig | None = None
    ) -> "GroqClient":
        config = config or GenerationConfig()
        return cls(
            api_key=config.api_key or os.getenv("GROQ_API_KEY", ""),
            model=config.model,
            timeout=config.timeout_seconds,
            reasoning_effort=getattr(config, "reasoning_effort", ""),
        )

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._reasoning_effort and self._reasoning_effort.lower() != "none":
            body["reasoning_effort"] = self._reasoning_effort
        payload = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Groq request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Groq request failed: {exc.reason}") from exc

        return body["choices"][0]["message"]["content"].strip()


@dataclass
class Citation:
    """One passage that the model actually referenced."""

    marker: int
    chunk_id: str
    source: str
    title: str
    breadcrumb: str
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False
    used_passage_count: int = 0
    retrieved_passage_count: int = 0
    grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "refused": self.refused,
            "grounded": self.grounded,
            "used_passage_count": self.used_passage_count,
            "retrieved_passage_count": self.retrieved_passage_count,
            "citations": [
                {
                    "marker": c.marker,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "title": c.title,
                    "breadcrumb": c.breadcrumb,
                    "snippet": c.snippet,
                    "score": c.score,
                }
                for c in self.citations
            ],
        }


def build_context_block(passages: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved passages as a numbered context block.

    The breadcrumb is included in the header so the model can see WHERE a
    passage came from. That single detail measurably reduces answers that
    confuse two similarly-worded API methods.
    """
    parts: List[str] = []
    for index, passage in enumerate(passages, start=1):
        breadcrumb = passage.get("breadcrumb") or passage.get("title", "")
        header = f"[{index}] {breadcrumb}".rstrip()
        source = passage.get("source", "")
        if source:
            header += f"  (source: {source})"
        parts.append(f"{header}\n{passage.get('text', '')}")
    return "\n\n---\n\n".join(parts)


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citation_markers(text: str) -> List[int]:
    """Return the distinct passage numbers cited, in order of first appearance."""
    seen: List[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        marker = int(match.group(1))
        if marker not in seen:
            seen.append(marker)
    return seen


def generate_answer(
    question: str,
    passages: Sequence[Dict[str, Any]],
    *,
    client: GroqClient,
    config: GenerationConfig | None = None,
) -> Answer:
    """Generate a cited answer, or refuse if there is nothing to ground it in."""
    config = config or GenerationConfig()

    # Refusal path 1: retrieval returned nothing. Never call the LLM.
    if not passages:
        return Answer(
            text=REFUSAL_MESSAGE,
            refused=True,
            retrieved_passage_count=0,
            grounded=True,
        )

    context = build_context_block(passages)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with bracketed citations."
    )

    raw = client.chat(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Refusal path 2: the model itself signalled insufficient context.
    if "INSUFFICIENT_CONTEXT" in raw.upper():
        return Answer(
            text=REFUSAL_MESSAGE,
            refused=True,
            retrieved_passage_count=len(passages),
            grounded=True,
        )

    markers = extract_citation_markers(raw)
    citations: List[Citation] = []
    for marker in markers:
        if not 1 <= marker <= len(passages):
            # A citation pointing at a passage we never supplied is a
            # hallucinated reference. Drop it and flag the answer.
            continue
        passage = passages[marker - 1]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=str(passage.get("id", "")),
                source=str(passage.get("source", "")),
                title=str(passage.get("title", "")),
                breadcrumb=str(passage.get("breadcrumb", "")),
                snippet=str(passage.get("text", ""))[:300],
                score=float(passage.get("score", 0.0)),
            )
        )

    # An answer with no valid citation is, by our own contract, ungrounded.
    # We surface that rather than hiding it.
    grounded = bool(citations)

    return Answer(
        text=raw,
        citations=citations,
        refused=False,
        used_passage_count=len(citations),
        retrieved_passage_count=len(passages),
        grounded=grounded,
    )


__all__ = (
    "GroqClient",
    "Answer",
    "Citation",
    "generate_answer",
    "build_context_block",
    "extract_citation_markers",
    "SYSTEM_PROMPT",
    "REFUSAL_MESSAGE",
)
