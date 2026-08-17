"""Header-aware Markdown chunking.

This is the highest-leverage part of the system, and the part most tutorial RAG
projects get wrong. The usual approach, a fixed 500-character recursive text
splitter, has three concrete defects on a documentation corpus:

1. Lost context. A chunk reading "It returns a new array and does not mutate the
   original." is useless in isolation. What is "it"? The heading said
   Array.prototype.map() four hundred characters earlier, so that information
   landed in a different chunk. Embeddings of orphaned text are near-meaningless.

2. Split code blocks. A fixed character window happily cuts a fenced code
   example in half, producing two chunks that are each syntactically broken and
   neither of which answers the question.

3. No structural awareness. Markdown already tells you where the semantic
   boundaries are, via headings. Splitting on character count throws away free,
   perfectly reliable signal.

Our approach:
  - Parse the document into sections using ATX headings, tracking a breadcrumb
    stack (h1 > h2 > h3).
  - Never split inside a fenced code block.
  - Prepend the breadcrumb to every chunk's embedded text so each chunk is
    self-describing, while keeping the raw body separately for display.
  - Fall back to paragraph-level packing only when a single section exceeds the
    token budget, applying overlap at paragraph granularity.

Interview framing: "I made chunk boundaries follow document structure instead of
character counts, and I measured the difference" is a much stronger claim than
"I used a text splitter".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Sequence

from .tokenize import approx_token_count

# ATX heading: 1-6 hashes, a space, then text.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Fenced code delimiter: three or more backticks or tildes, optional language.
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(\S*)")
# YAML frontmatter delimiter.
_FRONTMATTER = re.compile(r"^---\s*$")
# Sentence boundary, used only as a last-resort split point for prose that is
# too long to fit the budget as a single paragraph.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Unit separator, used only to build hash inputs so that fields cannot bleed
# into one another and cause id collisions.
_SEP = chr(31)


@dataclass
class Chunk:
    """A single retrievable unit.

    Attributes:
        id: Stable content-addressed identifier, deterministic across runs. This
            matters more than it looks: if chunk ids changed on every rebuild,
            every gold label in your eval set would silently break.
        text: Raw body text, used for display and citation.
        embed_text: What we actually embed, i.e. breadcrumb plus body. Keeping
            these separate means the user sees clean text while the retriever
            sees contextualized text.
        source: Logical document path or URL.
        title: Document title, usually the h1.
        breadcrumb: Heading path, e.g. "Array.prototype.map > Syntax".
        heading_level: Depth of the owning heading.
        token_count: Approximate token length of embed_text.
        position: Ordinal position within the source document.
    """

    id: str
    text: str
    embed_text: str
    source: str
    title: str = ""
    breadcrumb: str = ""
    heading_level: int = 0
    token_count: int = 0
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Section:
    """Intermediate parse result: one heading and the body beneath it."""

    breadcrumb: str
    heading_level: int
    body: str

    @property
    def title(self) -> str:
        """The leaf heading, i.e. the last breadcrumb segment.

        The breadcrumb is the authoritative field, so the title is derived from
        it rather than stored twice. Two copies of the same fact drift apart.
        """
        if not self.breadcrumb:
            return ""
        return self.breadcrumb.split(" > ")[-1].strip()

    @property
    def text(self) -> str:
        """Readable alias for ``body``, matching ``Chunk.text``."""
        return self.body


def strip_frontmatter(text: str) -> tuple[str, Dict[str, str]]:
    """Remove leading YAML frontmatter and return it as a flat dict.

    We parse only top-level scalar key/value pairs on purpose. Pulling in a YAML
    dependency to read a title is not worth it.
    """
    lines = text.splitlines()
    if not lines or not _FRONTMATTER.match(lines[0]):
        return text, {}

    meta: Dict[str, str] = {}
    for index in range(1, len(lines)):
        if _FRONTMATTER.match(lines[index]):
            for raw in lines[1:index]:
                if ":" in raw and not raw.startswith((" ", "-", "#")):
                    key, _, value = raw.partition(":")
                    meta[key.strip()] = value.strip().strip("'\"")
            return "\n".join(lines[index + 1 :]), meta

    # Unterminated frontmatter: treat the whole file as body rather than
    # silently discarding it.
    return text, {}


def parse_sections(text: str) -> List[Section]:
    """Split Markdown into heading-scoped sections, ignoring headings in code.

    The fence tracking is the important detail: a hash inside a Python code
    block is a comment, not a heading, and treating it as one shreds the doc.
    """
    sections: List[Section] = []
    stack: List[str] = []
    current_body: List[str] = []
    current_breadcrumb = ""
    current_level = 0

    in_fence = False
    fence_marker = ""

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body or current_breadcrumb:
            sections.append(
                Section(
                    breadcrumb=current_breadcrumb,
                    heading_level=current_level,
                    body=body,
                )
            )

    for line in text.splitlines():
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0] * 3
            elif marker.startswith(fence_marker):
                in_fence = False
            current_body.append(line)
            continue

        if in_fence:
            current_body.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            current_body = []
            level = len(heading.group(1))
            heading_title = heading.group(2).strip()
            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(heading_title)
            current_breadcrumb = " > ".join(p for p in stack if p)
            current_level = level
        else:
            current_body.append(line)

    flush()
    return sections


def _split_paragraphs(body: str) -> List[str]:
    """Split a section body into atomic units, keeping code fences intact.

    A fenced code block is one indivisible unit no matter how long it is.
    Truncating a code example is worse than exceeding the budget.
    """
    units: List[str] = []
    buffer: List[str] = []
    in_fence = False
    fence_marker = ""

    def flush_buffer() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            units.append(joined)

    for line in body.splitlines():
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                flush_buffer()
                buffer = [line]
                in_fence = True
                fence_marker = marker[0] * 3
            elif marker.startswith(fence_marker):
                buffer.append(line)
                flush_buffer()
                buffer = []
                in_fence = False
            else:
                buffer.append(line)
            continue

        if in_fence:
            buffer.append(line)
        elif line.strip():
            buffer.append(line)
        else:
            flush_buffer()
            buffer = []

    flush_buffer()
    return units


def _is_code_fence(unit: str) -> bool:
    return unit.lstrip().startswith(("```", "~~~"))


def _hard_split(unit: str, budget: int) -> List[str]:
    """Last-resort split for a single prose unit that busts the token budget.

    Paragraph-level packing cannot help when the paragraph ITSELF is larger than
    the budget, which happens in real corpora: wall-of-text release notes, long
    reference tables, generated API listings. Without this fallback the chunker
    silently emits one enormous chunk, which quietly breaks two things at once:
    the embedding model truncates at its context limit (so the tail of the text
    is never actually indexed), and a single retrieved chunk can blow the
    generator's context budget.

    Callers must not pass fenced code blocks here; a truncated code example is
    worse than an oversized chunk, so those stay intact by design.
    """
    if budget <= 0 or approx_token_count(unit) <= budget:
        return [unit]

    pieces: List[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(unit):
        sentence = sentence.strip()
        if not sentence:
            continue
        if approx_token_count(sentence) <= budget:
            pieces.append(sentence)
            continue

        # Still too long: fall back to fixed word windows. approx_token_count is
        # the max of word count and chars/4, so a window of `budget` words can
        # still exceed `budget` tokens. Halve until it actually fits.
        words = sentence.split()
        window = max(1, budget)
        while window > 1 and approx_token_count(" ".join(words[:window])) > budget:
            window = max(1, window // 2)
        for start in range(0, len(words), window):
            piece = " ".join(words[start : start + window])
            if piece:
                pieces.append(piece)

    # Re-pack the pieces so we emit as few chunks as the budget allows instead of
    # one chunk per sentence.
    packed: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = approx_token_count(piece)
        if current and current_tokens + piece_tokens > budget:
            packed.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        packed.append(" ".join(current))

    return packed or [unit]


def _chunk_id(source: str, breadcrumb: str, text: str) -> str:
    """Content-addressed and stable across rebuilds.

    Includes source and breadcrumb so identical boilerplate under different
    headings does not collapse into one chunk.
    """
    payload = _SEP.join((source, breadcrumb, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build_embed_text(title: str, breadcrumb: str, body: str) -> str:
    prefix = f"{title} | {breadcrumb}" if breadcrumb else title
    return f"{prefix}\n\n{body}"


def chunk_markdown(
    text: str,
    *,
    source: str,
    title: str = "",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 16,
) -> List[Chunk]:
    """Chunk one Markdown document.

    Args:
        max_tokens: Target upper bound per chunk. 512 is deliberate: large
            enough to hold a complete explanation plus a code example, small
            enough that a top-5 retrieval fits comfortably in context alongside
            the system prompt.
        overlap_tokens: Overlap applied at paragraph granularity when a section
            must be split, so a concept spanning a boundary survives intact in
            at least one chunk.
        min_tokens: Chunks shorter than this are merged forward. Prevents a
            corpus littered with "See also" fragments that match everything
            weakly and nothing well.
    """
    body_text, frontmatter = strip_frontmatter(text)
    doc_title = title or frontmatter.get("title", "") or source

    chunks: List[Chunk] = []
    position = 0

    for section in parse_sections(body_text):
        if not section.body.strip():
            continue

        breadcrumb = section.breadcrumb
        prefix = f"{doc_title} | {breadcrumb}" if breadcrumb else doc_title
        prefix_cost = approx_token_count(prefix)
        budget = max(max_tokens - prefix_cost, min_tokens)

        bodies: List[str] = []

        if approx_token_count(section.body) <= budget:
            bodies.append(section.body)
        else:
            units = []
            for raw_unit in _split_paragraphs(section.body):
                if _is_code_fence(raw_unit) or approx_token_count(raw_unit) <= budget:
                    units.append(raw_unit)
                else:
                    units.extend(_hard_split(raw_unit, budget))

            current: List[str] = []
            current_tokens = 0

            for unit in units:
                unit_tokens = approx_token_count(unit)

                if current and current_tokens + unit_tokens > budget:
                    bodies.append("\n\n".join(current))
                    tail: List[str] = []
                    tail_tokens = 0
                    for prev in reversed(current):
                        prev_tokens = approx_token_count(prev)
                        if tail_tokens + prev_tokens > overlap_tokens:
                            break
                        tail.insert(0, prev)
                        tail_tokens += prev_tokens
                    current = tail
                    current_tokens = tail_tokens

                current.append(unit)
                current_tokens += unit_tokens

            if current:
                bodies.append("\n\n".join(current))

        for body in bodies:
            body = body.strip()
            if not body:
                continue
            embed_text = _build_embed_text(doc_title, breadcrumb, body)
            chunks.append(
                Chunk(
                    id=_chunk_id(source, breadcrumb, body),
                    text=body,
                    embed_text=embed_text,
                    source=source,
                    title=doc_title,
                    breadcrumb=breadcrumb,
                    heading_level=section.heading_level,
                    token_count=approx_token_count(embed_text),
                    position=position,
                    metadata=dict(frontmatter),
                )
            )
            position += 1

    return _merge_tiny(chunks, min_tokens=min_tokens)


def _remake(base: Chunk, body: str, position: int) -> Chunk:
    embed_text = _build_embed_text(base.title, base.breadcrumb, body)
    return Chunk(
        id=_chunk_id(base.source, base.breadcrumb, body),
        text=body,
        embed_text=embed_text,
        source=base.source,
        title=base.title,
        breadcrumb=base.breadcrumb,
        heading_level=base.heading_level,
        token_count=approx_token_count(embed_text),
        position=position,
        metadata=base.metadata,
    )


def _merge_tiny(chunks: List[Chunk], *, min_tokens: int) -> List[Chunk]:
    """Merge undersized chunks into the next chunk, or the previous one."""
    if not chunks:
        return []

    out: List[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is not None:
            merged_body = f"{pending.text}\n\n{chunk.text}"
            chunk = _remake(chunk, merged_body, pending.position)
            pending = None

        if chunk.token_count < min_tokens:
            pending = chunk
            continue
        out.append(chunk)

    if pending is not None:
        if out:
            last = out[-1]
            merged_body = f"{last.text}\n\n{pending.text}"
            out[-1] = _remake(last, merged_body, last.position)
        else:
            out.append(pending)

    return out


def iter_chunk_dicts(chunks: Sequence[Chunk]) -> Iterator[Dict[str, Any]]:
    for chunk in chunks:
        yield chunk.to_dict()


__all__ = (
    "Chunk",
    "Section",
    "chunk_markdown",
    "parse_sections",
    "strip_frontmatter",
    "iter_chunk_dicts",
)
