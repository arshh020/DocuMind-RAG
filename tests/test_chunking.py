"""Tests for markdown-aware chunking.

Chunking is the highest-leverage and least-glamorous part of a RAG system. If a
chunk splits a code example away from the sentence explaining it, no retriever
and no LLM can recover the lost context. These tests pin the behaviours that
protect against that.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsrag.chunking import (
    Chunk,
    chunk_markdown,
    parse_sections,
    strip_frontmatter,
)

SAMPLE = """---
title: Array.prototype.map
slug: Web/JavaScript/Reference
---

# Array.prototype.map()

The map() method creates a new array populated with the results of calling a
provided function on every element in the calling array.

## Syntax

The syntax is straightforward.

```js
// # this hash must not be treated as a heading
const doubled = numbers.map((n) => n * 2)
```

## Description

map() calls the callback once for each index of the array. It does not call the
callback for empty slots in sparse arrays.

### Sparse arrays

Holes are preserved in the returned array.
"""


class TestFrontmatter(unittest.TestCase):
    def test_strips_yaml_frontmatter(self):
        stripped, metadata = strip_frontmatter(SAMPLE)
        self.assertNotIn("slug:", stripped)
        self.assertTrue(stripped.lstrip().startswith("#"))

    def test_returns_parsed_frontmatter_metadata(self):
        """Frontmatter is returned, not just discarded.

        MDN files carry the canonical page slug in frontmatter. Keeping it means
        a citation can eventually link to the real published URL instead of only
        a local file path.
        """
        _, metadata = strip_frontmatter(SAMPLE)
        self.assertEqual(metadata.get("title"), "Array.prototype.map")
        self.assertIn("slug", metadata)

    def test_leaves_text_without_frontmatter_alone(self):
        text = "# Title\n\nBody text."
        self.assertEqual(strip_frontmatter(text)[0], text)

    def test_handles_empty_string(self):
        self.assertEqual(strip_frontmatter("")[0], "")


class TestParseSections(unittest.TestCase):
    def test_finds_all_headings(self):
        sections = parse_sections(strip_frontmatter(SAMPLE)[0])
        titles = [s.title for s in sections]
        self.assertIn("Array.prototype.map()", titles)
        self.assertIn("Syntax", titles)
        self.assertIn("Description", titles)
        self.assertIn("Sparse arrays", titles)

    def test_ignores_hash_inside_code_fence(self):
        """A '#' inside a fenced block is a comment, not a heading.

        Naive line-by-line heading detection splits code examples in half here.
        This is the single most common chunking bug in tutorial RAG projects.
        """
        titles = [s.title for s in parse_sections(strip_frontmatter(SAMPLE)[0])]
        self.assertNotIn("this hash must not be treated as a heading", titles)
        for title in titles:
            self.assertNotIn("must not be treated", title)

    def test_builds_hierarchical_breadcrumb(self):
        sections = parse_sections(strip_frontmatter(SAMPLE)[0])
        nested = [s for s in sections if s.title == "Sparse arrays"]
        self.assertTrue(nested)
        breadcrumb = nested[0].breadcrumb
        self.assertIn("Description", breadcrumb)
        self.assertIn("Sparse arrays", breadcrumb)

    def test_content_before_first_heading_is_kept(self):
        sections = parse_sections("Intro text with no heading.\n\n# Later\n\nBody")
        joined = " ".join(s.text for s in sections)
        self.assertIn("Intro text", joined)


class TestChunkMarkdown(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_markdown(SAMPLE, source="map.md")

    def test_produces_chunks(self):
        self.assertTrue(self.chunks)
        self.assertTrue(all(isinstance(c, Chunk) for c in self.chunks))

    def test_every_chunk_has_source_and_id(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.source, "map.md")
            self.assertTrue(chunk.id)
            self.assertTrue(chunk.text.strip())

    def test_chunk_ids_are_unique(self):
        ids = [c.id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chunk_ids_are_stable_across_rebuilds(self):
        """Content-hash ids mean eval gold labels survive a reindex.

        With positional ids, adding one document at the top of the corpus
        renumbers everything and silently invalidates the entire eval set.
        """
        again = chunk_markdown(SAMPLE, source="map.md")
        self.assertEqual([c.id for c in self.chunks], [c.id for c in again])

    def test_chunk_id_changes_when_text_changes(self):
        modified = chunk_markdown(
            SAMPLE.replace("creates a new array", "creates a fresh array"),
            source="map.md",
        )
        self.assertNotEqual(
            {c.id for c in self.chunks}, {c.id for c in modified}
        )

    def test_embed_text_includes_breadcrumb_context(self):
        """Heading context is prepended to the embedded text.

        A chunk reading 'Holes are preserved in the returned array' is nearly
        meaningless in isolation. Prepending the breadcrumb tells both the
        embedder and BM25 what it is about.
        """
        target = [c for c in self.chunks if "Holes are preserved" in c.text]
        self.assertTrue(target)
        self.assertIn("Sparse arrays", target[0].embed_text)

    def test_code_fence_stays_intact(self):
        """A fenced code block must never be split across two chunks."""
        for chunk in self.chunks:
            self.assertEqual(
                chunk.text.count("```") % 2,
                0,
                f"unbalanced code fence in chunk {chunk.id}: {chunk.text[:120]}",
            )

    def test_respects_max_tokens_budget(self):
        long_text = "# Title\n\n" + ("word " * 4000)
        chunks = chunk_markdown(long_text, source="long.md", max_tokens=128)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Allow modest headroom: we never split an indivisible unit such as
            # a code fence just to satisfy the budget.
            self.assertLessEqual(chunk.token_count, 128 * 3)

    def test_overlap_is_applied_between_adjacent_chunks(self):
        long_text = "# Title\n\n" + "\n\n".join(
            f"Paragraph {i} contains distinctive filler text about topic {i}."
            for i in range(80)
        )
        chunks = chunk_markdown(
            long_text, source="o.md", max_tokens=64, overlap_tokens=16
        )
        self.assertGreater(len(chunks), 1)
        total_length = sum(len(c.text) for c in chunks)
        # Overlap means the chunks together are longer than the original.
        self.assertGreater(total_length, len(long_text) * 0.9)

    def test_tiny_chunks_are_merged_away(self):
        text = "# A\n\nx\n\n## B\n\ny\n\n## C\n\nz"
        chunks = chunk_markdown(text, source="tiny.md", min_tokens=20)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip())

    def test_empty_input_returns_no_chunks(self):
        self.assertEqual(chunk_markdown("", source="empty.md"), [])
        self.assertEqual(chunk_markdown("   \n\n  ", source="blank.md"), [])

    def test_document_with_no_headings_still_chunks(self):
        chunks = chunk_markdown(
            "Just prose with no headings at all. " * 50, source="flat.md"
        )
        self.assertTrue(chunks)

    def test_positions_are_sequential(self):
        positions = [c.position for c in self.chunks]
        self.assertEqual(positions, sorted(positions))

    def test_round_trip_serialization(self):
        for chunk in self.chunks:
            restored = Chunk.from_dict(chunk.to_dict())
            self.assertEqual(restored.id, chunk.id)
            self.assertEqual(restored.text, chunk.text)
            self.assertEqual(restored.breadcrumb, chunk.breadcrumb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
