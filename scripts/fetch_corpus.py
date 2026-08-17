#!/usr/bin/env python3
"""Fetch an openly-licensed documentation corpus.

Why this script exists at all: the tutorial project this replaces committed a
copyrighted commercial medical textbook into a public repo. That is a real legal
problem and a bad look on a portfolio. Every corpus offered here is openly
licensed and the license is recorded in the corpus directory.

Available corpora:
    mdn-js    MDN JavaScript reference       (CC-BY-SA 2.5+)  ~1500 files
    mdn-css   MDN CSS reference              (CC-BY-SA 2.5+)  ~1200 files
    mdn-http  MDN HTTP reference             (CC-BY-SA 2.5+)  ~600 files
    peps      Python Enhancement Proposals   (public domain)  ~700 files

Usage:
    python scripts/fetch_corpus.py --corpus mdn-js
    python scripts/fetch_corpus.py --corpus mdn-css --out data/corpus

Uses a sparse git checkout so you download only the subdirectory you need
instead of the whole multi-gigabyte repository.

Recommendation for a web developer: pick mdn-js or mdn-css. Domain familiarity is
the single biggest lever on how fast you can build and verify an eval set, and
the eval set is the bottleneck on a tight deadline.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CORPORA = {
    "mdn-js": {
        "repo": "https://github.com/mdn/content.git",
        "subdir": "files/en-us/web/javascript",
        "license": "CC-BY-SA 2.5+ (MDN Web Docs, https://github.com/mdn/content)",
    },
    "mdn-css": {
        "repo": "https://github.com/mdn/content.git",
        "subdir": "files/en-us/web/css",
        "license": "CC-BY-SA 2.5+ (MDN Web Docs, https://github.com/mdn/content)",
    },
    "mdn-http": {
        "repo": "https://github.com/mdn/content.git",
        "subdir": "files/en-us/web/http",
        "license": "CC-BY-SA 2.5+ (MDN Web Docs, https://github.com/mdn/content)",
    },
    "peps": {
        "repo": "https://github.com/python/peps.git",
        "subdir": "peps",
        "license": "Public domain / CC0 (Python Enhancement Proposals)",
    },
}


def run(command: list[str], cwd: Path | None = None) -> None:
    print("  $ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"command failed: {' '.join(command)}")


def fetch(name: str, out_dir: Path) -> int:
    spec = CORPORA[name]
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp:
        clone_dir = Path(temp) / "repo"
        print(f"Sparse-cloning {spec['repo']} ...")
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                spec["repo"],
                str(clone_dir),
            ]
        )
        run(["git", "sparse-checkout", "set", spec["subdir"]], cwd=clone_dir)

        source = clone_dir / spec["subdir"]
        if not source.is_dir():
            raise SystemExit(f"expected subdirectory missing: {spec['subdir']}")

        copied = 0
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".markdown", ".rst", ".txt"}:
                continue
            # Flatten the tree into readable filenames so `source` metadata in
            # citations stays short and human-meaningful.
            relative = path.relative_to(source)
            flat_name = str(relative).replace("/", "__").replace("\\", "__")
            shutil.copyfile(path, out_dir / flat_name)
            copied += 1

    (out_dir / "LICENSE.txt").write_text(
        f"Corpus: {name}\nLicense: {spec['license']}\n"
        f"Source repository: {spec['repo']}\nSubdirectory: {spec['subdir']}\n",
        encoding="utf-8",
    )
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default="mdn-js", choices=sorted(CORPORA), help="corpus to fetch"
    )
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove existing files in the output directory first",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git is required but was not found on PATH")

    if args.clean and args.out.is_dir():
        shutil.rmtree(args.out)

    count = fetch(args.corpus, args.out)
    print(f"\nFetched {count} files into {args.out}")
    print(f"License recorded in {args.out / 'LICENSE.txt'}")
    print("\nNext:  python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
