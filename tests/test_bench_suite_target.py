"""Docs consistency: the consolidated bench target named by this repository.

szl-holdings/frontier-bench is the sole publisher of the consolidated bench.
Its evidence controller pins ``SPACE_ID = "SZLHOLDINGS/szl-bench-suite"``
(deploy/bench-plane/finish_bench_plane.py, frontier-bench@e73f9f4e) and its
publisher workflow holds the writer lock
``hf-write/space/SZLHOLDINGS/szl-bench-suite`` (.github/workflows/bench.yml).
The earlier personal-namespace Space is a historical surface that the
publisher no longer writes.

This test keeps this repository's files on that canonical target. It reads
local files only, makes no network call, and asserts nothing about whether
the Space is published, current, or running.

Stdlib unittest only, so CI can run it as ``python tests/<this file>``.
"""
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_TARGET = "SZLHOLDINGS/szl-bench-suite"
CANONICAL_URL = "https://huggingface.co/spaces/SZLHOLDINGS/szl-bench-suite"

# The retired personal-namespace target, as a repo id or as its static-Space
# host. Written as a character class so this file does not match itself.
RETIRED = re.compile(r"betterwithage[/-]szl-bench-suite", re.IGNORECASE)
# Any "<owner>/szl-bench-suite" repo id, and any Hub link to that name.
# A trailing ".ext" (a filename such as szl-bench-suite.README.md) is not a ref.
OWNER_REF = re.compile(r"(?<![\w.-])([\w.-]+)/szl-bench-suite(?![\w-]|\.\w)")
HUB_LINK = re.compile(r"https://huggingface\.co/(?:spaces/|datasets/)?[\w.-]+/szl-bench-suite(?![\w-]|\.\w)")

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}


def text_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                yield path, path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable: not documentation


class BenchSuiteTargetTests(unittest.TestCase):
    def test_no_file_references_the_retired_target(self):
        hits = []
        for path, text in text_files():
            for lineno, line in enumerate(text.splitlines(), start=1):
                if RETIRED.search(line):
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}")
        self.assertEqual(hits, [], "retired bench target referenced")

    def test_every_named_target_is_canonical(self):
        wrong = []
        for path, text in text_files():
            rel = path.relative_to(ROOT).as_posix()
            for match in OWNER_REF.finditer(text):
                if match.group(0) != CANONICAL_TARGET:
                    wrong.append(f"{rel}: {match.group(0)}")
            for match in HUB_LINK.finditer(text):
                if match.group(0) != CANONICAL_URL:
                    wrong.append(f"{rel}: {match.group(0)}")
        self.assertEqual(wrong, [], "non-canonical bench target")

    def test_readme_public_surface_names_the_canonical_target(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = re.search(r"^## Public surface\s*$(.*?)(?=^## |\Z)", readme, re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(section, "README has no '## Public surface' section")
        body = section.group(1)
        self.assertIn(f"[{CANONICAL_TARGET}]({CANONICAL_URL})", body)
        self.assertIn("https://github.com/szl-holdings/frontier-bench", body)


if __name__ == "__main__":
    unittest.main()
