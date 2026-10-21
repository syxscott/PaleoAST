"""Which test file could plausibly cover which analysis?

A candidate is a test file that mentions BOTH the analysis' class (or
method) AND the module it lives in.  That is the file that would have to
declare the analysis, if it really exercises it.
"""

import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugins.catalog import BUILTIN_ANALYSES
from scripts.validation_matrix import SELF, TESTS

files = sorted(p for p in TESTS.rglob("test_*.py") if p.relative_to(REPO).as_posix() != SELF)
cache = {p: p.read_text(encoding="utf-8", errors="ignore") for p in files}

candidates: dict[pathlib.Path, list[str]] = collections.defaultdict(list)
for entry in BUILTIN_ANALYSES:
    needles = [n for n in (entry.class_name, entry.method) if n]
    module_tail = entry.module.rsplit(".", 1)[-1]
    for path, text in cache.items():
        if any(needle in text for needle in needles) and module_tail in text:
            candidates[path].append(entry.name)

print(f"{len(candidates)} candidate test files\n")
for path in sorted(candidates):
    rel = path.relative_to(REPO).as_posix()
    print(f"{rel}")
    print(f"    {len(candidates[path])}: {', '.join(sorted(candidates[path]))}")
