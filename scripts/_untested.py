"""List the analyses that no test file exercises."""

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugins.catalog import BUILTIN_ANALYSES
from scripts.validation_matrix import build

rows = {str(r["name"]): r for r in build()}
untested = [e for e in BUILTIN_ANALYSES if not rows[e.name]["any"]]
print(f"{len(untested)} analyses with no test at all:\n")
for entry in untested:
    print(f"  {entry.name:<30} {entry.module:<34} {entry.description}")
print("\nthe ones with SOME evidence but no plain unit test:")
for entry in BUILTIN_ANALYSES:
    row = rows[entry.name]
    if row["any"] and not row["unit"]:
        kinds = [k for k in ("R", "gold", "prop") if row[k]]
        print(f"  {entry.name:<30} {', '.join(kinds)}")
