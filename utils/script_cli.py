# =============================================================================
# FILE: utils/script_cli.py
# =============================================================================
"""
Headless entry point for the script session.

The console in ``views/script_console.py`` is for working interactively.
This is for the other half of why a scripting layer exists: the run that
takes forty minutes and produces one file, where a window would sit
useless on screen and the machine would be killed if it were closed by
accident.

    paleoast-run analysis.py --data sites.csv --var depth

Both paths build the same :class:`~utils.script_session.ScriptSession`,
so a script that works in one works in the other -- the only difference
is that here the data comes from a file rather than from whatever
spreadsheet happens to be open.

Exit codes are meaningful, because the whole point of running this from
a scheduler is that the scheduler can tell what happened:

    0   the script finished and nothing it ran raised
    1   the script raised; the traceback goes to stderr
    2   the file could not be read, or the arguments did not make sense

Data is read with the project's own loader when the suffix is one it
knows, so a .dat or .tps file behaves in a script the way it does in the
application. CSV and Excel go through pandas when it is available.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from utils.exceptions import PaleoASTError
from utils.script_session import ScriptSession

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_SCRIPT_ERROR = 1
EXIT_USAGE = 2


def load_data(path: Path) -> Any:
    """Read a data file into an ndarray.

    Prefers the project's own parsers, because they handle the formats
    this application exists to read -- grouped .dat files, TPS landmark
    sets, NEXUS -- and fall back to pandas for CSV and Excel.
    """
    suffix = path.suffix.lower()
    if suffix == ".dat":
        from parsers.dat_parser import DATParser

        # parse() takes a str, not a Path.
        return DATParser().parse(str(path)).data
    if suffix == ".tps":
        from parsers.tps_parser import TPSParser

        parsed = TPSParser().parse(str(path))
        # A TPS file is landmarks per specimen, not a rectangular table;
        # the natural array is one row per specimen, stacked.
        rows = [s.raw_data for s in parsed.specimens if s.raw_data is not None]
        if not rows:
            raise PaleoASTError(f"{path} contains no landmark coordinates")
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise PaleoASTError(
                f"{path} has specimens of differing landmark counts "
                f"{sorted(widths)}; they cannot be stacked into one array"
            )
        import numpy as _np

        return _np.vstack(rows)
    try:
        import pandas as pd
    except ImportError:
        raise PaleoASTError(f"reading {suffix or 'this file type'} needs pandas, which is not installed") from None
    if suffix in (".csv", ".txt", ""):
        frame = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")
    # Only numeric columns: a species table with a text column cannot
    # become an array, and silently dropping the column would change the
    # analysis the script asked for.
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] == 0:
        raise PaleoASTError(f"{path} has no numeric columns; a script needs an array")
    if numeric.shape[1] != frame.shape[1]:
        dropped = [c for c in frame.columns if c not in numeric.columns]
        logger.warning("dropped %d non-numeric column(s): %s", len(dropped), ", ".join(dropped))
    return numeric.to_numpy(dtype=float)


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="paleoast-run",
        description="Run a PaleoAST script against a data file.",
    )
    parser.add_argument("script", type=Path, help="Python file to execute")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Data file to expose as ``data``. Omit to run without one.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra name=value, parsed as JSON when possible. Repeatable.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the echoed results; only errors reach stderr.",
    )
    return parser


def _parse_vars(items: list[str]) -> dict[str, Any]:
    """Turn ``NAME=VALUE`` pairs into namespace entries.

    Values are JSON first, so ``n=999`` gives an int and
    ``groups=["a","b"]`` gives a list. A bare word stays a string, which
    is what someone writing ``metric=euclidean`` means.
    """
    import json

    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise PaleoASTError(f"--var expects NAME=VALUE, got {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name.isidentifier():
            raise PaleoASTError(f"--var name {name!r} is not a valid identifier")
        try:
            out[name] = json.loads(raw)
        except (ValueError, TypeError):
            out[name] = raw
    return out


def main(argv: list[str] | None = None) -> int:
    """Run one script and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.script.is_file():
        print(f"no such script: {args.script}", file=sys.stderr)
        return EXIT_USAGE
    try:
        data = load_data(args.data) if args.data else None
    except (PaleoASTError, OSError) as exc:
        print(f"could not load {args.data}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    session = ScriptSession(data_provider=(lambda: data) if data is not None else None)
    try:
        session.namespace.update(_parse_vars(args.var))
    except PaleoASTError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    source = args.script.read_text(encoding="utf-8")
    result = session.execute(source)
    if result.ok:
        if not args.quiet and result.text:
            print(result.text)
        return EXIT_OK
    sys.stderr.write(result.error or "the script raised\n")
    return EXIT_SCRIPT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
