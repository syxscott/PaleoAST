"""Validate the WiX authoring as XML before invoking candle.

candle's error for a malformed comment names a line and a column but not what
is wrong, and the message is localised, so a wrong character in a comment
costs a compile cycle each time. Two rules catch nearly all of it:

  * ``--`` is illegal inside an XML comment, which rules out the ASCII rules
    someone naturally writes under a heading;
  * ``<`` starts a tag, so a placeholder like ``<build_dir>`` in prose ends
    the comment early and the rest of the file is parsed as markup.

Both were hit while writing this file.

usage: python check_wxs.py <file> [<file> ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree

# A comment may not contain "--", so the delimiters themselves are stripped
# before the body is checked.
COMMENT = re.compile(r"<!--(.*?)-->", re.S)


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    for i, m in enumerate(COMMENT.finditer(text), 1):
        body = m.group(1)
        if "--" in body:
            line = text[: m.start()].count("\n") + 1
            problems.append(f"{path.name}:{line}  ' -- ' is illegal inside a comment")
        if "<" in body:
            line = text[: m.start()].count("\n") + 1
            problems.append(f"{path.name}:{line}  bare '<' inside a comment")

    # Anything left that looks like an unterminated comment.
    stripped = COMMENT.sub("", text)
    if "<!--" in stripped:
        problems.append(f"{path.name}  unterminated comment")

    try:
        ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        problems.append(f"{path.name}  XML parse error: {exc}")

    return problems


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    problems: list[str] = []
    for p in paths:
        found = check(p)
        print(f"{p.name}: {'ok' if not found else str(len(found)) + ' problem(s)'}")
        problems += found
    for line in problems:
        print("  " + line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
