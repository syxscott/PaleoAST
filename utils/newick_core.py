# =============================================================================
# FILE: utils/newick_core.py
# =============================================================================
"""
Shared Newick Recursive-Descent Parsing Core

One grammar implementation serving BOTH public parsers:

- ``parsers.newick_parser.NewickParser``  -> builds ``TreeNode``/``NewickTree``
- ``phylogenetics.tree._NewickParser``    -> builds ``PhyloNode``

Previously these were two independently written parsers that had drifted
apart (quote handling, comment nesting, branch-length validation, depth
guards).  This module merges the *grammar* into a single superset
implementation that emits model-agnostic :class:`NewickNodeSpec` trees;
each package keeps only its own node-building policy (NHX value typing,
metadata placement, negative branch-length handling).

Superset grammar features (union of both historic parsers):
    - Unquoted labels (stop on ``(),;:[]``, whitespace, quotes)
    - Single-quoted labels with ``''`` doubling *and* backslash escapes
    - Double-quoted labels with backslash escapes
    - Nesting-aware ``[ ... ]`` comments anywhere around a node
      (``&&NHX`` detection is left to the caller's comment policy)
    - ``!`` line comments (NEXUS-style reference lines)
    - Multiple trees per string, separated by ``;`` (or ``,``)
    - Recursion-depth guard (``MAX_NEWICK_DEPTH``)
    - Every syntax error carries line/column/offset diagnostics via
      :class:`utils.exceptions.NewickParseError`

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.exceptions import NewickParseError

# One Python frame per nesting level; the cap must stay comfortably below
# sys.getrecursionlimit() (1000 by default) so the guard, not a raw
# RecursionError, reports pathological input.
MAX_NEWICK_DEPTH = 400

_STRUCTURAL_CHARS = frozenset("(),;:[]")
_WHITESPACE = " \t\n\r"
_QUOTE_CHARS = "'\""


@dataclass
class NewickNodeSpec:
    """
    Raw parse result for one node, independent of any tree model.

    Attributes:
        name: Node label (``""`` when absent; quoting already resolved).
        branch_length: Parsed branch length or ``None``.
        comments: Bodies of ``[ ... ]`` comments attached to this node,
            in source order, *including* the ``&&NHX`` prefix when present.
        children: Child specs (empty for leaves).
    """

    name: str = ""
    branch_length: float | None = None
    comments: list[str] = field(default_factory=list)
    children: list[NewickNodeSpec] = field(default_factory=list)


class _Scanner:
    """Character-level cursor over the original text (offsets stay valid)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.length = len(text)
        self.depth = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def error(self, message: str, at: int | None = None) -> None:
        at = self.pos if at is None else at
        line_start = self.text.rfind("\n", 0, at) + 1
        line_end = self.text.find("\n", at)
        if line_end == -1:
            line_end = self.length
        raise NewickParseError(
            message,
            line=self.text.count("\n", 0, at) + 1,
            column=at - line_start + 1,
            position=at,
            line_text=self.text[line_start:line_end],
        )

    # ------------------------------------------------------------------
    # Low-level scanning
    # ------------------------------------------------------------------
    def current(self) -> str:
        return self.text[self.pos] if self.pos < self.length else ""

    def at_line_start(self) -> bool:
        idx = self.pos - 1
        while idx >= 0 and self.text[idx] in " \t":
            idx -= 1
        return idx < 0 or self.text[idx] == "\n"

    def skip_ws(self) -> None:
        while self.pos < self.length:
            char = self.text[self.pos]
            if char in _WHITESPACE:
                self.pos += 1
            elif char == "!" and self.at_line_start():
                newline = self.text.find("\n", self.pos)
                self.pos = self.length if newline == -1 else newline
            else:
                return

    def collect_comments(self, sink: list[str]) -> None:
        """Consume consecutive ``[ ... ]`` blocks at the cursor (nesting-aware)."""
        while True:
            self.skip_ws()
            if self.current() != "[":
                return
            start = self.pos
            self.pos += 1
            depth = 1
            while self.pos < self.length and depth > 0:
                char = self.text[self.pos]
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                self.pos += 1
            if depth > 0:
                self.error("Unterminated bracket comment", at=start)
            sink.append(self.text[start + 1 : self.pos - 1])

    # ------------------------------------------------------------------
    # Tokens
    # ------------------------------------------------------------------
    def parse_name(self) -> str:
        self.skip_ws()
        if self.current() in ("'", '"'):
            return self.parse_quoted_name()
        chars: list[str] = []
        while self.pos < self.length:
            char = self.text[self.pos]
            if char in _STRUCTURAL_CHARS or char in _WHITESPACE or char in _QUOTE_CHARS:
                break
            chars.append(char)
            self.pos += 1
        return "".join(chars)

    def parse_quoted_name(self) -> str:
        quote = self.current()
        start = self.pos
        self.pos += 1
        chars: list[str] = []
        while True:
            char = self.current()
            if char == "":
                self.error(f"Unterminated quoted Newick label: missing closing {quote!r}", at=start)
            if char == "\\" and self.pos + 1 < self.length and self.text[self.pos + 1] in (quote, "\\"):
                chars.append(self.text[self.pos + 1])
                self.pos += 2
                continue
            if char == quote:
                if quote == "'" and self.text[self.pos + 1 : self.pos + 2] == "'":
                    chars.append("'")
                    self.pos += 2
                    continue
                self.pos += 1
                return "".join(chars)
            chars.append(char)
            self.pos += 1

    def parse_number(self) -> float:
        start = self.pos
        chars: list[str] = []
        while self.pos < self.length:
            char = self.text[self.pos]
            if char in _STRUCTURAL_CHARS or char in _WHITESPACE or char in _QUOTE_CHARS:
                break
            chars.append(char)
            self.pos += 1
        raw = "".join(chars)
        try:
            return float(raw)
        except ValueError:
            detail = f"got {raw!r}" if raw else "branch length is empty"
            self.error(f"Invalid branch length after ':' ({detail})", at=start)

    # ------------------------------------------------------------------
    # Productions
    # ------------------------------------------------------------------
    def parse_node(self) -> NewickNodeSpec:
        # Single recursive frame per nesting level (keep the depth cap
        # well below ``sys.getrecursionlimit()`` and convert any
        # leaked RecursionError into a diagnostic parse error).
        self.depth += 1
        if self.depth > MAX_NEWICK_DEPTH:
            self.error("Newick nesting depth exceeded {0}; tree is malformed or too deeply nested".format(MAX_NEWICK_DEPTH))
        try:
            self.skip_ws()
            char = self.current()

            if char == "(":
                self.pos += 1
                spec = NewickNodeSpec()
                pending: list[str] = []
                while True:
                    self.collect_comments(pending)
                    self.skip_ws()
                    if self.current() == ")" and not spec.children:
                        self.error("Empty '()' in Newick tree: no children given")
                    child = self.parse_node()
                    child.comments = pending + child.comments
                    pending = []
                    self.collect_comments(child.comments)
                    spec.children.append(child)
                    self.skip_ws()
                    sep = self.current()
                    if sep == ",":
                        self.pos += 1
                        continue
                    if sep == ")":
                        self.pos += 1
                        break
                    if sep == "":
                        self.error("Unexpected end of input: expected ',' or ')'")
                    self.error(f"Expected ',' or ')', got {sep!r}")
                self.skip_ws()
                if self.current() not in ("", ",", ")", ";", ":", "["):
                    spec.name = self.parse_name()
                spec.name = spec.name or ""
                self._finish_fields(spec)
                return spec

            if char in ("", ",", ")", ";"):
                label = "end of input" if char == "" else repr(char)
                self.error(f"Expected a node name or '(', got {label}")

            spec = NewickNodeSpec()
            if char != ":":
                spec.name = self.parse_name()
                if not spec.name:
                    self.error(f"Unexpected character {self.current()!r} where a node name was expected")
            self._finish_fields(spec)
            return spec
        finally:
            self.depth -= 1

    def _finish_fields(self, spec: NewickNodeSpec) -> None:
        self.collect_comments(spec.comments)
        self.skip_ws()
        if self.current() == ":":
            self.pos += 1
            spec.branch_length = self.parse_number()
            self.collect_comments(spec.comments)


def parse_newick_trees(text: str, out: list[NewickNodeSpec] | None = None) -> list[NewickNodeSpec]:
    """
    Parse every tree in ``text`` into model-agnostic node specs.

    Parameters:
        text: Newick content (one or more trees; ``;``-terminated,
            ``!`` line comments and bracket comments allowed anywhere).
        out: Optional accumulator; already-parsed specs are appended to
            it *before* any error is raised, so callers that recover
            from a bad segment (lenient multi-tree files) keep the
            trees parsed so far.

    Returns:
        The ``out`` list (or a fresh one) with one spec per tree root.

    Raises:
        NewickParseError: On any syntax error, with line/column/offset.
    """
    scanner = _Scanner(text)
    specs = out if out is not None else []
    while True:
        pending: list[str] = []
        scanner.collect_comments(pending)
        char = scanner.current()
        if char == "":
            break
        if char in (";", ","):
            scanner.pos += 1
            continue
        start = scanner.pos
        try:
            root = scanner.parse_node()
        except RecursionError:
            scanner.error("Newick nesting too deep; tree is malformed or too deeply nested", at=start)
        if scanner.pos == start:  # pragma: no cover - defensive guard
            scanner.error("Cannot parse Newick tree")
        root.comments = pending + root.comments
        specs.append(root)
        tail: list[str] = []
        scanner.collect_comments(tail)
        if scanner.current() == ";":
            scanner.pos += 1
            scanner.collect_comments(tail)
        root.comments.extend(tail)
    return specs
