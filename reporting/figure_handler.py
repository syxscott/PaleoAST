"""LaTeX figure handling utilities for PaleoAST reports."""

import re

# Standard LaTeX escape table, keyed by the RAW character so it can be
# matched in a single alternation. The braces in \textasciitilde{} and
# \textasciicircum{} are load-bearing: TeX swallows the space after a
# control word, so the unbraced form would render as "x^2" and lose the
# space.
_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_ESCAPE_RE = re.compile("|".join(re.escape(char) for char in _LATEX_ESCAPES))


def _escape_latex(text: str) -> str:
    """Escape LaTeX-significant characters in user-supplied text.

    The previous implementation embedded ``caption`` and ``label``
    directly into LaTeX, allowing arbitrary injection (a caption of
    ``}\n\\input{secret.tex}`` would terminate the caption and pull
    in another file). Apply the standard LaTeX escape table to every
    caller-supplied string before interpolation.
    """
    if text is None:
        return ""
    # One pass over the input, never over our own output. The table used
    # to be applied as a sequence of str.replace() calls with the
    # backslash first, on the stated grounds that this avoided
    # re-escaping the substitutions below; it did the opposite -- the
    # braces emitted by \textbackslash{} were then rewritten by the later
    # { and } passes into \textbackslash\{\}, which LaTeX renders as the
    # literal words "textbackslash{}data". No ordering can fix that,
    # because the text being consumed is produced by the same pass.
    return _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_ESCAPES[m.group()], text)


class FigureHandler:
    """Manages LaTeX figure inclusions."""

    @staticmethod
    def include_figure(path: str, caption: str = "", label: str = "", width: str = "0.8\\textwidth") -> str:
        lines = ["\\begin{figure}[htbp]", "\\centering"]
        # ``path`` is treated as a file path; do not escape it because
        # \\includegraphics expects a verbatim path. ``caption`` and
        # ``label`` are user-supplied text and must be escaped.
        lines.append(f"\\includegraphics[width={width}]{{{path}}}")
        if caption:
            lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        if label:
            lines.append(f"\\label{{{_escape_latex(label)}}}")
        lines.append("\\end{figure}")
        return "\n".join(lines)
