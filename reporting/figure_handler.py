"""LaTeX figure handling utilities for PaleoAST reports."""


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
    # Backslash must be escaped first to avoid re-escaping the
    # substitutions below.
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = text
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


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