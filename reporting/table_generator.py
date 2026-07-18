"""LaTeX table generation utilities for PaleoAST reports."""

from typing import Any

from .figure_handler import _escape_latex


class TableGenerator:
    """Generates LaTeX table code from data."""

    @staticmethod
    def from_matrix(data: list[list[Any]], headers: list[str] | None = None, caption: str = "", label: str = "") -> str:
        if not data:
            return ""
        n_cols = len(data[0])
        col_spec = "|".join(["c"] * n_cols)
        lines = [f"\\begin{{tabular}}{{|{col_spec}|}}"]
        lines.append("\\hline")
        if headers:
            # Headers are user-supplied text — escape LaTeX-significant
            # characters to prevent injection (the previous version
            # embedded them raw, allowing arbitrary LaTeX commands).
            lines.append(" & ".join(_escape_latex(str(h)) for h in headers) + " \\\\")
            lines.append("\\hline")
        for row in data:
            lines.append(" & ".join(_escape_latex(str(v)) for v in row) + " \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        table = "\n".join(lines)
        if caption or label:
            env = ["\\begin{table}[htbp]", "\\centering", table]
            if caption:
                env.append(f"\\caption{{{_escape_latex(caption)}}}")
            if label:
                env.append(f"\\label{{{_escape_latex(label)}}}")
            env.append("\\end{table}")
            return "\n".join(env)
        return table