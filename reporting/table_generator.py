"""LaTeX table generation utilities for PaleoAST reports."""

from typing import Any

from .figure_handler import _escape_latex


class TableGenerator:
    """Generates LaTeX table code from data."""

    @staticmethod
    def from_matrix(data: list[list[Any]], headers: list[str] | None = None, caption: str = "", label: str = "") -> str:
        r"""Render a list of rows as a LaTeX ``tabular`` environment.

        The column spec is built from ``len(data[0])``, so a ragged input
        silently emits a row with the wrong number of cells and the
        document only fails at COMPILE time with "Extra alignment tab has
        been changed to \cr" -- hours from the call that caused it.
        ``MatrixConverter.to_latex`` validates the same two invariants
        (2-D-ness, label lengths) for exactly this reason, so the same
        guard is applied here rather than letting the compiler find it.

        Raises:
            ValueError: If a row's length differs from the first row's, or
                ``headers`` does not have one entry per column.
        """
        if not data:
            return ""
        n_cols = len(data[0])
        ragged = [(i, len(row)) for i, row in enumerate(data) if len(row) != n_cols]
        if ragged:
            detail = ", ".join(f"row {i} has {n} cell(s)" for i, n in ragged[:5])
            raise ValueError(
                f"from_matrix: the column spec declares {n_cols} column(s) but the "
                f"rows are ragged ({detail}"
                f"{', ...' if len(ragged) > 5 else ''}). Pad the short rows; a ragged "
                f"table is a LaTeX 'extra alignment tab' compile error."
            )
        if headers is not None and len(headers) != n_cols:
            raise ValueError(
                f"from_matrix: headers has {len(headers)} entries but the data has {n_cols} column(s)"
            )
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
