"""Matrix-to-LaTeX conversion utilities for PaleoAST reports."""

import numpy as np

from .figure_handler import _escape_latex


class MatrixConverter:
    """Converts numpy matrices to LaTeX representations."""

    @staticmethod
    def to_latex(
        matrix: np.ndarray, row_labels: list[str] | None = None, col_labels: list[str] | None = None, fmt: str = ".4f"
    ) -> str:
        """Render a 2-D matrix as a LaTeX ``tabular`` environment.

        Labels are escaped with the package's shared ``_escape_latex``, the
        same helper ``table_generator`` and ``report_builder`` already use.
        They were previously interpolated raw, and the strings that reach here
        are taxon names: ``Globigerinoides_ruber`` went into the document with
        a bare underscore, which is a LaTeX compile error rather than a
        cosmetic flaw. ``&``, ``%``, ``#``, ``$``, ``{`` and ``}`` are the same
        story, and a closing brace in a label can terminate the cell early.

        Label lengths are checked against the matrix rather than discovered by
        index error or by LaTeX. The column spec is built from the matrix
        shape, so a ``col_labels`` list of the wrong length emits a header row
        with a different number of cells than the spec declares -- again a
        compile error, but one raised by the document compiler hours away from
        the call that caused it.

        Parameters:
            matrix: A 2-D array. The formatted numbers need no escaping.
            row_labels: One label per row, or None.
            col_labels: One label per column, or None.
            fmt: Format spec applied to every cell.

        Raises:
            ValueError: If matrix is not 2-D, or a label list's length does
                not match the corresponding dimension.
        """
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError(f"to_latex needs a 2-D matrix, got {arr.ndim}-D with shape {arr.shape}")
        n_rows, n_cols = arr.shape

        if row_labels is not None and len(row_labels) != n_rows:
            raise ValueError(f"row_labels has {len(row_labels)} entries but the matrix has {n_rows} rows")
        if col_labels is not None and len(col_labels) != n_cols:
            raise ValueError(f"col_labels has {len(col_labels)} entries but the matrix has {n_cols} columns")

        col_spec = "|".join(["c"] * (n_cols + (1 if row_labels else 0)))
        lines = [f"\\begin{{tabular}}{{|{col_spec}|}}", "\\hline"]
        if col_labels:
            header = ""
            if row_labels:
                header = " & "
            header += " & ".join(_escape_latex(str(c)) for c in col_labels) + " \\\\"
            lines.append(header)
            lines.append("\\hline")
        for i in range(n_rows):
            parts = []
            if row_labels:
                parts.append(_escape_latex(str(row_labels[i])))
            parts.extend(f"{arr[i, j]:{fmt}}" for j in range(n_cols))
            lines.append(" & ".join(parts) + " \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)
