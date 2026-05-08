"""Matrix-to-LaTeX conversion utilities for PaleoAST reports."""
from typing import Any, List, Optional
import numpy as np


class MatrixConverter:
    """Converts numpy matrices to LaTeX representations."""

    @staticmethod
    def to_latex(
        matrix: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        fmt: str = ".4f"
    ) -> str:
        n_rows, n_cols = matrix.shape
        col_spec = "|".join(["c"] * (n_cols + (1 if row_labels else 0)))
        lines = [f"\\begin{{tabular}}{{|{col_spec}|}}", "\\hline"]
        if col_labels:
            header = ""
            if row_labels:
                header = " & "
            header += " & ".join(str(c) for c in col_labels) + " \\\\"
            lines.append(header)
            lines.append("\\hline")
        for i in range(n_rows):
            parts = []
            if row_labels:
                parts.append(str(row_labels[i]))
            parts.extend(f"{matrix[i, j]:{fmt}}" for j in range(n_cols))
            lines.append(" & ".join(parts) + " \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)
