"""LaTeX figure handling utilities for PaleoAST reports."""


class FigureHandler:
    """Manages LaTeX figure inclusions."""

    @staticmethod
    def include_figure(path: str, caption: str = "", label: str = "", width: str = "0.8\\textwidth") -> str:
        lines = ["\\begin{figure}[htbp]", "\\centering"]
        lines.append(f"\\includegraphics[width={width}]{{{path}}}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{figure}")
        return "\n".join(lines)
