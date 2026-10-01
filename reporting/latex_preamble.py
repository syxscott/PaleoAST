"""LaTeX preamble and document class definitions for PaleoAST report generation."""

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DocumentClass(Enum):
    """LaTeX document classes."""

    ARTICLE = auto()
    REPORT = auto()
    BOOK = auto()
    LETTER = auto()
    BEAMER = auto()


class LatexPreamble:
    """LaTeX preamble manager for document generation."""

    def __init__(
        self, document_class: DocumentClass = DocumentClass.ARTICLE, font_size: int = 11, paper_size: str = "a4paper"
    ):
        self._doc_class = document_class
        self._font_size = font_size
        self._paper_size = paper_size
        self._packages: list[str] = []
        self._extra_preamble: list[str] = []

    @property
    def packages(self) -> list[str]:
        return self._packages.copy()

    def add_package(self, name: str, options: str | None = None):
        """Add a ``\\usepackage`` line, ignoring a repeat of the same package.

        A package requested twice with *different* options is a hard LaTeX
        error ("Option clash for package ..."), and twice with the same
        options is at best a warning. Either way the second line is a mistake
        the caller should not have to know about, so a repeat is dropped and
        the first registration wins. Returning the stored line (or None when
        suppressed) lets callers see what happened.
        """
        key = name.strip()
        if any(self._package_key(line) == key for line in self._packages):
            logger.debug("LaTeX package %r already present; ignoring the repeat", key)
            return None
        if options:
            line = f"\\usepackage[{options}]{{{name}}}"
        else:
            line = f"\\usepackage{{{name}}}"
        self._packages.append(line)
        return line

    @staticmethod
    def _package_key(line: str) -> str:
        """The bare package name from a ``\\usepackage[..]{..}`` line."""
        start = line.find("{", line.find("]") + 1 if "[" in line else 0)
        if start == -1:
            return line.strip()
        end = line.find("}", start)
        return line[start + 1 : end] if end != -1 else line.strip()

    def add_preamble_line(self, line: str):
        self._extra_preamble.append(line)

    def generate_documentclass(self) -> str:
        class_map = {
            DocumentClass.ARTICLE: "article",
            DocumentClass.REPORT: "report",
            DocumentClass.BOOK: "book",
            DocumentClass.LETTER: "letter",
            DocumentClass.BEAMER: "beamer",
        }
        cls = class_map.get(self._doc_class, "article")
        return f"\\documentclass[{self._font_size}pt,{self._paper_size}]{{{cls}}}"
