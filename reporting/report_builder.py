"""
================================================================================
PaleoAST Reporting - Report Builder Module
================================================================================

自动化LaTeX学术报告构建器。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from .figure_handler import _escape_latex
from .latex_preamble import DocumentClass, LatexPreamble

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """章节类型"""

    TITLE = auto()
    ABSTRACT = auto()
    INTRODUCTION = auto()
    METHODS = auto()
    RESULTS = auto()
    DISCUSSION = auto()
    CONCLUSION = auto()
    ACKNOWLEDGMENTS = auto()
    REFERENCES = auto()
    APPENDIX = auto()
    CUSTOM = auto()


@dataclass
class Section:
    """报告章节"""

    title: str
    content: str
    section_type: SectionType = SectionType.CUSTOM
    level: int = 1  # 1=section, 2=subsection, 3=subsubsection
    label: str | None = None


@dataclass
class FigureReference:
    """图表引用"""

    figure_id: str
    caption: str
    path: str  # 图片路径
    width: str = "0.8\\textwidth"


@dataclass
class TableReference:
    """表格引用"""

    table_id: str
    caption: str
    content: str  # LaTeX表格代码


@dataclass
class StatisticalResult:
    """统计结果"""

    test_name: str
    statistic: float
    p_value: float | None
    df: int | None = None
    effect_size: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


class ReportBuilder:
    """
    LaTeX学术报告构建器

    自动从分析结果生成格式严谨的科学报告。

    核心功能:
        1. LaTeX文档结构管理
        2. 动态内容插入
        3. 图表和表格引用
        4. 数学公式渲染
        5. bibliography管理

    使用示例:
        >>> builder = ReportBuilder()
        >>> builder.set_title("系统发育分析报告")
        >>> builder.add_section("方法", "我们使用...")
        >>> builder.add_figure(fig, "系统发育树", "fig:tree")
        >>> builder.add_table(latex_table, "距离矩阵", "tab:dist")
        >>> builder.generate("report.tex")
    """

    def __init__(
        self, document_class: DocumentClass = DocumentClass.ARTICLE, font_size: int = 11, paper_size: str = "a4paper"
    ):
        """
        初始化报告构建器

        参数:
            document_class: 文档类
            font_size: 字体大小
            paper_size: 纸张大小
        """
        self._doc_class = document_class
        self._font_size = font_size
        self._paper_size = paper_size
        self._preamble = LatexPreamble(document_class, font_size, paper_size)

        # 文档内容
        self._title: str | None = None
        self._authors: list[str] = []
        self._affiliations: list[str] = []
        self._date: str | None = None
        self._abstract: str | None = None

        self._sections: list[Section] = []
        self._figures: list[FigureReference] = []
        self._tables: list[TableReference] = []
        self._statistical_results: list[StatisticalResult] = []
        self._references: list[str] = []

        self._logger = logging.getLogger(f"{__name__}.ReportBuilder")
        self._counter_figure = 0
        self._counter_table = 0

        # 默认设置
        self._preamble.add_package("graphicx")
        self._preamble.add_package("booktabs")
        self._preamble.add_package("amsmath")
        self._preamble.add_package("amssymb")
        self._preamble.add_package("hyperref")
        self._preamble.add_package("geometry", "margin=1in")
        self._preamble.add_package("natbib")

    def set_title(self, title: str) -> ReportBuilder:
        """设置标题"""
        self._title = title
        return self

    def add_author(self, name: str, affiliation: str | None = None, email: str | None = None) -> ReportBuilder:
        """添加作者"""
        self._authors.append(name)
        if affiliation:
            self._affiliations.append(affiliation)
        return self

    def set_date(self, date: str | None = None) -> ReportBuilder:
        """设置日期"""
        if date is None:
            date = datetime.now().strftime("%B %d, %Y")
        self._date = date
        return self

    def set_abstract(self, abstract: str, keywords: list[str] | None = None) -> ReportBuilder:
        """
        设置摘要

        参数:
            abstract: 摘要内容
            keywords: 关键词列表
        """
        if keywords:
            abstract += f"\\par\\textbf{{Keywords:}} {', '.join(keywords)}"
        self._abstract = abstract
        return self

    def add_section(self, title: str, content: str, level: int = 1, label: str | None = None) -> ReportBuilder:
        """
        添加章节

        参数:
            title: 章节标题
            content: 章节内容 (LaTeX格式)
            level: 章节级别 (1=section, 2=subsection, 3=subsubsection)
            label: 标签 (用于引用)
        """
        self._sections.append(Section(title=title, content=content, level=level, label=label))
        return self

    def add_figure(
        self, figure_path: str, caption: str, label: str | None = None, width: str = "0.8\\textwidth"
    ) -> str:
        """
        添加图表

        参数:
            figure_path: 图片路径
            caption: 图表标题
            label: 标签
            width: 图片宽度

        返回:
            figure_id
        """
        self._counter_figure += 1
        figure_id = f"fig:{label or self._counter_figure}"

        self._figures.append(FigureReference(figure_id=figure_id, caption=caption, path=figure_path, width=width))

        return figure_id

    def add_table(self, table_content: str, caption: str, label: str | None = None, placement: str = "htbp") -> str:
        """
        添加表格

        参数:
            table_content: LaTeX表格代码
            caption: 表格标题
            label: 标签
            placement: 位置参数

        返回:
            table_id
        """
        self._counter_table += 1
        table_id = f"tab:{label or self._counter_table}"

        self._tables.append(TableReference(table_id=table_id, caption=caption, content=table_content))

        return table_id

    def add_statistical_result(
        self,
        test_name: str,
        statistic: float,
        p_value: float | None = None,
        df: int | None = None,
        effect_size: float | None = None,
    ) -> ReportBuilder:
        """
        添加统计结果

        参数:
            test_name: 检验名称
            statistic: 统计量
            p_value: p值
            df: 自由度
            effect_size: 效应量
        """
        self._statistical_results.append(
            StatisticalResult(test_name=test_name, statistic=statistic, p_value=p_value, df=df, effect_size=effect_size)
        )
        return self

    def add_reference(self, bibtex_entry: str) -> ReportBuilder:
        """
        添加参考文献条目

        参数:
            bibtex_entry: BibTeX格式文献条目
        """
        self._references.append(bibtex_entry)
        return self

    def generate(self, output_path: str) -> str:
        """
        生成LaTeX文档

        参数:
            output_path: 输出文件路径

        返回:
            生成的LaTeX代码
        """
        # Validate the output path before doing any work — reject
        # suspicious paths that look like path traversal or attempt to
        # write outside an expected directory. ``Path.resolve(strict=False)``
        # normalises ``..`` components so we can check against an allow
        # list of parent directories if one is configured.
        from pathlib import Path

        out_path = Path(output_path).resolve()
        # Reject paths whose parent doesn't exist or that look like
        # device names on Windows (``C:\NUL``, etc.). The previous
        # implementation opened whatever string the caller supplied
        # without validation, allowing e.g. ``../../etc/passwd`` to
        # overwrite arbitrary files.
        if not str(out_path).strip():
            raise ValueError("generate: output_path is empty")
        # Disallow null bytes which would truncate the path on POSIX
        # file systems.
        if "\x00" in str(out_path):
            raise ValueError("generate: output_path contains a null byte")

        lines = []

        # 文档类
        lines.append(self._preamble.generate_documentclass())
        lines.append("")

        # 导言区
        lines.append("\\usepackage[utf8]{inputenc}")
        lines.append("\\usepackage[T1]{fontenc}")
        lines.append("")

        for pkg_str in self._preamble.packages:
            lines.append(pkg_str)

        lines.append("")
        lines.append("\\begin{document}")
        lines.append("")

        # 标题页
        lines.extend(self._generate_titlepage())

        # 摘要
        if self._abstract:
            lines.extend(self._generate_abstract())

        # 目录
        lines.append("\\newpage")
        lines.append("\\tableofcontents")
        lines.append("")

        # 章节
        for section in self._sections:
            lines.extend(self._generate_section(section))

        # 统计结果
        if self._statistical_results:
            lines.extend(self._generate_statistical_results())

        # 图表
        for figure in self._figures:
            lines.extend(self._generate_figure(figure))

        for table in self._tables:
            lines.extend(self._generate_table(table))

        # 参考文献
        if self._references:
            lines.extend(self._generate_references())

        lines.append("\\end{document}")

        # 写入文件
        latex_code = "\n".join(lines)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_code)

        self._logger.info(f"Generated LaTeX report: {out_path}")

        return latex_code

    def _generate_titlepage(self) -> list[str]:
        """生成标题页"""
        lines = []

        lines.append("\\begin{titlepage}")
        lines.append("\\centering")
        lines.append("")

        if self._title:
            lines.append(f"\\Large\\textbf{{{self._title}}}")
            lines.append("\\vspace{2cm}")

        lines.append("\\large")
        for author in self._authors:
            lines.append(f"{{{author}}}")
            lines.append("\\vspace{0.5cm}")

        if self._date:
            lines.append("\\vspace{1cm}")
            lines.append(f"{{{self._date}}}")

        lines.append("\\end{titlepage}")
        lines.append("\\newpage")

        return lines

    def _generate_abstract(self) -> list[str]:
        """生成摘要"""
        lines = []

        lines.append("\\begin{abstract}")
        lines.append(self._abstract)
        lines.append("\\end{abstract}")
        lines.append("\\newpage")

        return lines

    def _generate_section(self, section: Section) -> list[str]:
        """生成章节"""
        lines = []

        # 章节命令
        section_cmd = {1: "\\section", 2: "\\subsection", 3: "\\subsubsection"}.get(section.level, "\\section")

        label_part = f"\\label{{{section.label}}}" if section.label else ""
        # Caller-supplied title must be escaped (figure_handler and
        # table_generator do the same for their captions); previously a
        # title such as ``Body_mass & 95%`` was interpolated raw and
        # broke (or injected) LaTeX.
        lines.append(f"{section_cmd}{{{_escape_latex(section.title)}}}{label_part}")
        lines.append("")

        # 内容
        lines.append(section.content)
        lines.append("")

        return lines

    def _generate_references(self) -> list[str]:
        """生成参考文献"""
        lines = []

        lines.append("\\newpage")
        lines.append("\\section{References}")
        lines.append("")

        for ref in self._references:
            lines.append(ref)
            lines.append("")

        return lines

    def _generate_statistical_results(self) -> list[str]:
        """Generate statistical result table."""
        lines = ["\\section{Statistical Results}", "", "\\begin{table}[htbp]", "\\centering"]
        lines.append("\\begin{tabular}{llll}")
        lines.append("\\toprule")
        lines.append("Test & Statistic & p-value & Effect size \\\\")
        lines.append("\\midrule")
        for result in self._statistical_results:
            statistic = self._format_statistic("stat", result.statistic, result.df)
            p_value = self._format_pvalue(result.p_value) if result.p_value is not None else "--"
            effect = f"{result.effect_size:.4f}" if result.effect_size is not None else "--"
            # Test names are caller-supplied free text (e.g. "Mann-Whitney U
            # (body_mass ~ group)") and may contain LaTeX-significant
            # characters such as ``_`` or ``&`` — escape them.
            lines.append(f"{_escape_latex(result.test_name)} & {statistic} & {p_value} & {effect} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Summary of statistical tests}")
        lines.append("\\end{table}")
        lines.append("")
        return lines

    def _generate_figure(self, figure: FigureReference) -> list[str]:
        """Generate a figure block."""
        return [
            "\\begin{figure}[htbp]",
            "\\centering",
            f"\\includegraphics[width={figure.width}]{{{figure.path}}}",
            # ``path`` stays verbatim (\\includegraphics needs the real
            # path); caption and label are caller-supplied text and must
            # be escaped, matching FigureHandler.include_figure.
            f"\\caption{{{_escape_latex(figure.caption)}}}",
            f"\\label{{{_escape_latex(figure.figure_id)}}}",
            "\\end{figure}",
            "",
        ]

    def _generate_table(self, table: TableReference) -> list[str]:
        """Generate a table block."""
        return [
            "\\begin{table}[htbp]",
            "\\centering",
            table.content,
            f"\\caption{{{_escape_latex(table.caption)}}}",
            f"\\label{{{_escape_latex(table.table_id)}}}",
            "\\end{table}",
            "",
        ]

    def _format_pvalue(self, p: float) -> str:
        """格式化p值"""
        if p < 0.001:
            return "$p < 0.001$"
        elif p < 0.01:
            return f"$p = {p:.3f}$"
        else:
            return f"$p = {p:.4f}$"

    def _format_statistic(self, name: str, value: float, df: int | None = None) -> str:
        """格式化统计量"""
        if df is not None:
            return f"{name} = {value:.4f}, df = {df}"
        return f"{name} = {value:.4f}"

class LatexCompiler:
    """
    LaTeX编译器封装

    调用pdflatex进行编译。
    """

    def __init__(self, tex_path: str, output_dir: str | None = None, n_passes: int = 2):
        """
        初始化编译器

        参数:
            tex_path: TeX文件路径
            output_dir: 输出目录
            n_passes: 编译次数
        """
        self._tex_path = tex_path
        self._output_dir = output_dir or os.path.dirname(tex_path)
        self._n_passes = n_passes
        self._logger = logging.getLogger(f"{__name__}.LatexCompiler")

    def compile(self, output_format: str = "pdf") -> tuple[bool, str]:
        """
        编译LaTeX文档

        参数:
            output_format: 输出格式 (pdf, dvi)

        返回:
            (是否成功, 输出路径)
        """
        import subprocess

        tex_basename = os.path.splitext(self._tex_path)[0]

        # 构建命令
        if output_format == "pdf":
            cmd = ["pdflatex", "-interaction=nonstopmode", self._tex_path]
        else:
            cmd = ["latex", "-interaction=nonstopmode", self._tex_path]

        self._logger.info(f"Compiling with: {' '.join(cmd)}")

        # 多次编译以解决交叉引用
        for pass_num in range(self._n_passes):
            try:
                result = subprocess.run(cmd, cwd=self._output_dir, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    self._logger.warning(f"Pass {pass_num + 1} had errors")
                    self._logger.debug(result.stderr)

            except subprocess.TimeoutExpired:
                self._logger.error("Compilation timeout")
                return False, ""
            except FileNotFoundError:
                self._logger.error("pdflatex not found")
                return False, ""

        # 确定输出文件
        # pdflatex is executed with cwd=self._output_dir, so the PDF/DVI
        # is written into the output directory. The previous code
        # resolved ``tex_basename + ".pdf"`` against the *process* CWD,
        # reporting failure even though compilation succeeded whenever
        # output_dir differed from the CWD. Join the output directory
        # instead (falling back to the tex file's directory or ".")
        # and only compare basenames, because pdflatex writes the
        # output next to the job under its own name.
        out_dir = self._output_dir or os.path.dirname(self._tex_path) or "."
        extension = "pdf" if output_format == "pdf" else "dvi"
        output_path = os.path.join(out_dir, os.path.basename(tex_basename) + "." + extension)

        success = os.path.exists(output_path)

        if success:
            self._logger.info(f"Compilation successful: {output_path}")
        else:
            self._logger.error("Compilation failed")

        return success, output_path
