"""
================================================================================
PaleoAST Parsers - NEXUS Writer Module
================================================================================

本模块实现NEXUS格式文件的写入器，支持导出TAX, CHARACTERS, TREES块。

NEXUS格式是系统发育学领域最常用的数据交换格式，
由Maddison, Swofford和Maddison (1997)设计。

NEXUS格式规范 (Maddison et al. 1997):
==============================================================================

文件结构:
    #NEXUS

    BEGIN TAXA;
        TITLE <title>;
        NTAX <n>;
        TAXLABELS <taxon1> <taxon2> ... ;
    END;

    BEGIN CHARACTERS;
        TITLE <title>;
        NCHAR <n>;
        FORMAT DATATYPE=<type> INTERLEAVE=<yes|no> GAP=<gap_char> MISSING=<missing_char>;
        CHARLABELS <label1> <label2> ... ;
        MATRIX
            <taxon1> <sequence1>
            <taxon2> <sequence2>
            ...
        ;
    END;

    BEGIN TREES;
        TITLE <title>;
        FORMAT NEWICK;
        TREES
            TREE <name> = <newick_tree>
        ;
    END;

参考文献:
    Maddison, D. R., Swofford, D. L., & Maddison, W. P. (1997).
    NEXUS: An extensible file format for systematic data.
    Systematic Biology, 46(4), 590-621.

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NEXUSWriter:
    """
    NEXUS格式文件写入器

    支持写入TAX, CHARACTERS, TREES块，支持交错和非交错格式。

    Attributes:
        taxa: 分类单元列表
        data: 数据矩阵 (taxa x characters)
        trees: 树列表
        interleaved: 是否使用交错格式
        gap_char: Gap字符 (默认 '-')
        missing_char: Missing字符 (默认 '?')

    Example:
        >>> writer = NEXUSWriter()
        >>> writer.set_taxa(['TaxonA', 'TaxonB', 'TaxonC'])
        >>> writer.set_data([[0, 1, 0], [1, 0, 1], [0, 0, 0]],
        ...                char_labels=['Char1', 'Char2', 'Char3'])
        >>> nexus_str = writer.write()
    """

    def __init__(
        self,
        interleaved: bool = False,
        gap_char: str = "-",
        missing_char: str = "?",
    ) -> None:
        """
        初始化NEXUS写入器。

        Parameters:
            interleaved: 是否使用交错格式 (INTERLEAVE=YES)
            gap_char: Gap字符，用于表示缺失核苷酸/氨基酸位置
            missing_char: Missing字符，用于表示完全未知的数据
        """
        self._logger = logging.getLogger(f"{__name__}.NEXUSWriter")
        self._taxa: list[str] = []
        self._data: list[list[int | str]] = []
        self._char_labels: list[str] = []
        self._char_statlabels: dict[int, str] = {}
        self._taxa_metadata: dict[str, dict[str, Any]] = {}
        self._trees: list[tuple[str, str]] = []  # (tree_name, newick_string)
        self._interleaved = interleaved
        self._gap_char = gap_char
        self._missing_char = missing_char
        self._title: str | None = None

    def set_title(self, title: str) -> None:
        """设置文件标题"""
        self._title = title

    def set_taxa(self, taxa: list[str], metadata: dict[str, dict[str, Any]] | None = None) -> None:
        """
        设置分类单元列表。

        Parameters:
            taxa: 分类单元名称列表
            metadata: 可选的分类单元元数据字典
        """
        self._taxa = list(taxa)
        self._taxa_metadata = metadata.copy() if metadata else {}
        self._logger.debug(f"Set {len(taxa)} taxa")

    def set_data(
        self,
        data: list[list[int | str]],
        char_labels: list[str] | None = None,
        char_statlabels: dict[int, str] | None = None,
    ) -> None:
        """
        设置字符数据矩阵。

        Parameters:
            data: 数据矩阵 (n_taxa x n_chars)
            char_labels: 字符标签列表
            char_statlabels: 字符状态标签字典 {char_index: "label states"}
        """
        if len(data) != len(self._taxa):
            raise ValueError(
                f"Data row count ({len(data)}) must match taxa count ({len(self._taxa)})"
            )
        self._data = [list(row) for row in data]
        if char_labels is not None:
            self._char_labels = list(char_labels)
        if char_statlabels is not None:
            self._char_statlabels = dict(char_statlabels)
        self._logger.debug(f"Set data matrix: {len(data)} taxa x {len(data[0]) if data else 0} chars")

    def add_tree(self, name: str, newick: str) -> None:
        """
        添加一棵树。

        Parameters:
            name: 树名称
            newick: Newick格式树字符串
        """
        self._trees.append((name, newick))
        self._logger.debug(f"Added tree '{name}'")

    def write(self) -> str:
        """
        生成完整的NEXUS文件内容。

        Returns:
            NEXUS格式字符串
        """
        lines: list[str] = []

        # 文件头
        lines.append("#NEXUS")

        if self._title:
            lines.append(f"[Title: {self._title}]")

        # TAXA块
        taxa_block = self._write_taxa_block()
        if taxa_block:
            lines.append("")
            lines.append(taxa_block)

        # CHARACTERS块
        characters_block = self._write_characters_block()
        if characters_block:
            lines.append("")
            lines.append(characters_block)

        # TREES块
        trees_block = self._write_trees_block()
        if trees_block:
            lines.append("")
            lines.append(trees_block)

        lines.append("")
        self._logger.info(f"Generated NEXUS output: {len(lines)} lines")
        return "\n".join(lines)

    def _write_taxa_block(self) -> str:
        """生成TAX块内容"""
        if not self._taxa:
            return ""

        lines: list[str] = []
        lines.append("BEGIN TAXA;")
        lines.append(f"    NTAX={len(self._taxa)};")

        # TAXLABELS
        lines.append("    TAXLABELS")
        for taxon in self._taxa:
            # 处理带空格的分类单元名称
            if " " in taxon or "-" in taxon or "'" in taxon:
                lines.append(f"        '{taxon}'")
            else:
                lines.append(f"        {taxon}")
        lines.append("    ;")
        lines.append("END;")
        return "\n".join(lines)

    def _write_characters_block(self) -> str:
        """生成CHARACTERS块内容"""
        if not self._data:
            return ""

        n_taxa = len(self._taxa)
        n_chars = len(self._data[0]) if self._data else 0

        lines: list[str] = []
        lines.append("BEGIN CHARACTERS;")
        lines.append(f"    NCHAR={n_chars};")

        # FORMAT
        format_parts = [
            "DATATYPE=STANDARD",
            f"INTERLEAVE={'YES' if self._interleaved else 'NO'}",
            f"GAP={self._gap_char}",
            f"MISSING={self._missing_char}",
        ]
        lines.append(f"    FORMAT {' '.join(format_parts)};")

        # CHARSTATELABELS (如果提供)
        if self._char_statlabels:
            lines.append("    CHARSTATELABELS")
            for idx, label in sorted(self._char_statlabels.items()):
                lines.append(f"        {idx + 1} {label},")
            lines[-1] = lines[-1].rstrip(",") + ";"  # 最后一个逗号改分号

        # MATRIX
        lines.append("    MATRIX")

        if self._interleaved:
            lines.extend(self._write_interleaved_matrix())
        else:
            lines.extend(self._write_non_interleaved_matrix())

        lines.append("    ;")
        lines.append("END;")
        return "\n".join(lines)

    def _write_non_interleaved_matrix(self) -> list[str]:
        """生成非交错格式的MATRIX块"""
        lines: list[str] = []
        for i, taxon in enumerate(self._taxa):
            sequence = "".join(str(c) for c in self._data[i])
            # 处理带空格的分类单元名称
            if " " in taxon or "-" in taxon or "'" in taxon:
                taxon_str = f"'{taxon}'"
            else:
                taxon_str = taxon
            lines.append(f"        {taxon_str} {sequence}")
        return lines

    def _write_interleaved_matrix(self) -> list[str]:
        """生成交错格式的MATRIX块"""
        lines: list[str] = []
        n_taxa = len(self._taxa)
        n_chars = len(self._data[0]) if self._data else 0
        chars_per_line = max(50, n_chars // 3)  # 约3块

        for start in range(0, n_chars, chars_per_line):
            end = min(start + chars_per_line, n_chars)

            # 每个taxon一行片段
            for i, taxon in enumerate(self._taxa):
                fragment = "".join(str(c) for c in self._data[i][start:end])
                if " " in taxon or "-" in taxon or "'" in taxon:
                    taxon_str = f"'{taxon}'"
                else:
                    taxon_str = taxon

                if start == 0:
                    lines.append(f"        {taxon_str} {fragment}")
                else:
                    lines.append(f"        {taxon_str} {fragment}")

            if end < n_chars:
                lines.append("")  # 空行分隔

        return lines

    def _write_trees_block(self) -> str:
        """生成TREES块内容"""
        if not self._trees:
            return ""

        lines: list[str] = []
        lines.append("BEGIN TREES;")
        lines.append("    FORMAT NEWICK;")

        for tree_name, newick in self._trees:
            # 处理树名称中的空格
            if " " in tree_name or "'" in tree_name:
                lines.append(f"    TREE '{tree_name}' = {newick};")
            else:
                lines.append(f"    TREE {tree_name} = {newick};")

        lines.append("END;")
        return "\n".join(lines)

    def write_to_file(self, filepath: str) -> None:
        """
        写入NEXUS文件。

        Parameters:
            filepath: 输出文件路径
        """
        content = self.write()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        self._logger.info(f"Wrote NEXUS file to {filepath}")


def write_nexus(
    taxa: list[str],
    data: list[list[int | str]],
    trees: list[tuple[str, str]] | None = None,
    char_labels: list[str] | None = None,
    interleaved: bool = False,
    gap_char: str = "-",
    missing_char: str = "?",
) -> str:
    """
    便捷函数：生成NEXUS格式字符串。

    Parameters:
        taxa: 分类单元名称列表
        data: 数据矩阵 (n_taxa x n_chars)
        trees: 可选的 (tree_name, newick) 元组列表
        char_labels: 可选的字符标签
        interleaved: 是否交错格式
        gap_char: Gap字符
        missing_char: Missing字符

    Returns:
        NEXUS格式字符串

    Example:
        >>> taxa = ['A', 'B', 'C']
        >>> data = [[0, 1, 0], [1, 0, 1], [0, 1, 1]]
        >>> nexus_str = write_nexus(taxa, data)
    """
    writer = NEXUSWriter(interleaved=interleaved, gap_char=gap_char, missing_char=missing_char)
    writer.set_taxa(taxa)
    writer.set_data(data, char_labels=char_labels)
    if trees:
        for name, newick in trees:
            writer.add_tree(name, newick)
    return writer.write()
