"""
Tests for NEXUS Writer functionality.

这些测试验证NEXUS Writer的导出功能：
1. 基本TAX, CHARACTERS, TREES块写入
2. 交错和非交错格式
3. Gap和Missing字符编码
4. 往返解析（如果可能）
"""

from __future__ import annotations

import pytest

from parsers.nexus_writer import NEXUSWriter, write_nexus


class TestNEXUSWriterBasic:
    """测试NEXUS Writer基本功能"""

    def test_write_taxa_block(self):
        """测试写入TAX块"""
        writer = NEXUSWriter()
        writer.set_taxa(["Homo_sapiens", "Pan_troglodytes", "Gorilla_gorilla"])

        result = writer._write_taxa_block()
        assert "BEGIN TAXA" in result
        assert "NTAX=3" in result
        assert "Homo_sapiens" in result
        assert "Pan_troglodytes" in result
        assert "Gorilla_gorilla" in result
        assert "END" in result

    def test_write_taxa_block_with_special_names(self):
        """测试写入带空格的分类单元名称"""
        writer = NEXUSWriter()
        writer.set_taxa(["Homo sapiens", "Pan troglodytes"])

        result = writer._write_taxa_block()
        assert "'Homo sapiens'" in result
        assert "'Pan troglodytes'" in result

    def test_write_characters_block_basic(self):
        """测试写入CHARACTERS块"""
        writer = NEXUSWriter()
        writer.set_taxa(["A", "B"])
        writer.set_data(
            [[0, 1, 0], [1, 0, 1]],
            char_labels=["Char1", "Char2", "Char3"],
        )

        result = writer._write_characters_block()
        assert "BEGIN CHARACTERS" in result
        assert "NCHAR=3" in result
        assert "DATATYPE=STANDARD" in result
        assert "INTERLEAVE=NO" in result
        assert "GAP=-" in result
        assert "MISSING=?" in result

    def test_write_non_interleaved_matrix(self):
        """测试写入非交错格式矩阵"""
        writer = NEXUSWriter(interleaved=False)
        writer.set_taxa(["A", "B"])
        writer.set_data([[0, 1], [1, 0]])

        result = writer._write_characters_block()
        # 验证矩阵格式
        assert "A 01" in result or "A 01" in result
        assert "B 10" in result or "B 10" in result

    def test_write_interleaved_matrix(self):
        """测试写入交错格式矩阵"""
        writer = NEXUSWriter(interleaved=True)
        writer.set_taxa(["A", "B"])
        writer.set_data([[0, 1, 0, 1], [1, 0, 1, 0]])

        result = writer._write_characters_block()
        assert "INTERLEAVE=YES" in result

    def test_write_trees_block(self):
        """测试写入TREES块"""
        writer = NEXUSWriter()
        writer.add_tree("tree1", "(A:0.1,B:0.2)C:0.3;")
        writer.add_tree("tree2", "(A:0.15,B:0.25)D:0.4;")

        result = writer._write_trees_block()
        assert "BEGIN TREES" in result
        assert "FORMAT NEWICK" in result
        assert "TREE tree1" in result
        assert "TREE tree2" in result
        assert "END" in result

    def test_full_nexus_output(self):
        """测试完整的NEXUS输出"""
        writer = NEXUSWriter()
        writer.set_title("Test Data")
        writer.set_taxa(["A", "B", "C"])
        writer.set_data(
            [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
            char_labels=["C1", "C2", "C3"],
        )
        writer.add_tree("my_tree", "(A:0.1,B:0.2,C:0.3)D:0.01;")

        result = writer.write()

        assert "#NEXUS" in result
        assert "BEGIN TAXA" in result
        assert "BEGIN CHARACTERS" in result
        assert "BEGIN TREES" in result
        assert "Title: Test Data" in result


class TestNEXUSWriterCharacters:
    """测试字符数据处理"""

    def test_gap_and_missing_chars(self):
        """测试Gap和Missing字符"""
        writer = NEXUSWriter(gap_char="N", missing_char="X")
        writer.set_taxa(["A", "B"])
        writer.set_data([[0, "-", 1], ["?", 1, 0]])

        result = writer._write_characters_block()
        assert "GAP=N" in result
        assert "MISSING=X" in result

    def test_char_statlabels(self):
        """测试CHARSTATELABELS"""
        writer = NEXUSWriter()
        writer.set_taxa(["A", "B"])
        writer.set_data([[0, 1], [1, 0]])
        writer.set_data(
            [[0, 1], [1, 0]],
            char_statlabels={0: "Character_1 0-1", 1: "Character_2 0-1"},
        )

        result = writer._write_characters_block()
        assert "CHARSTATELABELS" in result


class TestNEXUSWriterEdgeCases:
    """测试边缘情况"""

    def test_empty_taxa(self):
        """测试空taxa"""
        writer = NEXUSWriter()
        result = writer._write_taxa_block()
        assert result == ""

    def test_empty_data(self):
        """测试空data"""
        writer = NEXUSWriter()
        writer.set_taxa(["A", "B"])
        result = writer._write_characters_block()
        assert result == ""

    def test_empty_trees(self):
        """测试空trees"""
        writer = NEXUSWriter()
        result = writer._write_trees_block()
        assert result == ""

    def test_data_row_count_mismatch(self):
        """测试数据行数与taxa不匹配"""
        writer = NEXUSWriter()
        writer.set_taxa(["A", "B", "C"])  # 3 taxa
        with pytest.raises(ValueError):
            writer.set_data([[0, 1], [1, 0]])  # 只有2行

    def test_write_with_taxa_metadata(self):
        """测试写入带元数据的taxa"""
        writer = NEXUSWriter()
        metadata = {
            "Homo_sapiens": {"common_name": "Human"},
            "Pan_troglodytes": {"common_name": "Chimp"},
        }
        writer.set_taxa(["Homo_sapiens", "Pan_troglodytes"], metadata=metadata)

        result = writer._write_taxa_block()
        assert "BEGIN TAXA" in result


class TestWriteNexusFunction:
    """测试便捷函数write_nexus"""

    def test_write_nexus_basic(self):
        """测试write_nexus基本功能"""
        taxa = ["A", "B", "C"]
        data = [[0, 1, 0], [1, 0, 1], [0, 1, 1]]

        result = write_nexus(taxa, data)

        assert "#NEXUS" in result
        assert "BEGIN TAXA" in result
        assert "BEGIN CHARACTERS" in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_write_nexus_with_trees(self):
        """测试write_nexus带trees"""
        taxa = ["A", "B"]
        data = [[0, 1], [1, 0]]
        trees = [("my_tree", "(A:0.1,B:0.2)AB:0.05;")]

        result = write_nexus(taxa, data, trees=trees)

        assert "BEGIN TREES" in result
        assert "TREE my_tree" in result

    def test_write_nexus_interleaved(self):
        """测试write_nexus交错格式"""
        taxa = ["A", "B"]
        data = [[0, 1, 0, 1], [1, 0, 1, 0]]

        result = write_nexus(taxa, data, interleaved=True)

        assert "INTERLEAVE=YES" in result

    def test_write_nexus_custom_gap_missing(self):
        """测试write_nexus自定义gap/missing"""
        taxa = ["A", "B"]
        data = [[0, "-"], ["?", 1]]

        result = write_nexus(taxa, data, gap_char="N", missing_char="X")

        assert "GAP=N" in result
        assert "MISSING=X" in result
