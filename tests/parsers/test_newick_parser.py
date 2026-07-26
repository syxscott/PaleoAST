"""
================================================================================
Tests for Newick Parser - Polytomy Bug Fixes
================================================================================

本测试文件验证 Newick parser 的 polytomy (多叉树) 处理是否正确。

Bug 修复历史:
    - 2026-07-27: 修复 _get_splits 只考虑前2个子节点的问题
      polytomy 树应保留所有子节点, 而非只取前2个

测试覆盖:
    1. 二叉树 (正常情况)
    2. 三叉树 polytomy
    3. 多叉树 polytomy (4+ 子节点)
    4. Robinson-Foulds 距离计算
    5. 深层嵌套 polytomy

作者: PaleoAST Development Team
"""

from __future__ import annotations

import pytest

from parsers.newick_parser import (
    NewickParser,
    NewickTree,
    TreeNode,
    TreeComparator,
    parse_newick,
)


class TestNewickParserBasic:
    """基础解析测试"""

    def setup_method(self):
        self.parser = NewickParser()

    def test_simple_tree(self):
        """测试简单树 (A,B)C;"""
        tree = self.parser.parse("(A,B)C;")
        assert tree.leaf_names == ["A", "B"]
        assert tree.root is not None
        assert tree.root.name == "C"

    def test_unnamed_tree(self):
        """测试无名树 (A,B);"""
        tree = self.parser.parse("(A,B);")
        assert tree.leaf_names == ["A", "B"]
        assert tree.root.name == ""

    def test_with_branch_lengths(self):
        """测试带枝长的树"""
        tree = self.parser.parse("(A:0.1,B:0.2)C:0.3;")
        assert tree.root.name == "C"
        assert tree.root.branch_length == 0.3
        assert tree.root.children[0].name == "A"
        assert tree.root.children[0].branch_length == 0.1


class TestNewickParserPolytomy:
    """Polytomy (多叉树) 测试 - 核心测试"""

    def setup_method(self):
        self.parser = NewickParser()

    def test_triple_polytomy(self):
        """测试三叉树 (A,B,C)D;"""
        tree = self.parser.parse("(A,B,C)D;")

        assert len(tree.root.children) == 3
        assert tree.leaf_names == ["A", "B", "C"]

    def test_quad_polytomy(self):
        """测试四叉树 (A,B,C,D)E;"""
        tree = self.parser.parse("(A,B,C,D)E;")

        assert len(tree.root.children) == 4
        assert tree.leaf_names == ["A", "B", "C", "D"]

    def test_quintuple_polytomy(self):
        """测试五叉树 (A,B,C,D,E)Root;"""
        tree = self.parser.parse("(A,B,C,D,E)Root;")

        assert len(tree.root.children) == 5
        assert set(tree.leaf_names) == {"A", "B", "C", "D", "E"}

    def test_nested_polytomy(self):
        """测试嵌套 polytomy ((A,B,C),(D,E,F))G;"""
        tree = self.parser.parse("((A,B,C),(D,E,F))G;")

        # 根节点有两个子节点
        assert len(tree.root.children) == 2
        # 第一个子节点是三叉 polytomy
        assert len(tree.root.children[0].children) == 3
        # 第二个子节点也是三叉 polytomy
        assert len(tree.root.children[1].children) == 3

    def test_mixed_polytomy(self):
        """测试混合 polytomy (A,(B,C,D),E)F;"""
        tree = self.parser.parse("(A,(B,C,D),E)F;")

        assert len(tree.root.children) == 3
        assert tree.root.children[0].name == "A"
        assert tree.root.children[1].name == ""
        assert len(tree.root.children[1].children) == 3  # polytomy
        assert tree.root.children[2].name == "E"


class TestTreeComparatorRF:
    """Robinson-Foulds 距离测试"""

    def test_rf_binary_identical(self):
        """测试相同二叉树的 RF 距离为 0"""
        tree1 = parse_newick("(A,B)C;")
        tree2 = parse_newick("(A,B)C;")

        distance = TreeComparator.rf_distance(tree1, tree2)
        assert distance == 0

    def test_rf_binary_different(self):
        """测试不同二叉树的 RF 距离"""
        tree1 = parse_newick("(A,B)C;")
        tree2 = parse_newick("(A,C)B;")

        # 这两棵树的分割不同
        distance = TreeComparator.rf_distance(tree1, tree2)
        assert distance > 0

    def test_rf_polytomy_identical(self):
        """测试相同 polytomy 树的 RF 距离为 0"""
        tree1 = parse_newick("(A,B,C)D;")
        tree2 = parse_newick("(A,B,C)D;")

        distance = TreeComparator.rf_distance(tree1, tree2)
        assert distance == 0

    def test_rf_polytomy_different(self):
        """测试不同 polytomy 树的 RF 距离"""
        tree1 = parse_newick("(A,B,C)D;")
        tree2 = parse_newick("(A,B,D)C;")

        distance = TreeComparator.rf_distance(tree1, tree2)
        assert distance > 0

    def test_rf_polytomy_preserves_all_splits(self):
        """测试 polytomy 的所有分割都被保留"""
        # 三叉树应该产生 3 个分割
        tree = parse_newick("(A,B,C)D;")

        splits = TreeComparator._get_splits(tree.root)

        # 对于 (A,B,C), 应该生成:
        # - {A} vs {B,C}
        # - {B} vs {A,C}
        # - {C} vs {A,B}
        assert len(splits) >= 3  # 至少 3 个分割


class TestTreeNodeOperations:
    """TreeNode 操作测试"""

    def test_polytomy_leaf_count(self):
        """测试 polytomy 的叶节点计数"""
        tree = parse_newick("(A,B,C,D)E;")

        assert tree.root.leaf_count == 4
        assert tree.leaf_count == 4

    def test_polytomy_get_leaves(self):
        """测试 polytomy 的 get_leaves"""
        tree = parse_newick("(A,B,C)Root;")

        leaves = tree.root.get_leaves()
        assert len(leaves) == 3
        assert set(l.name for l in leaves) == {"A", "B", "C"}

    def test_polytomy_to_newick(self):
        """测试 polytomy 转换为 Newick 格式"""
        tree = parse_newick("(A,B,C)D;")

        newick = tree.to_newick()
        # 验证可以重新解析
        tree2 = parse_newick(newick)
        assert tree2.leaf_names == tree.leaf_names


class TestNewickParserEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.parser = NewickParser()

    def test_large_polytomy(self):
        """测试大型 polytomy (10 个子节点)"""
        taxa = ",".join(f"taxon{i}" for i in range(10))
        newick = f"({taxa})Root;"
        tree = self.parser.parse(newick)

        assert len(tree.root.children) == 10
        assert tree.leaf_count == 10

    def test_deeply_nested_polytomy(self):
        """测试深层嵌套 polytomy"""
        # ((A,B,C),(D,E,F),(G,H,I))J;
        tree = self.parser.parse("((A,B,C),(D,E,F),(G,H,I))J;")

        assert len(tree.root.children) == 3
        for child in tree.root.children:
            assert len(child.children) == 3

    def test_single_child(self):
        """测试单子节点 (退化的树)"""
        tree = self.parser.parse("(A);")
        assert len(tree.root.children) == 1
        assert tree.root.children[0].name == "A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
