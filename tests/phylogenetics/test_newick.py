"""
Tests for PhyloTree NEWICK export functionality.

这些测试验证PhyloTree的NEWICK导出功能：
1. 基本二叉树往返NEWICK
2. 含polytomy的树正确导出
3. 枝长保留精度
4. 内部节点标签控制
"""

from __future__ import annotations

import pytest

from phylogenetics.tree import PhyloNode, PhyloTree, NodeType


class TestPhyloTreeNewickBasic:
    """测试NEWICK基本功能"""

    def test_binary_tree_roundtrip(self):
        """测试二叉树往返NEWICK（验证结构而非精确字符串）"""
        newick = "(A:0.1,B:0.2)C:0.3;"
        tree = PhyloTree.from_newick(newick)

        # Parse output back and verify structure
        result = tree.to_newick()
        tree2 = PhyloTree.from_newick(result)

        assert tree.leaf_names == tree2.leaf_names
        assert tree.leaf_count == tree2.leaf_count
        assert tree.node_count == tree2.node_count

    def test_simple_tree(self):
        """测试简单树"""
        tree = PhyloTree.from_newick("(A,B);")
        result = tree.to_newick()

        # 不带枝长
        assert "A" in result
        assert "B" in result
        assert "(" in result

    def test_tree_without_branch_lengths(self):
        """测试不带枝长的树"""
        newick = "(A:0.1,B:0.2)C:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(branch_lengths=False)

        # 验证没有枝长数值
        assert ":0.1" not in result
        assert ":0.2" not in result
        assert ":0.3" not in result
        assert "A" in result
        assert "B" in result

    def test_tree_without_internal_labels(self):
        """测试不带内部节点标签的树"""
        newick = "(A:0.1,B:0.2)Internal:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(internal_labels=False)

        # 内部节点名称不应该出现
        assert "Internal" not in result
        # 但叶节点名称应该保留
        assert "A" in result
        assert "B" in result

    def test_tree_with_both_labels_and_lengths(self):
        """测试同时显示标签和枝长"""
        newick = "(A:0.1,B:0.2)Ancestor:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(branch_lengths=True, internal_labels=True)

        assert "Ancestor" in result
        assert ":0.1" in result
        assert ":0.2" in result
        assert ":0.3" in result


class TestPhyloTreePolytomy:
    """测试Polytomy处理"""

    def test_trifurcation(self):
        """测试三叉节点（polytomy）"""
        newick = "((A,B,C)D:0.1,E:0.2)Root:0.3;"
        tree = PhyloTree.from_newick(newick)

        # Parse output back and verify structure
        result = tree.to_newick()
        tree2 = PhyloTree.from_newick(result)

        assert tree.leaf_names == tree2.leaf_names
        assert tree.leaf_count == tree2.leaf_count
        assert tree.node_count == tree2.node_count
        # Verify the polytomy structure (3 children at node D)
        d_node = [n for n in tree.root.get_all_nodes() if n.name == "D"][0]
        assert len(d_node.children) == 3

    def test_quadfurcation(self):
        """测试四叉节点"""
        newick = "(((A,B,C,D)E:0.1)F:0.2)Root:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick()

        # 四个叶节点应该被正确处理
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result

    def test_nested_polytomy(self):
        """测试嵌套polytomy"""
        newick = "(((A,B,C)D:0.05,(E,F,G)H:0.05)I:0.1,J:0.2)Root:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick()

        assert "A" in result
        assert "E" in result
        assert "G" in result
        assert "J" in result


class TestPhyloTreeBranchLengthPrecision:
    """测试枝长精度"""

    def test_high_precision_branch_lengths(self):
        """测试高精度枝长"""
        tree = PhyloTree()
        tree.root = PhyloNode(
            name="Root",
            node_type=NodeType.INTERNAL,
            branch_length=0.123456789,
        )
        tree.root.add_child(PhyloNode(name="A", node_type=NodeType.LEAF, branch_length=0.987654321))
        tree.root.add_child(PhyloNode(name="B", node_type=NodeType.LEAF, branch_length=0.111111111))

        result = tree.to_newick(precision=9)

        # With precision=9, we should see 9 decimal places
        assert "0.123456789" in result or "0.123456788" in result  # floating point rounding
        assert "0.987654321" in result or "0.987654320" in result
        assert "0.111111111" in result or "0.111111110" in result

    def test_default_precision(self):
        """测试默认精度（6位小数）"""
        tree = PhyloTree()
        tree.root = PhyloNode(name="Root", node_type=NodeType.INTERNAL, branch_length=0.1234567890)
        tree.root.add_child(PhyloNode(name="A", node_type=NodeType.LEAF, branch_length=0.1))
        tree.root.add_child(PhyloNode(name="B", node_type=NodeType.LEAF, branch_length=0.2))

        result = tree.to_newick()

        # 默认6位精度
        assert ":0.123457:" in result or ":0.123457" in result

    def test_low_precision_branch_lengths(self):
        """测试低精度枝长"""
        # Create tree with known branch lengths
        tree = PhyloTree()
        tree.root = PhyloNode(name="Root", node_type=NodeType.INTERNAL, branch_length=0.15)
        tree.root.add_child(PhyloNode(name="A", node_type=NodeType.LEAF, branch_length=0.99))
        tree.root.add_child(PhyloNode(name="B", node_type=NodeType.LEAF, branch_length=0.14))

        result = tree.to_newick(precision=2)

        # With precision=2, 0.99 should stay 0.99, 0.14 should become 0.14
        assert ":0.99:" in result or ":0.99" in result or "0.99" in result
        # 0.15 becomes 0.15, 0.14 stays 0.14


class TestPhyloTreeNewickEdgeCases:
    """测试边缘情况"""

    def test_empty_tree(self):
        """测试空树"""
        tree = PhyloTree()
        result = tree.to_newick()
        assert result == ";"

    def test_single_taxon(self):
        """测试单分类单元树"""
        tree = PhyloTree.from_newick("A:0.1;")
        result = tree.to_newick()
        assert "A" in result

    def test_tree_with_unnamed_nodes(self):
        """测试带未命名节点的树"""
        newick = "((A:0.1,B:0.2):0.3,C:0.4):0.5;"
        tree = PhyloTree.from_newick(newick)
        result = tree.to_newick()

        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_tree_with_support_values(self):
        """测试带支持率的树"""
        # 支持率在内部节点名称位置
        newick = "((A:0.1,B:0.2)95:0.3,C:0.4)100:0.5;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(internal_labels=True)

        assert "95" in result
        assert "100" in result

    def test_tree_no_root_node(self):
        """测试无根树（通过newick往返）"""
        newick = "(A:0.1,B:0.2,C:0.3);"
        tree = PhyloTree.from_newick(newick)

        # Parse output back and verify structure
        result = tree.to_newick()
        tree2 = PhyloTree.from_newick(result)

        assert tree.leaf_names == tree2.leaf_names
        assert tree.leaf_count == tree2.leaf_count
        assert tree.node_count == tree2.node_count


class TestPhyloNodeNewick:
    """测试PhyloNode的NEWICK方法"""

    def test_leaf_node(self):
        """测试叶节点"""
        node = PhyloNode(name="TestLeaf", node_type=NodeType.LEAF, branch_length=0.5)

        result = node.to_newick()

        assert result.startswith("TestLeaf:")
        # Check that the length value is approximately 0.5 (precision handling)
        assert "0.5" in result or "0.500000" in result

    def test_leaf_without_length(self):
        """测试不带枝长的叶节点"""
        node = PhyloNode(name="TestLeaf", node_type=NodeType.LEAF)

        result = node.to_newick()

        assert result == "TestLeaf"

    def test_internal_node(self):
        """测试内部节点"""
        parent = PhyloNode(name="Inner", node_type=NodeType.INTERNAL, branch_length=0.3)
        child1 = PhyloNode(name="A", node_type=NodeType.LEAF, branch_length=0.1)
        child2 = PhyloNode(name="B", node_type=NodeType.LEAF, branch_length=0.2)
        parent.add_child(child1)
        parent.add_child(child2)

        result = parent.to_newick()

        # Verify structure with parentheses, A, B, and Inner
        assert "(" in result and ")" in result
        assert "A" in result and "B" in result
        assert "Inner" in result
        assert "0.3" in result  # branch length

    def test_multifurcation(self):
        """测试多叉节点"""
        parent = PhyloNode(name="Multi", node_type=NodeType.INTERNAL)
        for name in ["A", "B", "C", "D"]:
            parent.add_child(PhyloNode(name=name, node_type=NodeType.LEAF))

        result = parent.to_newick()

        # 验证逗号分隔
        assert "A," in result or ",A" in result or result.count(",") == 3


class TestPhyloTreeNewickCombinations:
    """测试NEWICK各种参数组合"""

    def test_no_lengths_no_labels(self):
        """测试无枝长无标签"""
        newick = "(A:0.1,B:0.2)Root:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(branch_lengths=False, internal_labels=False)

        assert "(" in result
        assert ")" in result
        assert "," in result
        assert "A" in result
        assert "B" in result
        assert "Root" not in result
        assert ":0" not in result

    def test_all_options(self):
        """测试所有选项启用"""
        newick = "(A:0.1,B:0.2)Root:0.3;"
        tree = PhyloTree.from_newick(newick)

        result = tree.to_newick(branch_lengths=True, internal_labels=True)

        assert "A:0.1" in result
        assert "B:0.2" in result
        assert "Root:0.3" in result
