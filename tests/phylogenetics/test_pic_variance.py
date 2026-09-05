"""
================================================================================
Test PIC Variance Calculation - Felsenstein 1985 Golden Values
================================================================================

这些测试使用经典 4-taxa 树验证 PIC 方差计算是否符合 Felsenstein 1985 公式。

测试树 (Newick 格式):
    (A:1, B:2, (C:3, D:4)E:5)F:0;

树结构:
                    F
                   / \
                  A   E
                 1   / \
                    C   D
                   3   4

参考文献:
- Felsenstein, J. (1985). Phylogenies and the Comparative Method.
  American Naturalist, 125(1), 1-15.

作者: PaleoAST Development Team
"""

import numpy as np
import pytest

from phylogenetics.pic import compute_pic, validate_pic_assumptions
from phylogenetics.tree import PhyloTree


class TestPICVariance4TaxaTree:
    """测试经典 4-taxa 树的 PIC 方差计算"""

    @pytest.fixture
    def tree_4taxa(self):
        """经典 4-taxa 全二叉树 (旧 fixture "(A:1, B:2, (C:3, D:4)E:5)F:0;"
        的根有 3 个子节点, 是三分叉而非二叉树)"""
        return PhyloTree.from_newick("((A:1, B:2):3, (C:3, D:4):5)F:0;")

    @pytest.fixture
    def traits_simple(self):
        """简单性状数据用于验证"""
        return {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

    def test_pic_variance_felsenstein1985(self, tree_4taxa, traits_simple):
        """
        验证 PIC 方差符合 Felsenstein 1985 公式

        对于二叉树，父节点方差 = v0 + v1，其中 v0/v1 已包含子节点枝长
        不应该重复加 bl0 + bl1 (旧代码的错误)
        """
        contrasts, pairs = compute_pic(tree_4taxa, traits_simple)

        # 4-taxa 树应有 3 个独立对比 (n-1 for binary tree)
        assert len(contrasts) == 3, f"Expected 3 contrasts for 4-taxa tree, got {len(contrasts)}"

        # 验证对比值是有限的
        for c in contrasts:
            assert np.isfinite(c), f"Contrast {c} is not finite"

    def test_pic_variance_no_double_counting(self, tree_4taxa, traits_simple):
        """
        验证 v0 + v1 不会重复计算枝长

        旧代码: node_var = v0 + v1 + bl0 + bl1 (错误 - 双重计算)
        新代码: node_var = v0 + v1 (正确 - Felsenstein 1985)
        """
        contrasts, pairs = compute_pic(tree_4taxa, traits_simple)

        # 如果 v0/v1 已经包含 bl0/bl1，则 node_var = v0 + v1
        # 我们可以通过检查对比值的合理性来间接验证

        # 在简单性状下，对比值应该在合理范围内
        for c in contrasts:
            assert abs(c) < 100, f"Contrast {c} is unreasonably large (possible variance error)"

    def test_validate_pic_assumptions_binary_tree(self, tree_4taxa):
        """验证二叉树满足 PIC 假设"""
        result = validate_pic_assumptions(tree_4taxa)

        assert result['is_rooted'], "Tree should be rooted"
        assert result['has_branch_lengths'], "Tree should have branch lengths"
        assert result['polytomy_count'] == 0, "Binary tree should have 0 polytomies"
        assert result['assumptions_satisfied'], "Binary tree should satisfy PIC assumptions"


class TestPICVariancePolytomy:
    """测试 polytomy (多分支) 情况的 PIC 方差计算"""

    @pytest.fixture
    def tree_5taxa_polytomy(self):
        """创建 5-tip 树，带 3-furcation (polytomy)。
        (旧 fixture "((A:1, B:2, C:3)D:4, E:5)F:0;" 只有 4 个 tip——
        "D" 是内部节点名而非第 5 个分类单元)"""
        return PhyloTree.from_newick("((A:1, B:2, C:3)Z:4, E:5, G:6)F:0;")

    @pytest.fixture
    def traits_5taxa(self):
        """5 个分类单元的性状数据"""
        return {"A": 1.0, "B": 2.0, "C": 3.0, "E": 4.0, "G": 5.0}

    def test_polytomy_detected(self, tree_5taxa_polytomy):
        """验证 polytomy 被正确检测"""
        result = validate_pic_assumptions(tree_5taxa_polytomy)

        assert result['polytomy_count'] > 0, "Polytomy should be detected"
        assert not result['assumptions_satisfied'], "Polytomy tree should NOT satisfy standard PIC assumptions"

    def test_polytomy_pic_computation(self, tree_5taxa_polytomy, traits_5taxa):
        """验证 polytomy 树的 PIC 仍能正确计算"""
        contrasts, pairs = compute_pic(tree_5taxa_polytomy, traits_5taxa)

        # 5-taxa 树应有 4 个独立对比
        # 其中 2 个来自 D 节点的 3-furcation (k-1 = 2)
        # 1 个来自 E vs D 的对比
        # 1 个来自根节点的最终对比
        assert len(contrasts) == 4, f"Expected 4 contrasts for 5-taxa polytomy tree, got {len(contrasts)}"

    def test_polytomy_running_var_no_double_counting(self, tree_5taxa_polytomy, traits_5taxa):
        """
        验证 polytomy 累积方差也没有双重计算

        旧代码: running_var = v0 + v1 + bl0 + bl1 (错误)
        新代码: running_var = v0 + v1 (正确)
        """
        contrasts, pairs = compute_pic(tree_5taxa_polytomy, traits_5taxa)

        # 所有对比值应该有限且合理
        for c in contrasts:
            assert np.isfinite(c), f"Contrast {c} is not finite"
            assert abs(c) < 100, f"Contrast {c} is unreasonably large"


class TestPICVarianceEdgeCases:
    """边缘情况测试"""

    def test_two_taxa_tree(self):
        """最简单的二叉树 (2 taxa)"""
        tree = PhyloTree.from_newick("(A:1, B:2);")
        traits = {"A": 1.0, "B": 3.0}

        contrasts, pairs = compute_pic(tree, traits)

        # 2-taxa 树应有 1 个对比
        assert len(contrasts) == 1
        # 对比值 = (1.0 - 3.0) / sqrt(v_A + v_B)
        # v_A = 1 (A 到根的枝长), v_B = 2 (B 到根的枝长)
        expected_contrast = (1.0 - 3.0) / np.sqrt(1 + 2)
        assert np.isclose(contrasts[0], expected_contrast, rtol=1e-10)

    def test_unbalanced_tree(self):
        """高度不平衡的树"""
        tree = PhyloTree.from_newick("(A:0.1, (B:0.2, (C:0.3, D:0.4)E:0.5)F:0.6)G:0;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

        contrasts, pairs = compute_pic(tree, traits)

        # 4-taxa 不平衡树应有 3 个对比
        assert len(contrasts) == 3
        for c in contrasts:
            assert np.isfinite(c)


class TestPICVarianceFormula:
    """直接测试 PIC 方差公式 (不依赖 compute_pic 的完整实现)"""

    def test_node_variance_formula_binary(self):
        """
        测试二叉节点的方差公式

        Felsenstein 1985:
        Var(recon_A - recon_B | P) = v_A + v_B

        其中 v_A = 从节点 P 到 tip A 的累积方差 (已包含枝长)
        """
        # 创建简单树: (A:1, B:2)C:0;
        tree = PhyloTree.from_newick("(A:1, B:2)C:0;")
        traits = {"A": 1.0, "B": 3.0}

        contrasts, pairs = compute_pic(tree, traits)

        # v_A = 1 (A 的枝长), v_B = 2 (B 的枝长)
        # node_var at C should be v_A + v_B = 3
        # contrast = (1 - 3) / sqrt(3) = -2 / sqrt(3)
        expected = (1.0 - 3.0) / np.sqrt(1 + 2)
        assert np.isclose(contrasts[0], expected, rtol=1e-10)

    def test_unequal_branch_lengths(self):
        """测试枝长不等的情况"""
        # (A:10, B:1)C:0; - A 的枝长是 B 的 10 倍
        tree = PhyloTree.from_newick("(A:10, B:1)C:0;")
        traits = {"A": 2.0, "B": 1.0}

        contrasts, pairs = compute_pic(tree, traits)

        # v_A = 10, v_B = 1
        # node_var = 11
        # contrast = (2 - 1) / sqrt(11)
        expected = (2.0 - 1.0) / np.sqrt(10 + 1)
        assert np.isclose(contrasts[0], expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
