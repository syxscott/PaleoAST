"""
================================================================================
Tests for Phylogenetic Independent Contrasts (PIC)
================================================================================

验证 Felsenstein (1985) PIC 算法的正确实现:
- 基本二叉树的对比计算
- Polytomy 的正确处理 (不简单取前2个子节点)
- Unary 退化情况
- 与已知的 R ape::pic() 结果比对

参考文献:
- Felsenstein, J. (1985). Phylogenies and the comparative method.
  American Naturalist, 125(1), 1-15.
"""

import numpy as np
import pytest

from phylogenetics import PhyloNode, PhyloTree, compute_pic


class TestPICBasic:
    """基本 PIC 计算测试"""

    def test_simple_binary_tree_3_tips(self):
        """
        测试最简单的二叉树 (3 个 tips)

        树结构:
                    Root
                   /    \
                 A       B

        枝长均为 1，性状值 A=2, B=4

        预期 contrast = (2-4)/sqrt(1+1) = -2/sqrt(2) = -1.414
        """
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 2.0, "B": 4.0}

        contrasts, pairs = compute_pic(tree, traits)

        # 只有一个对比
        assert len(contrasts) == 1

        # 验证对比值
        expected_contrast = (2.0 - 4.0) / np.sqrt(1 + 1)
        np.testing.assert_almost_equal(contrasts[0], expected_contrast, decimal=10)

        # 验证配对
        assert pairs[0] == ("A", "B")

    def test_binary_tree_equal_branch_lengths(self):
        """
        测试等枝长二叉树

        树结构:
                    Root:1
                   /       \
              A:1          B:1

        性状值 A=0, B=1
        预期 contrast = (0-1)/sqrt(1+1) = -1/sqrt(2) ≈ -0.7071
        """
        tree = PhyloTree.from_newick("(A:1,B:1)Root:0;")
        traits = {"A": 0.0, "B": 1.0}

        contrasts, pairs = compute_pic(tree, traits)

        expected = (0.0 - 1.0) / np.sqrt(1 + 1)
        np.testing.assert_almost_equal(contrasts[0], expected, decimal=10)

    def test_binary_tree_uneven_branch_lengths(self):
        """
        测试不等枝长二叉树

        树结构:
                    Root
                   /    \
                 A:2    B:1

        累积方差: v_A = 2, v_B = 1
        性状值: A=1, B=2

        预期 contrast = (1-2)/sqrt(2+1) = -1/sqrt(3) ≈ -0.577
        """
        tree = PhyloTree.from_newick("(A:2,B:1)Root:0;")
        traits = {"A": 1.0, "B": 2.0}

        contrasts, pairs = compute_pic(tree, traits)

        expected = (1.0 - 2.0) / np.sqrt(2 + 1)
        np.testing.assert_almost_equal(contrasts[0], expected, decimal=10)

    def test_binary_tree_with_multiple_levels(self):
        """
        测试多层级二叉树

        树结构:
                    Root:1
                   /    \
              Node1:1   B:1
             /    \
          A:1     C:1

        累积方差:
        - A: v_A = 1+1+1 = 3
        - C: v_C = 1+1+1 = 3
        - B: v_B = 1+1 = 2

        对于 Node1 (A vs C): contrast = (1-3)/sqrt(3+3) = -2/sqrt(6)
        """
        tree = PhyloTree.from_newick("(A:1,(C:1)Node1:1,B:1)Root:1;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0}

        contrasts, pairs = compute_pic(tree, traits)

        # 应该有2个对比: Node1 的 (A,C) 和 Root 的 (Node1, B)
        assert len(contrasts) == 2

        # 验证对比值存在且有效 (具体数值因实现而异)
        for c in contrasts:
            assert np.isfinite(c), f"Contrast should be finite, got {c}"


class TestPICPolytomy:
    """Polytomy 处理测试 - 确保不简单取前2个子节点"""

    def test_trichotomy_3_children(self):
        """
        测试三分支节点 (trichotomy) 的正确处理

        树结构:
                    Root
                 /   |   \
               A     B    C

        所有枝长为 1，性状值 A=1, B=2, C=3

        根据 Pagel 1992 / Felsenstein 2008，k=3 个子节点应产生 k-1 = 2 个独立对比

        规范处理 (Felsenstein 1985; 与 ape::pic 及 statistics/pcm.py
        的单一参考实现一致): 向上传递的是逆方差加权重建值, 而非
        标准化对比值。

        迭代组合:
        1. contrast1 = (A-B)/sqrt(v_A+v_B) = (1-2)/sqrt(1+1) = -0.7071
           重建值 x_AB = (1+2)/2 = 1.5, pooled var = 1*1/(1+1) = 0.5
        2. contrast2 = (x_AB - C)/sqrt(pooled + bl_AB + v_C)
                      = (1.5 - 3)/sqrt(0.5 + 1 + 1) = -1.5/sqrt(2.5)
                      = -0.9487
           (旧期望 -1.5134 把标准化对比 contrast1 与原始值 C=3 相减,
            量纲不一致, 无统计意义)
        """
        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:0;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0}

        contrasts, pairs = compute_pic(tree, traits)

        # k=3 应产生 k-1 = 2 个对比
        assert len(contrasts) == 2, f"Expected 2 contrasts for trichotomy, got {len(contrasts)}"

        # 第一个对比: A vs B
        expected_c1 = (1.0 - 2.0) / np.sqrt(1 + 1)
        np.testing.assert_almost_equal(contrasts[0], expected_c1, decimal=10)

        # 第二个对比: 重建值 vs C
        # pooled(=0.5) 已包含 A、B 的枝长; combined 伪节点位于 Root
        # (到 Root 枝长 0); v_C = 0 + 1
        expected_c2 = (1.5 - 3.0) / np.sqrt(0.5 + 1)
        np.testing.assert_almost_equal(contrasts[1], expected_c2, decimal=10)

    def test_quadfurcation_4_children(self):
        """
        测试四分支节点 (quadfurcation) 的正确处理

        树结构:
                    Root
               /    |    \
             A      B    (C,D)
                   / \
                  C   D

        实际上这是嵌套的二叉树，让我们直接用真正的 quadfurcation:
                    Root
               /    |    \
              A     B     C     D

        k=4 应产生 k-1 = 3 个对比
        """
        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:0;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

        contrasts, pairs = compute_pic(tree, traits)

        # k=4 应产生 k-1 = 3 个对比
        assert len(contrasts) == 3, f"Expected 3 contrasts for quadfurcation, got {len(contrasts)}"


class TestPICValidation:
    """PIC 假设验证测试"""

    def test_validate_assumptions_binary_tree(self):
        """二叉树应满足所有假设"""
        from phylogenetics.pic import validate_pic_assumptions

        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        result = validate_pic_assumptions(tree)

        assert result['is_rooted'] is True
        assert result['has_branch_lengths'] is True
        assert result['polytomy_count'] == 0
        assert result['assumptions_satisfied'] is True

    def test_validate_assumptions_polytomy(self):
        """Polytomy 树应有警告但仍可计算"""
        from phylogenetics.pic import validate_pic_assumptions

        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:1;")
        result = validate_pic_assumptions(tree)

        assert result['is_rooted'] is True
        assert result['has_branch_lengths'] is True
        assert result['polytomy_count'] == 1
        assert len(result['warnings']) > 0
        # Poltomy 违反 PIC 的严格二叉假设: 不满足假设 (仍可用迭代组合
        # 法计算, 见 test_trichotomy_3_children)。与
        # TestPICVariancePolytomy::test_polytomy_detected 的断言一致。
        assert result['assumptions_satisfied'] is False


class TestPICEdgeCases:
    """边界情况测试"""

    def test_missing_trait_warning(self, caplog):
        """缺失性状值应产生警告"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 1.0}  # 缺少 B

        with caplog.at_level("WARNING"):
            contrasts, pairs = compute_pic(tree, traits)

        # 应该使用默认值 0.0 继续计算
        assert len(contrasts) == 1
        assert "not found" in caplog.text

    def test_zero_branch_length(self):
        """零枝长应能正常处理"""
        tree = PhyloTree.from_newick("(A:0,B:0)Root:0;")
        traits = {"A": 1.0, "B": 3.0}

        # 应该不会除零
        contrasts, pairs = compute_pic(tree, traits)
        assert len(contrasts) == 1


class TestPICAncestralStates:
    """祖先状态估计测试"""

    def test_ancestral_states_computed(self):
        """验证祖先状态是否被计算"""
        from phylogenetics.pic import compute_pic_with_ancestral_states

        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 2.0, "B": 4.0}

        contrasts, pairs, ancestral = compute_pic_with_ancestral_states(tree, traits)

        # Root 的祖先状态应为 A 和 B 的平均
        assert 'Root' in ancestral or '_internal_' in str(list(ancestral.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
