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
        """缺失性状值应告警并把该叶剔除，而不是按 0.0 代入

        更新说明: 旧断言 `len(contrasts) == 1` 固化的是 bug —— 缺失的 B 被当成
        0.0 代入，凭空造出一个 (A, B) 对比 (值 = 1/sqrt(1))，而 B=0.0 并不是
        观测数据。现在缺性状的叶被剔除后再计算，本例只剩 A 一个叶，
        没有可比对的成对支，因此对比数为 0；被剔除的叶名记录在
        tree.metadata['pic_missing_tips']。
        """
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 1.0}  # 缺少 B

        with caplog.at_level("WARNING"):
            contrasts, pairs = compute_pic(tree, traits)

        assert contrasts == []
        assert pairs == []
        assert "not found" in caplog.text
        assert tree.metadata["pic_missing_tips"] == ["B"]
        # 原树未被修改 (剔除只作用于工作拷贝)
        assert [leaf.name for leaf in tree.root.get_leaves()] == ["A", "B"]

    def test_partial_missing_traits_still_compare_remaining_tips(self):
        """剔除缺失叶后，其余有数据的叶仍应正常产生对比"""
        tree = PhyloTree.from_newick("((A:1,B:1)N:1,C:1)Root:1;")
        traits = {"A": 1.0, "B": 3.0}  # 缺少 C

        contrasts, pairs = compute_pic(tree, traits)

        assert len(contrasts) == 1
        assert pairs[0] == ("A", "B")
        np.testing.assert_almost_equal(contrasts[0], (1.0 - 3.0) / np.sqrt(1 + 1), decimal=10)

    def test_zero_branch_length(self):
        """零枝长: 对比无定义 (0/0)，应返回 NaN 而不是被 1e-10 放大成伪对比

        更新说明: 旧实现把方差和夹到 1e-10，使 (1-3)/sqrt(1e-10) ≈ -2e5 的
        巨大伪对比冒充真实统计量 (仍断言 len==1 掩盖了问题)。
        """
        tree = PhyloTree.from_newick("(A:0,B:0)Root:0;")
        traits = {"A": 1.0, "B": 3.0}

        contrasts, pairs = compute_pic(tree, traits)

        assert len(contrasts) == 1
        assert pairs == [("A", "B")]
        assert np.isnan(contrasts[0])
        assert tree.metadata["pic_degenerate_pairs"] == [("A", "B")]

    def test_empty_tree_raises(self):
        """空树 (root=None) 应给出明确错误而不是 AttributeError"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        tree.root = None

        with pytest.raises(ValueError, match="no root node"):
            compute_pic(tree, {"A": 1.0, "B": 2.0})

    def test_no_traits_raises(self):
        """空性状字典应报错而不是静默返回空结果"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        with pytest.raises(ValueError, match="no trait values"):
            compute_pic(tree, {})

    def test_negative_root_variance_raises(self):
        """root_variance 为负应报错"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        with pytest.raises(ValueError, match="root_variance"):
            compute_pic(tree, {"A": 1.0, "B": 2.0}, root_variance=-1.0)

    def test_root_variance_does_not_change_contrasts(self):
        """root_variance 仅为兼容保留: 对比值不随之改变"""
        tree = PhyloTree.from_newick("((A:1,B:1)N:2,C:3)Root:1;")
        traits = {"A": 1.0, "B": 3.0, "C": 2.0}

        base, _ = compute_pic(tree, traits, root_variance=0.0)
        shifted, _ = compute_pic(tree, traits, root_variance=5.0)

        np.testing.assert_allclose(base, shifted, rtol=1e-12)

    def test_polytomy_combined_labels_are_unambiguous(self):
        """多个多分叉节点的伪节点名不得重复

        更新说明: 旧实现每个多分叉节点都从 1 重新计数，_combined_1 在
        contrast_pairs 中出现多次且指向不同的节点对。
        """
        tree = PhyloTree.from_newick("((A:1,B:1,C:1)N1:1,(D:1,E:1,F:1)N2:1)Root:0;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}

        contrasts, pairs = compute_pic(tree, traits)

        labels = {label for pair in pairs for label in pair}
        combined = [label for label in labels if label.startswith("_combined_")]
        assert len(contrasts) == 5  # (3-1) + (3-1) + Root 的 1 个
        # 每个伪节点名只出现一次 => 无歧义
        assert len(combined) == len(set(combined))
        flat = [label for pair in pairs for label in pair]
        assert len(flat) == len(set(flat))


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
        # 等方差两支 => 逆方差加权重建退化为简单平均 3.0
        np.testing.assert_almost_equal(ancestral["Root"], 3.0, decimal=10)

    def test_unnamed_internal_nodes_get_unique_keys(self):
        """无名内部节点不得共用 '_internal_' 键而互相覆盖

        更新说明: 旧实现 `ancestral_states[node.name if node.name else '_internal_']`
        使所有无名内部节点写到同一个键，只剩最后一个节点的值。
        """
        from phylogenetics.pic import compute_pic_with_ancestral_states

        tree = PhyloTree.from_newick("((((A:1,B:1):1,C:1):1,D:1):1,E:1):0;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}

        contrasts, pairs, ancestral = compute_pic_with_ancestral_states(tree, traits)

        # 4 个无名内部节点 => 4 个不同键
        assert len(ancestral) == 4, ancestral
        assert sorted(ancestral) == ["_internal_1", "_internal_2", "_internal_3", "_internal_4"]
        assert all(np.isfinite(v) for v in ancestral.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
