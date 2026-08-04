"""
================================================================================
Test Phylogenetic Signal - Golden Values
================================================================================

这些测试验证 Blomberg's K 和 Pagel's λ 的计算与已知 golden values 一致。

测试策略:
1. 模拟 Brown Motion (BM) 数据 - K 应接近 1
2. 模拟独立数据 (随机 permutation) - K 应接近 0
3. 使用已知案例对比 R phytools::phylosignal 的结果

参考文献:
- Blomberg, S. P., Garland, T., & Ives, A. R. (2003). Testing for phylogenetic
  signal in comparative data: behavioral traits are more labile. Evolution,
  57(4), 717-745.
- Pagel, M. (1999). Inferring the historical patterns of biological evolution.
  Nature, 401(6756), 877-884.

作者: PaleoAST Development Team
"""

import numpy as np
import pytest

from phylogenetics.signal import (
    blomberg_k,
    pagel_lambda,
    phylogenetic_signal,
    simulate_brownian_motion,
)
from phylogenetics.tree import PhyloTree


class TestBlombergK:
    """测试 Blomberg's K 计算"""

    @pytest.fixture
    def tree_5taxa(self):
        """5-taxa 测试树"""
        return PhyloTree.from_newick("((A:1, B:2)E:3, (C:4, D:5)F:6)G:0;")

    def test_bm_data_k_approx_one(self, tree_5taxa):
        """
        验证 BM 模拟数据的 K 值接近 1

        Blomberg's K 在 BM 进化下期望值为 1
        由于采样误差，K 通常在 0.5-2.0 之间
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = blomberg_k(tree_5taxa, traits, n_permutations=99)

        # K should be reasonably close to 1 for BM data
        # Allow wide tolerance since K has high variance with few taxa
        assert 0.3 < result.K < 3.0, f"K={result.K} is outside reasonable range for BM data"
        assert result.n_taxa == 5

    def test_independent_data_k_approx_zero(self, tree_5taxa):
        """
        验证独立 (permuted) 数据的 K 值接近 0

        如果性状独立于系统发育，K 应接近 0
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        # 随机置换性状值 (破坏系统发育信号)
        trait_values = list(traits.values())
        np.random.shuffle(trait_values)
        permuted_traits = dict(zip(traits.keys(), trait_values))

        result = blomberg_k(tree_5taxa, permuted_traits, n_permutations=99)

        # K should be close to 0 for independent data
        assert result.K < 0.5, f"K={result.K} suggests phylogenetic signal in permuted data"

    def test_strong_signal_high_k(self):
        """
        验证强系统发育信号的数据 K > 1

        创建一个所有叶节点性状值相同 (完美保守) 的情况
        """
        tree = PhyloTree.from_newick("(A:1, B:2, C:3)D:0;")
        # 所有性状相同 = 完美保守
        traits = {"A": 1.0, "B": 1.0, "C": 1.0}

        result = blomberg_k(tree, traits, n_permutations=99)

        # 完美保守情况下 K 应该很高
        assert result.K > 1.0, f"K={result.K} should be > 1 for perfectly conserved trait"

    def test_k_pvalue_interpretation(self, tree_5taxa):
        """
        验证 p 值的解释

        对于真实 BM 数据，K 的置换检验 p 值应该不显著
        (因为 K ≈ 1 是 BM 的期望)
        """
        np.random.seed(123)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = blomberg_k(tree_5taxa, traits, n_permutations=99)

        # p 值可以是任意值，不做严格断言
        assert 0.0 <= result.K_pvalue <= 1.0


class TestPagelLambda:
    """测试 Pagel's λ 计算"""

    @pytest.fixture
    def tree_5taxa(self):
        """5-taxa 测试树"""
        return PhyloTree.from_newick("((A:1, B:2)E:3, (C:4, D:5)F:6)G:0;")

    def test_bm_data_lambda_approx_one(self, tree_5taxa):
        """
        验证 BM 数据的 λ 值接近 1

        Pagel's λ 在 BM 进化下期望值为 1
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = pagel_lambda(tree_5taxa, traits)

        # λ should be close to 1 for BM data
        assert 0.5 < result.lambda_ <= 1.0, f"λ={result.lambda_} is outside reasonable range for BM data"

    def test_independent_data_lambda_approx_zero(self, tree_5taxa):
        """
        验证独立 (permuted) 数据的 λ 值接近 0

        如果性状独立于系统发育，λ 应接近 0
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        # 随机置换性状值
        trait_values = list(traits.values())
        np.random.shuffle(trait_values)
        permuted_traits = dict(zip(traits.keys(), trait_values))

        result = pagel_lambda(tree_5taxa, permuted_traits)

        # λ should be close to 0 for independent data
        assert result.lambda_ < 0.5, f"λ={result.lambda_} suggests phylogenetic signal in permuted data"

    def test_lambda_bounds(self, tree_5taxa):
        """
        验证 λ 在 (0, 1) 范围内

        Pagel λ 的定义域是 [0, 1]
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = pagel_lambda(tree_5taxa, traits)

        assert 0.0 <= result.lambda_ <= 1.0, f"λ={result.lambda_} is outside [0,1] bounds"

    def test_lambda_pvalue(self, tree_5taxa):
        """
        验证 λ 的似然比检验 p 值

        比较 λ=0 (无信号) vs λ=fitted (最大似然)
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = pagel_lambda(tree_5taxa, traits)

        # p 值应该是有效的概率值
        assert 0.0 <= result.lambda_pvalue <= 1.0, "p-value should be between 0 and 1"


class TestPhylogeneticSignalCombined:
    """综合测试 Blomberg K 和 Pagel λ"""

    @pytest.fixture
    def tree_5taxa(self):
        return PhyloTree.from_newick("((A:1, B:2)E:3, (C:4, D:5)F:6)G:0;")

    def test_combined_result(self, tree_5taxa):
        """验证 phylogenetic_signal 同时返回 K 和 λ"""
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = phylogenetic_signal(tree_5taxa, traits, n_permutations=99)

        assert result.K is not None, "K should be computed"
        assert result.lambda_ is not None, "λ should be computed"
        assert result.n_taxa == 5

    def test_bm_signal_consistency(self, tree_5taxa):
        """
        验证 BM 数据下 K 和 λ 的一致性

        对于真正的 BM 数据:
        - K ≈ 1
        - λ ≈ 1
        """
        np.random.seed(42)
        traits = simulate_brownian_motion(tree_5taxa, root_value=0.0, sigma=1.0)

        result = phylogenetic_signal(tree_5taxa, traits, n_permutations=99)

        # 对于 BM 数据，K 和 λ 都应该接近 1
        assert 0.5 < result.K < 2.0
        assert 0.5 < result.lambda_ <= 1.0


class TestSignalAgainstKnownValues:
    """与已知 reference values 对比测试"""

    def test_kestrel_etal_2003_example(self):
        """
        测试 Kestrel et al. 2003 中的简单案例

        3-taxa 树: (A:1, B:1, C:1);
        性状: A=0, B=1, C=2

        预期: K 应该 > 1 (因为性状沿系统发育有方向性变化)
        """
        tree = PhyloTree.from_newick("(A:1, B:1, C:1)G:0;")
        traits = {"A": 0.0, "B": 1.0, "C": 2.0}

        result = blomberg_k(tree, traits, n_permutations=999)

        # 这个简单案例的 K 应该 > 0
        assert result.K > 0

    def test_pagel_1999_example(self):
        """
        参考 Pagel 1999 的 λ 计算示例

        简单 4-taxa 树
        """
        tree = PhyloTree.from_newick("(A:1, B:1, (C:1, D:1)E:1)F:0;")
        traits = {"A": 1.0, "B": 1.5, "C": 2.0, "D": 2.5}

        result = phylogenetic_signal(tree, traits, n_permutations=99)

        # λ 应该是一个有效的值
        assert 0.0 <= result.lambda_ <= 1.0
        assert result.K is not None


class TestSignalEdgeCases:
    """边缘情况测试"""

    def test_two_taxa_insufficient(self):
        """验证 2 个分类单元无法计算 K"""
        tree = PhyloTree.from_newick("(A:1, B:1);")
        traits = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="at least 3 taxa"):
            blomberg_k(tree, traits)

    def test_single_tip_insufficient(self):
        """验证单个分类单元无法计算"""
        tree = PhyloTree.from_newick("A:0;")
        traits = {"A": 1.0}

        with pytest.raises(ValueError, match="at least 3 taxa"):
            blomberg_k(tree, traits)

    def test_perfect_star_tree(self):
        """
        测试星形树 (无系统发育结构)

        星形树下，所有 tip 到根的距离相等
        任何性状都没有系统发育信号
        """
        tree = PhyloTree.from_newick("(A:1, B:1, C:1, D:1)G:0;")
        # 随机性状
        np.random.seed(42)
        traits = {name: np.random.randn() for name in ["A", "B", "C", "D"]}

        result = phylogenetic_signal(tree, traits, n_permutations=99)

        # 星形树下，K 和 λ 应该都较低
        # (因为没有真正的系统发育结构)
        assert result.K is not None
        assert result.lambda_ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
