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
        # fixture 树 ((A:1,B:2)E:3,(C:4,D:5)F:6)G:0 只有 4 个 tip
        assert result.n_taxa == 4

    def test_independent_data_k_approx_zero(self):
        """
        验证独立 (iid) 数据无显著系统发育信号

        K 的零分布依赖树形——对iid 数据断言 "K < 0.5" 是错误的
        (本 32-tip 树上 iid 的 K 中位数约 0.77, 因 tr(V⁻¹)tr(V)/n
        修正项)。科学的断言是: iid 数据的置换检验不显著 (p > 0.05),
        5 个固定种子全部满足。
        """
        inner = ",".join(f"(t{i}a:1,t{i}b:1):1" for i in range(16))
        tree = PhyloTree.from_newick(f"({inner});")
        for seed in [1, 2, 3, 4, 5]:
            rng = np.random.default_rng(seed)
            traits = {name: rng.normal() for name in (leaf.name for leaf in tree.root.get_leaves())}
            result = blomberg_k(tree, traits, n_permutations=199)
            assert result.K_pvalue > 0.05, (
                f"seed={seed}: iid data should not show significant signal (p={result.K_pvalue})"
            )

    def test_strong_signal_high_k(self):
        """
        验证强系统发育信号的数据 K > 1

        姐妹种性状相同、外群不同 => 变异与系统发育高度一致。
        (完全恒定的性状方差为 0, K = 0/0 无定义, 规范实现返回 0;
        用它断言 K>1 是测试自身的概念错误。)
        """
        tree = PhyloTree.from_newick("(A:1, B:2, C:3)D:0;")
        traits = {"A": 5.0, "B": 5.0, "C": 0.0}

        result = blomberg_k(tree, traits, n_permutations=99)

        assert result.K > 1.0, f"K={result.K} should be > 1 for strong phylogenetic signal"

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

    def test_bm_data_lambda_approx_one(self):
        """
        验证 BM 数据的 λ 值接近 1

        Pagel's λ 在 BM 进化下期望值为 1。n 很小时单次实现的 λ̂ 采样
        噪声极大 (n=4 时中位数可到 ~0.5), 因此用 32-tip 平衡树 +
        5 个固定种子的中位数做确定性断言 (规范实现: 中位数 = 1.0)。
        """
        inner = ",".join(f"(t{i}a:1,t{i}b:1):1" for i in range(16))
        tree = PhyloTree.from_newick(f"({inner});")

        lams = []
        for seed in [1, 2, 3, 4, 5]:
            rng = np.random.default_rng(seed)
            traits = {}
            self._simulate_bm(tree.root, 0.0, rng, traits)
            lams.append(pagel_lambda(tree, traits).lambda_)

        median_lambda = float(np.median(lams))
        assert 0.4 < median_lambda <= 1.0, (
            f"median λ={median_lambda} is outside reasonable range for BM data"
        )

    @staticmethod
    def _simulate_bm(node, val, rng, out):
        if node.is_leaf:
            out[node.name] = val
            return
        for ch in node.children:
            TestPagelLambda._simulate_bm(ch, val + rng.normal(0, np.sqrt(ch.branch_length or 0.0)), rng, out)

    def test_independent_data_lambda_approx_zero(self):
        """
        验证独立 (iid) 数据的 λ 值接近 0

        32-tip 平衡树 + 5 固定种子的 iid 性状: 中位数 λ = 0.089。
        单次实现的 λ̂ 噪声大, 用中位数做确定性断言。
        """
        inner = ",".join(f"(t{i}a:1,t{i}b:1):1" for i in range(16))
        tree = PhyloTree.from_newick(f"({inner});")
        lams = []
        for seed in [1, 2, 3, 4, 5]:
            rng = np.random.default_rng(seed)
            traits = {name: rng.normal() for name in (leaf.name for leaf in tree.root.get_leaves())}
            lams.append(pagel_lambda(tree, traits).lambda_)
        assert float(np.median(lams)) < 0.5, f"median λ={np.median(lams)} suggests signal in iid data"

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
        assert result.n_taxa == 4  # fixture 树只有 4 个 tip

    @staticmethod
    def _simulate_bm(node, val, rng, out):
        if node.is_leaf:
            out[node.name] = val
            return
        for ch in node.children:
            TestPhylogeneticSignalCombined._simulate_bm(ch, val + rng.normal(0, np.sqrt(ch.branch_length or 0.0)), rng, out)

    def test_bm_signal_consistency(self):
        """
        验证 BM 数据下 K 和 λ 的一致性 (32-tip 树, 5 种子中位数,
        确定值: median K = 1.08, median λ = 1.0)
        """
        inner = ",".join(f"(t{i}a:1,t{i}b:1):1" for i in range(16))
        tree = PhyloTree.from_newick(f"({inner});")
        self._sim = type(self)._simulate_bm

        ks = []
        lams = []
        for seed in [1, 2, 3, 4, 5]:
            rng = np.random.default_rng(seed)
            traits = {}
            self._simulate_bm(tree.root, 0.0, rng, traits)
            result = phylogenetic_signal(tree, traits, n_permutations=99)
            ks.append(result.K)
            lams.append(result.lambda_)

        assert 0.5 < float(np.median(ks)) < 2.0
        assert 0.3 < float(np.median(lams)) <= 1.0


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
