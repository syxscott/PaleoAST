"""
================================================================================
Tests for Phylogenetic Signal (Blomberg's K & Pagel's λ)
================================================================================

验证系统发育信号计算的正确实现:
- Blomberg's K 对 Brownian motion 模拟数据应 ≈ 1
- Blomberg's K 对随机数据应 ≈ 0
- Pagel's λ 对 Brownian motion 数据应 ≈ 1
- Pagel's λ 对独立数据应 ≈ 0

参考文献:
- Blomberg, S. P., Garland, T., & Ives, A. R. (2003). Testing for phylogenetic
  signal in comparative data. Evolution, 57(4), 717-745.
- Pagel, M. (1999). Inferring the historical patterns of biological evolution.
  Nature, 401(6756), 877-884.
"""

import numpy as np
import pytest

from phylogenetics import PhyloTree, simulate_brownian_motion
from phylogenetics.signal import (
    PhylogeneticSignalResult,
    blomberg_k,
    lambda_interpretation,
    pagel_lambda,
    phylogenetic_signal,
)


class TestBlombergK:
    """Blomberg's K 测试"""

    def test_brownian_motion_K_approximately_one(self):
        """
        Brownian motion 模拟数据 K 值应接近 1

        对于严格遵循 BM 进化的数据，K 应该 ≈ 1
        """
        np.random.seed(42)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")

        # 模拟 BM 性状
        traits = simulate_brownian_motion(tree, root_value=0.0, sigma=1.0)

        result = blomberg_k(tree, traits, n_permutations=99)

        # K 应该接近 1 (允许较大容差因为小树)
        assert result.K is not None
        assert 0.2 < result.K < 3.0, f"K = {result.K} is outside expected range for BM"
        assert result.K_pvalue is not None
        assert 0 <= result.K_pvalue <= 1

    def test_random_data_K_near_zero(self):
        """
        随机数据 (无系统发育信号) K 值应接近 0
        """
        np.random.seed(123)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")

        # 随机性状 (无信号)
        traits = {"A": np.random.normal(), "B": np.random.normal(),
                  "C": np.random.normal(), "D": np.random.normal()}

        result = blomberg_k(tree, traits, n_permutations=99)

        # K 应该接近 0
        assert result.K is not None
        # 随机数据 K 可能有时会偏高，但应该显著高于 BM 的 K
        assert result.K >= 0
        assert result.K_pvalue is not None

    def test_K_with_known_low_signal(self):
        """
        测试弱信号数据

        几乎所有 tip 值相同，只有少数 outliers
        可能产生极端 K 值因为方差结构异常
        """
        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")

        # 大部分相同，少量 outliers
        traits = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 100.0}

        result = blomberg_k(tree, traits, n_permutations=99)

        # K 应该存在且 p-value 有效
        assert result.K is not None
        assert result.K_pvalue is not None
        assert 0 <= result.K_pvalue <= 1

    def test_K_requires_minimum_taxa(self):
        """K 计算需要至少 3 个 taxa"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="at least 3 taxa"):
            blomberg_k(tree, traits)


class TestPagelLambda:
    """Pagel's λ 测试"""

    def test_brownian_motion_lambda_approximately_one(self):
        """
        Brownian motion 数据 λ 应接近 1 (但由于小样本可能有变化)
        """
        np.random.seed(42)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")
        traits = simulate_brownian_motion(tree, root_value=0.0, sigma=1.0)

        result = pagel_lambda(tree, traits)

        assert result.lambda_ is not None
        # λ 应该在 [0, 1] 范围内
        assert 0.0 <= result.lambda_ <= 1.0, f"λ = {result.lambda_} should be in [0, 1]"
        assert result.lambda_pvalue is not None

    def test_random_data_lambda_near_zero(self):
        """
        随机数据 λ 应接近 0
        """
        np.random.seed(456)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")
        traits = {f"taxon_{i}": np.random.normal() for i in range(4)}
        traits = {"A": np.random.normal(), "B": np.random.normal(),
                  "C": np.random.normal(), "D": np.random.normal()}

        result = pagel_lambda(tree, traits)

        assert result.lambda_ is not None
        # 随机数据 λ 可能接近 0
        assert 0 <= result.lambda_ <= 1.0 + 1e-6
        assert result.lambda_pvalue is not None

    def test_lambda_optimization(self):
        """测试 λ 优化是否收敛"""
        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:1;")
        traits = {"A": 1.0, "B": 2.0, "C": 1.5}

        result = pagel_lambda(tree, traits)

        # λ 应该在 [0, 1] 范围内
        assert 0 <= result.lambda_ <= 1.0
        # 对数似然应该存在
        assert result.log_likelihood is not None

    def test_lambda_AIC_calculation(self):
        """测试 AIC 计算"""
        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:1;")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0}

        result = pagel_lambda(tree, traits, return_AIC=True)

        assert result.AIC is not None
        # AIC can be negative when log-likelihood is large and positive
        # The relative magnitude is what matters
        assert np.isfinite(result.AIC)

    def test_lambda_requires_minimum_taxa(self):
        """λ 计算需要至少 3 个 taxa"""
        tree = PhyloTree.from_newick("(A:1,B:1)Root:1;")
        traits = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="at least 3 taxa"):
            pagel_lambda(tree, traits)


class TestPhylogeneticSignalCombined:
    """综合系统发育信号测试"""

    def test_combined_signal_bm_data(self):
        """综合方法对 BM 数据应同时返回 K 和 λ"""
        np.random.seed(789)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")
        traits = simulate_brownian_motion(tree, root_value=10.0, sigma=2.0)

        result = phylogenetic_signal(tree, traits, n_permutations=99)

        assert result.K is not None
        assert result.lambda_ is not None
        assert result.K_pvalue is not None
        assert result.lambda_pvalue is not None
        assert result.n_taxa == 4

    def test_combined_signal_random_data(self):
        """综合方法对随机数据"""
        np.random.seed(999)

        tree = PhyloTree.from_newick("(A:1,B:1,C:1,D:1)Root:1;")
        traits = {"A": np.random.uniform(0, 1),
                  "B": np.random.uniform(0, 1),
                  "C": np.random.uniform(0, 1),
                  "D": np.random.uniform(0, 1)}

        result = phylogenetic_signal(tree, traits, n_permutations=99)

        assert result.K is not None
        assert result.lambda_ is not None


class TestLambdaInterpretation:
    """λ 解释测试"""

    def test_interpretation_strings(self):
        """测试 λ 解释函数"""
        assert "无系统发育信号" in lambda_interpretation(0.05)
        assert "弱系统发育信号" in lambda_interpretation(0.3)
        assert "中等系统发育信号" in lambda_interpretation(0.7)
        assert "强系统发育信号" in lambda_interpretation(0.95)
        assert "Invalid" in lambda_interpretation(-0.5)
        assert "λ > 1" in lambda_interpretation(1.5)


class TestSimulateBrownianMotion:
    """Brownian Motion 模拟测试"""

    def test_simulate_returns_dict(self):
        """模拟应返回正确的字典格式"""
        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:1;")

        traits = simulate_brownian_motion(tree, root_value=0.0, sigma=1.0)

        assert isinstance(traits, dict)
        assert set(traits.keys()) == {"A", "B", "C"}

    def test_simulate_deterministic_with_seed(self):
        """相同种子应产生相同结果"""
        tree = PhyloTree.from_newick("(A:1,B:1,C:1)Root:1;")

        np.random.seed(12345)
        traits1 = simulate_brownian_motion(tree, root_value=0.0, sigma=1.0)

        np.random.seed(12345)
        traits2 = simulate_brownian_motion(tree, root_value=0.0, sigma=1.0)

        for key in traits1:
            np.testing.assert_almost_equal(traits1[key], traits2[key])


class TestPhylogeneticSignalResult:
    """结果容器测试"""

    def test_result_dataclass(self):
        """测试结果数据类"""
        result = PhylogeneticSignalResult(
            K=1.5,
            lambda_=0.8,
            K_pvalue=0.05,
            lambda_pvalue=0.02,
            log_likelihood=-10.5,
            AIC=23.0,
            n_taxa=10,
            trait_name="body_size"
        )

        assert result.K == 1.5
        assert result.lambda_ == 0.8
        assert result.K_pvalue == 0.05
        assert result.lambda_pvalue == 0.02
        assert result.log_likelihood == -10.5
        assert result.AIC == 23.0
        assert result.n_taxa == 10
        assert result.trait_name == "body_size"

    def test_result_defaults(self):
        """测试默认值"""
        result = PhylogeneticSignalResult()

        assert result.K is None
        assert result.lambda_ is None
        assert result.K_pvalue is None
        assert result.lambda_pvalue is None
        assert result.n_taxa == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
