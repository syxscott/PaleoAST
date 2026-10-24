# =============================================================================
# FILE: tests/golden/test_spectral_moments.py
# =============================================================================
"""
``stratigraphy.spectral_moments`` 的测试：谱矩、谱形描述量与 AR(1) 估计。

期望值来自**定义本身**，而不是"跑一遍把输出冻结下来"：

* 谱矩 ``m_k = ∫ f^k P(f) df`` 对"白噪声地板 + 单条谱线"的组合有解析值：
  白噪声是常数，积分解析可算；离散的单条谱线按梯形权重折算成等效质量，
  同样解析可算。矩的比较用相对误差判据（离散谱有偏差，判据取 20%）。
* 谱质心等于 ``m_1 / m_0``、谱宽等于频率分布的标准差，这两条是定义。
* ``estimate_ar1_rho`` 对 AR(1) 模拟序列应接近真值；估计量是一致且几乎
  无偏的，抽样误差由 ``1 / sqrt(N)`` 控制。
* ``minimum_rho`` 取的是保守上界，因此**不会低于**数据的真实 AR(1) 持续性。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"estimate_ar1_rho", "minimum_rho", "spectral_moments"}


import numpy as np
import pytest

from stratigraphy.spectral_moments import (
    estimate_ar1_rho,
    minimum_rho,
    spectral_moments,
)
from utils.exceptions import ValidationError

#: 谱矩测试用的频率轴：0.01 到 1.0，步长 0.001（单条谱线正落在网格点上）。
GRID = 0.01 + 0.001 * np.arange(991)
#: 网格的步长，谱线的等效质量按它折算。
GRID_STEP = 0.001
#: 白噪声地板的谱密度。
WHITE_LEVEL = 0.5
#: 单条谱线的位置与高度。
LINE_FREQUENCY = 0.3
LINE_HEIGHT = 40.0


def _ar1_series(rho: float, size: int, seed: int) -> np.ndarray:
    """生成平稳 AR(1) 序列 ``x[t] = rho * x[t-1] + eps[t]``。"""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=size)
    series = np.zeros(size, dtype=float)
    for index in range(1, size):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _spiky_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """白噪声地板加一条离散谱线。"""
    power = np.full_like(GRID, WHITE_LEVEL)
    line_index = round((LINE_FREQUENCY - GRID[0]) / GRID_STEP)
    power[line_index] += LINE_HEIGHT
    return power, GRID


def _analytic_moments(order: int) -> np.ndarray:
    """白噪声地板加单条谱线的解析谱矩。"""
    moments = np.zeros(order + 1, dtype=float)
    for k in range(order + 1):
        floor = WHITE_LEVEL * (GRID[-1] ** (k + 1) - GRID[0] ** (k + 1)) / (k + 1)
        line = LINE_HEIGHT * GRID_STEP * LINE_FREQUENCY**k
        moments[k] = floor + line
    return moments


class TestSpectralMoments:
    """谱矩必须与解析值一致，描述量必须符合定义。"""

    def test_the_moments_match_the_analytic_values(self):
        """谱矩与"白噪声 + 单谱线"的解析值接近。"""
        power, frequencies = _spiky_spectrum()
        result = spectral_moments(power, frequencies)
        expected = _analytic_moments(4)
        for order in range(3):
            assert result.moments[order] == pytest.approx(expected[order], rel=0.2)

    def test_the_orders_match_the_moments(self):
        """矩阶与谱矩一一对应。"""
        power, frequencies = _spiky_spectrum()
        result = spectral_moments(power, frequencies)
        assert result.orders.tolist() == [0, 1, 2, 3, 4]
        assert result.moments.size == result.orders.size

    def test_the_centroid_is_the_moment_ratio(self):
        """谱质心等于 m_1 / m_0。"""
        power, frequencies = _spiky_spectrum()
        result = spectral_moments(power, frequencies)
        assert result.centroid == pytest.approx(result.moments[1] / result.moments[0], rel=1e-12)

    def test_the_centroid_matches_the_analytic_value(self):
        """谱质心等于解析的功率加权平均频率。"""
        power, frequencies = _spiky_spectrum()
        result = spectral_moments(power, frequencies)
        expected = _analytic_moments(4)
        assert result.centroid == pytest.approx(expected[1] / expected[0], rel=1e-3)

    def test_the_bandwidth_is_the_frequency_standard_deviation(self):
        """谱宽等于频率分布的标准差。"""
        power, frequencies = _spiky_spectrum()
        result = spectral_moments(power, frequencies)
        variance = result.moments[2] / result.moments[0] - result.centroid**2
        assert result.bandwidth == pytest.approx(np.sqrt(variance), rel=1e-12)
        assert result.bandwidth > 0.0

    def test_a_flat_spectrum_is_centred_on_the_band(self):
        """平坦谱的频率分布是均匀分布：质心在中点，谱宽是均匀分布的标准差。"""
        power = np.full_like(GRID, 0.5)
        result = spectral_moments(power, GRID)
        span = GRID[-1] - GRID[0]
        assert result.bandwidth == pytest.approx(span / np.sqrt(12.0), rel=1e-5)
        assert result.centroid == pytest.approx(0.5 * (GRID[0] + GRID[-1]), rel=1e-9)

    def test_a_narrow_line_is_centred_on_its_frequency(self):
        """谱越窄，质心越靠近谱线频率。"""
        power = np.full_like(GRID, 1e-6)
        power[round((LINE_FREQUENCY - GRID[0]) / GRID_STEP)] = 1.0
        result = spectral_moments(power, GRID)
        assert result.centroid == pytest.approx(LINE_FREQUENCY, abs=0.01)

    def test_the_ar1_coefficient_of_white_noise_is_near_zero(self):
        """白噪声谱的 AR(1) 系数接近 0。"""
        power = np.full_like(GRID, 0.5)
        result = spectral_moments(power, GRID)
        assert abs(result.ar1_rho) < 0.2

    def test_an_all_zero_spectrum_raises(self):
        """全零谱没有定义好的矩。"""
        power = np.zeros_like(GRID)
        with pytest.raises(ValidationError):
            spectral_moments(power, GRID)

    def test_non_positive_frequencies_raise(self):
        """频率轴必须严格为正。"""
        power = np.full_like(GRID, 0.5)
        with pytest.raises(ValidationError):
            spectral_moments(power, np.concatenate(([0.0], GRID[1:])))
        with pytest.raises(ValidationError):
            spectral_moments(power, GRID - 1.0)

    def test_a_non_increasing_frequency_axis_raises(self):
        """频率轴必须严格递增。"""
        power = np.full_like(GRID, 0.5)
        scrambled = GRID.copy()
        scrambled[10], scrambled[11] = scrambled[11], scrambled[10]
        with pytest.raises(ValidationError):
            spectral_moments(power, scrambled)

    def test_negative_power_raises(self):
        """功率谱不能为负。"""
        power = np.full_like(GRID, 0.5)
        power[5] = -0.1
        with pytest.raises(ValidationError):
            spectral_moments(power, GRID)

    def test_too_few_bins_raise(self):
        """点数太少无法给出谱矩。"""
        with pytest.raises(ValidationError):
            spectral_moments(np.array([1.0, 2.0]), np.array([0.5, 1.0]))

    def test_mismatched_lengths_raise(self):
        """功率与频率必须等长。"""
        with pytest.raises(ValidationError):
            spectral_moments(np.full_like(GRID, 0.5), GRID[:-1])

    def test_an_out_of_range_max_order_raises(self):
        """``max_order`` 必须在允许范围内。"""
        power, frequencies = _spiky_spectrum()
        with pytest.raises(ValidationError):
            spectral_moments(power, frequencies, max_order=0)
        with pytest.raises(ValidationError):
            spectral_moments(power, frequencies, max_order=99)


class TestEstimateAr1Rho:
    """AR(1) 系数估计必须接近真值。"""

    def test_the_estimate_is_close_to_the_true_coefficient(self):
        """对 AR(1) 模拟序列，估计值接近真值。"""
        for true_rho in (0.3, 0.6, 0.9):
            series = _ar1_series(true_rho, 5000, seed=17)
            assert estimate_ar1_rho(series) == pytest.approx(true_rho, abs=0.1)

    def test_white_noise_gives_a_coefficient_near_zero(self):
        """白噪声没有持续性。"""
        series = np.random.default_rng(5).normal(size=5000)
        assert estimate_ar1_rho(series) == pytest.approx(0.0, abs=0.1)

    def test_too_few_samples_raise(self):
        """点数太少无法估计。"""
        with pytest.raises(ValidationError):
            estimate_ar1_rho(np.array([1.0, 2.0]))

    def test_a_constant_series_raises(self):
        """常数序列的 AR(1) 系数没有定义。"""
        with pytest.raises(ValidationError):
            estimate_ar1_rho(np.full(100, 3.0))

    def test_a_non_finite_series_raises(self):
        """含非有限值的序列不能估计。"""
        series = _ar1_series(0.5, 100, seed=1)
        series[10] = np.nan
        with pytest.raises(ValidationError):
            estimate_ar1_rho(series)


class TestMinimumRho:
    """最小 rho 必须保守（不低于真实持续性）且可复现。"""

    def test_the_result_is_not_below_the_true_coefficient(self):
        """对 AR(1) 模拟序列，返回值不低于真值。"""
        series = _ar1_series(0.6, 5000, seed=11)
        result = minimum_rho(series, random_seed=11)
        assert result >= 0.6

    def test_the_result_is_not_below_the_true_coefficient_for_another_seed(self):
        """换一组随机种子，结论不变。"""
        series = _ar1_series(0.75, 5000, seed=99)
        result = minimum_rho(series, random_seed=99)
        assert result >= 0.75

    def test_the_result_is_reproducible_with_the_same_seed(self):
        """给定种子时结果完全可复现。"""
        series = _ar1_series(0.5, 2000, seed=7)
        first = minimum_rho(series, random_seed=3)
        second = minimum_rho(series, random_seed=3)
        assert first == second

    def test_the_result_stays_inside_the_unit_interval(self):
        """返回值落在 [0, 0.999] 内。"""
        series = _ar1_series(0.9, 2000, seed=21)
        result = minimum_rho(series, random_seed=4)
        assert 0.0 <= result <= 0.999

    def test_white_noise_gives_a_small_coefficient(self):
        """白噪声没有可检出的持续性。"""
        series = np.random.default_rng(13).normal(size=2000)
        assert minimum_rho(series, random_seed=2) < 0.2

    def test_too_few_samples_raise(self):
        """点数太少无法做检验。"""
        with pytest.raises(ValidationError):
            minimum_rho(np.arange(5.0))

    def test_an_invalid_candidate_count_raises(self):
        """候选数至少为 2。"""
        series = _ar1_series(0.5, 500, seed=1)
        with pytest.raises(ValidationError):
            minimum_rho(series, n_candidates=1)

    def test_an_invalid_alpha_raises(self):
        """显著性水平必须落在 (0, 1) 内。"""
        series = _ar1_series(0.5, 500, seed=1)
        with pytest.raises(ValidationError):
            minimum_rho(series, alpha=0.0)
        with pytest.raises(ValidationError):
            minimum_rho(series, alpha=1.0)

    def test_an_unseeded_run_warns_about_reproducibility(self):
        """不给种子时按仓库约定发出 RuntimeWarning。"""
        series = _ar1_series(0.5, 500, seed=1)
        with pytest.warns(RuntimeWarning):
            minimum_rho(series)
