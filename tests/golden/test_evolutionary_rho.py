# =============================================================================
# FILE: tests/golden/test_evolutionary_rho.py
# =============================================================================
"""
``stratigraphy.evolutionary_rho`` 的测试：演化 ρ1（滑窗滞后 1 阶自相关）。

期望值来自**定义本身**，而不是"跑一遍把输出冻结下来"：

* 逐窗 ρ1 的估计量是 Yule–Walker 形式 ``Σz(2:)·z(1:-1) / Σz(1:-1)²``。因此对一
  条 AR(1) 模拟序列，只要窗口足够长，窗内 ρ1 必须接近真系数——这是估计量的定义
  直接给出的结论，不依赖任何调参。
* 合成一段"前半段强红噪声、后半段弱白噪声"的记录，演化 ρ1 必须呈现出**方向性**
  的强弱变化：前半段的平均 ρ1 高于后半段。这正是演化 ρ1 存在的意义。
* 窗口中心是窗内首末坐标的中点；窗口点数是 ``round(window / dt)``，因此窗数是
  ``N - K + 1``，两者都是定义给出的。
* 不确定度由移动块自助法给出：同一个种子必须给出逐点相同的结果；不给种子时必
  须发出 ``RuntimeWarning``（仓库对随机分析的统一约定）。
* 边界（窗口太短/太长、步长非法、点数太少、时间轴非严格递增）必须抛
  ``ValidationError``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"evolutionary_rho1"}


import numpy as np
import pytest

from stratigraphy.evolutionary_rho import evolutionary_rho1
from utils.exceptions import ValidationError

#: 合成记录的长度（采样间隔为 1）。
RECORD_LENGTH = 800

#: 前后两段的分界点。
SPLIT = 400

#: 滑窗长度（时间单位）；必须明显小于半段长度，窗才不会跨越分界点。
WINDOW = 120.0


def _ar1_series(rho: float, size: int, seed: int, innovation_sd: float = 1.0) -> np.ndarray:
    """生成平稳 AR(1) 序列 ``x[t] = rho * x[t-1] + eps[t]``。"""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, innovation_sd * np.sqrt(1.0 - rho * rho), size=size)
    series = np.zeros(size, dtype=float)
    for index in range(1, size):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _two_regime_record() -> tuple[np.ndarray, np.ndarray]:
    """前半段强红噪声、后半段弱白噪声的合成记录。"""
    time = np.arange(RECORD_LENGTH, dtype=float)
    red = _ar1_series(0.85, SPLIT, seed=21, innovation_sd=1.0)
    white = np.random.default_rng(22).normal(0.0, 0.2, size=RECORD_LENGTH - SPLIT)
    return time, np.concatenate([red, white])


class TestEvolutionaryRho:
    """演化 ρ1 必须反映出局部持续性的强弱变化。"""

    def test_the_window_rho_follows_the_local_persistence(self):
        """前半段（红噪声）的平均 ρ1 必须高于后半段（白噪声）。"""
        time, values = _two_regime_record()
        result = evolutionary_rho1(time, values, window=WINDOW, random_seed=3)
        front = result.rho[result.window_centres < SPLIT - 150.0]
        back = result.rho[result.window_centres > SPLIT + 150.0]
        assert front.size > 10
        assert back.size > 10
        assert float(np.mean(front)) > float(np.mean(back))

    def test_the_window_rho_of_an_ar1_series_matches_the_true_coefficient(self):
        """AR(1) 序列的窗内 ρ1 就是它的滞后 1 阶自相关。"""
        time = np.arange(1000, dtype=float)
        values = _ar1_series(0.7, 1000, seed=8)
        result = evolutionary_rho1(time, values, window=400.0, n_bootstrap=0)
        assert float(np.mean(result.rho)) == pytest.approx(0.7, abs=0.1)

    def test_the_window_count_follows_the_window_definition(self):
        """窗数是 ``N - K + 1``，``K = round(window / dt)``。"""
        time = np.arange(500, dtype=float)
        values = np.sin(time / 5.0)
        result = evolutionary_rho1(time, values, window=100.0, n_bootstrap=0)
        assert result.rho.size == 500 - 100 + 1
        assert result.window_centres.size == result.rho.size

    def test_the_window_centres_are_the_midpoints_of_the_windows(self):
        """窗中心是窗内首末坐标的中点，因此等距且递增。"""
        time = np.arange(500, dtype=float)
        values = np.sin(time / 5.0)
        result = evolutionary_rho1(time, values, window=100.0, n_bootstrap=0)
        assert result.window_centres[0] == pytest.approx(49.5)
        assert result.window_centres[-1] == pytest.approx(449.5)
        assert np.all(np.diff(result.window_centres) > 0.0)

    def test_a_fixed_seed_reproduces_the_uncertainty(self):
        """同一个种子必须给出逐点相同的不确定度。"""
        time, values = _two_regime_record()
        first = evolutionary_rho1(time, values, window=WINDOW, random_seed=99)
        second = evolutionary_rho1(time, values, window=WINDOW, random_seed=99)
        assert np.array_equal(first.uncertainty, second.uncertainty)
        assert np.array_equal(first.rho, second.rho)

    def test_without_a_seed_the_result_is_not_reproducible_but_warns(self):
        """不给种子时结果不可复现，并且必须发出 ``RuntimeWarning``。"""
        time, values = _two_regime_record()
        with pytest.warns(RuntimeWarning):
            first = evolutionary_rho1(time, values, window=WINDOW)
        with pytest.warns(RuntimeWarning):
            second = evolutionary_rho1(time, values, window=WINDOW)
        assert not np.array_equal(first.uncertainty, second.uncertainty)

    def test_the_uncertainty_is_positive_and_scaled_with_the_window(self):
        """不确定度是抽样标准差：非负，且窗口越长越小。"""
        time, values = _two_regime_record()
        short = evolutionary_rho1(time, values, window=60.0, random_seed=5)
        long = evolutionary_rho1(time, values, window=240.0, random_seed=5)
        assert np.all(short.uncertainty > 0.0)
        assert float(np.mean(short.uncertainty)) > float(np.mean(long.uncertainty))

    def test_the_summary_reports_the_windows(self):
        """摘要必须给出窗数与 ρ1 的范围。"""
        time, values = _two_regime_record()
        result = evolutionary_rho1(time, values, window=WINDOW, random_seed=3)
        text = result.summary()
        assert str(result.rho.size) in text

    def test_a_straight_line_leaves_nothing_to_correlate(self):
        """一段完全线性的趋势，残差是空的，ρ1 必须是 0。

        去趋势之后窗内只剩下舍入噪声，于是分子分母都是舍入量，它们的比值
        在 ``-1`` 与 ``1`` 之间随机取值——那是噪声，不是自相关。Acycle 在
        同样的情形下给出的正是这种随机数（例如直线记录上实测到
        ``-0.31 .. +0.98``）。本模块在残差相对窗内散布可以忽略时返回 0，
        这是一个**有意**的差别，模块的 docstring 里写明了。
        """
        time = np.arange(400.0)
        values = np.linspace(0.0, 10.0, 400)
        result = evolutionary_rho1(time, values, window=40.0, n_bootstrap=0)
        assert np.all(result.rho == 0.0)

    def test_the_window_matches_acycles_own_formula(self):
        """与 Acycle 的公式逐点比对（机器精度）。

        下面的参考实现是照着 Acycle ``acycleEvolutionaryRho1.m`` 的每一步
        抄下来的（1 起始的样本序号基 ``sampleIndex-(K+1)/2``、同一能量项、
        同一 Mudelsee 分子分母约定），不是照着本模块抄的，所以两者一致
        是证据而不是同义反复。
        """
        time, values = _two_regime_record()
        result = evolutionary_rho1(time, values, window=WINDOW, n_bootstrap=0)

        window_points = int(WINDOW)
        sample_index = np.arange(1, window_points + 1, dtype=float)
        detrend_basis = sample_index - (window_points + 1) / 2.0
        basis_sum_squares = float(np.sum(detrend_basis**2))

        expected = np.empty(values.size - window_points + 1, dtype=float)
        for row in range(expected.size):
            window = values[row : row + window_points]
            centered = window - float(np.mean(window))
            slope = float(np.sum(detrend_basis * centered)) / basis_sum_squares
            detrended = centered - slope * detrend_basis
            z = detrended - float(np.mean(detrended))
            expected[row] = float(np.sum(z[1:] * z[:-1]) / float(np.sum(z[:-1] ** 2)))

        assert np.allclose(np.asarray(result.rho, dtype=float), expected, rtol=0.0, atol=1e-12)


class TestEvolutionaryRhoBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_a_window_shorter_than_three_points_raises(self):
        """``round(window / dt)`` 至少要有 3 个点。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=1.0)

    def test_a_window_longer_than_the_record_raises(self):
        """窗点数不能超过重采样后的点数。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=100_000.0)

    def test_a_non_positive_sampling_interval_raises(self):
        """步长必须是正的有限数。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=WINDOW, sampling_interval=0.0)
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=WINDOW, sampling_interval=-1.0)

    def test_a_sampling_interval_longer_than_the_record_raises(self):
        """步长不能超过记录跨度。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=WINDOW, sampling_interval=1.0e6)

    def test_mismatched_lengths_raise(self):
        """时间与数值必须等长。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values[:-1], window=WINDOW)

    def test_too_few_points_raise(self):
        """点数太少无法做去趋势与滞后 1 阶估计。"""
        with pytest.raises(ValidationError):
            evolutionary_rho1(np.arange(4.0), np.arange(4.0), window=2.0)

    def test_a_non_increasing_time_axis_raises(self):
        """时间轴必须严格递增（重复时间戳没有单一采样间隔）。"""
        time, values = _two_regime_record()
        broken = time.copy()
        broken[10] = broken[9]
        with pytest.raises(ValidationError):
            evolutionary_rho1(broken, values, window=WINDOW)

    def test_a_negative_bootstrap_count_raises(self):
        """自助次数不能为负。"""
        time, values = _two_regime_record()
        with pytest.raises(ValidationError):
            evolutionary_rho1(time, values, window=WINDOW, n_bootstrap=-1)
