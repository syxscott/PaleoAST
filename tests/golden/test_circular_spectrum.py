# =============================================================================
# FILE: tests/golden/test_circular_spectrum.py
# =============================================================================
"""
``stratigraphy.circular_spectrum``（圆周谱）的测试。

测试围绕三条主线组织：

* **主方向与集中度。** 构造"角度集中在 90° 附近、带噪声"的序列，主方向必须
  落在 90° 的容差范围内（±20°），合成长度 ``R1`` 必须明显大于 0.5。

* **循环性（最关键的一条）。** 角度是**循环变量**：350° 与 10° 只差 20°。
  把角度均匀铺在 350°–360° 与 0°–10° 两段上，主方向必须落在 0°/360°
  附近，而**不是 180°** —— 180° 正是"把角度当线性变量"会算出来的算术
  平均值。测试同时把那个线性平均值算出来钉在断言里，使得这条用例真正
  区分了"循环"与"线性"两种处理。

* **圆谐结构。** 一阶/二阶圆谐的合成长度与方向必须符合定义：轴向（双峰）
  数据的 ``R1`` 趋近 0 而 ``R2`` 趋近 1；均匀分布的数据没有任何主方向；
  各阶谐波功率等于合成长度的平方。

另外覆盖 Acycle "circular spectrum" 原义的**合成向量长度谱**：一串等间距
事件的合成向量长度必须在真周期处取到 1。

边界测试断言抛 ``utils.exceptions.ValidationError``；可复现性断言固定种子
给出完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge
  University Press.（圆谐、合成长度与主方向的经典定义）
* Mardia, K. V. & Jupp, P. E. (2000). Circular Statistics. Wiley.（圆谐功率、
  二阶谐波与双峰/轴向判别）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "circular spectrum" 子功能）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"circular_spectrum"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.circular_spectrum import circular_spectrum, circular_strength_spectrum
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成角度序列的样本数。
N_ANGLES = 400

#: 集中方向的合成角度序列所用的噪声标准差（度）。
DIRECTION_NOISE = 20.0

#: 集中方向的真值（度）。
TRUE_DIRECTION = 90.0

#: 主方向断言的容差（度）。
DIRECTION_TOLERANCE = 20.0

#: 跨零度分布的合成角度序列所用的角度宽度（度）。
WRAP_HALF_WIDTH = 10.0


# =============================================================================
# 辅助函数
# =============================================================================


def _concentrated_angles(
    direction: float = TRUE_DIRECTION,
    spread: float = DIRECTION_NOISE,
    seed: int = 4,
) -> npt.NDArray:
    """集中在 ``direction`` 附近、带高斯噪声的角度序列（度）。"""
    rng = np.random.default_rng(seed)
    return rng.normal(direction, spread, N_ANGLES)


def _wrapped_angles(seed: int = 2024) -> npt.NDArray:
    """均匀铺在 350°–360° 与 0°–10° 两段上的角度序列（度）。

    这批角度本质上是**集中在 0°** 的：算术平均值却是 180°，正是把角度当
    线性变量会犯的错。
    """
    rng = np.random.default_rng(seed)
    lower = rng.uniform(360.0 - WRAP_HALF_WIDTH, 360.0, N_ANGLES // 2)
    upper = rng.uniform(0.0, WRAP_HALF_WIDTH, N_ANGLES // 2)
    return np.concatenate([lower, upper])


def _angular_difference(first: float, second: float) -> float:
    """两个方向（度）之间的最小夹角（度，``0..180``）。"""
    difference = abs(float(first) - float(second)) % 360.0
    return min(difference, 360.0 - difference)


# =============================================================================
# 圆谐：主方向与集中度
# =============================================================================


class TestCircularHarmonics:
    """圆谐必须给出正确的主方向与集中度。"""

    def test_concentrated_angles_recover_the_mean_direction(self):
        """集中在 90° 附近的序列：主方向必须接近 90°（±20°）。"""
        angles = _concentrated_angles()
        result = circular_spectrum(angles)

        assert _angular_difference(result.mean_direction, TRUE_DIRECTION) <= DIRECTION_TOLERANCE, (
            f"the mean direction {result.mean_direction:.2f} degrees is far from {TRUE_DIRECTION}"
        )

    def test_concentrated_angles_have_a_large_resultant_length(self):
        """集中在 90° 附近的序列：合成长度必须明显大于 0.5。"""
        angles = _concentrated_angles()
        result = circular_spectrum(angles)

        assert result.resultant_length > 0.5, (
            f"the resultant length {result.resultant_length:.4f} is too small for concentrated angles"
        )
        assert result.resultant_length <= 1.0

    def test_the_circularity_of_the_angle_scale_is_respected(self):
        """跨零度的分布：主方向必须落在 0°/360° 附近，而不是 180°。

        这条用例是本文件的核心：350° 与 10° 只差 20°，把它们当线性变量求
        平均会得到 180°。测试把那个线性平均值一并算出来并钉在断言里。
        """
        angles = _wrapped_angles()
        result = circular_spectrum(angles)

        assert _angular_difference(result.mean_direction, 0.0) <= DIRECTION_TOLERANCE, (
            f"the mean direction {result.mean_direction:.2f} degrees is not close to 0/360 degrees"
        )
        # 对照：把角度当线性变量的算术平均恰好落在 180° 附近。
        linear_mean = float(np.mean(angles))
        assert _angular_difference(linear_mean, 180.0) <= DIRECTION_TOLERANCE, (
            "the linear mean no longer reproduces the 180-degree trap; the test data changed"
        )

    def test_axial_data_is_caught_by_the_second_harmonic(self):
        """轴向（双峰）数据：一阶谐波消失，二阶谐波占主导。

        数据全部落在 0° 与 180° 上：一阶合成长度趋近 0（正负抵消），二阶
        合成长度趋近 1（``2 * 180° = 360°`` 是整圈）。
        """
        angles = np.array([0.0, 180.0] * (N_ANGLES // 2))
        result = circular_spectrum(angles)

        assert result.resultant_length < 0.05, (
            f"the first harmonic of axial data must vanish, got {result.resultant_length:.4f}"
        )
        assert float(np.sqrt(result.power[1])) > 0.95, "the second harmonic must dominate axial data"

    def test_uniform_angles_have_no_preferred_direction(self):
        """均匀分布的角度没有任何主方向，合成长度趋近 0。"""
        angles = np.linspace(0.0, 360.0, 361)[:-1]
        result = circular_spectrum(angles)

        assert result.resultant_length < 0.01, (
            f"uniform angles must have no concentration, got {result.resultant_length:.4f}"
        )

    def test_the_harmonic_spectrum_has_the_expected_shape(self):
        """频率轴是谐波数，功率是各阶合成长度的平方，且落在 ``[0, 1]``。"""
        angles = _concentrated_angles()
        result = circular_spectrum(angles, n_harmonics=5)

        assert result.frequencies.shape == (5,)
        assert result.power.shape == (5,)
        assert np.allclose(result.frequencies, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.all(result.power >= 0.0)
        assert np.all(result.power <= 1.0)
        # 一阶谐波必须解释最多的圆方差。
        assert float(result.power[0]) == pytest.approx(result.resultant_length**2, rel=1e-12)

    def test_results_are_reproducible(self):
        """同一输入必须给出完全相同的结果。"""
        angles = _concentrated_angles()
        one = circular_spectrum(angles)
        two = circular_spectrum(angles)

        assert np.array_equal(one.power, two.power)
        assert one.mean_direction == two.mean_direction
        assert one.resultant_length == two.resultant_length


# =============================================================================
# 合成向量长度谱（Acycle 原义）
# =============================================================================


class TestCircularStrengthSpectrum:
    """周期栅格上的合成向量长度必须在真周期处达到 1。"""

    def test_strength_peaks_at_the_true_period(self):
        """等间距事件：真周期处的合成长度必须是 1，且是唯一的最大值。

        周期栅格刻意取 ``(3.0, 20.0)`` 共 35 个点，使真周期 5.0 正好落在
        栅格上（否则栅格上离 5.0 最近的两点各差一点，合成长度就达不到 1）。
        """
        events = np.arange(0.0, 100.0, 5.0)
        result = circular_strength_spectrum(events, period_range=(3.0, 20.0), n_periods=35)

        peak = int(np.argmax(result.strength))
        assert result.periods[peak] == pytest.approx(5.0)
        assert result.strength[peak] == pytest.approx(1.0, abs=1e-9)
        assert int(np.sum(result.strength > 0.99)) == 1, "the peak must be unique"
        assert float(np.sort(result.strength)[-2]) < 0.5, "the background strength must be low"

    def test_phase_coordinates_stay_inside_the_period(self):
        """折回相位坐标必须落在 ``[0, p)``。"""
        events = np.arange(0.0, 100.0, 5.0)
        result = circular_strength_spectrum(events, period_range=(3.0, 20.0), n_periods=35)

        assert np.all(result.phase_coordinates >= 0.0)
        assert np.all(result.phase_coordinates < result.periods)
        assert np.all(result.strength >= 0.0)
        assert np.all(result.strength <= 1.0)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_angles_are_rejected(self):
        with pytest.raises(ValidationError):
            circular_spectrum(np.array([90.0]))

    def test_non_finite_angles_are_rejected(self):
        angles = _concentrated_angles()
        angles[5] = np.inf
        with pytest.raises(ValidationError):
            circular_spectrum(angles)

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValidationError):
            circular_spectrum(np.zeros((4, 2)))

    @pytest.mark.parametrize("n_harmonics", [0, -1, 1000])
    def test_invalid_harmonic_counts_are_rejected(self, n_harmonics: int):
        angles = _concentrated_angles()
        with pytest.raises(ValidationError):
            circular_spectrum(angles, n_harmonics=n_harmonics)

    @pytest.mark.parametrize(
        "period_range",
        [(0.0, 20.0), (-5.0, 20.0), (20.0, 20.0), (20.0, 5.0)],
    )
    def test_invalid_period_ranges_are_rejected(self, period_range: tuple[float, float]):
        events = np.arange(0.0, 100.0, 5.0)
        with pytest.raises(ValidationError):
            circular_strength_spectrum(events, period_range=period_range)

    def test_duplicate_event_coordinates_are_rejected(self):
        events = np.array([0.0, 5.0, 5.0, 10.0])
        with pytest.raises(ValidationError):
            circular_strength_spectrum(events, period_range=(2.0, 20.0))

    def test_unknown_grid_mode_is_rejected(self):
        events = np.arange(0.0, 100.0, 5.0)
        with pytest.raises(ValidationError):
            circular_strength_spectrum(events, period_range=(2.0, 20.0), grid="quadratic")
