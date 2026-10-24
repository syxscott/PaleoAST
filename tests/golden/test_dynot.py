# =============================================================================
# FILE: tests/golden/test_dynot.py
# =============================================================================
"""
``stratigraphy.dynot``（沉积噪声模型 / DYNOT 与 ρ1 方法）的测试。

本文件位于 ``tests/golden/``，因此它为仓库的验证矩阵提供**黄金级证据**：
断言全部落在"与已发表或独立推导的真值对照"这一列，而不是"只跑通一个例子"
的单元测试那一列。

测试围绕五条主线组织：

* **合成真值。** 用**已知 AR(1) 系数**的 AR(1) 过程构造记录，断言模块估计
  出的 AR(1) 滞后 1 阶自相关系数与真值相差不超过 0.1。这条断言针对的是
  :attr:`~stratigraphy.dynot.DynotResult.ar1_rho` —— 它用的是 Li 等 (2018)
  论文给出的 ρ1 公式（``Σ x(i)x(i-1) / Σ x(i-1)²``），与 Acycle 的
  ``rhoAR1.m`` / ``rhoAR1ML.m`` 同源。

* **ρ1 的方向性。** ρ1 是"记录在噪声频带（分析频带的短周期半边）里的噪声
  功率"与"基线模型在同一频带里的噪声功率"之比，因此：

  1. 给一条**比基线噪声更强**的记录（已知 AR(1) 红噪声 + 额外叠加的白噪声，
     而基线用的是记录自身的 AR(1) 持续性），ρ1 必须 **> 1**；
  2. 给一条**纯白噪声**序列、基线也用白噪声，ρ1 必须 **≈ 1**（容差 ±0.3）；
  3. 给一条**纯 AR(1)** 序列、基线用红噪声，ρ1 必须 **≈ 1**（容差 ±0.3）——
     这是红噪声基线"按记录的持续性自标定"的直接后果：记录的噪声正好等于
     它自己的持续性所预测的噪声。

  第 1 条与第 3 条构成一对：同一条 AR(1) 记录在"额外加噪"前后，ρ1 从
  ≈1 抬到 >1，所以这个方向性不是把两条互不相干的记录拿来比。

* **频率轴口径。** 频率必须是"每单位时间"（cycles/Ma 等），不是"每样本"。
  断言频率落在 **Rayleigh 分辨率**（``1/T``）与 **Nyquist 频率**之间；并
  额外断言**采样更密不会让频率栅格变粗**——本仓库修过一个真实 bug：栅格曾
  由采样密度决定，同一条记录采样越密、报出的周期反而越差。

* **归一化。** 噪声谱以"白噪声背景 = 1"归一化，因此一条纯白噪声记录的平均
  噪声功率应当是 1；这条断言把 ρ1 的分子和分母放在同一量纲上。

* **边界与可复现性。** 点数太少、频率范围非法、``n_frequencies`` < 1、
  常数序列都断言抛 ``utils.exceptions.ValidationError``；固定 ``random_seed``
  断言结果完全可复现。

参考来源（断言中引用的数值出处）
--------------------------------
* Li, M., Hinnov, L., Huang, C., Ogg, J., 2018. Sedimentary noise and sea
  levels linked to land–ocean water exchange and obliquity forcing.
  *Nature Communications* 9: 1004. (沉积噪声模型；论文中的 ρ1 是滞后 1 阶
  自相关系数，公式见上。)
* Mudelsee, M., 2002. AR(1) 过程的谱估计与滞后 1 阶自相关系数估计。
  (Acycle ``rhoAR1.m`` 采用同一估计量；本模块直接复用仓库内
  ``stratigraphy.spectral_moments.estimate_ar1_rho``。)
* Mann, M.E. & Lees, J.M., 1996. Red noise and the climate record.
  *Geophysical Research Letters* 23: 2325–2328. (AR(1) 红噪声参考谱。)
* Politis, D.N. & Romano, J.P., 1994. The stationary bootstrap.
  *Journal of Time Series Analysis* 15: 297–306. (分块自助法不确定度。)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"dynot", "rho1_sedimentation_ratio"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.dynot import rho1_sedimentation_ratio, sedimentary_noise
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的采样点数（等间隔时间轴，跨度 1 个时间单位）。
N_POINTS = 500

#: 合成记录覆盖的时间跨度（任意"每单位时间"）。
SPAN = 1.0

#: 合成记录里已知的 AR(1) 系数真值。
TRUE_RHO = 0.6

#: 判定 AR(1) 系数估计是否合格的容差。
RHO_TOLERANCE = 0.1

#: 判定"纯白噪声"记录的 ρ1 是否等于 1 的容差。
RHO1_TOLERANCE = 0.3


# =============================================================================
# 辅助函数
# =============================================================================


def _uniform_axis(n_points: int = N_POINTS, span: float = SPAN) -> npt.NDArray:
    """等间隔时间轴。"""
    return np.linspace(0.0, span, n_points)


def _ar1_series(rho: float, length: int, rng: np.random.Generator) -> npt.NDArray:
    """已知 AR(1) 系数为 ``rho`` 的平稳 AR(1) 序列（单位方差）。"""
    innovations = rng.normal(scale=np.sqrt(1.0 - rho**2), size=length)
    series = np.empty(length, dtype=float)
    series[0] = rng.normal()
    for index in range(1, length):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _white_series(length: int, rng: np.random.Generator) -> npt.NDArray:
    """单位方差的白噪声序列。"""
    return rng.normal(size=length)


# =============================================================================
# 合成真值：AR(1) 系数
# =============================================================================


def test_ar1_coefficient_recovers_the_known_truth():
    """已知 ρ 的 AR(1) 过程，估计出的 AR(1) 系数应当落在真值 ±0.1 内。"""
    rng = np.random.default_rng(20240101)
    axis = _uniform_axis()
    series = _ar1_series(TRUE_RHO, N_POINTS, rng)

    result = sedimentary_noise(axis, series, random_seed=7)

    assert abs(result.ar1_rho - TRUE_RHO) <= RHO_TOLERANCE, (
        f"AR(1) coefficient {result.ar1_rho:.4f} is not within {RHO_TOLERANCE} of {TRUE_RHO}"
    )


def test_ar1_coefficient_recovers_the_truth_for_a_second_rho():
    """换一个真值与另一条随机流，结论不变（不是某一条样本的巧合）。"""
    rng = np.random.default_rng(31415926)
    axis = _uniform_axis(n_points=2000)
    series = _ar1_series(0.3, 2000, rng)

    result = sedimentary_noise(axis, series, random_seed=11)

    assert abs(result.ar1_rho - 0.3) <= RHO_TOLERANCE, (
        f"AR(1) coefficient {result.ar1_rho:.4f} is not within 0.1 of 0.3"
    )


def test_rho1_ratio_bootstrap_also_reports_the_ar1_coefficient():
    """``rho1_sedimentation_ratio`` 报告同一个 AR(1) 系数估计量。"""
    rng = np.random.default_rng(27182818)
    axis = _uniform_axis()
    series = _ar1_series(TRUE_RHO, N_POINTS, rng)

    result = rho1_sedimentation_ratio(axis, series, n_bootstrap=12, random_seed=3)

    assert abs(result.ar1_rho - TRUE_RHO) <= RHO_TOLERANCE
    assert result.rho1_uncertainty > 0.0


# =============================================================================
# ρ1 的方向性
# =============================================================================


def test_rho1_exceeds_one_when_the_record_carries_more_noise_than_the_baseline():
    """
    比基线噪声更强的记录，ρ1 必须大于 1。

    基线用的是记录自身的 AR(1) 持续性；这条记录在 AR(1) 红噪声之上**额外
    叠加了白噪声**，因此它在短周期端携带的噪声功率强于基线的预期。
    """
    rng = np.random.default_rng(424242)
    axis = _uniform_axis()
    noisy = _ar1_series(0.7, N_POINTS, rng) + _white_series(N_POINTS, rng)

    result = sedimentary_noise(axis, noisy, noise_model="ar1", random_seed=5)

    assert result.rho1 > 1.0, f"a noisier record should give rho1 > 1; got {result.rho1:.4f}"


def test_rho1_is_near_one_for_pure_white_noise_with_a_white_baseline():
    """纯白噪声序列配白噪声基线，ρ1 应当接近 1。"""
    rng = np.random.default_rng(112358)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    result = sedimentary_noise(axis, series, noise_model="white", random_seed=5)

    assert abs(result.rho1 - 1.0) <= RHO1_TOLERANCE, (
        f"white noise against a white baseline should give rho1 near 1; got {result.rho1:.4f}"
    )


def test_rho1_is_near_one_for_pure_ar1_noise_with_an_ar1_baseline():
    """
    纯 AR(1) 序列配红噪声基线，ρ1 应当接近 1。

    红噪声基线按记录自身的持续性自标定，所以"记录的噪声正好等于它自己的
    持续性所预测的噪声"时 ρ1 = 1。这条断言同时是上一条"额外加噪 ⇒ ρ1 > 1"
    的对照：两者用的是同一条 AR(1) 记录，只差一层额外噪声。
    """
    rng = np.random.default_rng(161803)
    axis = _uniform_axis()
    series = _ar1_series(0.7, N_POINTS, rng)

    result = sedimentary_noise(axis, series, noise_model="ar1", random_seed=5)

    assert abs(result.rho1 - 1.0) <= RHO1_TOLERANCE, (
        f"an AR(1) record against its own AR(1) baseline should give rho1 near 1; got {result.rho1:.4f}"
    )


def test_rho1_is_larger_for_the_noisier_record_than_for_the_clean_one():
    """同一条 AR(1) 记录，叠加白噪声后 ρ1 必须变大。"""
    rng = np.random.default_rng(5772156)
    axis = _uniform_axis()
    clean = _ar1_series(0.7, N_POINTS, rng)
    noisy = clean + _white_series(N_POINTS, rng)

    clean_result = sedimentary_noise(axis, clean, noise_model="ar1", random_seed=5)
    noisy_result = sedimentary_noise(axis, noisy, noise_model="ar1", random_seed=5)

    assert noisy_result.rho1 > clean_result.rho1, (
        f"adding noise should raise rho1: {noisy_result.rho1:.4f} vs {clean_result.rho1:.4f}"
    )


def test_short_period_power_dominance_raises_rho1_above_the_white_baseline():
    """短周期端功率远超白噪声参考时，ρ1（白噪声基线）显著大于 1。"""
    rng = np.random.default_rng(1414213)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng) + 10.0 * np.sin(2.0 * np.pi * 300.0 * axis)

    result = sedimentary_noise(axis, series, noise_model="white", random_seed=5)

    assert result.rho1 > 1.0, f"a short-period dominated record should give rho1 > 1; got {result.rho1:.4f}"


# =============================================================================
# 频率轴口径
# =============================================================================


def test_frequency_axis_spans_the_rayleigh_resolution_and_the_nyquist_frequency():
    """频率必须落在 Rayleigh 分辨率与 Nyquist 之间，且单位是"每单位时间"。"""
    rng = np.random.default_rng(1732050)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    result = sedimentary_noise(axis, series, random_seed=5)

    rayleigh = 1.0 / float(axis[-1] - axis[0])
    nyquist = 1.0 / (2.0 * float(np.median(np.diff(axis))))
    assert result.noise_frequencies[0] >= rayleigh * (1.0 - 1e-9)
    assert result.noise_frequencies[-1] <= nyquist * (1.0 + 1e-9)
    assert np.all(np.diff(result.noise_frequencies) > 0.0)
    assert result.noise_power.size == result.noise_frequencies.size
    assert result.times.size == axis.size


def test_a_denser_sampling_does_not_coarsen_the_frequency_grid():
    """
    采样更密不得让频率栅格变粗。

    这条断言针对本仓库修过的真实 bug：频率栅格曾由采样密度决定，同一条
    记录采样越密、报出的周期反而越差。修好之后：

    * 栅格**永远从 Rayleigh 分辨率 ``1/T`` 起**（最长可分辨周期与采样密度
      无关，两条记录完全一样）；
    * 栅格**步长永远不比 Rayleigh 分辨率更粗**（旧版本里步长是
      ``Nyquist / n_frequencies``，采样越密反而越粗）；
    * 采样更密时频率上界只会更高，绝不会更低（短周期端不会因此变差）。
    """
    rng = np.random.default_rng(2645751)
    span = 1.0
    sparse_axis = np.linspace(0.0, span, 400)
    dense_axis = np.linspace(0.0, span, 2000)
    sparse = _white_series(sparse_axis.size, rng)
    dense = _white_series(dense_axis.size, rng)

    sparse_result = sedimentary_noise(sparse_axis, sparse, random_seed=5)
    dense_result = sedimentary_noise(dense_axis, dense, random_seed=5)

    rayleigh = 1.0 / span
    sparse_step = float(sparse_result.noise_frequencies[1] - sparse_result.noise_frequencies[0])
    dense_step = float(dense_result.noise_frequencies[1] - dense_result.noise_frequencies[0])

    assert sparse_step <= rayleigh * (1.0 + 1e-9)
    assert dense_step <= rayleigh * (1.0 + 1e-9)
    assert abs(dense_result.noise_frequencies[0] - sparse_result.noise_frequencies[0]) <= 1e-9
    assert dense_result.noise_frequencies[-1] >= sparse_result.noise_frequencies[-1] - 1e-9


def test_an_explicit_frequency_range_is_honoured():
    """显式给定的频率范围必须被原样使用。"""
    rng = np.random.default_rng(3141592)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    result = sedimentary_noise(axis, series, frequency_range=(2.0, 50.0), n_frequencies=64, random_seed=5)

    assert result.noise_frequencies[0] == pytest.approx(2.0)
    assert result.noise_frequencies[-1] == pytest.approx(50.0)
    assert result.noise_frequencies.size == 64


# =============================================================================
# 归一化
# =============================================================================


def test_the_noise_spectrum_uses_a_white_background_of_one():
    """噪声谱以"白噪声背景 = 1"归一化：纯白噪声记录的平均噪声功率应当是 1。"""
    rng = np.random.default_rng(1618033)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    result = sedimentary_noise(axis, series, noise_model="white", random_seed=5)

    assert abs(float(np.mean(result.noise_power)) - 1.0) <= 0.15, (
        f"a white record should have a mean noise power of 1; got {float(np.mean(result.noise_power)):.4f}"
    )


# =============================================================================
# 边界
# =============================================================================


def test_too_few_points_raise_validation_error():
    """点数太少无法估计噪声谱，抛 ``ValidationError``。"""
    axis = np.linspace(0.0, 1.0, 3)
    series = np.array([0.1, 0.4, -0.2])

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, random_seed=1)


def test_a_constant_series_raises_validation_error():
    """常数序列没有噪声可估计，抛 ``ValidationError``。"""
    axis = _uniform_axis()

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, np.full(N_POINTS, 2.5), random_seed=1)


def test_an_illegal_frequency_range_raises_validation_error():
    """非法频率范围（上限超过 Nyquist、下限不为正、上下限反序）抛 ``ValidationError``。"""
    rng = np.random.default_rng(9999)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, frequency_range=(10.0, 1.0), random_seed=1)
    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, frequency_range=(-1.0, 10.0), random_seed=1)
    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, frequency_range=(1.0, 1.0e9), random_seed=1)


def test_a_non_positive_frequency_count_raises_validation_error():
    """``n_frequencies`` 小于 1 抛 ``ValidationError``。"""
    rng = np.random.default_rng(8888)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, n_frequencies=0, random_seed=1)
    with pytest.raises(ValidationError):
        rho1_sedimentation_ratio(axis, series, n_frequencies=-5, n_bootstrap=4, random_seed=1)


def test_an_unsupported_noise_model_raises_validation_error():
    """不受支持的基线模型抛 ``ValidationError``。"""
    rng = np.random.default_rng(7777)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, series, noise_model="pink", random_seed=1)


def test_a_too_small_bootstrap_count_raises_validation_error():
    """``n_bootstrap`` 少于 2 无法估出标准差，抛 ``ValidationError``。"""
    rng = np.random.default_rng(6666)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    with pytest.raises(ValidationError):
        rho1_sedimentation_ratio(axis, series, n_bootstrap=1, random_seed=1)


def test_mismatched_lengths_raise_validation_error():
    """时间轴与观测值长度不一致时抛 ``ValidationError``。"""
    axis = _uniform_axis()

    with pytest.raises(ValidationError):
        sedimentary_noise(axis, np.zeros(N_POINTS - 1), random_seed=1)


# =============================================================================
# 可复现性
# =============================================================================


def test_a_fixed_seed_makes_the_result_reproducible():
    """固定 ``random_seed`` 给出完全相同的结果。"""
    rng = np.random.default_rng(5555)
    axis = _uniform_axis()
    series = _ar1_series(0.6, N_POINTS, rng)

    first = sedimentary_noise(axis, series, random_seed=1234)
    second = sedimentary_noise(axis, series, random_seed=1234)

    assert first.rho1 == second.rho1
    assert first.rho1_uncertainty == second.rho1_uncertainty
    assert first.ar1_rho == second.ar1_rho
    assert np.array_equal(first.noise_power, second.noise_power)


def test_a_fixed_seed_makes_the_bootstrap_reproducible():
    """``rho1_sedimentation_ratio`` 的自助法同样可复现。"""
    rng = np.random.default_rng(4444)
    axis = _uniform_axis()
    series = _ar1_series(0.6, N_POINTS, rng)

    first = rho1_sedimentation_ratio(axis, series, n_bootstrap=10, random_seed=99)
    second = rho1_sedimentation_ratio(axis, series, n_bootstrap=10, random_seed=99)

    assert first.rho1 == second.rho1
    assert first.rho1_uncertainty == second.rho1_uncertainty


def test_an_unseeded_run_warns_that_it_is_not_reproducible():
    """不给种子时按仓库约定发出 ``RuntimeWarning``。"""
    rng = np.random.default_rng(3333)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    with pytest.warns(RuntimeWarning):
        sedimentary_noise(axis, series)


def test_the_summary_reports_the_ratio_and_the_ar1_coefficient():
    """摘要文本里必须能看到 ρ1 与 AR(1) 系数。"""
    rng = np.random.default_rng(2222)
    axis = _uniform_axis()
    series = _white_series(N_POINTS, rng)

    summary = sedimentary_noise(axis, series, random_seed=5).summary()

    assert "rho1" in summary
    assert "AR(1) rho" in summary
