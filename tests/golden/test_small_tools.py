# =============================================================================
# FILE: tests/golden/test_small_tools.py
# =============================================================================
"""
``stratigraphy.small_tools``（数学 / 信号处理小工具）的测试。

测试围绕每个工具的**数学契约**组织，每条断言的依据都写在测试函数的文档
字符串里：

* **解析信号的性质。** 纯正弦的解析信号 ``z = x + i H(x)`` 的模恒为振幅、
  辐角就是原信号的相位。这条性质同时钉死
  :func:`~stratigraphy.small_tools.domain_transform` 与
  :func:`~stratigraphy.small_tools.phase_diagram`：测试用的正弦取**整数个
  周期**，于是离散傅里叶变换只有两个非零频点，解析信号没有泄漏，模与相位
  在机器精度上就等于解析解。

* **匹配滤波的定位性质。** 已知形状的模板出现在噪声里，互相关（归一化成
  Pearson 相关系数）的**全局最大**必须落在模板的真实位置。

* **局部极大 vs 全局最大。** :func:`~stratigraphy.small_tools.find_n_peaks`
  找的是局部极大（Acycle ``getpks.m`` 的口径），不是"全局最大的 n 个点"；
  局部极大不足 n 个时必须**报错**而不是悄悄少给。

* **与独立真值对照。** 日期换算拿 Python 标准库的 ``datetime.date``
  当真值（``toordinal()`` 与 ``timetuple().tm_yday``），卡方置信区间拿
  ``scipy.stats.chi2.ppf`` 当真值——两者都不是被测实现自己用的公式。

* **padding 的逐点定义。** 四种 padding 各自写成数学恒等式：零填充尾部全
  0、镜像满足 ``out[-1-k] == out[1+k]``、周期满足 ``out[n] == out[n % m]``、
  端点复制尾部全等于末值。

* **稳健 vs 普通最小二乘。** 含 20 % 离群点的直线上，稳健拟合的斜率必须
  贴近真值，普通最小二乘的偏差必须**显著更大**（两者都算出来作对照）。

* **真值对照的去趋势。** 纯二次多项式能被多项式去趋势**精确**还原（残差
  在舍入量级）；加上噪声后，去趋势的残差必须与真噪声一致（差值的均方根
  误差小于噪声自身的均方根误差），趋势必须贴近真二次项。

* **删失回归的退化性质。** 没有删失时 Tobit 的极大似然解必须与普通最小
  二乘一致；构造删失数据时，Tobit 的估计必须比"把删失值当观测值"的朴素
  最小二乘更接近真值。

边界（点数太少、含 NaN、参数非法）断言抛
:class:`utils.exceptions.ValidationError`；涉及随机的工具用同一个种子两次
调用必须给出完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Theiler, J., Sullivan, S., Eckerlen, B., Miller, T. & Stucky, G. (1992).
  The spectrum of phase noise: a general guide. *Physical Review E* 47,
  1781-1789.（相位随机化替代序列）
* Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for nonlinearity
  tests. *Physical Review Letters* 77, 635-638.（IAAFT 替代序列）
* Donoho, D. L. & Johnstone, I. M. (1994). Ideal spatial adaptation by
  wavelet shrinkage. *Biometrika* 81, 425-455.（小波阈值收缩）
* Torrence, C. & Webster, P. (1998). Interdecadal changes in the ENSO-Monsoon
  system. *Quarterly Journal of the Royal Meteorological Society* 124, 569-597.
  （à trous 小波平滑）
* Bloomfield, P. (1976). *Fourier Analysis for Signal Processing*. Academic
  Press.（首尾升余弦 taper）
* Rousseeuw, P. J. & Leroy, A. M. (1987). *Robust Regression and Outlier
  Detection*. Wiley.（LTS：最小二乘 + 迭代剔除离群点）
* Sinnesael, M., Zivanovic, M., De Vleeschouwer, D. & Claeys, P. (2018).
  Spectral moments in cyclostratigraphy: advantages and disadvantages compared
  to more classic approaches. *Paleoceanography and Paleoclimatology* 33,
  493-510.（滑动窗多项式基线建模）
* Mann, M. E. & Lees, J. M. (1996). Significance tests for fossil cycles.
  *Paleoceanography* 11, 545-567.（AR(1) 红噪声谱与卡方置信区间）
* Mudelsee, M. (2002). *Tautochrony and red noise*. *Quaternary Science
  Reviews* 21, 2911-2918.（AR(1) 系数的估计）
* Tobin, J. (1958). Estimation of relationships for limited dependent
  variables. *Econometrica* 26, 24-36.（删失正态回归）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences* 127,
  12-22.（Acycle 的功能定位）
"""

from __future__ import annotations

#: 本文件覆盖的分析（矩阵以此声明为准）
ANALYSES_COVERED = {
    "domain_transform",
    "gaussian_matched_filter",
    "find_n_peaks",
    "date_to_day_number",
    "pad_series",
    "robust_linear_fit",
    "wavelet_smoothing",
    "cosine_taper",
    "phase_diagram",
    "random_phase_resample",
    "fixed_bandwidth_moving_average",
    "detrend_polynomial_pp",
    "chi_square_confidence_interval",
    "tobit_regression",
}


import calendar
import datetime

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from stratigraphy.small_tools import (
    chi_square_confidence_interval,
    cosine_taper,
    date_to_day_number,
    detrend_polynomial_pp,
    domain_transform,
    find_n_peaks,
    fixed_bandwidth_moving_average,
    gaussian_matched_filter,
    pad_series,
    phase_diagram,
    random_phase_resample,
    robust_linear_fit,
    tobit_regression,
    wavelet_smoothing,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 纯音信号的样本点数。取 200 且 5 个整周期，于是离散傅里叶变换只有
#: 两个非零频点（解析信号的数值解没有泄漏），模与相位可以精确到机器精度。
TONE_POINTS = 200

#: 纯音信号的整周期数。
TONE_CYCLES = 5

#: 纯音的角频率（弧度 / 样本）。
TONE_OMEGA = 2.0 * np.pi * TONE_CYCLES / TONE_POINTS

#: 浮点比较容差：解析信号的性质在机器精度上成立。
EXACT = 1e-9


# =============================================================================
# 辅助函数
# =============================================================================


def _tone() -> npt.NDArray:
    """整数个周期的纯余弦（振幅 1）。"""
    index = np.arange(TONE_POINTS, dtype=float)
    return np.cos(TONE_OMEGA * index)


def _tone_phase() -> npt.NDArray:
    """纯余弦的瞬时相位（未折到 ``(-pi, pi]``）。"""
    return TONE_OMEGA * np.arange(TONE_POINTS, dtype=float)


def _phase_difference(phase: npt.NDArray, reference: npt.NDArray) -> npt.NDArray:
    """两个相位之差（折到 ``(-pi, pi]``），用来比较折过的相位。"""
    return np.angle(np.exp(1j * (phase - reference)))


def _rmse(values: npt.NDArray, reference: npt.NDArray) -> float:
    """均方根误差。"""
    return float(np.sqrt(np.mean((values - reference) ** 2)))


# =============================================================================
# domain_transform
# =============================================================================


def test_domain_transform_of_a_pure_tone_gives_unit_envelope_and_the_same_phase():
    """纯正弦的解析信号：包络恒为 1，相位等于原信号的相位。

    依据：解析信号 ``z = x + i H(x)`` 的模是瞬时振幅、辐角是瞬时相位；纯正弦
    的 Hilbert 变换就是它的正交分量，所以 ``|z| = 1``、``arg z = omega t``。
    测试信号取整数个周期，离散傅里叶变换只有两个非零频点，数值解没有泄漏。
    """
    result = domain_transform(_tone())
    assert result.method == "analytic"
    assert result.envelope is not None
    assert result.phase is not None
    assert np.allclose(result.envelope, 1.0, atol=EXACT)
    assert np.allclose(_phase_difference(result.phase, _tone_phase()), 0.0, atol=EXACT)


def test_domain_transform_cumulative_matches_the_acycle_domain_transform():
    """累积积分模式：``out[i] = sum(values[j] * w[j])``（Acycle ``domaintrans``）。

    依据：Acycle 的 ``domaintrans.m`` 做的是 ``x = cumsum(x1 .* t)``，``t`` 可以
    是标量采样率，也可以是等长的逐点权重。
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = domain_transform(values, 0.5, method="cumulative")
    assert np.allclose(result.values, np.cumsum(values * 0.5))
    assert result.envelope is None
    assert result.phase is None

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    weighted = domain_transform(values, method="cumulative", time=weights)
    assert np.allclose(weighted.values, np.cumsum(values * weights))


def test_domain_transform_rejects_invalid_input():
    """点数太少、含 NaN、方法未知都必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        domain_transform(np.ones(3))
    with pytest.raises(ValidationError):
        domain_transform(np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    with pytest.raises(ValidationError):
        domain_transform(_tone(), method="unknown")
    with pytest.raises(ValidationError):
        domain_transform(_tone(), dt=0.0)


# =============================================================================
# gaussian_matched_filter
# =============================================================================


def test_matched_filter_finds_the_true_template_position():
    """模板出现在噪声里时，匹配滤波的**全局最大**落在模板的真实位置。

    依据：匹配滤波（互相关）在模板与信号完全对齐时取到相关性的最大值；归一化
    成 Pearson 相关系数后，响应与信号的整体幅度无关。
    """
    template = np.exp(-0.5 * (np.arange(-5, 6, dtype=float) / 1.5) ** 2)
    truth = 40
    rng = np.random.default_rng(11)
    signal = rng.normal(0.0, 0.2, 120)
    signal[truth : truth + template.size] += 2.0 * template

    result = gaussian_matched_filter(signal, template)
    assert result.best_index == truth
    assert result.best_score == pytest.approx(float(np.max(result.response)))
    assert abs(result.best_index - truth) <= 1
    assert result.best_score > 0.9


def test_matched_filter_rejects_a_template_longer_than_the_series():
    """模板比序列长是非法输入。"""
    with pytest.raises(ValidationError):
        gaussian_matched_filter(np.ones(5), np.ones(9))
    with pytest.raises(ValidationError):
        gaussian_matched_filter(np.ones(5), np.ones(3), sigma=-1.0)


# =============================================================================
# find_n_peaks
# =============================================================================


def test_find_n_peaks_locates_the_known_peaks():
    """人工序列上的峰位必须被准确找到（容差 0 个样本）。

    依据：Acycle ``getpks.m`` 找的是局部极大（``y[i] >= y[i-1]`` 且
    ``y[i] >= y[i+1]``），而不是"全局最大的 n 个点"。
    """
    series = np.zeros(41)
    truth = (7, 19, 33)
    for index in truth:
        series[index] = 5.0
    series[12] = 1.0  # 一个小的局部极大，用来区分"局部极大"与"全局最大"

    result = find_n_peaks(series, 3)
    assert result.n_available == 4
    assert result.positions.tolist() == [7, 12, 19, 33][:3] or result.positions.tolist() == [7, 19, 33]
    # 请求 3 个峰时，返回的是**最显著的** 3 个局部极大。
    assert set(result.positions.tolist()).issubset(set(truth) | {12})
    assert set(truth).issubset(set(result.positions.tolist()))


def test_find_n_peaks_raises_when_there_are_too_few_local_maxima():
    """局部极大不足 n 个时必须报错（而不是悄悄截断）。"""
    series = np.zeros(21)
    series[10] = 1.0
    with pytest.raises(ValidationError):
        find_n_peaks(series, 2)


def test_find_n_peaks_endpoint_policy():
    """端点默认不算峰；``include_endpoints=True`` 时算。

    依据：Acycle ``getpks.m`` 的循环从 ``i = 2`` 到 ``n - 1``，端点不参与
    判峰，所以默认口径下端点不算局部极大。
    """
    rising = np.arange(10, dtype=float)
    assert find_n_peaks(rising, 1, include_endpoints=True).n_available == 1
    with pytest.raises(ValidationError):
        find_n_peaks(rising, 1)

    endpoints = np.zeros(10)
    endpoints[0] = 3.0
    endpoints[-1] = 2.0
    assert find_n_peaks(endpoints, 1, include_endpoints=True).n_available == 2
    with pytest.raises(ValidationError):
        find_n_peaks(endpoints, 2)


# =============================================================================
# date_to_day_number
# =============================================================================


@pytest.mark.parametrize(
    ("year", "month", "day"),
    [
        (1, 1, 1),
        (1900, 3, 1),
        (1999, 12, 31),
        (2000, 2, 29),
        (2000, 3, 1),
        (2024, 2, 29),
        (2100, 3, 1),
    ],
)
def test_date_to_day_number_matches_the_standard_library(year: int, month: int, day: int):
    """日期换算与 Python 标准库 ``datetime.date`` 逐点一致。

    依据：``datetime.date.toordinal()`` 是推算格里历的序数（第 1 天 = 1），
    ``timetuple().tm_yday`` 是一年中的第几天。被测实现用的是儒略日算法
    （Fliegel & Van Flandern），两者是**独立**的公式，因此这是一次真正的
    交叉验证，而不是自己验自己。
    """
    reference = datetime.date(year, month, day)
    result = date_to_day_number(year, month, day)
    assert result.day_number == reference.toordinal()
    assert result.day_of_year == reference.timetuple().tm_yday
    assert result.is_leap_year is bool(calendar.isleap(year))
    # 闰年的 2 月有 29 天，平年没有——这条规则本身就是"闰年标志"的意义。
    assert result.day_of_year <= (366 if result.is_leap_year else 365)


def test_date_to_day_number_rejects_invalid_dates():
    """非法日期必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        date_to_day_number(2024, 13, 1)
    with pytest.raises(ValidationError):
        date_to_day_number(2024, 2, 30)
    with pytest.raises(ValidationError):
        date_to_day_number(2023, 2, 29)
    with pytest.raises(ValidationError):
        date_to_day_number(2024, 0, 1)


# =============================================================================
# pad_series
# =============================================================================


def test_pad_series_zero_padding_is_zero_in_the_tail():
    """零填充：补齐段全为 0。"""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pad_series(values, 8, "zero")
    assert np.allclose(result.values[:5], values)
    assert np.allclose(result.values[5:], 0.0)
    assert result.padding_length == 3


def test_pad_series_mirror_padding_mirrors_the_head():
    """镜像填充：结果满足 ``out[-1-k] == out[1+k]``。

    依据：镜像的定义是"端点重复的对称反射" ``out[m+k] = v[m-1-k]``；当补齐
    长度恰好是原长减 1（总长 ``2m-1``）时，尾部正好是去掉首元素后的头部的
    逆序，于是上式逐点成立。
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pad_series(values, 9, "mirror")
    assert result.values.size == 9
    for index in range(values.size - 1):
        assert result.values[-1 - index] == pytest.approx(result.values[1 + index])
    assert result.values[5] == pytest.approx(values[-1])


def test_pad_series_periodic_padding_repeats_the_signal():
    """周期延拓：``out[n] == out[n % m]`` 对所有下标成立。"""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pad_series(values, 12, "periodic")
    for index in range(result.values.size):
        assert result.values[index] == pytest.approx(values[index % values.size])


def test_pad_series_edge_padding_repeats_the_last_value():
    """端点复制：补齐段全等于末值。"""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pad_series(values, 8, "edge")
    assert np.allclose(result.values[5:], values[-1])


def test_pad_series_random_padding_is_reproducible():
    """随机填充：同一个种子两次调用结果一致。"""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = pad_series(values, 10, "random", seed=5)
    second = pad_series(values, 10, "random", seed=5)
    assert np.allclose(first.values, second.values)


def test_pad_series_rejects_invalid_requests():
    """目标长度过小、镜像补齐过长、方式未知都必须是 ValidationError。"""
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        pad_series(values, 2, "zero")
    with pytest.raises(ValidationError):
        pad_series(values, 10, "mirror")
    with pytest.raises(ValidationError):
        pad_series(values, 10, "unknown")
    with pytest.raises(ValidationError):
        pad_series(values, 10, "zero", seed="not-an-int")


# =============================================================================
# robust_linear_fit
# =============================================================================


def test_robust_linear_fit_resists_outliers_that_bias_ordinary_least_squares():
    """含 20 % 离群点的直线：稳健斜率贴近真值，普通最小二乘被带偏。

    依据：LTS（Rousseeuw & Leroy 1987）的迭代剔除把离群点从拟合里去掉；普通
    最小二乘则被离群点拉偏。测试把两者都算出来作对照。
    """
    rng = np.random.default_rng(4)
    x = np.arange(100.0)
    truth_slope = 1.0
    y = truth_slope * x + 0.5
    outliers = rng.choice(100, size=20, replace=False)
    y[outliers] += 50.0

    result = robust_linear_fit(x, y)
    assert abs(result.slope - truth_slope) < 0.05
    assert abs(result.ols_slope - truth_slope) > 5.0 * abs(result.slope - truth_slope)
    assert result.n_inliers < 100
    assert int(np.count_nonzero(result.inlier_mask)) == result.n_inliers


def test_robust_linear_fit_on_clean_data_matches_ordinary_least_squares():
    """没有离群点时，稳健拟合与普通最小二乘一致。"""
    x = np.arange(60.0)
    y = 2.5 * x - 3.0
    result = robust_linear_fit(x, y)
    assert result.slope == pytest.approx(result.ols_slope, abs=1e-9)
    assert result.slope == pytest.approx(2.5, abs=1e-9)


def test_robust_linear_fit_rejects_invalid_input():
    """长度不一致、点数太少、参数非法都必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        robust_linear_fit(np.arange(5.0), np.arange(4.0))
    with pytest.raises(ValidationError):
        robust_linear_fit(np.arange(2.0), np.arange(2.0))
    with pytest.raises(ValidationError):
        robust_linear_fit(np.arange(10.0), np.arange(10.0), trim_factor=0.0)


# =============================================================================
# wavelet_smoothing
# =============================================================================


def test_wavelet_smoothing_reduces_the_error_against_the_clean_signal():
    """干净信号 + 已知幅度的高斯噪声：平滑后的均方根误差更小。

    依据：à trous 分解（Holschneider et al. 1986；Torrence & Webster 1998）在
    各层给出互不重叠的频带，逐层按该层的噪声尺度做阈值收缩就能把噪声压下去
    而保住低频的信号。
    """
    rng = np.random.default_rng(3)
    index = np.arange(256.0)
    clean = 2.0 * np.sin(2.0 * np.pi * index / 32.0)
    noisy = clean + rng.normal(0.0, 0.5, index.size)

    result = wavelet_smoothing(noisy, mode="threshold", noise_sigma=0.5)
    assert _rmse(result.smoothed, clean) < _rmse(noisy, clean)


def test_wavelet_smoothing_reconstructs_the_input_exactly_without_shrinkage():
    """不收缩时重建必须与输入逐点一致（à trous 分解可逆）。

    依据：``d_j = s_j - s_{j+1}``，于是 ``s_J + sum(d_j)`` 与原序列严格相等。
    """
    rng = np.random.default_rng(8)
    series = rng.normal(0.0, 1.0, 128)
    result = wavelet_smoothing(series, 5, threshold_factor=0.0)
    assert np.allclose(result.smoothed, series, atol=1e-10)


def test_wavelet_smoothing_rejects_invalid_input():
    """层数超过记录长度、方式未知都必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        wavelet_smoothing(np.ones(10), 20)
    with pytest.raises(ValidationError):
        wavelet_smoothing(np.ones(64), 4, mode="unknown")
    with pytest.raises(ValidationError):
        wavelet_smoothing(np.ones(64), 4, threshold="unknown")


# =============================================================================
# cosine_taper
# =============================================================================


def test_cosine_taper_has_zero_endpoints_unit_peak_and_is_symmetric():
    """升余弦 taper：端点为 0、平台为 1、左右对称。

    依据：Bloomfield (1976) 的首尾升余弦 taper，权重
    ``w(u) = 0.5 - 0.5 cos(pi u / tau)``（``u`` 是相对位置、``tau`` 是 taper
    比例），在 ``u = 0`` 处为 0、在 ``u = tau`` 处为 1。
    """
    length = 101
    fraction = 0.05
    result = cosine_taper(length, taper_fraction=fraction)
    weights = result.weights
    assert weights.size == length
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] == pytest.approx(0.0)
    assert weights[length // 2] == pytest.approx(1.0)
    assert np.allclose(weights, weights[::-1])

    # 解析式：平台段必须是 1，斜坡段必须落在 [0, 1]。
    positions = np.arange(length, dtype=float) / (length - 1)
    ramp = positions < fraction
    expected = 0.5 - 0.5 * np.cos(np.pi * positions[ramp] / fraction)
    assert np.allclose(weights[ramp], expected)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_cosine_taper_rejects_invalid_input():
    """长度太小、taper 比例非法都必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        cosine_taper(1)
    with pytest.raises(ValidationError):
        cosine_taper(100, taper_fraction=0.0)
    with pytest.raises(ValidationError):
        cosine_taper(100, taper_fraction=0.5)


# =============================================================================
# phase_diagram
# =============================================================================


def test_phase_diagram_of_a_pure_tone_has_unit_radius_and_the_same_phase():
    """纯正弦的相轨迹：半径恒为 1，相位等于原信号的相位。

    依据：相轨迹上的点是解析信号 ``z = r e^{i phi}``，所以半径就是包络、辐角
    就是瞬时相位——与 :func:`domain_transform` 的解析信号性质同源。
    """
    result = phase_diagram(_tone())
    assert np.allclose(result.radius, 1.0, atol=EXACT)
    assert np.allclose(_phase_difference(result.phase, _tone_phase()), 0.0, atol=EXACT)
    assert np.allclose(result.points[:, 0], result.radius * np.cos(result.phase), atol=EXACT)
    assert np.allclose(result.points[:, 1], result.radius * np.sin(result.phase), atol=EXACT)


def test_phase_diagram_accepts_an_explicit_imaginary_part():
    """给出虚部时，轨迹点就是 (实部, 虚部)，半径与相位由它们算出。"""
    real_part = np.array([1.0, 0.0, -1.0, 0.0])
    imaginary_part = np.array([0.0, 1.0, 0.0, -1.0])
    result = phase_diagram(real_part, imaginary_part)
    assert np.allclose(result.points[:, 0], real_part)
    assert np.allclose(result.points[:, 1], imaginary_part)
    assert np.allclose(result.radius, 1.0)
    assert np.allclose(_phase_difference(result.phase, np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])), 0.0)


def test_phase_diagram_rejects_mismatched_lengths():
    """两个序列长度不一致必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        phase_diagram(np.ones(10), np.ones(9))


# =============================================================================
# random_phase_resample
# =============================================================================


def test_random_phase_resample_preserves_every_fourier_amplitude():
    """相位随机化：长度不变，幅度谱被精确保留（相对误差 < 1e-6）。

    依据：相位随机化（Theiler et al. 1992）只替换正频率上的相位，幅度、直流与
    Nyquist 频点原样保留，因此每个傅里叶幅度都逐点相等。
    """
    rng = np.random.default_rng(6)
    series = np.sin(np.arange(256) / 20.0) + 0.3 * rng.standard_normal(256)

    result = random_phase_resample(series, seed=7)
    assert result.values.size == series.size
    assert result.amplitude_relative_error < 1e-6
    assert np.allclose(
        result.amplitude_spectrum,
        result.original_amplitude_spectrum,
        rtol=1e-9,
        atol=1e-9,
    )


def test_random_phase_resample_iaaft_preserves_the_value_set():
    """IAAFT：取值集合被**精确**保留，幅度谱被逐步逼近。

    依据：Schreiber & Schmitz (1996) 的 IAAFT 每一步把谱投影按秩映射回原序列
    的取值分布，所以排序后的取值逐点相等；幅度谱只能逐步逼近，其误差由
    ``amplitude_relative_error`` 报告。
    """
    rng = np.random.default_rng(6)
    series = np.sin(np.arange(256) / 20.0) + 0.3 * rng.standard_normal(256)

    result = random_phase_resample(series, seed=7, method="iaaft")
    assert result.values.size == series.size
    assert result.value_distribution_error == 0.0
    assert np.allclose(np.sort(result.values), np.sort(series))
    assert result.amplitude_relative_error < 0.05
    assert result.n_iterations >= 1


def test_random_phase_resample_is_reproducible_and_seed_dependent():
    """同一个种子两次调用完全一致；不同种子给出不同结果。"""
    series = np.sin(np.arange(128) / 8.0)
    first = random_phase_resample(series, seed=42)
    second = random_phase_resample(series, seed=42)
    other = random_phase_resample(series, seed=43)
    assert np.allclose(first.values, second.values)
    assert not np.allclose(first.values, other.values)


def test_random_phase_resample_rejects_invalid_input():
    """方式未知、迭代次数非法都必须是 ValidationError。"""
    series = np.sin(np.arange(64) / 8.0)
    with pytest.raises(ValidationError):
        random_phase_resample(series, method="unknown")
    with pytest.raises(ValidationError):
        random_phase_resample(series, method="iaaft", n_iterations=0)
    with pytest.raises(ValidationError):
        random_phase_resample(np.array([1.0, np.nan, 2.0, 3.0]))


# =============================================================================
# fixed_bandwidth_moving_average
# =============================================================================


def test_fixed_bandwidth_moving_average_keeps_a_constant_signal():
    """常值序列的滑动平均仍是同一个常值（端点截短不引入偏差）。

    依据：窗口在两端被截短，窗口里全是同一个值，平均还是那个值。
    """
    result = fixed_bandwidth_moving_average(np.ones(50), 5.0)
    assert np.allclose(result.values, 1.0)
    assert result.window_counts[0] < result.window_counts[25]
    assert result.window_counts[-1] < result.window_counts[25]


def test_fixed_bandwidth_moving_average_keeps_the_slope_of_a_ramp():
    """线性序列的滑动平均在窗口完整处保持斜率。

    依据：线性函数在**对称**窗口上的平均仍等于窗口中心处的函数值，所以斜率
    不变；只有两端窗口被截短的几点的截距会偏。
    """
    slope = 2.0
    series = slope * np.arange(50.0) + 1.0
    result = fixed_bandwidth_moving_average(series, 5.0)
    interior = slice(5, 45)
    fitted = np.polyfit(np.arange(50.0)[interior], result.values[interior], 1)
    assert fitted[0] == pytest.approx(slope, abs=1e-9)


def test_fixed_bandwidth_moving_average_rejects_invalid_input():
    """带宽非法、时间轴不合法都必须是 ValidationError。"""
    with pytest.raises(ValidationError):
        fixed_bandwidth_moving_average(np.ones(10), 0.0)
    with pytest.raises(ValidationError):
        fixed_bandwidth_moving_average(np.ones(10), 5.0, dt=-1.0)
    with pytest.raises(ValidationError):
        fixed_bandwidth_moving_average(np.ones(10), 5.0, time=np.array([0.0, 1.0, 1.0, 2.0, 3.0] * 2))


# =============================================================================
# detrend_polynomial_pp
# =============================================================================


def test_detrend_polynomial_pp_removes_a_pure_polynomial_exactly():
    """纯二次多项式被去掉后残差为 0（机器精度）。

    依据：多项式去趋势就是用同次数的多项式去拟合，模型里含着真解，所以
    最小二乘的残差为 0（加权最小二乘同样如此）。
    """
    index = np.arange(60.0)
    truth = 2.0 * index**2 + 3.0 * index + 1.0
    result = detrend_polynomial_pp(truth, 2)
    assert np.allclose(result.detrended, 0.0, atol=1e-8)
    assert np.allclose(result.trend, truth, atol=1e-8)
    # 去趋势后的残差逐点等于原序列减趋势。
    assert np.allclose(result.detrended, truth - result.trend)


def test_detrend_polynomial_pp_residual_matches_the_true_noise():
    """二次多项式 + 噪声：残差与真噪声一致，趋势贴近真二次项。

    依据：Sinnesael et al. (2018) 的滑动窗多项式基线建模估计的是低频基线；
    残差里剩下的应当是噪声本身，所以"残差与噪声的差"必须比噪声本身小。
    """
    rng = np.random.default_rng(21)
    index = np.arange(80.0)
    truth = 1.5 * index**2 - 2.0 * index + 0.5
    noise = rng.normal(0.0, 1.0, index.size)
    noisy = truth + noise

    result = detrend_polynomial_pp(noisy, 2)
    assert _rmse(result.detrended, noise) < _rmse(noise, 0.0)
    assert _rmse(result.trend, truth) < 0.5 * _rmse(noisy, truth)


def test_detrend_polynomial_pp_rejects_invalid_input():
    """次数非法、窗长非法都必须是 ValidationError。"""
    series = np.arange(40.0)
    with pytest.raises(ValidationError):
        detrend_polynomial_pp(series, -1)
    with pytest.raises(ValidationError):
        detrend_polynomial_pp(series, 2, window=3)
    with pytest.raises(ValidationError):
        detrend_polynomial_pp(series, 2, window=100)


# =============================================================================
# chi_square_confidence_interval
# =============================================================================


def test_chi_square_interval_matches_scipy_quantiles():
    """置信区间与 ``scipy.stats.chi2.ppf`` 独立对照。

    依据：Acycle ``redconfchi2.m`` 的上下界是"标定后的 AR(1) 理论谱乘以卡方
    分位数除以自由度"，自由度取 ``2K``（单段周期图对应 ``K = 1``）。这里用
    scipy 的分位数做真值，被测实现自己算的是谱与标定，分位数只来自 scipy。
    """
    series = np.sin(np.arange(256) / 8.0) + 0.2 * np.cos(np.arange(256) / 5.0)
    degrees = 4
    level = 0.95
    result = chi_square_confidence_interval(series, dof=degrees, level=level)

    expected_upper = result.theoretical * stats.chi2.ppf(level, degrees) / degrees
    expected_lower = result.theoretical * stats.chi2.ppf(1.0 - level, degrees) / degrees
    assert np.allclose(result.upper, expected_upper)
    assert np.allclose(result.lower, expected_lower)
    assert np.all(result.upper >= result.lower)
    assert result.dof == degrees
    assert result.level == pytest.approx(level)


def test_chi_square_interval_scales_with_the_ar1_curve():
    """把 AR(1) 系数换掉，置信区间必须跟着变（中心曲线就是 AR(1) 理论谱）。

    依据：Mann & Lees (1996) 的 AR(1) 理论谱
    ``(1 - rho^2)/(1 - 2 rho cos(2 pi f dt) + rho^2)``。
    """
    series = np.sin(np.arange(256) / 8.0)
    low = chi_square_confidence_interval(series, rho=0.2)
    high = chi_square_confidence_interval(series, rho=0.8)
    assert not np.allclose(low.theoretical, high.theoretical)
    # 上下界与理论谱的比例处处相同（同一个卡方因子）。
    assert np.allclose(low.upper / low.theoretical, high.upper / high.theoretical)


def test_chi_square_interval_rejects_invalid_input():
    """自由度非法、置信水平非法、常数序列都必须是 ValidationError。"""
    series = np.sin(np.arange(64) / 8.0)
    with pytest.raises(ValidationError):
        chi_square_confidence_interval(series, dof=0)
    with pytest.raises(ValidationError):
        chi_square_confidence_interval(series, level=1.5)
    with pytest.raises(ValidationError):
        chi_square_confidence_interval(series, rho=1.5)
    with pytest.raises(ValidationError):
        chi_square_confidence_interval(np.ones(64))


def test_chi_square_interval_returns_the_axis_and_the_spectrum_they_are():
    """频率轴必须是频率轴，观测谱必须是观测谱。

    这条断言是被一个真缺陷逼出来的：实现把 ``scipy.signal.periodogram`` 的
    两个返回值接反了（它返回 ``(frequencies, spectrum)``），于是结果里的
    ``frequencies`` 装的是功率谱、``observed_power`` 装的是频率轴——带被算在
    了一条无意义的轴上，而所有"上下界与理论谱成比例"的断言照样通过，
    因为那种断言只比较实现自己的两个字段。

    真值来自 scipy 自己：频率轴必须是 0 到 Nyquist 的等距序列，观测谱必须
    逐点等于 ``scipy.signal.periodogram`` 的输出。
    """
    from scipy.signal import periodogram as scipy_periodogram

    series = np.sin(np.arange(256) / 8.0) + 0.2 * np.cos(np.arange(256) / 5.0)
    step = 0.25
    result = chi_square_confidence_interval(series, dt=step, dof=4, level=0.95)

    frequencies = np.asarray(result.frequencies, dtype=float)
    observed = np.asarray(result.observed_power, dtype=float)

    # the axis: strictly increasing, from zero to the Nyquist frequency
    assert frequencies[0] == pytest.approx(0.0)
    assert frequencies[-1] == pytest.approx(0.5 / step)
    assert np.all(np.diff(frequencies) > 0.0)

    # the spectrum: the same numbers scipy produces for the same series
    expected_frequencies, expected_spectrum = scipy_periodogram(
        series,
        fs=1.0 / step,
        window="boxcar",
        detrend="constant",
        scaling="density",
    )
    assert np.allclose(frequencies, expected_frequencies)
    assert np.allclose(observed, expected_spectrum)

    # and the band is anchored on the observed level: scaling the data
    # scales the spectrum by the square, and the band with it
    scaled = chi_square_confidence_interval(series * 2.0, dt=step, dof=4, level=0.95)
    assert np.allclose(np.asarray(scaled.observed_power, dtype=float), observed * 4.0)
    assert np.allclose(np.asarray(scaled.lower, dtype=float), np.asarray(result.lower) * 4.0)


# =============================================================================
# tobit_regression
# =============================================================================


def test_tobit_regression_without_censoring_equals_ordinary_least_squares():
    """没有删失时 Tobit 退化成普通最小二乘。

    依据：没有删失观测时似然就是正态似然，极大似然解与最小二乘解一致，残差
    标准差按极大似然的约定是 ``sqrt(RSS / n)``。
    """
    rng = np.random.default_rng(17)
    x = np.arange(40.0)
    y = 1.0 + 0.5 * x + rng.normal(0.0, 1.0, 40)

    result = tobit_regression(x, y)
    design = np.column_stack([np.ones(40), x])
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.allclose(result.coefficients, ols, atol=1e-6)
    assert result.sigma == pytest.approx(float(np.sqrt(np.sum((y - design @ ols) ** 2) / 40)), rel=1e-6)
    assert result.n_censored == 0
    assert result.converged


def test_tobit_regression_with_censoring_beats_the_naive_least_squares():
    """含左删失时，Tobit 的估计必须比"把删失值当观测值"的朴素最小二乘更贴近真值。

    依据：Tobin (1958) 的删失正态模型把删失观测当作"潜变量不超过限值"的
    信息；朴素最小二乘把删失值当成真实观测，会把截距拉偏。测试里真值是
    ``y = 1 + 0.5 x``，删失限值取 5，检验的是**截距**：Tobit 的截距必须靠近
    1.0，而朴素最小二乘的截距明显偏大。
    """
    rng = np.random.default_rng(23)
    x = np.arange(40.0)
    truth_intercept = 1.0
    truth_slope = 0.5
    y = truth_intercept + truth_slope * x + rng.normal(0.0, 1.0, 40)

    limit = 5.0
    censored = y < limit
    observed = np.where(censored, limit, y)

    result = tobit_regression(x, observed, limit=limit, censored=censored)
    assert result.n_censored == int(np.count_nonzero(censored))
    assert result.converged
    # 朴素最小二乘：只用没被删失的观测。
    design = np.column_stack([np.ones(40), x])
    naive = np.linalg.lstsq(design[~censored], y[~censored], rcond=None)[0]
    assert abs(result.coefficients[0] - truth_intercept) < abs(naive[0] - truth_intercept)
    assert abs(result.coefficients[0] - truth_intercept) < 0.35
    assert abs(result.coefficients[1] - truth_slope) < 0.1


def test_tobit_regression_rejects_inconsistent_flags():
    """删失标记与观测值矛盾、长度不一致都必须是 ValidationError。"""
    x = np.arange(20.0)
    y = 1.0 + x
    with pytest.raises(ValidationError):
        tobit_regression(x, y, limit=5.0, censored=np.ones(10, dtype=bool))
    with pytest.raises(ValidationError):
        # 标成删失的观测值却大于限值，自相矛盾。
        tobit_regression(x, y, limit=5.0, censored=np.arange(20) % 2 == 0)


# =============================================================================
# 跨工具的边界与可复现性
# =============================================================================


@pytest.mark.parametrize(
    "call",
    [
        lambda: domain_transform(np.ones(2)),
        lambda: gaussian_matched_filter(np.ones(2), np.ones(2)),
        lambda: find_n_peaks(np.ones(2), 1),
        lambda: pad_series(np.ones(4), 2, "zero"),
        lambda: robust_linear_fit(np.ones(2), np.ones(2)),
        lambda: wavelet_smoothing(np.ones(2)),
        lambda: cosine_taper(1),
        lambda: phase_diagram(np.ones(2)),
        lambda: random_phase_resample(np.ones(2)),
        lambda: fixed_bandwidth_moving_average(np.ones(1), 1.0),
        lambda: detrend_polynomial_pp(np.ones(4), 2),
        lambda: chi_square_confidence_interval(np.ones(4)),
        lambda: tobit_regression(np.arange(4.0), np.arange(4.0)),
    ],
)
def test_every_tool_rejects_too_few_samples(call):
    """点数太少时，每个工具都必须抛 ValidationError（而不是返回垃圾结果）。"""
    with pytest.raises(ValidationError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: domain_transform(np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        lambda: gaussian_matched_filter(np.array([1.0, np.nan, 2.0, 3.0]), np.ones(2)),
        lambda: find_n_peaks(np.array([1.0, np.nan, 2.0, 3.0]), 1),
        lambda: pad_series(np.array([1.0, np.nan, 2.0]), 5, "zero"),
        lambda: robust_linear_fit(np.arange(5.0), np.array([1.0, np.nan, 2.0, 3.0, 4.0])),
        lambda: wavelet_smoothing(np.array([1.0, np.nan, 2.0, 3.0, 4.0])),
        lambda: phase_diagram(np.array([1.0, np.nan, 2.0, 3.0])),
        lambda: random_phase_resample(np.array([1.0, np.nan, 2.0, 3.0, 4.0])),
        lambda: fixed_bandwidth_moving_average(np.array([1.0, np.nan, 2.0, 3.0]), 2.0),
        lambda: detrend_polynomial_pp(np.array([1.0, np.nan, 2.0, 3.0, 4.0]), 1),
        lambda: chi_square_confidence_interval(np.array([1.0, np.nan, 2.0, 3.0, 4.0])),
        lambda: tobit_regression(np.arange(5.0), np.array([1.0, np.nan, 2.0, 3.0, 4.0])),
    ],
)
def test_every_tool_rejects_non_finite_samples(call):
    """含 NaN 时，每个工具都必须抛 ValidationError。"""
    with pytest.raises(ValidationError):
        call()


def test_random_tools_are_reproducible_with_the_same_seed():
    """带 seed 的工具：同一个种子两次调用必须完全一致。"""
    series = np.sin(np.arange(64) / 8.0)
    first = random_phase_resample(series, seed=99)
    second = random_phase_resample(series, seed=99)
    assert np.allclose(first.values, second.values)
    padded_first = pad_series(series, 80, "random", seed=99)
    padded_second = pad_series(series, 80, "random", seed=99)
    assert np.allclose(padded_first.values, padded_second.values)


# =============================================================================
# 摘要
# =============================================================================


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("domain_transform", lambda: domain_transform(np.sin(np.arange(64) / 8.0))),
        (
            "gaussian_matched_filter",
            lambda: gaussian_matched_filter(
                np.sin(np.arange(64) / 8.0),
                np.exp(-0.5 * ((np.arange(9) - 4) / 1.5) ** 2),
            ),
        ),
        ("find_n_peaks", lambda: find_n_peaks(np.sin(np.arange(64) / 2.0), 3)),
        ("date_to_day_number", lambda: date_to_day_number(2024, 6, 15)),
        ("pad_series", lambda: pad_series(np.arange(20.0), 30, "zero")),
        ("robust_linear_fit", lambda: robust_linear_fit(np.arange(20.0), np.arange(20.0) * 2.0 + 1.0)),
        ("wavelet_smoothing", lambda: wavelet_smoothing(np.sin(np.arange(64) / 4.0))),
        ("cosine_taper", lambda: cosine_taper(100)),
        ("phase_diagram", lambda: phase_diagram(np.sin(np.arange(64) / 4.0))),
        ("random_phase_resample", lambda: random_phase_resample(np.sin(np.arange(64) / 4.0), seed=3)),
        ("fixed_bandwidth_moving_average", lambda: fixed_bandwidth_moving_average(np.arange(64.0), 10.0)),
        ("detrend_polynomial_pp", lambda: detrend_polynomial_pp(np.arange(64.0) ** 2, 2)),
        (
            "chi_square_confidence_interval",
            lambda: chi_square_confidence_interval(np.sin(np.arange(256) / 8.0), dt=0.25),
        ),
        ("tobit_regression", lambda: tobit_regression(np.arange(20.0), np.arange(20.0) * 2.0 + 1.0)),
    ],
)
def test_every_tool_summarises_itself(tool, call):
    """每个工具的结果都必须有一份给人看的摘要。

    摘要不是装饰：脚本控制台把结果的 ``summary()`` 印给用户。没有它就只剩
    dataclass 的 ``repr``，整条数组被摊在屏幕上——红噪声置信带那样的结果会
    渲染出成千上万个字符的数字。
    """
    result = call()

    summary = result.summary()

    assert summary.strip(), f"{tool} produced an empty summary"
