# =============================================================================
# FILE: tests/golden/test_bispectrum.py
# =============================================================================
"""
``stratigraphy.bispectrum``（双谱与双相干）的测试。

测试围绕四条主线组织：

* **合成真值（最关键的一条）。** 构造三个正弦分量且满足 ``f3 = f1 + f2`` 的
  信号（``f1 = 17/221``、``f2 = 13/221``、``f3 = 30/221``，记录长度取 221 的
  整数倍使每个分量都是整周期），双谱在 ``(f1, f2)`` 处必须出现**压倒性的
  峰值**：断言用"峰值 / 中位数"之比，而不是把某个具体数值钉死——双谱幅度的
  绝对值依赖分段长度与窗函数，比例才是与实现无关的判据。

* **相位。** 双谱的物理意义在相位上。零相位差的三波共振给出
  ``arg B(f1, f2) ≈ 0``（三个分量的相位 0 + 0 - 0），测试在这个三元组上钉一个
  容差。

* **纯噪声。** 白噪声没有任何三元组耦合：双相干应该**普遍很低**，双谱的峰值
  与中位数之比也远小于合成真值里的量级。

* **显著性。** 代理数据给出的阈值必须是一份**校准过的零假设**：白噪声下
  超过阈值的频点比例应当接近名义的 5%（点态）。另外覆盖代理数据路径的
  形状、主域之外的 ``NaN``、固定种子的可复现性。

边界测试断言抛 ``utils.exceptions.ValidationError``。

参考来源（断言中引用的数值出处）
--------------------------------
* Swaminathan, J. K., Bruno, R. S. & Pierce, J. D. (1999). Multifractal analysis
  of the turbulent energy dissipation field. *Physical Review Letters*, 83,
  2216.（双谱方法）
* Kim, M. C. & Powers, E. J. (1992). Measurement of turbulent energy
  dissipation statistics using the bispectrum. *Journal of Fluid Mechanics*,
  248, 51--68.（双相干的 Kim--Powers 归一化，``0..1`` 的取值范围）
* Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for nonlinearity
  tests. *Physical Review Letters*, 77, 635--638.（IAAFT 代理数据）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"bispectrum", "bicoherence_significance"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.bispectrum import BispectrumResult, bicoherence_significance, bispectrum
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 采样间隔（每单位时间的步长）。
DT = 1.0

#: FFT（分段）长度。221 = 13 * 17，因此 ``f1 = 17/221``、``f2 = 13/221`` 恰好
#: 落在频点上，``f1 + f2 = 30/221`` 也落在轴内。
NFFT = 221

#: 第一个分量的频点序号。
BIN_F1 = 17

#: 第二个分量的频点序号。
BIN_F2 = 13

#: 记录长度：221 的整数倍，保证每个正弦分量都是整周期（无频谱泄漏）。
N_SAMPLES = 4 * 221

#: 合成真值里 ``f1``、``f2`` 的频率（每单位时间）。
F1 = BIN_F1 / NFFT
F2 = BIN_F2 / NFFT

#: 双谱相位在真值三元组上的容差（弧度）。
PHASE_TOLERANCE = 0.25

#: 真值三元组处双相干的下限（Kim--Powers 归一化后应接近 1）。
TRIAD_BICOHERENCE_MIN = 0.9

#: 真值三元组处"峰值 / 中位数"的下限。
TRIAD_PEAK_RATIO_MIN = 100.0

#: 白噪声双相干的中位数上限。
NOISE_MEDIAN_BICOHERENCE_MAX = 0.30

#: 白噪声双相干的 95% 分位上限。
NOISE_BICOHERENCE_P95_MAX = 0.70

#: 白噪声双谱"峰值 / 中位数"的上限（真值里是 1e13 量级）。
NOISE_PEAK_RATIO_MAX = 50.0

#: 显著性检验使用的代理序列条数。
N_SURROGATES = 20

#: 显著性检验的名义水平（点态 5%）。
ALPHA = 0.05

#: 允许的显著频点比例上限（名义 5%，留出抽样误差的余量）。
SIGNIFICANT_FRACTION_MAX = 0.20


# =============================================================================
# 辅助函数
# =============================================================================


def _triad_signal(n_samples: int = N_SAMPLES, noise: float = 0.0, seed: int = 0) -> npt.NDArray:
    """构造满足 ``f3 = f1 + f2`` 的三个正弦分量（零初相位）。"""
    time = np.arange(n_samples, dtype=float) * DT
    signal = np.cos(2.0 * np.pi * F1 * time) + np.cos(2.0 * np.pi * F2 * time) + np.cos(2.0 * np.pi * (F1 + F2) * time)
    if noise > 0.0:
        generator = np.random.default_rng(seed)
        signal = signal + noise * generator.standard_normal(n_samples)
    return signal


def _triad_index() -> tuple[int, int]:
    """真值三元组在结果矩阵中的下标（频率轴不含直流，因此要减 1）。"""
    return BIN_F1 - 1, BIN_F2 - 1


def _white_noise(n_samples: int = N_SAMPLES, seed: int = 5) -> npt.NDArray:
    """构造白噪声序列。"""
    return np.random.default_rng(seed).standard_normal(n_samples)


# =============================================================================
# 结果容器与频率轴
# =============================================================================


def test_result_is_square_and_matches_the_frequency_axes() -> None:
    """结果矩阵应是 ``(n_f1, n_f2)`` 的方阵，两个频率轴长度一致。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    assert isinstance(result, BispectrumResult)
    assert result.bispectrum.shape == (result.frequencies.size, result.frequencies2.size)
    assert result.bicoherence.shape == result.bispectrum.shape
    assert result.phase.shape == result.bispectrum.shape
    assert result.significance is None


def test_frequency_axis_is_cycles_per_unit_time() -> None:
    """频率轴是每单位时间的循环频率，且随 ``dt`` 反比缩放。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    expected = np.arange(1, NFFT // 2) / (NFFT * DT)
    assert np.allclose(result.frequencies, expected)
    assert result.frequencies.size == NFFT // 2 - 1

    # dt 加倍，单位时间内的频率数减半。
    doubled = bispectrum(_triad_signal(), dt=2.0 * DT, nfft=NFFT)
    assert np.allclose(doubled.frequencies, expected / 2.0)

    # 真值频率必须落在轴上。
    row, column = _triad_index()
    assert result.frequencies[row] == pytest.approx(F1)
    assert result.frequencies2[column] == pytest.approx(F2)


def test_summary_reports_the_grid() -> None:
    """摘要文本应当报告网格与频率范围。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    summary = result.summary()
    assert isinstance(summary, str)
    assert "Bispectrum" in summary
    assert str(result.frequencies.size) in summary


# =============================================================================
# 合成真值：峰值、相位、双相干
# =============================================================================


def test_bispectrum_peaks_at_the_triad() -> None:
    """真值三元组上必须出现压倒性的双谱峰值。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    magnitude = np.abs(result.bispectrum)
    finite = np.isfinite(magnitude)
    assert finite.any()

    row, column = _triad_index()
    peak_value = magnitude[row, column]
    median_value = float(np.median(magnitude[finite]))
    assert median_value > 0.0
    assert peak_value / median_value > TRIAD_PEAK_RATIO_MIN

    # 峰值必须落在真值三元组上（对称位置 (f2, f1) 的幅度相同，故比较数值）。
    peak_row, peak_column = np.unravel_index(int(np.nanargmax(magnitude)), magnitude.shape)
    assert magnitude[peak_row, peak_column] == pytest.approx(peak_value, rel=1e-9)


def test_bispectrum_phase_is_zero_at_the_triad() -> None:
    """零相位差的三波共振给出 ``arg B(f1, f2) ≈ 0``。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    row, column = _triad_index()
    assert abs(float(result.phase[row, column])) < PHASE_TOLERANCE
    assert np.all(np.abs(result.phase[np.isfinite(result.phase)]) <= np.pi + 1e-12)


def test_bicoherence_is_near_one_at_the_triad() -> None:
    """Kim--Powers 归一化的双相干在真值三元组上应接近 1，且不超过 1。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT)
    row, column = _triad_index()
    assert result.bicoherence[row, column] > TRIAD_BICOHERENCE_MIN
    assert float(np.nanmax(result.bicoherence)) <= 1.0


def test_normalize_false_keeps_the_raw_bispectrum_only() -> None:
    """``normalize=False`` 时不做归一化，双相干全为 ``NaN``。"""
    result = bispectrum(_triad_signal(), dt=DT, nfft=NFFT, normalize=False)
    assert not np.isfinite(result.bicoherence).any()
    row, column = _triad_index()
    assert np.isfinite(result.bispectrum[row, column])


# =============================================================================
# 纯噪声
# =============================================================================


def test_white_noise_has_low_bicoherence_and_no_peak() -> None:
    """白噪声没有三元组耦合：双相干普遍低，双谱也没有压倒性峰值。"""
    result = bispectrum(_white_noise(), dt=DT, nfft=NFFT)
    magnitude = np.abs(result.bispectrum)
    finite = np.isfinite(magnitude)
    coherence = result.bicoherence[finite]

    assert float(np.median(coherence)) < NOISE_MEDIAN_BICOHERENCE_MAX
    assert float(np.percentile(coherence, 95)) < NOISE_BICOHERENCE_P95_MAX

    median_value = float(np.median(magnitude[finite]))
    assert float(np.max(magnitude[finite])) / median_value < NOISE_PEAK_RATIO_MAX


def test_white_noise_has_no_significant_triad() -> None:
    """噪声下超过显著性阈值的频点比例应当接近名义的 5%。"""
    result = bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=N_SURROGATES, random_seed=3)
    assert result.significance is not None
    finite = np.isfinite(result.bicoherence)
    threshold = np.where(np.isfinite(result.significance), result.significance, np.inf)
    significant = finite & (result.bicoherence > threshold)
    assert int(significant.sum()) / int(finite.sum()) < SIGNIFICANT_FRACTION_MAX


# =============================================================================
# 显著性
# =============================================================================


def test_significance_threshold_shape_and_domain() -> None:
    """阈值矩阵与结果同形，主域之外为 ``NaN``，主域之内为正。"""
    result = bicoherence_significance(_triad_signal(), dt=DT, nfft=NFFT, n_surrogates=N_SURROGATES, random_seed=1)
    assert result.significance is not None
    assert result.significance.shape == result.bicoherence.shape

    frequency_sum = result.frequencies[:, None] + result.frequencies2[None, :]
    principal = frequency_sum <= result.frequencies[-1]
    assert np.isnan(result.significance[~principal]).all()
    assert (result.significance[principal] > 0.0).all()


def test_significance_is_reproducible_with_a_fixed_seed() -> None:
    """固定种子的代理数据给出完全相同的结果；不同种子给出不同结果。"""
    first = bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=N_SURROGATES, random_seed=42)
    second = bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=N_SURROGATES, random_seed=42)
    third = bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=N_SURROGATES, random_seed=43)

    assert first.significance is not None and second.significance is not None and third.significance is not None
    assert np.array_equal(first.significance, second.significance, equal_nan=True)
    assert np.array_equal(first.bicoherence, second.bicoherence, equal_nan=True)
    assert not np.array_equal(first.significance, third.significance, equal_nan=True)


def test_unseeded_surrogates_warn_about_reproducibility() -> None:
    """不给种子时必须发出 ``RuntimeWarning`` 提示结果不可复现。"""
    with pytest.warns(RuntimeWarning):
        bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=4)


def test_iaaft_and_phase_randomized_surrogates_both_run() -> None:
    """两种代理方式都必须给出形状正确、有限值合理的阈值矩阵。"""
    iaaft_result = bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=4, iaaft=True, random_seed=9)
    phase_result = bicoherence_significance(
        _white_noise(), dt=DT, nfft=NFFT, n_surrogates=4, iaaft=False, random_seed=9
    )
    assert iaaft_result.significance is not None and phase_result.significance is not None
    assert iaaft_result.significance.shape == phase_result.significance.shape
    assert np.isfinite(iaaft_result.significance).any()


# =============================================================================
# 边界
# =============================================================================


@pytest.mark.parametrize(
    ("values", "kwargs"),
    [
        (np.ones(10), {}),
        (np.arange(100.0).reshape(10, 10), {}),
        (np.concatenate([np.ones(99), [np.nan]]), {}),
        (np.ones(200), {"dt": 0.0}),
        (np.ones(200), {"dt": -1.0}),
        (np.ones(200), {"nfft": 8}),
        (np.ones(200), {"nfft": 256.5}),
        (np.ones(200), {"normalize": "yes"}),
    ],
)
def test_invalid_inputs_raise_validation_error(values: npt.NDArray, kwargs: dict) -> None:
    """非法输入必须抛 ``ValidationError``。"""
    with pytest.raises(ValidationError):
        bispectrum(values, **kwargs)


def test_constant_series_raises_validation_error() -> None:
    """常数序列没有谱结构，必须被拒绝。"""
    with pytest.raises(ValidationError):
        bispectrum(np.ones(200), dt=DT, nfft=NFFT)


def test_invalid_surrogate_count_raises_validation_error() -> None:
    """代理条数必须至少为 1。"""
    with pytest.raises(ValidationError):
        bicoherence_significance(_white_noise(), dt=DT, nfft=NFFT, n_surrogates=0)
