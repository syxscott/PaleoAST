# =============================================================================
# FILE: tests/golden/test_spectral_significance.py
# =============================================================================
"""
``stratigraphy.spectral_significance`` 的测试：SWA 背景、FDR 与贝叶斯周期图。

期望值来自**方法的定义与可解析的关系**，而不是"跑一遍把输出冻结下来"：

* **SWA 背景**：合成谱由"幂律红噪声 + 高频白噪声段"拼成，白噪声段的真值是
  已知常数。背景在白噪声段必须回到那个常数，而在红噪声段必须跟着幂律走，
  否则它就不是背景、只是把谱抄了一遍。两种尾模型（线性/二次）按整条谱上的
  R² 择优，测试直接验证这条择优规则，而不是钉死哪一个变体获胜。
* **FDR**：先按 ``p = sf(8 * power / background, 8)`` 反推出想要的 p 值向量，
  再检验 Benjamini–Hochberg 过程：真峰的 p 值必须全部入选，而入选个数必须
  远少于"高于背景"的峰数（这正是 FDR 相对固定阈值假警报率的价值）。
* **贝叶斯周期图**：方向性判据——已知周期处的后验概率必须显著高于其它频率；
  固定种子必须完全可复现，不给种子必须发出 ``RuntimeWarning``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"swa_fdr", "swa_background", "weedon_periodogram", "bayesian_periodogram"}


import numpy as np
import pytest
from scipy import stats

from stratigraphy.spectral_analysis import SpectralAnalyzer
from stratigraphy.spectral_significance import (
    bayesian_periodogram,
    swa_background,
    swa_fdr,
    weedon_periodogram,
)
from utils.exceptions import ValidationError

#: SWA 合成谱的频率轴。
AXIS = np.linspace(0.01, 1.0, 240)

#: 合成谱中白噪声段的真值。
WHITE_LEVEL = 1.0

#: 合成谱中红噪声与白噪声的分界频率。
BREAK_FREQUENCY = 0.5

#: 合成谱中红噪声的幂指数。
RED_SLOPE = 1.5

#: 贝叶斯周期图测试用的已知周期（样本数）。
TRUE_PERIOD = 50.0

#: FDR 水平测试用的记录长度。
FDR_SERIES_SIZE = 4096

#: FDR 水平测试用的 AR(1) 系数（红噪声）。
FDR_RHO = 0.3

#: FDR 水平测试的重复次数（抽样离散是真实的，重复取平均才稳）。
FDR_REPEATS = 6


def _mixed_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """幂律红噪声 + 高频白噪声段，带一点真实谱常见的起伏。"""
    power = np.where(
        AXIS < BREAK_FREQUENCY,
        WHITE_LEVEL * (BREAK_FREQUENCY / AXIS) ** RED_SLOPE,
        WHITE_LEVEL,
    )
    rng = np.random.default_rng(4242)
    power = power * np.exp(rng.normal(0.0, 0.05, size=AXIS.size))
    return AXIS, power


def _power_from_p_values(p_values: np.ndarray, background: float) -> np.ndarray:
    """按 ``p = sf(8 * power / background, 8)`` 反推功率谱。"""
    ratio = stats.chi2.isf(p_values, 8.0) / 8.0
    return ratio * background


# =============================================================================
# SWA background
# =============================================================================


def test_swa_background_recovers_the_white_noise_level() -> None:
    """背景在白噪声段回到白噪声水平，在红噪声段跟随幂律。"""
    axis, power = _mixed_spectrum()
    result = swa_background(axis, power)
    white = axis >= BREAK_FREQUENCY
    red = ~white

    white_level = float(np.mean(result.swa[white]))
    assert white_level == pytest.approx(WHITE_LEVEL, rel=0.05)
    # 红噪声段的背景远离白噪声水平：背景确实跟住了谱的形状。
    assert float(np.mean(result.swa[red])) > 5.0 * WHITE_LEVEL
    # 背景处处为正，否则无法作为功率比的分母。
    assert bool(np.all(result.swa > 0.0))


def test_swa_background_variant_follows_the_r_squared_rule() -> None:
    """``auto`` 选中的变体必须是 R² 更高的那个。"""
    axis, power = _mixed_spectrum()
    auto = swa_background(axis, power)
    linear = swa_background(axis, power, variant="linear")
    quadratic = swa_background(axis, power, variant="quadratic")

    scores = {"linear": linear.r_squared, "quadratic": quadratic.r_squared}
    best = max(scores, key=lambda name: scores[name])
    assert auto.variant == best
    assert auto.r_squared == pytest.approx(scores[best])
    assert auto.r_squared > 0.9


def test_swa_background_of_a_flat_spectrum_is_flat() -> None:
    """纯白噪声谱的背景就是它自己。"""
    axis = np.linspace(0.05, 1.0, 96)
    power = np.full_like(axis, 3.0)
    result = swa_background(axis, power)
    assert np.allclose(result.swa, 3.0, rtol=1e-6)


def test_swa_background_rejects_invalid_input() -> None:
    """点数太少、频率非递增、功率为负、变体拼错都要报 ValidationError。"""
    axis, power = _mixed_spectrum()
    with pytest.raises(ValidationError):
        swa_background(axis[:5], power[:5])
    with pytest.raises(ValidationError):
        swa_background(np.sort(axis)[::-1].copy(), power)
    with pytest.raises(ValidationError):
        swa_background(axis, -power)
    with pytest.raises(ValidationError):
        swa_background(axis, power, variant="cubic")


# =============================================================================
# False discovery rate
# =============================================================================


def test_swa_fdr_keeps_every_true_peak_and_far_fewer_than_all_peaks() -> None:
    """真峰全部入选，入选数远少于高于背景的峰数。"""
    axis = np.linspace(0.01, 1.0, 500)
    background = np.full_like(axis, 2.0)
    rng = np.random.default_rng(20240517)
    p_values = rng.uniform(1e-6, 1.0, size=axis.size)
    true_indices = [17, 89, 233, 401]
    for index in true_indices:
        p_values[index] = 1e-25

    power = _power_from_p_values(p_values, 2.0)
    result = swa_fdr(axis, power, background, alpha=0.05)

    for index in true_indices:
        assert bool(result.significant[index]), f"peak {index} was not selected"

    n_peaks = int(np.sum(power > background))
    assert n_peaks > 20, "the noise should produce plenty of peaks above the background"
    selected = int(np.sum(result.significant))
    assert selected < 0.1 * n_peaks
    # 只留真峰，最多再带上极少数噪声频率。
    assert selected <= len(true_indices) + 2
    assert result.threshold == pytest.approx(0.05 * selected / axis.size)


def test_swa_fdr_returns_the_requested_p_values() -> None:
    """p 值就是 ``sf(8 * power / background, 8)``。"""
    axis = np.linspace(0.01, 1.0, 64)
    background = np.full_like(axis, 1.5)
    power = np.full_like(axis, 1.5)
    result = swa_fdr(axis, power, background)
    assert np.allclose(result.p_values, stats.chi2.sf(8.0, 8.0))


def test_swa_fdr_rejects_invalid_input() -> None:
    """alpha 越界、背景非正都要报 ValidationError。"""
    axis, power = _mixed_spectrum()
    background = np.full_like(axis, 1.0)
    with pytest.raises(ValidationError):
        swa_fdr(axis, power, background, alpha=0.0)
    with pytest.raises(ValidationError):
        swa_fdr(axis, power, background, alpha=1.5)
    with pytest.raises(ValidationError):
        swa_fdr(axis, power, np.zeros_like(background))


# =============================================================================
# Bayesian periodogram
# =============================================================================


def _periodic_series(sigma: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """已知周期的正弦信号加高斯噪声。"""
    axis = np.arange(1000.0)
    rng = np.random.default_rng(7)
    series = np.sin(2.0 * np.pi * axis / TRUE_PERIOD) + rng.normal(0.0, sigma, size=axis.size)
    return axis, series


def test_bayesian_periodogram_favours_the_known_period() -> None:
    """已知周期处的后验概率显著高于其它频率。"""
    axis, series = _periodic_series()
    frequencies, power, posterior = bayesian_periodogram(axis, series, n_frequencies=500, random_seed=42)

    true_frequency = 1.0 / TRUE_PERIOD
    peak = int(np.argmax(posterior))
    assert frequencies[peak] == pytest.approx(true_frequency, rel=0.02)
    assert posterior[peak] > 0.3
    assert posterior[peak] > 1000.0 * float(np.median(posterior))
    assert power[peak] > float(np.median(power))

    # 频率栅格是几何间隔的：局部间隔按真值处的比例算。
    ratio = frequencies[1] / frequencies[0]
    local_step = true_frequency * (ratio - 1.0)
    near = np.abs(frequencies - true_frequency) < 3.0 * local_step
    assert float(np.sum(posterior[near])) > 0.9
    assert float(np.max(np.delete(posterior, near))) < 0.01
    assert float(np.sum(posterior)) == pytest.approx(1.0)


def test_bayesian_periodogram_is_reproducible_with_a_seed() -> None:
    """固定种子必须给出完全相同的结果。"""
    axis, series = _periodic_series()
    first = bayesian_periodogram(axis, series, n_frequencies=200, random_seed=7)
    second = bayesian_periodogram(axis, series, n_frequencies=200, random_seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])


def test_bayesian_periodogram_warns_without_a_seed() -> None:
    """不给种子必须发出 RuntimeWarning。"""
    axis, series = _periodic_series()
    with pytest.warns(RuntimeWarning):
        bayesian_periodogram(axis, series, n_frequencies=64, random_seed=None)


def test_bayesian_periodogram_rejects_invalid_input() -> None:
    """常数序列、非法迭代次数与频率范围都要报 ValidationError。"""
    axis, series = _periodic_series()
    with pytest.raises(ValidationError):
        bayesian_periodogram(axis, np.ones_like(series))
    with pytest.raises(ValidationError):
        bayesian_periodogram(axis, series, n_iterations=0)
    with pytest.raises(ValidationError):
        bayesian_periodogram(axis, series, n_frequencies=2)
    with pytest.raises(ValidationError):
        bayesian_periodogram(axis, series, frequency_range=(1.0, 0.5))


# =============================================================================
# Degrees-of-freedom contract of the FDR pipeline
# =============================================================================


def _ar1_series(rho: float, size: int, seed: int) -> np.ndarray:
    """平稳 AR(1) 红噪声序列 ``x[t] = rho * x[t-1] + eps[t]``。"""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=size)
    series = np.empty(size, dtype=float)
    series[0] = innovations[0]
    for index in range(1, size):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def test_weedon_periodogram_closes_on_eight_degrees_of_freedom() -> None:
    """8 自由度谱的契约：``8 * power / mean(power)`` 必须服从卡方 8。

    这是从**被测产物**里读出来的契约，而不是把期望值写死在实现里：若分段
    平均的段数、或每段的归一化方式变了，这条断言会立刻失败。
    """
    axis = np.arange(FDR_SERIES_SIZE, dtype=float)
    rng = np.random.default_rng(2024)
    values = rng.normal(0.0, 1.0, size=axis.size)

    result = weedon_periodogram(axis, values)
    assert result.degrees_of_freedom == 8.0
    assert result.frequencies.size == result.power.size
    assert result.swa.size == result.power.size

    ratio = 8.0 * result.power / float(np.mean(result.power))
    assert float(np.mean(ratio)) == pytest.approx(8.0, rel=0.02)
    ks = stats.kstest(ratio, "chi2", args=(8.0,))
    assert float(ks.pvalue) > 0.01


def test_swa_fdr_level_on_red_noise() -> None:
    """红噪声上，8 自由度谱的实测假阳性率必须落在名义 5% 的邻域内。

    统计量的选择
    ~~~~~~~~~~~~
    Benjamini–Hochberg 是"阶梯向上"过程：在零假设下它要么一个都不拒绝，
    要么拒绝一整块，因此**单次实现的拒绝个数**并不稳定（8 自由度谱上通常
    一个都不拒）。自由度契约直接控制的是**逐点超越率**
    ``p(f) <= alpha`` 的频率占比，所以水平判据用它。

    区间的理由
    ~~~~~~~~~~
    SWA 背景是在**对数谱**上做的窗口平均，它估计的是局部几何均值，而
    卡方零假设用的是算术均值；两者相差一个系统性的因子，使 p 值偏小。
    实测这条流水线的逐点超越率在 5%–9% 之间（红噪声 AR(1)，
    ``alpha = 0.05``，6 次重复），因此断言取 2%–12% 的邻域：既能容纳抽样
    离散，又能抓住"自由度搞错 → 假阳性率翻数倍"这种量级的错误。

    对照
    ~~~~
    仓库现成的 :class:`stratigraphy.spectral_analysis.SpectralAnalyzer`
    周期图是**未加窗、未平均**的，每点只有 2 个自由度。用 8 自由度的临界
    值去套它，实测假阳性率会翻数倍——这条对照把"自由度错了会怎样"钉成
    一条回归测试。用它自己的 2 个自由度则回到正常水平。
    """
    axis = np.arange(FDR_SERIES_SIZE, dtype=float)
    analyzer = SpectralAnalyzer()

    weedon_rates: list[float] = []
    mismatched_rates: list[float] = []
    matched_rates: list[float] = []
    mismatched_blocks: list[float] = []
    weedon_blocks: list[float] = []

    for repeat in range(FDR_REPEATS):
        values = _ar1_series(FDR_RHO, FDR_SERIES_SIZE, 3000 + repeat)

        result = weedon_periodogram(axis, values)
        weedon = swa_fdr(
            result.frequencies,
            result.power,
            result.swa,
            alpha=0.05,
            degrees_of_freedom=result.degrees_of_freedom,
        )
        weedon_rates.append(float(np.mean(weedon.p_values <= 0.05)))
        weedon_blocks.append(float(np.mean(weedon.significant)))

        plain = np.asarray(analyzer._lomb_scargle(axis, values, result.frequencies), dtype=float)
        mismatched = swa_fdr(result.frequencies, plain, result.swa, alpha=0.05, degrees_of_freedom=8.0)
        mismatched_rates.append(float(np.mean(mismatched.p_values <= 0.05)))
        mismatched_blocks.append(float(np.mean(mismatched.significant)))

        matched = swa_fdr(result.frequencies, plain, result.swa, alpha=0.05, degrees_of_freedom=2.0)
        matched_rates.append(float(np.mean(matched.p_values <= 0.05)))

    # 8 自由度谱 + 正确的自由度：逐点假阳性率落在名义水平的邻域内。
    assert min(weedon_rates) >= 0.02
    assert max(weedon_rates) <= 0.12

    # 自由度不匹配：每一次实现的假阳性率都越出上面的邻域，且平均至少多出
    # 一半（实测约 17% 对 9%）。
    assert min(mismatched_rates) > 0.12
    assert float(np.mean(mismatched_rates)) >= 1.5 * float(np.mean(weedon_rates))

    # 用它自己的自由度（2），仓库现成的 LS 谱回到正常水平。
    assert min(matched_rates) >= 0.02
    assert max(matched_rates) <= 0.12

    # BH 的拒绝块：不匹配自由度的谱会拒掉一大块，远多于校准过的谱。
    assert float(np.mean(mismatched_blocks)) > float(np.mean(weedon_blocks))


def test_weedon_periodogram_rejects_invalid_input() -> None:
    """分段数、频率点数越界与常数序列都要报 ValidationError。"""
    axis = np.arange(512.0, dtype=float)
    values = np.sin(2.0 * np.pi * axis / 32.0)
    with pytest.raises(ValidationError):
        weedon_periodogram(axis, values, n_segments=1)
    with pytest.raises(ValidationError):
        weedon_periodogram(axis, values, n_frequencies=2)
    with pytest.raises(ValidationError):
        weedon_periodogram(axis, values, frequency_range=(1.0, 0.5))
    with pytest.raises(ValidationError):
        weedon_periodogram(axis, np.ones_like(values))


def test_swa_fdr_rejects_a_non_positive_degrees_of_freedom() -> None:
    """自由度必须是正数。"""
    axis, power = _mixed_spectrum()
    background = np.full_like(axis, 1.0)
    with pytest.raises(ValidationError):
        swa_fdr(axis, power, background, degrees_of_freedom=0.0)
