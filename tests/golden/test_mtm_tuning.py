# =============================================================================
# FILE: tests/golden/test_mtm_tuning.py
# =============================================================================
"""
``stratigraphy.mtm_tuning``（自适应 taper 数 + 谱的 F 检验）的测试。

本文件提供三条证据：

1. **自适应 taper 数的方向性判据。** taper 数必须是**奇数**（Thomson 的
   ``K = 2·NW − 1`` 只在整半带宽上给出奇数），并且在记录变长时**单调不减**
   ——记录越长，能用的锥越多，直到被带宽本身限住为止。这是从"规则"推出
   的方向性判据，不是冻结的数字。

2. **F 检验在零假设下的标定。** 对纯白噪声序列，各频率的 p 值应当近似
   **均匀分布**：用 KS 统计量加四分位数判据检查（不断言任何精确的
   p 值）；对含强周期的序列，该周期所在频率的 p 值必须**极小**。

3. **频率轴的口径。** 频率是"每单位时间"（cycles/Ma 之类），不是每样本：
   已知周期的合成信号，其谱峰必须落在 ``1 / 周期`` 上。

可复现性
--------
F 检验的解析 p 值是确定性的；``n_simulations > 0`` 时改用蒙特卡洛标定，
这条路径接受 ``random_seed``：固定种子必须给出**完全相同**的 p 值；不给
种子时按仓库的约定发出 ``RuntimeWarning``（结果不可复现）。

参考来源
--------
* Thomson, D.J. 1982. Spectrum estimation and harmonic analysis. Proceedings
  of the IEEE 70: 1055-1096.（多锥方法、``K = 2NW − 1``、有效自由度。）
* Percival, D.B. & Walden, A.T. 1993. Spectral Analysis and its
  Applications. Cambridge University Press.（DPSS 聚集特征值与有效自由度。）
* Phipson, S.M. & Smyth, G.K. 2010. Permutation P-values should never be
  zero. Statistical Applications in Genetics and Molecular Biology 9: 39.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"multitaper_taper_count", "tuned_multitaper_spectrum"}


import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats

from stratigraphy.mtm_tuning import (
    adaptive_degrees_of_freedom,
    multitaper_f_test,
    select_taper_count,
    tuned_multitaper_spectrum,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的长度。
N_SAMPLES = 512

#: 白噪声记录用的种子（固定，保证判据可复现）。
NOISE_SEED = 20240517

#: 强周期信号的周期（样本）。
CYCLE_PERIOD = 20.0

#: 强周期信号的振幅（标准差单位）。
CYCLE_AMPLITUDE = 3.0


# =============================================================================
# 辅助函数
# =============================================================================


def _white_noise(n: int = N_SAMPLES, seed: int = NOISE_SEED) -> npt.NDArray:
    """纯白噪声：没有任何周期成分，是 F 检验零假设的代表。"""
    return np.random.default_rng(seed).standard_normal(n)


def _signal_with_cycle(
    n: int = N_SAMPLES,
    period: float = CYCLE_PERIOD,
    amplitude: float = CYCLE_AMPLITUDE,
    seed: int = NOISE_SEED,
) -> npt.NDArray:
    """白噪声 + 一个已知的强周期。"""
    t = np.arange(n, dtype=float)
    noise = np.random.default_rng(seed).standard_normal(n)
    return noise + amplitude * np.sin(2.0 * np.pi * t / period)


# =============================================================================
# 1. 自适应 taper 数
# =============================================================================


def test_taper_count_is_odd_and_grows_with_the_record_length():
    """taper 数必须是奇数，且随记录变长而单调不减（方向性判据）。"""
    counts = [select_taper_count(n, time_bandwidth=6.0).n_tapers for n in (12, 16, 20, 24, 40)]
    assert all(count % 2 == 1 for count in counts), f"taper counts must be odd: {counts}"
    assert counts == sorted(counts), f"taper counts must not shrink with the record: {counts}"
    assert counts[0] < counts[-1], f"a longer record must allow more tapers: {counts}"


def test_taper_count_grows_with_the_requested_bandwidth():
    """带宽越大，taper 数越多；且始终是奇数。"""
    counts = [select_taper_count(256, time_bandwidth=nw).n_tapers for nw in (2.0, 3.0, 4.0, 5.0)]
    assert counts == sorted(counts), f"taper counts must grow with the bandwidth: {counts}"
    assert counts[0] < counts[-1]
    assert all(count % 2 == 1 for count in counts)


def test_the_reported_bandwidth_matches_the_taper_count():
    """返回的带宽必须与 taper 数自洽：NW = (K + 1) / 2。"""
    for n_samples in (32, 64, 128, 256):
        selection = select_taper_count(n_samples, time_bandwidth=4.0)
        count = selection.n_tapers
        bandwidth = selection.time_bandwidth
        assert bandwidth == pytest.approx((count + 1) / 2.0)
        assert count % 2 == 1


def test_an_explicit_taper_count_is_used_as_given():
    """调用方给了 taper 数就用它（并据此反推带宽）。"""
    selection = select_taper_count(256, n_tapers=7)
    count = selection.n_tapers
    bandwidth = selection.time_bandwidth
    assert count == 7
    assert bandwidth == pytest.approx(4.0)


def test_a_short_record_adapts_the_bandwidth_down():
    """记录太短时带宽被自适应地收窄，而不是硬报一个用不了的 taper 数。"""
    selection = select_taper_count(12, time_bandwidth=6.0)
    count = selection.n_tapers
    bandwidth = selection.time_bandwidth
    assert count <= 12 // 2
    assert count % 2 == 1
    assert bandwidth == pytest.approx((count + 1) / 2.0)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"n_samples": 3}, "too short"),
        ({"n_samples": 100, "time_bandwidth": 0.0}, "bandwidth must be positive"),
        ({"n_samples": 100, "time_bandwidth": -1.0}, "bandwidth must be positive"),
        ({"n_samples": 100, "n_tapers": 0}, "taper count must be positive"),
        ({"n_samples": 100, "n_tapers": 80}, "more tapers than the record supports"),
    ],
)
def test_bad_taper_selections_are_rejected(kwargs, reason):
    """非法参数 -> ValidationError。"""
    with pytest.raises(ValidationError):
        select_taper_count(**kwargs)


def test_the_taper_selection_summarises_itself():
    """结果必须有一份给人看的摘要。

    脚本控制台把结果的 ``summary()`` 印给用户；没有它就只剩 dataclass 的
    ``repr``，两个裸数字，看不出这是选参的结果。
    """
    result = select_taper_count(256, time_bandwidth=4.0)

    summary = result.summary()

    assert summary.strip()
    assert str(result.n_tapers) in summary


# =============================================================================
# 2. F 检验
# =============================================================================


def test_f_test_p_values_are_approximately_uniform_under_the_null():
    """纯白噪声下，各频率的 p 值应近似均匀分布（KS + 分位数判据）。"""
    result = tuned_multitaper_spectrum(_white_noise(), time_bandwidth=3.0)
    p_values = result.p_values
    assert p_values is not None
    assert p_values.size == result.spectrum.size
    assert np.all((p_values > 0.0) & (p_values <= 1.0))

    ks = sp_stats.kstest(p_values, "uniform")
    assert ks.statistic < 0.13, f"the p values are not uniform (KS D = {ks.statistic})"

    # 分位数判据：均匀分布的四分位数是 0.25 / 0.5 / 0.75。
    quartiles = np.quantile(p_values, [0.25, 0.5, 0.75])
    assert 0.12 < quartiles[0] < 0.38, f"the lower quartile is off: {quartiles}"
    assert 0.38 < quartiles[1] < 0.62, f"the median is off: {quartiles}"
    assert 0.62 < quartiles[2] < 0.88, f"the upper quartile is off: {quartiles}"

    false_alarm_rate = float(np.mean(p_values < 0.05))
    assert 0.005 < false_alarm_rate < 0.12, f"the rate of p < 0.05 is off: {false_alarm_rate}"


def test_f_test_flags_a_strong_period():
    """含强周期的序列，该周期所在频率的 p 值必须极小。"""
    result = tuned_multitaper_spectrum(_signal_with_cycle(), time_bandwidth=3.0)
    target = 1.0 / CYCLE_PERIOD
    index = int(np.argmin(np.abs(result.frequencies - target)))
    assert result.p_values[index] < 1.0e-6, f"the p value at the injected period is {result.p_values[index]}"
    assert int(np.argmin(result.p_values)) == index, "the smallest p value must sit on the injected period"


def test_f_test_accepts_a_scalar_background_and_a_scalar_dof():
    """标量背景与标量自由度必须被接受，并给出 (0, 1] 的 p 值。"""
    result = tuned_multitaper_spectrum(_white_noise(), time_bandwidth=3.0, background=1.0)
    p_values = multitaper_f_test(
        result.spectrum,
        result.frequencies,
        np.ones_like(result.spectrum),
        result.n_tapers,
        result.degrees_of_freedom,
    )
    assert p_values.size == result.spectrum.size
    assert np.all((p_values > 0.0) & (p_values <= 1.0))


def test_f_test_can_use_per_frequency_degrees_of_freedom():
    """逐频率自由度（自适应权重）必须被接受，且 p 值仍然合法。"""
    series = _white_noise()
    result = tuned_multitaper_spectrum(series, time_bandwidth=3.0)
    dof = adaptive_degrees_of_freedom(series, time_bandwidth=3.0)
    assert dof.size == result.spectrum.size
    assert np.all(dof > 0.0)
    p_values = multitaper_f_test(
        result.spectrum,
        result.frequencies,
        np.ones_like(result.spectrum),
        result.n_tapers,
        dof,
    )
    assert np.all((p_values > 0.0) & (p_values <= 1.0))


def test_n_frequencies_restricts_the_tested_range():
    """n_frequencies 只检验谱的前若干个频率点。"""
    result = tuned_multitaper_spectrum(_white_noise(), time_bandwidth=3.0)
    subset = multitaper_f_test(
        result.spectrum,
        result.frequencies,
        np.ones_like(result.spectrum),
        result.n_tapers,
        result.degrees_of_freedom,
        n_frequencies=32,
    )
    assert subset.size == 32


@pytest.mark.parametrize(
    ("spectrum", "frequencies", "background", "tapers", "dof"),
    [
        ([1.0, 2.0], [0.1], [1.0, 1.0], 3, 6.0),
        ([1.0, 2.0], [0.1, 0.2], [1.0, 0.0], 3, 6.0),
        ([1.0, 2.0], [0.1, 0.2], [1.0, 1.0], 1, 6.0),
        ([1.0, 2.0], [0.1, 0.2], [1.0, 1.0], 3, 0.0),
    ],
)
def test_bad_f_test_inputs_are_rejected(spectrum, frequencies, background, tapers, dof):
    """长度不符、背景非正、taper 不足、自由度非正 -> ValidationError。"""
    with pytest.raises(ValidationError):
        multitaper_f_test(
            np.asarray(spectrum, dtype=float),
            np.asarray(frequencies, dtype=float),
            np.asarray(background, dtype=float),
            tapers,
            dof,
        )


# =============================================================================
# 3. 频率轴与蒙特卡洛标定
# =============================================================================


def test_the_frequency_axis_is_per_unit_time():
    """频率是每单位时间：已知周期的谱峰必须落在 1 / 周期 上。"""
    dt = 0.02
    period_ma = 0.4
    n = 1024
    t = np.arange(n, dtype=float)
    series = np.sin(2.0 * np.pi * t * dt / period_ma)
    result = tuned_multitaper_spectrum(series, time_bandwidth=3.0, dt=dt)
    target = 1.0 / period_ma
    index = int(np.argmin(np.abs(result.frequencies - target)))
    assert result.frequencies[index] == pytest.approx(target, abs=0.05)
    assert abs(int(np.argmax(result.spectrum)) - index) <= 1, "the spectral peak must sit on the injected period"


def test_the_monte_carlo_calibration_is_reproducible_with_a_fixed_seed():
    """固定种子 -> 完全相同的 p 值。"""
    series = _white_noise()
    first = tuned_multitaper_spectrum(series, time_bandwidth=3.0, n_simulations=20, random_seed=7)
    second = tuned_multitaper_spectrum(series, time_bandwidth=3.0, n_simulations=20, random_seed=7)
    assert first.p_values is not None and second.p_values is not None
    assert np.array_equal(first.p_values, second.p_values)


def test_the_monte_carlo_calibration_warns_when_no_seed_is_given():
    """不给种子 -> RuntimeWarning（仓库的随机过程约定）。"""
    with pytest.warns(RuntimeWarning):
        tuned_multitaper_spectrum(_white_noise(), time_bandwidth=3.0, n_simulations=5, random_seed=None)


def test_the_monte_carlo_calibration_finds_the_injected_period():
    """蒙特卡洛标定后，强周期处的 p 值同样极小。"""
    result = tuned_multitaper_spectrum(
        _signal_with_cycle(),
        time_bandwidth=3.0,
        n_simulations=40,
        random_seed=3,
    )
    target = 1.0 / CYCLE_PERIOD
    index = int(np.argmin(np.abs(result.frequencies - target)))
    assert result.p_values[index] <= 1.0 / 41.0


def test_bad_multitaper_inputs_are_rejected():
    """样本太少、含非有限值、dt 非正 -> ValidationError。"""
    with pytest.raises(ValidationError):
        tuned_multitaper_spectrum(np.arange(8.0), time_bandwidth=3.0)
    with pytest.raises(ValidationError):
        tuned_multitaper_spectrum(np.concatenate([np.arange(40.0), [np.nan]]), time_bandwidth=3.0)
    with pytest.raises(ValidationError):
        tuned_multitaper_spectrum(_white_noise(), time_bandwidth=3.0, dt=0.0)
