# =============================================================================
# FILE: tests/property/test_signal_generator_properties.py
# =============================================================================
"""
``stratigraphy.signal_generator`` 的测试：正弦信号、多分量合成与三种噪声。

期望值来自**信号与随机过程本身的定义**，而不是"跑一遍把输出冻结下来"：

* 正弦信号的频谱主峰必须落在 ``1 / period`` 上（用 FFT 独立验证）；
* 多分量合成后，每个分量的频率都必须在谱里出现（同样的 FFT 验证）；
* AR(1) 过程的滞后 1 阶自相关理论值为 ``rho``；
* 粉噪声的功率谱斜率理论值为 ``-1``，用 log-log 回归独立估计；
* 白噪声的理论标准差就是 ``sigma``；
* 固定 ``random_seed`` 必须给出逐点相同的结果。

随机过程的样本量取足够大（白噪声 20000、红噪声 20000、粉噪声 8192），
使统计量的抽样误差远小于断言容差；粉噪声用固定种子，因此它的谱斜率
既是统计量，也是完全可复现的。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"generate_signal"}


import numpy as np
import pytest

from stratigraphy.signal_generator import (
    generate_pink_noise,
    generate_red_noise,
    generate_signal,
    generate_sine,
    generate_white_noise,
)
from utils.exceptions import ValidationError

# ---------------------------------------------------------------------------
# 固定夹具
# ---------------------------------------------------------------------------

#: 采样间隔 0.01、记录长度 100.0 的等间距时间轴。记录长度是周期的整数
#: 倍，因此各分量的频率恰好落在 FFT 的整数频点上（没有频谱泄漏）。
SAMPLING_INTERVAL = 0.01
N_SAMPLES = 10000
GRID = np.arange(N_SAMPLES) * SAMPLING_INTERVAL
FREQUENCIES = np.fft.rfftfreq(N_SAMPLES, d=SAMPLING_INTERVAL)


def _dominant_frequency(signal: np.ndarray) -> float:
    """用 FFT 求信号的主频。"""
    power = np.abs(np.fft.rfft(signal)) ** 2
    return float(FREQUENCIES[int(np.argmax(power))])


class TestGenerateSine:
    """正弦信号的周期与相位必须与定义一致。"""

    def test_the_fft_peak_sits_at_the_inverse_period(self):
        """主频必须是 1 / period。"""
        signal = generate_sine(GRID, amplitude=1.0, period=10.0)
        assert _dominant_frequency(signal) == pytest.approx(0.1, abs=1e-9)

    def test_the_amplitude_is_respected(self):
        """振幅是信号的最大绝对值。"""
        signal = generate_sine(GRID, amplitude=2.5, period=7.0)
        assert np.max(np.abs(signal)) == pytest.approx(2.5, abs=1e-9)

    def test_the_phase_shifts_the_signal(self):
        """相位是弧度：phase = pi/2 时 sin 变成 cos。"""
        signal = generate_sine(GRID, amplitude=1.0, period=10.0, phase=np.pi / 2.0)
        expected = np.cos(2.0 * np.pi * GRID / 10.0)
        assert np.allclose(signal, expected, rtol=0.0, atol=1e-12)

    def test_a_non_positive_period_raises(self):
        """周期必须为正。"""
        with pytest.raises(ValidationError):
            generate_sine(GRID, amplitude=1.0, period=0.0)
        with pytest.raises(ValidationError):
            generate_sine(GRID, amplitude=1.0, period=-5.0)


class TestGenerateSignal:
    """多分量合成后每个分量都必须在谱里出现。"""

    def test_every_component_shows_up_as_a_spectral_peak(self):
        """两个分量的频率都必须是谱里的强峰。"""
        components = [(1.0, 10.0), (0.5, 4.0, 0.7)]
        signal = generate_signal(GRID, components)
        power = np.abs(np.fft.rfft(signal)) ** 2
        median_power = float(np.median(power[1:]))
        for period in (10.0, 4.0):
            expected_frequency = 1.0 / period
            bin_index = int(np.argmin(np.abs(FREQUENCIES - expected_frequency)))
            window = power[bin_index - 2 : bin_index + 3]
            assert power[bin_index] == pytest.approx(float(np.max(window)))
            assert power[bin_index] > 100.0 * median_power

    def test_components_may_omit_the_phase(self):
        """(振幅, 周期) 与 (振幅, 周期, 0.0) 必须等价。"""
        without_phase = generate_signal(GRID, [(1.0, 10.0)])
        with_zero_phase = generate_signal(GRID, [(1.0, 10.0, 0.0)])
        assert np.allclose(without_phase, with_zero_phase, rtol=0.0, atol=1e-12)

    def test_noise_is_added_when_requested(self):
        """加了噪声之后信号被改变，但分量本身还在。"""
        clean = generate_signal(GRID, [(1.0, 10.0)])
        noisy = generate_signal(GRID, [(1.0, 10.0)], noise="white", noise_sigma=0.1, random_seed=5)
        assert not np.allclose(clean, noisy)
        assert _dominant_frequency(noisy) == pytest.approx(0.1, abs=1e-9)

    def test_no_noise_when_sigma_is_zero(self):
        """噪声强度为 0 时不叠加噪声，也不消耗随机数。"""
        clean = generate_signal(GRID, [(1.0, 10.0)])
        unchanged = generate_signal(GRID, [(1.0, 10.0)], noise="white", noise_sigma=0.0)
        assert np.allclose(clean, unchanged)

    def test_an_unknown_noise_type_raises(self):
        """未知噪声类型必须报错。"""
        with pytest.raises(ValidationError):
            generate_signal(GRID, [(1.0, 10.0)], noise="brown")

    def test_a_malformed_component_raises(self):
        """分量必须给出两个或三个数。"""
        with pytest.raises(ValidationError):
            generate_signal(GRID, [(1.0, 10.0, 0.0, 0.0)])


class TestNoiseProcesses:
    """三种噪声的统计性质必须与定义一致。"""

    def test_white_noise_has_the_requested_standard_deviation(self):
        """白噪声的标准差就是 sigma。"""
        noise = generate_white_noise(20000, sigma=2.0, random_seed=1)
        assert float(np.std(noise)) == pytest.approx(2.0, rel=0.05)
        assert abs(float(np.mean(noise))) < 0.2

    def test_red_noise_lag_one_autocorrelation_matches_rho(self):
        """AR(1) 的滞后 1 阶自相关理论值是 rho。"""
        rho = 0.7
        noise = generate_red_noise(20000, rho=rho, sigma=1.0, random_seed=3)
        lag_one = float(np.corrcoef(noise[:-1], noise[1:])[0, 1])
        assert lag_one == pytest.approx(rho, abs=0.02)

    def test_red_noise_has_the_requested_standard_deviation(self):
        """AR(1) 的边际标准差是 sigma。"""
        noise = generate_red_noise(20000, rho=0.7, sigma=1.5, random_seed=4)
        assert float(np.std(noise)) == pytest.approx(1.5, rel=0.05)

    def test_pink_noise_has_a_spectral_slope_of_minus_one(self):
        """粉噪声的功率谱斜率是 -1，用 log-log 回归独立估计。"""
        n = 8192
        noise = generate_pink_noise(n, sigma=1.0, random_seed=7)
        frequencies = np.fft.rfftfreq(n)
        power = np.abs(np.fft.rfft(noise)) ** 2
        usable = (frequencies > 0) & (power > 0)
        slope = float(np.polyfit(np.log10(frequencies[usable]), np.log10(power[usable]), 1)[0])
        assert slope == pytest.approx(-1.0, abs=0.3)

    def test_a_fixed_seed_is_reproducible(self):
        """同一个种子必须给出逐点相同的结果。"""
        first = generate_red_noise(1000, rho=0.6, sigma=1.0, random_seed=11)
        second = generate_red_noise(1000, rho=0.6, sigma=1.0, random_seed=11)
        assert np.array_equal(first, second)
        pink_first = generate_pink_noise(1000, random_seed=11)
        pink_second = generate_pink_noise(1000, random_seed=11)
        assert np.array_equal(pink_first, pink_second)

    def test_different_seeds_give_different_results(self):
        """不同种子必须给出不同结果。"""
        first = generate_white_noise(1000, random_seed=1)
        second = generate_white_noise(1000, random_seed=2)
        assert not np.array_equal(first, second)

    def test_an_unseeded_run_warns_that_it_is_not_reproducible(self):
        """不给种子时必须提醒结果不可复现。"""
        with pytest.warns(RuntimeWarning, match="random_seed"):
            generate_white_noise(100)
        with pytest.warns(RuntimeWarning, match="random_seed"):
            generate_red_noise(100)
        with pytest.warns(RuntimeWarning, match="random_seed"):
            generate_pink_noise(100)


class TestEdgeCases:
    """非法参数必须报错。"""

    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_a_non_positive_length_raises(self, n: int):
        """样本数必须为正。"""
        with pytest.raises(ValidationError):
            generate_white_noise(n)
        with pytest.raises(ValidationError):
            generate_red_noise(n)
        with pytest.raises(ValidationError):
            generate_pink_noise(n)

    @pytest.mark.parametrize("rho", [-1.5, -1.0, 1.0, 1.5])
    def test_an_out_of_range_rho_raises(self, rho: float):
        """rho 必须严格落在 (-1, 1) 之内，否则过程不平稳。"""
        with pytest.raises(ValidationError):
            generate_red_noise(100, rho=rho)

    def test_a_negative_sigma_raises(self):
        """标准差不能为负。"""
        with pytest.raises(ValidationError):
            generate_white_noise(100, sigma=-1.0)

    def test_an_empty_time_axis_raises(self):
        """空时间轴必须报错。"""
        with pytest.raises(ValidationError):
            generate_sine(np.array([]))
        with pytest.raises(ValidationError):
            generate_signal(np.array([]), [(1.0, 10.0)])
