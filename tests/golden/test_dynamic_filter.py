# =============================================================================
# FILE: tests/golden/test_dynamic_filter.py
# =============================================================================
"""
``stratigraphy.dynamic_filter`` 的测试：时变滤波的输出与边界。

判据来自滤波的**定义**，而不是"跑一遍把输出冻结下来"：

* 输入是"白噪声 + 已知正弦"。滤波后正弦的幅度应当基本保留，而噪声功率
  应当下降，因此用最小二乘投影测得的信噪比必须高于输入的信噪比。
* 输出与输入等长、全部有限；逐点的信噪比曲线与有效自由度都必须为正。
* 参数越界（点数太少、段数非法、门槛非正）必须抛出 ``ValidationError``。

信噪比的测法：把序列投影到已知的 ``cos``/``sin`` 基上，投影出的正弦分量的
方差除以残差方差。这个定义对相位与幅度的小偏移都稳健。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"dynamic_filter"}


import numpy as np
import pytest

from stratigraphy.dynamic_filter import dynamic_filter
from utils.exceptions import ValidationError

#: 测试序列的长度与信号周期。
N_SAMPLES = 2000
PERIOD = 60.0
#: 白噪声的标准差。
NOISE_SIGMA = 0.5


def _noisy_sine(seed: int = 4242) -> tuple[np.ndarray, np.ndarray]:
    """生成"单位正弦 + 白噪声"序列及其时间轴。"""
    rng = np.random.default_rng(seed)
    time = np.arange(N_SAMPLES, dtype=float)
    series = np.sin(2.0 * np.pi * time / PERIOD) + rng.normal(0.0, NOISE_SIGMA, size=N_SAMPLES)
    return time, series


def _projection_snr(series: np.ndarray) -> float:
    """用最小二乘投影测得信号功率与噪声功率之比。"""
    time = np.arange(series.size, dtype=float)
    design = np.column_stack(
        [
            np.cos(2.0 * np.pi * time / PERIOD),
            np.sin(2.0 * np.pi * time / PERIOD),
            np.ones_like(time),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
    harmonic = coefficients[0] * design[:, 0] + coefficients[1] * design[:, 1]
    residual = series - design @ coefficients
    return float(np.var(harmonic) / np.var(residual))


class TestDynamicFilter:
    """滤波必须提高信噪比，并给出逐点的诊断量。"""

    def test_the_filter_improves_the_signal_to_noise_ratio(self):
        """滤波后的信噪比高于输入。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series, n_segments=10)
        assert _projection_snr(result.filtered) > _projection_snr(series)

    def test_the_filter_improves_the_signal_to_noise_ratio_for_more_segments(self):
        """换一种分段数结论不变。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series, n_segments=4)
        assert _projection_snr(result.filtered) > _projection_snr(series)

    def test_the_output_has_the_same_length_as_the_input(self):
        """输出与输入等长。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series)
        assert result.filtered.size == series.size
        assert result.signal_to_noise.size == series.size
        assert result.effective_dof.size == series.size

    def test_every_output_value_is_finite(self):
        """输出与诊断曲线全部有限。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series)
        assert np.all(np.isfinite(result.filtered))
        assert np.all(np.isfinite(result.signal_to_noise))
        assert np.all(np.isfinite(result.effective_dof))

    def test_the_local_signal_to_noise_is_positive(self):
        """逐点信噪比为正。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series)
        assert np.all(result.signal_to_noise > 0.0)

    def test_the_effective_degrees_of_freedom_is_positive(self):
        """逐点有效自由度为正。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series)
        assert np.all(result.effective_dof > 0.0)

    def test_the_signal_is_preserved(self):
        """被保留的正弦幅度不低于输入的一半。"""
        _time, series = _noisy_sine()
        result = dynamic_filter(series)
        assert _projection_snr(result.filtered) > 0.0
        time = np.arange(N_SAMPLES, dtype=float)
        design = np.column_stack([np.cos(2.0 * np.pi * time / PERIOD), np.sin(2.0 * np.pi * time / PERIOD)])
        before, *_ = np.linalg.lstsq(design, series, rcond=None)
        after, *_ = np.linalg.lstsq(design, result.filtered, rcond=None)
        amplitude_before = float(np.hypot(before[0], before[1]))
        amplitude_after = float(np.hypot(after[0], after[1]))
        assert amplitude_after > 0.5 * amplitude_before

    def test_a_constant_series_is_reduced_to_its_level(self):
        """常数序列没有可提取的信号，输出只剩局部水平。"""
        series = np.full(N_SAMPLES, 4.0)
        result = dynamic_filter(series)
        assert np.allclose(result.filtered, 4.0)
        assert np.all(np.isfinite(result.filtered))

    def test_too_few_samples_raise(self):
        """点数太少无法滤波。"""
        with pytest.raises(ValidationError):
            dynamic_filter(np.arange(5.0))

    def test_a_non_finite_series_raises(self):
        """含非有限值的序列不能滤波。"""
        series = np.zeros(N_SAMPLES)
        series[3] = np.inf
        with pytest.raises(ValidationError):
            dynamic_filter(series)

    def test_an_invalid_segment_count_raises(self):
        """段数至少为 1。"""
        _time, series = _noisy_sine()
        with pytest.raises(ValidationError):
            dynamic_filter(series, n_segments=0)

    def test_an_invalid_minimum_snr_raises(self):
        """信噪比门槛必须为正。"""
        _time, series = _noisy_sine()
        with pytest.raises(ValidationError):
            dynamic_filter(series, min_snr=0.0)
        with pytest.raises(ValidationError):
            dynamic_filter(series, min_snr=-1.0)
