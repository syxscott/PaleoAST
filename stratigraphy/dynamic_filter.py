# =============================================================================
# FILE: stratigraphy/dynamic_filter.py
# =============================================================================
"""
时变（动态）滤波：按局部噪声水平调节滤波强度。

Acycle 的 "Dynamic Filter"（``code/misc/dynamic_filter.m`` 与
``code/seriesanalysis/acycleDynamicFilter.m``）是**时变**的：滤波强度随局部
估计的噪声水平逐点变化，而不是全序列用一套固定参数。本模块以分段的方式
复现这一思路，并且完全在**频域**完成（Acycle 的动态滤波同样是频域的
"频率稳定化"操作）：

1. 把序列切成 ``n_segments`` 段；
2. 每段算周期图 ``P(f)``（每单位时间的频率轴），并**只在高频段**拟合
   AR(1) 噪声模型 ``S_noise(f) = s * shape(f; rho_s)``：
   ``shape`` 是 AR(1) 的功率谱形状 ``(1 - rho^2) / (1 - 2 rho cos(pi u) + rho^2)``，
   ``u`` 是以 Nyquist 归一化的频率。之所以只在高频段拟合，是因为周期信号
   （要保留的目标）通常集中在低频段，在整段频带上拟合会把信号误判成
   红噪声、从而把信号一起滤掉；
3. 逐频点做 Wiener 收缩：``g(f) = max(0, 1 - S_noise(f) / P(f))``，
   信号明显高于局部噪声的频点几乎原样保留，被噪声淹没的频点被压掉；
4. 该段的**局部信噪比**由收缩前后的功率估计给出；低于 ``min_snr`` 的段被
   认为“没有可信信号”，其波动按 ``snr / min_snr`` 的比例压低；
5. 每段的信噪比与**有效自由度**（保留的频率个数乘 2）再插值回每个样本点，
   得到逐点曲线。

分段的意义在于：``rho_s`` 与 ``s`` 逐段变化，因此滤波强度随时间变化——
这正是“动态”滤波与固定滤波的区别。

采样约定
~~~~~~~~
输入是等间距采样的序列（采样间隔视为 1 个时间单位），因此频谱用
``numpy.fft.rfft`` 计算，频率轴按每单位时间的循环频率给出。滤波后的序列
保留各段的局部均值（直流分量不被当作信号保留，只作为基准加回）。

参考文献
~~~~~~~~
* Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and
  Control. Holden & Day.（AR(1) 的 Yule–Walker 估计与创新理论）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. Computers & Geosciences 127, 12-22.
* Thibault, N. & Rizzi, G. (2019). 动态滤波（频率稳定化）工作流的设计者。

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考其公开的行为约定
（分段、逐段按局部 AR(1) 参数调节滤波强度、频域滤波），未翻译或复制其
MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 滤波至少需要的样本点数。
MIN_SAMPLES = 16

#: 单段至少需要的样本点数（低于此值只做退化的白噪声收缩）。
MIN_SEGMENT_SAMPLES = 4

#: AR(1) 系数搜索网格的分辨率与上限。
RHO_GRID_SIZE = 51
MAX_AR1_RHO = 0.999

#: 信噪比的显示上限（避免除以零时出现无穷大）。
MAX_REPORTED_SNR = 1.0e12


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class DynamicFilterResult:
    """
    动态滤波的输出。

    Attributes:
        filtered: 滤波后的序列，与输入等长。
        signal_to_noise: 逐点的局部信噪比曲线。
        effective_dof: 逐点的局部有效自由度（保留的频率个数乘 2）。
    """

    filtered: npt.NDArray
    signal_to_noise: npt.NDArray
    effective_dof: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Dynamic filter\n"
            f"{'=' * 48}\n"
            f"Samples: {int(self.filtered.size)}\n"
            f"Median SNR: {float(np.median(self.signal_to_noise)):.6g}\n"
            f"Median effective DOF: {float(np.median(self.effective_dof)):.6g}"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _as_series(values: object, name: str = "values") -> npt.NDArray:
    """把输入转成一维有限数组，并检查长度下限。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if array.ndim != 1:
        raise ValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"parameter": name, "ndim": int(array.ndim)},
        )
    if array.size < MIN_SAMPLES:
        raise ValidationError(
            _("{0} needs at least {1} samples; got {2}").format(name, MIN_SAMPLES, int(array.size)),
            details={"parameter": name, "length": int(array.size), "minimum": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


# =============================================================================
# Local noise model
# =============================================================================


def _ar1_shape(frequencies: npt.NDArray, nyquist: float, rho: float) -> npt.NDArray:
    """AR(1) 的功率谱形状（以 Nyquist 归一化的频率）。"""
    normalised = frequencies / nyquist
    denominator = 1.0 - 2.0 * rho * np.cos(np.pi * normalised) + rho * rho
    return (1.0 - rho * rho) / denominator


def _fit_ar1_rho(power: npt.NDArray, frequencies: npt.NDArray, nyquist: float) -> float:
    """
    在 log 空间把 AR(1) 形状拟合到给定的频段，返回最佳 rho。

    尺度（谱的整体水平）不参与拟合，因此拟合只反映频谱的**形状**。
    """
    if power.size < 2:
        return 0.0
    maximum = float(np.max(power))
    if maximum <= 0.0:
        return 0.0
    floor = maximum * 1e-12
    log_power = np.log(np.maximum(power, floor))
    best_rho = 0.0
    best_distance = np.inf
    for rho in np.linspace(0.0, MAX_AR1_RHO, RHO_GRID_SIZE):
        log_shape = np.log(_ar1_shape(frequencies, nyquist, float(rho)))
        residual = log_power - log_shape
        distance = float(np.sum((residual - residual.mean()) ** 2))
        if distance < best_distance:
            best_distance = distance
            best_rho = float(rho)
    return best_rho


# =============================================================================
# Public API
# =============================================================================


def dynamic_filter(
    values: Sequence[float] | npt.NDArray,
    n_segments: int = 10,
    min_snr: float = 1.0,
) -> DynamicFilterResult:
    """
    对序列做时变（分段）动态滤波。

    Parameters:
        values: 一维有限序列，等间距采样，至少 16 个点。
        n_segments: 分段数，至少 1；实际段数不超过样本点数。
        min_snr: 局部信噪比的门槛。低于该门槛的段被认为没有可信信号，
            其波动按 ``snr / min_snr`` 的比例被压低。

    Returns:
        DynamicFilterResult: 滤波序列、逐点信噪比曲线与逐点有效自由度。

    Raises:
        ValidationError: 输入非法、``n_segments`` 小于 1 或 ``min_snr``
            不是正数。

    Examples:
        >>> import numpy as np
        >>> time = np.arange(1000)
        >>> series = np.sin(2 * np.pi * time / 50) + 0.5 * np.random.default_rng(0).normal(size=1000)
        >>> result = dynamic_filter(series)
        >>> result.filtered.size == series.size
        True
    """
    array = _as_series(values, "values")
    segments = int(n_segments)
    if segments != n_segments or segments < 1:
        raise ValidationError(
            _("n_segments must be an integer of at least 1"),
            details={"n_segments": n_segments},
        )
    threshold = float(min_snr)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValidationError(
            _("min_snr must be a positive finite number"),
            details={"min_snr": min_snr},
        )

    total = int(array.size)
    count = min(segments, total)
    bounds = np.array_split(np.arange(total), count)

    filtered = np.empty(total, dtype=float)
    segment_snr = np.empty(count, dtype=float)
    segment_dof = np.empty(count, dtype=float)

    for position, indices in enumerate(bounds):
        segment = array[indices]
        length = int(segment.size)
        level = float(segment.mean())
        centred = segment - level

        spectrum = np.fft.rfft(centred)
        power = (spectrum.real**2 + spectrum.imag**2) / float(length)
        frequencies = np.fft.rfftfreq(length, d=1.0)
        nyquist = float(frequencies[-1]) if frequencies[-1] > 0.0 else 0.5

        # 高频段用于拟合噪声模型：周期信号集中在低频，整段拟合会把信号
        # 当成红噪声。
        band = frequencies >= 0.5 * nyquist
        if int(np.count_nonzero(band)) < 2:
            band = np.ones_like(power, dtype=bool)
        rho = _fit_ar1_rho(power[band], frequencies[band], nyquist)
        shape = _ar1_shape(frequencies, nyquist, rho)
        scale = float(np.mean(power[band]) / np.mean(shape[band]))
        if not np.isfinite(scale) or scale < 0.0:
            scale = 0.0
        noise_model = scale * shape

        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(power > 0.0, np.maximum(0.0, 1.0 - noise_model / power), 0.0)
        gain = np.where(np.isfinite(gain), gain, 0.0)

        reconstructed = np.fft.irfft(spectrum * gain, n=length)
        reconstructed = reconstructed - reconstructed.mean()

        squared = gain**2
        signal_power = float(np.sum(np.maximum(power - noise_model, 0.0) * squared))
        noise_power = float(np.sum(noise_model * squared))
        if float(np.sum(power)) <= 0.0:
            snr = 0.0
        elif noise_power > 0.0:
            snr = signal_power / noise_power
        else:
            snr = MAX_REPORTED_SNR
        snr = float(min(snr, MAX_REPORTED_SNR))

        weight = 1.0 if snr >= threshold else max(snr / threshold, 0.0)
        filtered[indices] = level + weight * reconstructed

        segment_snr[position] = snr
        segment_dof[position] = 2.0 * float(np.count_nonzero(gain > 0.0))

    knots = np.array([0.5 * (float(indices[0]) + float(indices[-1])) for indices in bounds])
    positions = np.arange(total, dtype=float)
    signal_to_noise = np.interp(positions, knots, segment_snr)
    effective_dof = np.interp(positions, knots, segment_dof)

    return DynamicFilterResult(
        filtered=filtered,
        signal_to_noise=signal_to_noise,
        effective_dof=effective_dof,
    )
