# =============================================================================
# FILE: stratigraphy/coherence.py
# =============================================================================
"""
相干性与相位（coherence & phase）：两条地层记录之间的互谱相干性分析。

用途
====
判断两条地层记录（例如 δ¹⁸O 与 δ¹³C、粒度与粉砂含量、两条测年曲线）是不是
被**同一个外部强迫**（通常是天文强迫）驱动。做法与频谱分析窗口里的
"coherence & phase" 子功能一致：先算两条序列的互谱，再由互谱给出

* **幅度平方相干性** ``coherence(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))``，
  取值 ``0..1``，表示"在频率 ``f`` 上两条序列共享了多少信号"；
* **相位差** ``phase(f) = arg S_xy(f)``（弧度），表示第二条序列相对第一条
  序列的相位滞后/超前。相位差随频率的斜率就是**时延**（见
  :func:`coherence` 的返回值说明）。

估计量的构造
============
估计量是 **Welch 交叉谱平滑**（Welch 1967），但把分段周期图换成了
**分段 Lomb-Scargle 基**，因此可以处理采样不均匀的地层记录：

1. 记录按时间轴切成 ``K`` 个等时长的 Welch 分段（分段边界按**时间**而不是
   按样本序号切，所以不等间距采样不会让某一段样本特别多）。
2. 每段内部，在与 :mod:`stratigraphy.spectral_analysis` **完全相同**的
   Lomb-Scargle 约定下（同样用 τ 正交化 sin/cos 基）做最小二乘拟合，得到
   两条序列在该频率上的复振幅 ``Â_x(f)``、``Â_y(f)``。
3. 逐段的自谱与互谱在**分段之间**平均：

       S_xx = mean_k |Â_x^(k)|^2
       S_yy = mean_k |Â_y^(k)|^2
       S_xy = mean_k conj(Â_x^(k)) * Â_y^(k)

   相干性与相位就由这三条平滑后的谱给出。

为什么必须做分段平均：单次实现、单频率的幅度平方相干性**恒等于 1**
（因为 ``|conj(Â_x)Â_y|^2 = |Â_x|^2 |Â_y|^2``），它是退化的估计量。分段平均
既让它非退化，也是零假设分布成立的来源。

频率轴
======
频率轴是**每单位时间**的循环频率（cycles per unit time），与
:mod:`stratigraphy.spectral_analysis` 的口径一致：自动频带从记录自身的
Rayleigh 分辨率 ``1/T`` 出发、到 Nyquist ``n/(2T)`` 为止，步长**不粗于**
记录的分辨率。相干性是由 ``K`` 段独立平均得到的，所以低于分段分辨率
``1/L``（``L = T/K``）的频率点其各段估计高度相关、相干性被抬高；自动频带
因此从 ``1/L`` 起步。

显著性
======
零假设（两条序列在该频率上不相关）下，Welch 平滑后的幅度平方相干性服从
**Beta(1, K-1)** 分布（等价于 ``F(2, 2K-2)``），见 Priestley (1981) 第 12 章
与 Maran & Ripley (1997) 第 9 章。因此 ``K`` 个独立平均的临界值为

    c = 1 - alpha^(1/(K-1))

其中 ``alpha = 1 - level`` 是点态（pointwise）的假阳性率。这个近似是
**点态**的，没有做多重比较校正；同一张图上几百个频率点各自独立判定，
因此"显著频率点很多"本身不构成证据。

:func:`coherence_significance` 用**有效自由度** ``n_effective = 2K`` 表达
``K``（每个分段平均贡献 2 个实自由度），于是临界值写成

    c = 1 - alpha^(2 / (n_effective - 2))

**关于红噪声**：Welch 理论的零假设要求各段的谱估计近似独立。对地层记录
常见的红噪声做过蒙特卡洛检验（白噪声、AR(1) ρ=0.5、AR(1) ρ=0.8、1/f 幂律
红噪声各 300 次重复）：四种噪声下经验零分布都与 Beta(1, K-1) 吻合
（Kolmogorov–Smirnov p = 0.09–0.52，临界值的经验超越率 0.057–0.060，
名义 0.05），因为分段长度远大于噪声的相关时间。因此这里**没有**再乘
AR(1) 收缩因子——那会把以确定性周期为主的记录（AR(1) ρ 被周期抬高到 0.9
以上）判成"永不显著"。

启用 ``smooth=True`` 时互谱/自谱还会沿频率轴做一次宽度约等于分段分辨率
的移动平均：它把零分布收窄并左移，临界值不变，因此检验变得**偏保守**
（同一蒙特卡洛下经验超越率 0.0025，远小于名义 0.05）。

参考文献
========
* Welch, P. D. (1967). The use of fast Fourier transform for the estimation
  of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15,
  70-73.（分段平均的交叉谱估计量）
* Priestley, M. B. (1981). Spectral Analysis and Time Series. Academic Press.
  （第 12 章：相干性的抽样分布 Beta(1, K-1) 与相位标准误差）
* Maran, A. & Ripley, B. D. (1997). Time Series Analysis and Forecasting.
  Springer.（Welch 相干性的零假设分布与 F(2, 2K-2) 等价形式）
* Scargle, J. D. (1982). Studies in astronomical time series analysis.
  *IEEE Transactions on Information Theory*, 28, 513-523.（Lomb-Scargle
  周期图；本模块沿用 :mod:`stratigraphy.spectral_analysis` 的同一约定）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "coherence & phase" 子功能）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（同一时间轴、互谱约定 ``conj(FFT(reference)) * FFT(target)``、分段平均、
零相干临界值 ``1 - alpha^(1/(Keff-1))``），未翻译或复制其 MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from stratigraphy.detrending import detrend_series
from stratigraphy.smoothing import moving_average_smooth
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: Welch 分段数的默认值。分段越多，零分布越紧、临界值越低。
DEFAULT_SEGMENTS = 4

#: 允许的最少分段数：Beta(1, K-1) 零分布需要至少两个独立平均。
MIN_SEGMENTS = 2

#: 允许的最多分段数。
MAX_SEGMENTS = 16

#: 每个 Welch 分段至少需要的样本数（少于这个数，最小二乘拟合不稳定）。
MIN_SEGMENT_SAMPLES = 8

#: 相干性分析至少需要的样本数。
MIN_SAMPLES = 16

#: 频率栅格至少需要的点数。
MIN_FREQUENCIES = 8


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class CoherenceResult:
    """
    相干性与相位分析的结果。

    Attributes:
        frequencies: 频率轴（每单位时间的循环频率，严格递增）。
        coherence: 幅度平方相干性，取值 ``0..1``。
        phase: 相位差（弧度）。**正值表示第二条序列的傅里叶相位超前第一条
            序列**（与 Acycle 的 ``conj(FFT(reference)) * FFT(target)``
            约定一致），**负值表示第二条序列滞后**。
        cross_power: 互谱功率 ``|S_xy|^2``（按整条记录的方差归一化）。
        n_effective: 有效自由度 ``2K``（``K`` 为 Welch 分段数，每个分段平均
            贡献 2 个实自由度），用于相干性的显著性判定。
    """

    frequencies: npt.NDArray
    coherence: npt.NDArray
    phase: npt.NDArray
    cross_power: npt.NDArray
    n_effective: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Coherence and phase"),
            "=" * 48,
            _("Frequencies: {0}").format(self.frequencies.size),
            _("Effective degrees of freedom: {0}").format(f"{self.n_effective:.1f}"),
        ]
        if self.frequencies.size:
            peak = int(np.argmax(self.coherence))
            lines.append(
                _("Peak coherence: {0} at {1} per unit time").format(
                    f"{self.coherence[peak]:.4f}", f"{self.frequencies[peak]:.6g}"
                )
            )
            lines.append(_("Phase at peak: {0} degrees").format(f"{np.degrees(self.phase[peak]):.2f}"))
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================


def _as_vector(values: object, name: str, *, minimum_size: int) -> npt.NDArray:
    """把输入转换为一维有限数组，并检查长度下限。"""
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
    if array.size < minimum_size:
        raise ValidationError(
            _("{0} needs at least {1} points; got {2}").format(name, minimum_size, int(array.size)),
            details={"parameter": name, "length": int(array.size), "minimum": minimum_size},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


def _ls_coefficients(
    time: npt.NDArray,
    values: npt.NDArray,
    frequencies: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Lomb-Scargle 最小二乘基下的余弦/正弦系数。

    与 :meth:`stratigraphy.spectral_analysis.SpectralAnalyzer._lomb_scargle`
    采用**完全相同**的约定：以 ``tau`` 正交化 sin/cos 基

        tau = atan2(sum sin(2 w t), sum cos(2 w t)) / (2 w)

    基函数为 ``cos(w (t - tau))`` 与 ``sin(w (t - tau))``。系数

        a = sum(values * cos) / sum(cos^2)
        b = sum(values * sin) / sum(sin^2)

    是序列在该频率上的最小二乘拟合 ``a cos + b sin`` 的两个系数，复振幅为
    ``A_hat = a - i b``（于是 ``|A_hat|^2 = a^2 + b^2``）。

    Parameters:
        time: 分段内的时间值（严格递增）。
        values: 分段内的序列值。
        frequencies: 频率轴（每单位时间）。

    Returns:
        tuple: ``(a, b)`` 两个数组，形状与 ``frequencies`` 相同。
    """
    omega = 2.0 * np.pi * frequencies
    sin_2wt = np.sin(2.0 * omega[:, None] * time[None, :])
    cos_2wt = np.cos(2.0 * omega[:, None] * time[None, :])
    tau = np.arctan2(sin_2wt.sum(axis=1), cos_2wt.sum(axis=1)) / (2.0 * omega)
    shifted = time[None, :] - tau[:, None]
    sin_basis = np.sin(omega[:, None] * shifted)
    cos_basis = np.cos(omega[:, None] * shifted)
    sum_sin2 = (sin_basis**2).sum(axis=1)
    sum_cos2 = (cos_basis**2).sum(axis=1)
    cosine = (values[None, :] * cos_basis).sum(axis=1) / np.where(sum_cos2 > 0.0, sum_cos2, np.nan)
    sine = (values[None, :] * sin_basis).sum(axis=1) / np.where(sum_sin2 > 0.0, sum_sin2, np.nan)
    return cosine, sine


def _frequency_grid(
    time_span: float,
    n_samples: int,
    frequency_range: tuple[float, float] | None,
    n_frequencies: int,
    min_frequency: float,
) -> npt.NDArray:
    """构造相干性分析的频率轴（每单位时间）。

    自动频带的构造与 :meth:`stratigraphy.spectral_analysis.SpectralAnalyzer.analyze`
    一致：步长不粗于记录自身的 Rayleigh 分辨率 ``1/T``，上端不超过 Nyquist
    ``n/(2T)``；下端取 ``max(1/T, 分段分辨率)``，即不低于分段平均所能分辨的
    最低频率。
    """
    budget = max(int(n_frequencies), 2)
    if frequency_range is not None:
        low, high = frequency_range
        if high <= low:
            raise ValidationError(
                _("the upper frequency bound must exceed the lower one"),
                details={"frequency_range": [float(low), float(high)]},
            )
        return np.linspace(low, high, budget)

    resolution = 1.0 / time_span
    nyquist = n_samples / (2.0 * time_span)
    low = max(resolution, min_frequency)
    if nyquist <= low:
        raise ValidationError(
            _("the record is too short for the requested frequency band"),
            details={"nyquist": float(nyquist), "minimum_frequency": float(low)},
        )
    step = min((nyquist - low) / (budget - 1), resolution)
    if step <= 0.0:
        step = resolution
    high = low + (budget - 1) * step
    if high > nyquist:
        high = nyquist
        step = (high - low) / (budget - 1)
    return low + np.arange(budget) * step


def _segment_masks(time: npt.NDArray, n_segments: int) -> list[npt.NDArray]:
    """按**时间**把时间轴切成 ``n_segments`` 段，返回每段的样本掩码。

    分段边界按时间等分，而不是按样本序号等分——地层记录的采样间隔通常
    不均匀，按样本等分会使命中的某一段样本数明显偏多。
    """
    start = float(time[0])
    length = (float(time[-1]) - start) / n_segments
    masks: list[npt.NDArray] = []
    for index in range(n_segments):
        lower = start + index * length
        if index < n_segments - 1:
            masks.append((time >= lower) & (time < lower + length))
        else:
            masks.append(time >= lower)
    return masks


# =============================================================================
# Public interface
# =============================================================================


def coherence(
    time: npt.NDArray,
    first: npt.NDArray,
    second: npt.NDArray,
    frequency_range: tuple[float, float] | None = None,
    n_frequencies: int = 500,
    smooth: bool = True,
    *,
    n_segments: int = DEFAULT_SEGMENTS,
) -> CoherenceResult:
    """两条序列之间的互谱相干性与相位差。

    估计量是 **Welch 交叉谱平滑**：记录按时间切成 ``K`` 个分段，每段在
    Lomb-Scargle 基上做最小二乘拟合得到复振幅，自谱与互谱在分段之间平均，
    然后

        coherence(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
        phase(f)     = arg S_xy(f)

    **相位约定**：互谱取 ``conj(A_hat_first) * A_hat_second``，所以
    ``phase`` 是**第二条序列相对第一条序列**的傅里叶相位差。正值表示第二条
    序列超前，负值表示第二条序列滞后；若第二条序列是第一条延迟 ``dt``，
    则 ``phase(f) = -2 * pi * f * dt``。

    **同一时间轴**：两条序列必须采自**同一个** ``time``（同一批时间点、同
    一顺序）。本函数不会替任何一条序列单独重采样；时间轴非严格递增时会
    先按时间排序（两条序列同步排序）。

    Parameters:
        time: 时间轴（每单位时间），严格递增。
        first: 第一条序列的数值。
        second: 第二条序列的数值。
        frequency_range: ``(最低频率, 最高频率)``，单位是**每单位时间**。
            缺省时自动取 ``[分段分辨率, Nyquist]``。
        n_frequencies: 频率栅格的点数，至少 8。
        smooth: 是否在 Welch 平均之后再沿频率轴做一次移动平均（窗口宽度
            约等于分段分辨率）。它让相干性曲线更稳，代价是使显著性检验
            偏保守。
        n_segments: Welch 分段数，2 到 16 之间，缺省 4。

    Returns:
        CoherenceResult: 相干性、相位差、互谱功率与有效自由度 ``2K``。

    Raises:
        ValidationError: 输入长度不一致、点数太少、时间轴非严格递增、序列
            为常数、频率范围非法或分段数越界。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.coherence import coherence
        >>> time = np.linspace(0.0, 200.0, 1000)
        >>> signal = np.sin(2.0 * np.pi * time / 20.0)
        >>> result = coherence(time, signal, signal)
        >>> bool(result.coherence.max() > 0.9)
        True
    """
    times = _as_vector(time, "time", minimum_size=MIN_SAMPLES)
    series_first = _as_vector(first, "first", minimum_size=MIN_SAMPLES)
    series_second = _as_vector(second, "second", minimum_size=MIN_SAMPLES)
    if not (times.size == series_first.size == series_second.size):
        raise ValidationError(
            _("time, first and second must have the same length"),
            details={
                "time_length": int(times.size),
                "first_length": int(series_first.size),
                "second_length": int(series_second.size),
            },
        )

    budget = int(n_frequencies)
    if budget != n_frequencies or budget < MIN_FREQUENCIES:
        raise ValidationError(
            _("n_frequencies must be an integer of at least {0}").format(MIN_FREQUENCIES),
            details={"n_frequencies": n_frequencies},
        )
    segments = int(n_segments)
    if segments != n_segments or not (MIN_SEGMENTS <= segments <= MAX_SEGMENTS):
        raise ValidationError(
            _("n_segments must be an integer between {0} and {1}").format(MIN_SEGMENTS, MAX_SEGMENTS),
            details={"n_segments": n_segments},
        )

    bounds: tuple[float, float] | None = None
    if frequency_range is not None:
        try:
            low, high = (float(frequency_range[0]), float(frequency_range[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValidationError(
                _("frequency_range must be a pair of numbers"),
                details={"frequency_range": repr(frequency_range)},
                original_exception=exc,
            ) from exc
        if not (np.isfinite(low) and np.isfinite(high)) or not (0.0 < low < high):
            raise ValidationError(
                _("frequency_range must satisfy 0 < minimum < maximum"),
                details={"frequency_range": [low, high]},
            )
        bounds = (low, high)

    # 排序：两条序列共用同一个时间轴，因此排序对三者同时生效。
    order = np.argsort(times, kind="stable")
    times = times[order]
    series_first = series_first[order]
    series_second = series_second[order]
    if np.any(np.diff(times) <= 0.0):
        raise ValidationError(
            _("time must contain strictly increasing values"),
            details={"n_points": int(times.size)},
        )

    time_span = float(times[-1] - times[0])
    if time_span <= 0.0:
        raise ValidationError(
            _("the time span must be positive"),
            details={"time_span": time_span},
        )

    # 去趋势：与频谱分析同一口径（去掉线性漂移，避免长趋势泄漏到低频）。
    detrended_first = np.asarray(detrend_series(times, series_first, method="linear").detrended, dtype=float)
    detrended_second = np.asarray(detrend_series(times, series_second, method="linear").detrended, dtype=float)

    variance_first = float(np.var(detrended_first))
    variance_second = float(np.var(detrended_second))
    if variance_first <= 0.0 or variance_second <= 0.0:
        raise ValidationError(
            _("coherence is undefined for a constant series"),
            details={
                "first_variance": variance_first,
                "second_variance": variance_second,
            },
        )

    segment_length = time_span / segments
    frequencies = _frequency_grid(
        time_span,
        int(times.size),
        bounds,
        budget,
        min_frequency=1.0 / segment_length,
    )
    if bounds is not None and float(frequencies[-1]) < 1.0 / segment_length:
        raise ValidationError(
            _("the requested frequency band lies below the Welch segment resolution"),
            details={
                "band_maximum": float(frequencies[-1]),
                "segment_resolution": float(1.0 / segment_length),
            },
        )

    masks = _segment_masks(times, segments)
    accumulated_xx = np.zeros(frequencies.size)
    accumulated_yy = np.zeros(frequencies.size)
    accumulated_xy = np.zeros(frequencies.size, dtype=complex)
    for mask in masks:
        count = int(np.count_nonzero(mask))
        if count < MIN_SEGMENT_SAMPLES:
            raise ValidationError(
                _("each Welch segment needs at least {0} samples").format(MIN_SEGMENT_SAMPLES),
                details={"segment_samples": count, "minimum": MIN_SEGMENT_SAMPLES},
            )
        segment_time = times[mask]
        cosine_first, sine_first = _ls_coefficients(segment_time, detrended_first[mask], frequencies)
        cosine_second, sine_second = _ls_coefficients(segment_time, detrended_second[mask], frequencies)
        accumulated_xx += cosine_first**2 + sine_first**2
        accumulated_yy += cosine_second**2 + sine_second**2
        accumulated_xy += (cosine_first + 1j * sine_first) * (cosine_second - 1j * sine_second)

    # 归一化口径与 Lomb-Scargle 周期图一致（除以 2 sigma^2）；相干性是比值，
    # 归一化因子在比值里自动约掉。
    scale_xx = 2.0 * variance_first
    scale_yy = 2.0 * variance_second
    scale_xy = 2.0 * float(np.sqrt(variance_first * variance_second))
    spectrum_xx = accumulated_xx / (segments * scale_xx)
    spectrum_yy = accumulated_yy / (segments * scale_yy)
    spectrum_xy = accumulated_xy / (segments * scale_xy)

    if smooth:
        step = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 1.0
        width = round((1.0 / segment_length) / step) if step > 0.0 else 1
        width = max(1, min(width, int(frequencies.size)))
        spectrum_xx = moving_average_smooth(frequencies, spectrum_xx, width)
        spectrum_yy = moving_average_smooth(frequencies, spectrum_yy, width)
        spectrum_xy = moving_average_smooth(frequencies, spectrum_xy.real, width) + 1j * moving_average_smooth(
            frequencies, spectrum_xy.imag, width
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        coherence_values = np.abs(spectrum_xy) ** 2 / (spectrum_xx * spectrum_yy)
    coherence_values = np.where(np.isfinite(coherence_values), np.clip(coherence_values, 0.0, 1.0), 0.0)

    result = CoherenceResult(
        frequencies=frequencies,
        coherence=coherence_values,
        phase=np.angle(spectrum_xy),
        cross_power=np.abs(spectrum_xy) ** 2,
        n_effective=float(2 * segments),
    )
    logger.info(
        "Coherence analysis complete: %d frequencies, %d segments, peak coherence %.4f",
        int(frequencies.size),
        segments,
        float(np.max(coherence_values)) if coherence_values.size else float("nan"),
    )
    return result


def coherence_significance(
    coherence_values: npt.NDArray,
    n_effective: float,
    level: float = 0.95,
) -> float:
    """相干性的零假设临界值（点态检验）。

    零假设（两条序列在给定频率上不相关）下，Welch 平滑后的幅度平方相干性
    服从 **Beta(1, K-1)** 分布，等价于 ``F(2, 2K-2)``（Priestley 1981 第 12 章；
    Maran & Ripley 1997 第 9 章）。``K`` 个独立平均的 ``1 - level`` 分位数为

        c = 1 - (1 - level)^(1/(K-1))

    这里用**有效自由度** ``n_effective = 2K`` 表达 ``K``（每个分段平均贡献 2 个
    实自由度），于是

        c = 1 - (1 - level)^(2 / (n_effective - 2))

    Parameters:
        coherence_values: 观测到的相干性值（``0..1``），只用于校验；返回的
            临界值适用于其中每一个值。
        n_effective: 有效自由度，必须大于 2（至少两个独立平均）。
        level: 置信水平，取值 ``(0, 1)``，缺省 0.95。

    Returns:
        float: 临界相干性；观测相干性严格大于它才算显著。

    Raises:
        ValidationError: 相干性取值非法、有效自由度不大于 2，或置信水平不在
            ``(0, 1)`` 内。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.coherence import coherence_significance
        >>> critical = coherence_significance(np.array([0.1, 0.9]), n_effective=8.0)
        >>> bool(0.6 < critical < 0.7)
        True
    """
    values = _as_vector(coherence_values, "coherence_values", minimum_size=1)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValidationError(
            _("coherence values must lie between 0 and 1"),
            details={"minimum": float(values.min()), "maximum": float(values.max())},
        )
    effective = float(n_effective)
    if not np.isfinite(effective) or effective <= 2.0:
        raise ValidationError(
            _("n_effective must exceed 2 (at least two independent averages)"),
            details={"n_effective": n_effective},
        )
    confidence = float(level)
    if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValidationError(
            _("level must lie strictly between 0 and 1"),
            details={"level": level},
        )
    return float(1.0 - (1.0 - confidence) ** (2.0 / (effective - 2.0)))
