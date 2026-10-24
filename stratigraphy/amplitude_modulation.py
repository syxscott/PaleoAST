# =============================================================================
# FILE: stratigraphy/amplitude_modulation.py
# =============================================================================
"""
振幅调制（amplitude modulation）分析。

把一条记录拆成"载波"（carrier）与"包络"（envelope）两半：载波是快速振荡的
那部分，包络是它振幅随时间的缓慢变化。调幅信号

    x(t) = a(t) * cos(2 * pi * fc * t + phi(t))

的解析信号（analytic signal）近似为 ``a(t) * exp(i * (2 * pi * fc * t + phi(t)))``，
所以解析信号的模就是调制包络 ``a(t)``，解析信号的辐角就是瞬时相位。地质记录
里的用法：一条钻孔记录里，某个轨道周期（载波）的能量/振幅随地质时间起伏，
起伏本身就是更深一级的调制信号——例如"岁差周期（载波）的振幅被百万年尺度
的偏心率周期调制"。

本模块做三件事：

* :func:`tanner_hilbert` —— 对一列等间距（或可重采样为等间距的）数据做 Hilbert
  变换，得到瞬时振幅包络与瞬时相位。Hilbert 变换的定义与 Taner-Hilbert 方法
  一致（Taner, 2000；Acycle 的 ``tanerhilbertML.m`` 也走同一路线）。
* :func:`amplitude_modulation` —— 先按可选的频带做零相位带通滤波，再取包络，
  同时把带通后的载波一并返回，方便画"包络 + 载波"图。
* :class:`AmplitudeModulationResult` —— 结果容器。

方法依据：

* Kodama, K.P. & Hinnov, L.A. (2015). *Rock Magnetic Cyclostratigraphy*.
  Wiley-Blackwell. —— 轨道周期的振幅调制是旋回地层学里"调幅"分析的经典
  用途（Taner 加权带通 + 包络提取）。
* Taner, T. (2000). Attributes revisited. Rock Solid Images, Inc. ——
  Taner-Hilbert 方法：带通滤波后取解析信号，得到包络与瞬时频率。
* Hill, H.A. et al. (2022). Acycle: a new open-source MATLAB toolbox for
  astrochronology. —— 本模块按 Acycle 的 ``acycleAmplitudeModulation.m`` /
  ``tanerhilbertML.m`` 的算法描述独立实现，未移植其 GPL 源码。

单位与约定
----------

* ``time`` 的单位是 Ma（与仓库其它模块一致），``band`` 用"每 Ma 的周期数"
  表示，即 ``f = 1 / T``。
* ``band = (f_low, f_high)`` 是带通滤波的两个带边频率（周期数 / Ma），必须
  满足 ``0 < f_low < f_high < 奈奎斯特频率``。
* 相位以弧度给出，落在 ``(-pi, pi]``；需要连续相位时自己解包
  （``numpy.unwrap``）。

已知边界效应
------------

解析信号用 FFT 计算，隐含"记录首尾相接"的假设：记录长度不是周期的整数倍
时，首尾会出现振铃式的包络过冲。需要干净的包络时，请让记录覆盖整数个周期，
或者自行截掉两端若干个周期后再调用本模块。

References:
    1. Kodama, K.P. & Hinnov, L.A. (2015). Rock Magnetic Cyclostratigraphy.
       Wiley-Blackwell.
    2. Taner, T. (2000). Attributes revisited. Rock Solid Images, Inc.
    3. Hill, H.A. et al. (2022). Acycle: a new open-source MATLAB toolbox for
       astrochronology. (Amplitude modulation / Taner-Hilbert.)
    4. Schreiber, T. & Schmitz, A. (2000). Broadband time-series analysis with
       the Hilbert transform. Vistas in Astronomy, 44: 3139-3174. (Hilbert
       transform and instantaneous amplitude/phase.)
    5. Oppenheim, A.V. & Schafer, R.W. (2010). Discrete-Time Signal Processing
       (3rd ed.). Pearson. (Butterworth band-pass design, zero-phase
       filtering by forward-backward filtering.)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, hilbert, sosfiltfilt

from config.i18n import _
from utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

#: 最少需要多少个有效样本。Hilbert 变换需要至少几个样本来构造解析信号，
#: 带通滤波（零相位前后向滤波）需要更多样本。
MIN_SAMPLES = 8

#: 带通滤波的默认阶数（Butterworth）。用二阶节级联（sos）实现，避免高阶
#: 带通在 ``ba`` 形式下的数值失稳。
DEFAULT_FILTER_ORDER = 4

#: 采样是否"等间距"的相对容差：相邻间隔相对中位间隔的最大偏离超过它，
#: 就认为是不等间距采样（先线性重采样到等间距栅格）。
REGULARITY_TOLERANCE = 1e-6


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class AmplitudeModulationResult:
    """振幅调制的结果容器。

    Attributes:
        times: 时间轴，与输入等长（若做了重采样，则仍返回原始时间）。
        envelope: 瞬时振幅包络，长度与 ``times`` 相同。
        carrier: 载波：给了 ``band`` 时是零相位带通滤波后的信号，否则是
            去均值后的原始序列。
    """

    times: npt.NDArray
    envelope: npt.NDArray
    carrier: npt.NDArray

    def modulation_depth(self) -> float:
        """返回调幅深度 ``(max - min) / mean``。

        纯载波（没有调制）时该值接近 0；均值为 1 的正弦调制（包络在
        ``1 - m`` 与 ``1 + m`` 之间摆动）该值等于调制指数 ``m`` 的两倍。
        """
        mean = float(np.mean(self.envelope))
        if mean <= 0.0:
            return 0.0
        return float((np.max(self.envelope) - np.min(self.envelope)) / mean)

    def summary(self) -> str:
        """生成摘要文本。"""
        return "\n".join(
            [
                _("Amplitude Modulation"),
                "=" * 50,
                _("Points: {0}").format(int(self.times.size)),
                _("Envelope mean: {0:.6g}").format(float(np.mean(self.envelope))),
                _("Modulation depth: {0:.6g}").format(self.modulation_depth()),
                _("Carrier std: {0:.6g}").format(float(np.std(self.carrier))),
            ]
        )


# =============================================================================
# 输入准备
# =============================================================================


def _prepare_series(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """校验并整理输入：去掉非有限值、按时间排序。"""
    t = np.asarray(time, dtype=float).flatten()
    v = np.asarray(values, dtype=float).flatten()
    if t.size != v.size:
        raise DataValidationError(
            _("time and values must have the same length; got {0} and {1}").format(t.size, v.size)
        )
    if t.size == 0:
        raise DataValidationError(_("the series is empty"))
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < MIN_SAMPLES:
        raise DataValidationError(
            _("an amplitude modulation needs at least {0} finite points; got {1}").format(MIN_SAMPLES, t.size),
            details={"n_finite": int(t.size)},
        )
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if t.size < 2 or bool(np.any(np.diff(t) <= 0.0)):
        raise DataValidationError(_("time must be strictly increasing and contain no duplicate timestamps"))
    if float(np.ptp(v)) == 0.0:
        raise DataValidationError(_("the series is constant, so it has no amplitude to modulate"))
    return t, v


def _is_regular(t: npt.NDArray, dt: float) -> bool:
    """判断采样是否等间距（相对中位间隔的最大偏离是否可忽略）。"""
    steps = np.diff(t)
    deviation = float(np.max(np.abs(steps - dt)))
    return deviation <= REGULARITY_TOLERANCE * dt


def _uniform_grid(t: npt.NDArray, v: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """把序列放到等间距栅格上。

    Hilbert 变换按离散时间定义，隐含等间距采样。不等间距时先按中位间隔
    线性重采样到等间距栅格，滤波与 Hilbert 变换都在该栅格上做，最后再插值
    回原始时间轴。

    Returns:
        ``(grid_time, grid_values)``；等间距时 ``grid_time`` 与 ``t`` 相同。
    """
    dt = float(np.median(np.diff(t)))
    if _is_regular(t, dt):
        return t, v
    span = float(t[-1] - t[0])
    n = int(np.floor(span / dt + 1e-9)) + 1
    grid = t[0] + np.arange(n, dtype=float) * dt
    logger.warning(
        "amplitude_modulation: unevenly sampled series, resampling %d points onto a "
        "uniform grid with dt=%.6g before the Hilbert transform",
        int(t.size),
        dt,
    )
    return grid, np.interp(grid, t, v)


def _analytic_signal(values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """解析信号的瞬时振幅与瞬时相位。

    解析信号 ``z = x + i * H(x)``（``H`` 为 Hilbert 变换）由
    ``scipy.signal.hilbert`` 给出；瞬时振幅是 ``|z|``，瞬时相位是
    ``angle(z)``。
    """
    analytic = hilbert(values)
    return np.abs(analytic), np.angle(analytic)


def _bandpass(values: npt.NDArray, dt: float, band: tuple[float, float]) -> npt.NDArray:
    """零相位 Butterworth 带通滤波。

    参数:
        values: 待滤波的序列（已去均值）。
        dt: 采样间隔（时间单位）。
        band: ``(f_low, f_high)``，频率单位为"每时间单位的周期数"。

    Returns:
        零相位（前后向）滤波后的序列。
    """
    f_low, f_high = band
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    if not (0.0 < f_low < f_high):
        raise DataValidationError(_("band must satisfy 0 < f_low < f_high; got ({0}, {1})").format(f_low, f_high))
    if f_high >= nyquist:
        raise DataValidationError(
            _("band upper frequency {0} must stay below the Nyquist frequency {1}").format(f_high, nyquist),
            details={"f_high": f_high, "nyquist": nyquist},
        )
    sos = butter(
        DEFAULT_FILTER_ORDER,
        [f_low, f_high],
        btype="band",
        fs=fs,
        output="sos",
    )
    try:
        filtered = sosfiltfilt(sos, values)
    except ValueError as exc:
        raise DataValidationError(
            _("the series is too short for a zero-phase band-pass filter: {0}").format(exc)
        ) from exc
    return np.asarray(filtered, dtype=float)


# =============================================================================
# 主入口
# =============================================================================


def tanner_hilbert(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """对序列做 Hilbert 变换，返回瞬时振幅包络与瞬时相位。

    序列先去掉均值，再做解析信号变换；瞬时振幅是解析信号的模，瞬时相位是
    解析信号的辐角（弧度，落在 ``(-pi, pi]``）。不等间距的时间轴会先被线性
    重采样到等间距栅格，结果再插值回原始时间轴。

    Parameters:
        time: 时间轴，与 ``values`` 等长。
        values: 待分析的序列。

    Returns:
        ``(envelope, phase)``：两个长度都与 ``time`` 相同的数组。

    Raises:
        DataValidationError: 序列为空、长度不一致、点数太少、时间轴非严格
            递增或序列是常数。
    """
    t, v = _prepare_series(time, values)
    grid_t, grid_v = _uniform_grid(t, v - float(np.mean(v)))
    envelope, phase = _analytic_signal(grid_v)
    if grid_t.size != t.size:
        envelope = np.interp(t, grid_t, envelope)
        unwrapped = np.unwrap(phase)
        phase = np.interp(t, grid_t, unwrapped)
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
    logger.info(
        "tanner_hilbert: %d points, envelope mean=%.6g",
        int(t.size),
        float(np.mean(envelope)),
    )
    return np.asarray(envelope, dtype=float), np.asarray(phase, dtype=float)


def amplitude_modulation(
    time: npt.NDArray,
    values: npt.NDArray,
    band: tuple[float, float] | None = None,
) -> AmplitudeModulationResult:
    """计算振幅调制：可选带通滤波 + 瞬时振幅包络。

    Parameters:
        time: 时间轴，与 ``values`` 等长。
        values: 待分析的序列。
        band: 可选的带通频带 ``(f_low, f_high)``，频率单位为"每时间单位的
            周期数"（``f = 1 / T``）。给了 ``band`` 时先做零相位 Butterworth
            带通滤波再取包络；不给则对整条序列（去均值后）取包络。

    Returns:
        AmplitudeModulationResult: 时间轴、包络与载波。

    Raises:
        DataValidationError: 输入不合法，或 ``band`` 不是合法的频带。
    """
    t, v = _prepare_series(time, values)
    centered = v - float(np.mean(v))
    grid_t, grid_v = _uniform_grid(t, centered)

    if band is None:
        carrier = grid_v
    else:
        try:
            f_low, f_high = (float(band[0]), float(band[1]))
        except (TypeError, IndexError, ValueError) as exc:
            raise DataValidationError(
                _("band must be a pair of frequencies (f_low, f_high); got {0}").format(band)
            ) from exc
        if not (np.isfinite(f_low) and np.isfinite(f_high)):
            raise DataValidationError(_("band frequencies must be finite numbers"))
        grid_dt = float(np.median(np.diff(grid_t)))
        carrier = _bandpass(grid_v, grid_dt, (f_low, f_high))

    envelope, _phase = _analytic_signal(carrier)
    if grid_t.size != t.size:
        envelope = np.interp(t, grid_t, envelope)
        carrier = np.interp(t, grid_t, carrier)

    logger.info(
        "amplitude_modulation: %d points, band=%s, envelope mean=%.6g",
        int(t.size),
        "none" if band is None else str(band),
        float(np.mean(envelope)),
    )
    return AmplitudeModulationResult(
        times=t,
        envelope=np.asarray(envelope, dtype=float),
        carrier=np.asarray(carrier, dtype=float),
    )
