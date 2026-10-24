# =============================================================================
# FILE: stratigraphy/evolutionary.py
# =============================================================================
"""
演化谱（evolutionary spectrum）分析。

演化谱就是"沿时间轴滑动的谱"：把一条记录按固定长度的时间窗切开，每个窗内
独立估计一次功率谱，再把所有窗的谱按时间排成一张"频率 × 时间"图。它回答的
问题是"这个周期在什么时候存在、什么时候消失、周期本身有没有变长变短"——
轨道节律随地质时间漂移、沉积速率变化引起的周期伸缩，都靠它来看。

方法依据：

* Kodama, K.P. & Hinnov, L.A. (2015). *Rock Magnetic Cyclostratigraphy*.
  Wiley-Blackwell. —— 演化谱的基本思想：常规谱估计在时间轴上滑动，每个
  时间窗给出一条谱。
* Hill, H.A. et al. (2022). Acycle: a new open-source MATLAB toolbox for
  astrochronology. —— 本模块按 Acycle 的 evofft / evoplomb / evopmtm 系列
  算法描述独立实现，未移植其 GPL 源码。

与 Acycle 的对应关系与差异：

* 滑窗方式相同：窗长 ``window``（与 ``time`` 同单位），窗间重叠，窗中心
  时间取窗内中间样本的时间（Acycle 的 ``taxis`` / ``y_grid``）。
* Acycle 用绝对步长 ``step``（样本数）表示窗的推进，本模块用重叠比例
  ``overlap`` 表示，二者等价（``step = (1 - overlap) * window``）。
* Acycle 的频率范围由用户给的 ``fmin`` / ``fmax`` 指定；本模块按记录的
  Rayleigh 分辨率自动生成栅格，用户只给 ``n_frequencies``。

频率栅格为什么不随采样密度漂移（本模块最关键的设计）
--------------------------------------------------------
栅格的步长由**记录本身**决定，而不是由采样点数决定：

* 窗长 ``W``（时间单位）给出 Rayleigh 分辨率 ``1 / W``：长度 ``W`` 的
  记录无法把比 ``1 / W`` 更细的频率分辨出来。
* 栅格步长取 ``(1 / W) / INTERPOLATION``（默认零填充 4 倍，与 Acycle
  ``evofft.m`` 的 ``kpts = 4 * mpts`` 一致），因此**加密采样不会改变
  任何一个栅格点的位置**，只会让栅格向上延伸（上限是采样奈奎斯特频率与
  ``n_frequencies`` 中较小者）。
* 栅格的下限是 ``1 / W``：比窗长更长的周期在这个窗里本来就不可分辨。

这正是本仓库此前修过的一个真实缺陷的教训：频率栅格一旦由采样密度决定，
把同一条记录采得更密反而会让周期估得更差（见
``stratigraphy/spectral_analysis.py`` 中同样的注释）。

三个估计器共用同一套频率栅格
------------------------------
FFT（零填充周期图）、Lomb-Scargle 周期图、Thomson 多锥谱（MTM）在同一
频率栅格上求值，所以三者的谱可以直接相减、相加、逐频率比较。多锥分支复用
``stratigraphy.cycles`` 里的 DPSS 构造与 Thomson 权重，数值上与
``cycles.multitaper_spectrum`` 一致（差别只有两处：窗先做了线性去趋势，
以及锥乘序列用零填充 DFT 在公共栅格上求值，而不是用记录自身 FFT 栅格）。

不等间距时间轴
--------------
Lomb-Scargle 分支原生支持不等间距采样；FFT 与 MTM 分支需要等间距数据，
遇到不等间距时本模块自动改用 Lomb-Scargle 分支并在日志中给出提示，
结果的 ``method`` 字段记录实际使用的估计器。

References:
    1. Kodama, K.P. & Hinnov, L.A. (2015). Rock Magnetic Cyclostratigraphy.
       Wiley-Blackwell.
    2. Hill, H.A. et al. (2022). Acycle: a new open-source MATLAB toolbox
       for astrochronology. (Evolutionary FFT / Lomb-Scargle / MTM.)
    3. Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
       Proceedings of the IEEE 70: 1055-1096. (Multitaper method.)
    4. Percival, D.B. & Walden, A.T. (1993). Spectral Analysis for Physical
       Applications. Cambridge University Press. (DPSS construction.)
    5. Horne, J.H. & Baliunas, S.L. (1986). Iterative techniques for
       frequency domain studies of unevenly spaced data. Astrophysics
       Journal 302: 755-763. (Lomb-Scargle periodogram.)
    6. Press, W.H. et al. (1992). Numerical Recipes in C, 2nd ed.
       Cambridge University Press. (Rayleigh resolution of a finite record.)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import periodogram

from config.i18n import _
from stratigraphy.cycles import dpss_tapers
from stratigraphy.spectral_analysis import SpectralAnalyzer, _linear_detrend
from utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

#: 零填充倍数：栅格步长 = Rayleigh 分辨率 / 该倍数。Acycle 的 ``evofft.m``
#: 用 ``kpts = 4 * mpts``，``evofftML.m`` 用 10 倍，这里取 4 倍。
INTERPOLATION = 4

#: 多锥分支的时间带宽积 NW（Acycle ``evopmtmML.m`` 的默认参数同量级）。
MTM_NW = 3.0

#: 一个时间窗至少要有的样本数（多锥估计至少需要 16 个样本）。
MIN_WINDOW_SAMPLES = 16

#: 未指定窗长时，窗长取记录总时长的这个比例（相当于约 4 个窗铺满记录）。
DEFAULT_WINDOW_FRACTION = 0.25

#: 采样是否"等间距"的相对容差：相邻间隔相对中位间隔的最大偏离超过它，
#: 就认为是不等间距采样（改用 Lomb-Scargle 分支）。
REGULARITY_TOLERANCE = 1e-6

#: 支持的估计器名称。
METHODS = ("fft", "lombscargle", "mtm")


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class EvolutionarySpectrumResult:
    """演化谱的结果容器。

    Attributes:
        times: 每个时间窗的中心时间，长度 ``n_windows``。
        frequencies: 所有时间窗共用的频率栅格（周期 / 时间单位），长度
            ``n_freqs``。栅格由记录时长决定，与采样密度无关。
        power: 功率谱矩阵，形状 ``(n_windows, n_freqs)``。
        method: 实际使用的估计器，``"fft"`` / ``"lombscargle"`` / ``"mtm"``。
    """

    times: npt.NDArray
    frequencies: npt.NDArray
    power: npt.NDArray
    method: str

    def peak_periods(self, threshold: float = 0.5) -> list[float]:
        """返回每个时间窗的主周期（按时间顺序）。

        第 ``i`` 个元素对应 ``times[i]``。一个窗口的主周期取该窗口谱中
        功率最大的频率所对应的周期。``threshold`` 是相对功率阈值：主峰
        功率低于全部窗口最强主峰的 ``threshold`` 倍时，说明该时段没有
        可信的周期，用 ``nan`` 表示。

        Parameters:
            threshold: 相对功率阈值，取值 ``(0, 1]``。

        Returns:
            每个时间窗的主周期列表，长度与窗口数相同。
        """
        if self.power.size == 0 or self.frequencies.size == 0:
            return []
        if not 0.0 < threshold <= 1.0:
            raise DataValidationError(_("threshold must lie in (0, 1]; got {0}").format(threshold))
        peak_idx = np.argmax(self.power, axis=1)
        peak_power = self.power[np.arange(self.power.shape[0]), peak_idx]
        strongest = float(np.max(peak_power))
        periods: list[float] = []
        for idx, power in zip(peak_idx, peak_power, strict=True):
            if strongest <= 0.0 or float(power) >= threshold * strongest:
                periods.append(float(1.0 / self.frequencies[int(idx)]))
            else:
                periods.append(float("nan"))
        return periods

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [
            _("Evolutionary Spectrum"),
            "=" * 50,
            _("Method: {0}").format(self.method),
            _("Windows: {0}").format(self.times.size),
            _("Frequencies: {0}").format(self.frequencies.size),
            _("Frequency range: {0:.6g} - {1:.6g}").format(float(self.frequencies[0]), float(self.frequencies[-1])),
            _("Window span: {0:.6g} - {1:.6g}").format(float(self.times[0]), float(self.times[-1])),
        ]
        peaks = self.peak_periods()
        finite = [p for p in peaks if np.isfinite(p)]
        if finite:
            lines.append(_("Dominant periods: {0:.6g} - {1:.6g}").format(min(finite), max(finite)))
        return "\n".join(lines)


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
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < MIN_WINDOW_SAMPLES:
        raise DataValidationError(
            _("an evolutionary spectrum needs at least {0} finite points; got {1}").format(MIN_WINDOW_SAMPLES, t.size),
            details={"n_finite": int(t.size)},
        )
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if t.size < 2 or bool(np.any(np.diff(t) <= 0.0)):
        raise DataValidationError(_("time must be strictly increasing and contain no duplicate timestamps"))
    if float(np.ptp(v)) == 0.0:
        raise DataValidationError(_("the series is constant, so it has no periodicity to track"))
    return t, v


def _is_regular(t: npt.NDArray, dt: float) -> bool:
    """判断采样是否等间距（相对中位间隔的最大偏离是否可忽略）。"""
    steps = np.diff(t)
    deviation = float(np.max(np.abs(steps - dt)))
    return deviation <= REGULARITY_TOLERANCE * dt


def _resolve_window(window: float | None, t: npt.NDArray, dt: float) -> float:
    """确定窗长（时间单位），并校验它落在记录范围内。"""
    span = float(t[-1] - t[0])
    if window is None:
        if span <= 0.0:
            raise DataValidationError(_("the record must span a positive time range"))
        resolved = span * DEFAULT_WINDOW_FRACTION
    else:
        resolved = float(window)
        if not np.isfinite(resolved):
            raise DataValidationError(_("window must be a finite number"))
        if resolved <= 0.0:
            raise DataValidationError(_("window must be greater than 0; got {0}").format(resolved))
    if resolved > span:
        raise DataValidationError(
            _("window ({0}) must not be longer than the record ({1})").format(resolved, span),
            details={"window": resolved, "record_span": span},
        )
    return resolved


def _window_layout(t: npt.NDArray, dt: float, window: float, overlap: float) -> tuple[list[tuple[int, int]], bool]:
    """计算每个时间窗的样本区间。

    等间距数据按样本数切窗（与 Acycle 的 ``mpts`` / ``inc`` 一致）；不等
    间距数据按时间区间切窗。

    Returns:
        ``(windows, regular)``：windows 是 ``(start, stop)`` 的样本下标列表
        （左闭右开），regular 表示是否等间距采样。
    """
    step_time = window * (1.0 - overlap)
    if step_time <= 0.0:
        raise DataValidationError(_("overlap must be smaller than 1; got {0}").format(overlap))
    if _is_regular(t, dt):
        mpts = int(np.round(window / dt))
        if mpts < MIN_WINDOW_SAMPLES:
            raise DataValidationError(
                _("a window of {0} time units holds only {1} samples, fewer than the {2} needed").format(
                    window, mpts, MIN_WINDOW_SAMPLES
                ),
                details={"window": window, "samples_per_window": mpts},
            )
        if mpts > t.size:
            raise DataValidationError(
                _("the window holds {0} samples but the record only has {1}").format(mpts, t.size),
                details={"samples_per_window": mpts, "n_points": int(t.size)},
            )
        step = max(1, int(np.round(mpts * (1.0 - overlap))))
        windows = [(s, s + mpts) for s in range(0, t.size - mpts + 1, step)]
        return windows, True

    windows = []
    start_time = float(t[0])
    while start_time < float(t[-1]):
        i0 = int(np.searchsorted(t, start_time, side="left"))
        i1 = int(np.searchsorted(t, start_time + window, side="left"))
        if i1 - i0 >= MIN_WINDOW_SAMPLES:
            windows.append((i0, i1))
        start_time += step_time
    if not windows:
        raise DataValidationError(
            _("no time window holds the {0} samples a spectral estimate needs").format(MIN_WINDOW_SAMPLES),
            details={"window": window, "n_points": int(t.size)},
        )
    return windows, False


# =============================================================================
# 频率栅格
# =============================================================================


def _frequency_grid(window_span: float, dt: float, n_frequencies: int) -> tuple[npt.NDArray, npt.NDArray, int]:
    """生成所有时间窗共用的频率栅格。

    栅格步长 = Rayleigh 分辨率 / ``INTERPOLATION``，只由记录时长决定；零
    填充 FFT 的长度取 ``INTERPOLATION * n_samples``，于是零填充 FFT 的
    频率轴与该栅格逐点重合，不需要任何插值。

    Returns:
        ``(frequencies, bin_indices, n_fft)``：bin_indices 是零填充 FFT
        中对应栅格点的下标。
    """
    resolution = 1.0 / window_span
    step = resolution / INTERPOLATION
    n_fft = max(1, int(np.round(window_span / dt)) * INTERPOLATION)
    max_bin = n_fft // 2
    n_grid = max(1, min(int(n_frequencies), max_bin - INTERPOLATION + 1))
    frequencies = resolution + np.arange(n_grid, dtype=float) * step
    bin_indices = INTERPOLATION + np.arange(n_grid)
    return frequencies, bin_indices, n_fft


# =============================================================================
# 三个估计器
# =============================================================================


def _fft_spectrum(values: npt.NDArray, dt: float, n_fft: int, bin_indices: npt.NDArray) -> npt.NDArray:
    """零填充周期图（Acycle ``evofft.m`` 的做法）。"""
    _, psd = periodogram(
        values,
        fs=1.0 / dt,
        window="boxcar",
        detrend=False,
        nfft=n_fft,
        scaling="spectrum",
    )
    return np.asarray(psd[bin_indices], dtype=float)


def _lomb_spectrum(time: npt.NDArray, values: npt.NDArray, frequencies: npt.NDArray) -> npt.NDArray:
    """Lomb-Scargle 周期图。

    复用 ``stratigraphy.spectral_analysis.SpectralAnalyzer`` 里已有的
    Lomb-Scargle 实现（同一份估计器与归一化，保证与本仓库其它地方算出来
    的谱可以直接比较），该实现原生支持不等间距采样。
    """
    analyzer = SpectralAnalyzer()
    return np.asarray(analyzer._lomb_scargle(time, values, frequencies), dtype=float)


def _mtm_spectrum(values: npt.NDArray, n_fft: int, bin_indices: npt.NDArray, nw: float = MTM_NW) -> npt.NDArray:
    """Thomson 多锥谱（MTM），在公共频率栅格上求值。

    锥函数与本仓库 ``stratigraphy.cycles.multitaper_spectrum`` 完全一致
    （同一套 DPSS 与同一套特征值加权、同一套归一化）；区别只在于锥乘序列
    用零填充 DFT 求值，从而与 FFT / Lomb-Scargle 分支落在同一组频率上。
    """
    v = values - float(np.mean(values))
    n = v.size
    n_tapers = max(1, min(int(np.floor(2.0 * nw - 1.0)), n // 2))
    eigenvalues, tapers = dpss_tapers(n, nw, n_tapers)
    variance = float(np.sum(v**2) / n)
    if variance <= 0.0:
        return np.zeros(bin_indices.size, dtype=float)
    products = np.empty((n_tapers, bin_indices.size), dtype=float)
    for i in range(n_tapers):
        spec = np.abs(np.fft.rfft(tapers[i] * v, n=n_fft)) ** 2
        products[i] = spec[bin_indices] / variance
    weight = float(np.sum(eigenvalues))
    return (eigenvalues[:, None] * products).sum(axis=0) / weight


# =============================================================================
# 主入口
# =============================================================================


def evolutionary_spectrum(
    time: npt.NDArray,
    values: npt.NDArray,
    window: float | None = None,
    method: str = "fft",
    n_frequencies: int = 500,
    overlap: float = 0.5,
) -> EvolutionarySpectrumResult:
    """计算演化谱（沿时间轴滑动的功率谱）。

    Parameters:
        time: 时间轴，与 ``values`` 等长。等间距与不等间距都支持。
        values: 待分析的序列。
        window: 窗长（与 ``time`` 同单位）。``None`` 表示取记录总时长的
            四分之一。
        method: 估计器，``"fft"``（零填充周期图）、``"lombscargle"``
            （Lomb-Scargle 周期图）、``"mtm"``（Thomson 多锥谱）。不等
            间距的数据只能用 Lomb-Scargle，此时本函数自动改用它并在
            ``result.method`` 中如实记录。
        n_frequencies: 频率栅格的点数上限。
        overlap: 相邻窗的重叠比例，取值 ``[0, 1)``。

    Returns:
        EvolutionarySpectrumResult: 频率-时间功率矩阵。
    """
    name = str(method).lower()
    if name not in METHODS:
        raise DataValidationError(_("method must be one of {0}; got '{1}'").format(", ".join(METHODS), method))
    if int(n_frequencies) < 1:
        raise DataValidationError(_("n_frequencies must be >= 1; got {0}").format(n_frequencies))
    if not 0.0 <= float(overlap) < 1.0:
        raise DataValidationError(_("overlap must lie in [0, 1); got {0}").format(overlap))

    t, v = _prepare_series(time, values)
    dt = float(np.median(np.diff(t)))
    window_span = _resolve_window(window, t, dt)
    windows, regular = _window_layout(t, dt, window_span, float(overlap))

    if not regular and name != "lombscargle":
        logger.warning(
            "evolutionary_spectrum: unevenly sampled series, falling back to the "
            "Lomb-Scargle estimator (requested method was '%s')",
            name,
        )
        name = "lombscargle"

    nominal_span = window_span
    if regular:
        nominal_span = windows[0][1] - windows[0][0]
        nominal_span = nominal_span * dt
    frequencies, bin_indices, n_fft = _frequency_grid(nominal_span, dt, int(n_frequencies))

    rows = np.empty((len(windows), frequencies.size), dtype=float)
    times = np.empty(len(windows), dtype=float)
    for row, (i0, i1) in enumerate(windows):
        segment = v[i0:i1]
        times[row] = float(t[i0 + (i1 - i0 - 1) // 2])
        detrended = _linear_detrend(segment)
        if name == "fft":
            rows[row] = _fft_spectrum(detrended, dt, n_fft, bin_indices)
        elif name == "mtm":
            rows[row] = _mtm_spectrum(detrended, n_fft, bin_indices)
        else:
            rows[row] = _lomb_spectrum(t[i0:i1], detrended, frequencies)

    logger.info(
        "evolutionary_spectrum: method=%s, %d windows, %d frequencies, resolution=%.6g, dt=%.6g",
        name,
        len(windows),
        frequencies.size,
        1.0 / nominal_span,
        dt,
    )
    return EvolutionarySpectrumResult(
        times=times,
        frequencies=frequencies,
        power=rows,
        method=name,
    )


def evolutionary_lomb_scargle(
    time: npt.NDArray,
    values: npt.NDArray,
    window: float | None = None,
    n_frequencies: int = 500,
    overlap: float = 0.5,
) -> EvolutionarySpectrumResult:
    """用 Lomb-Scargle 周期图计算演化谱（原生支持不等间距时间轴）。"""
    return evolutionary_spectrum(
        time,
        values,
        window=window,
        method="lombscargle",
        n_frequencies=n_frequencies,
        overlap=overlap,
    )


def evolutionary_multitaper(
    time: npt.NDArray,
    values: npt.NDArray,
    window: float | None = None,
    n_frequencies: int = 500,
    overlap: float = 0.5,
) -> EvolutionarySpectrumResult:
    """用 Thomson 多锥谱（MTM）计算演化谱。

    锥函数、特征值加权与归一化与 ``stratigraphy.cycles.multitaper_spectrum``
    一致，区别是锥乘序列在公共的零填充频率栅格上求值，从而与其它估计器
    逐频率可比。
    """
    return evolutionary_spectrum(
        time,
        values,
        window=window,
        method="mtm",
        n_frequencies=n_frequencies,
        overlap=overlap,
    )
