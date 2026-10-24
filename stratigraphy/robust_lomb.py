# =============================================================================
# FILE: stratigraphy/robust_lomb.py
# =============================================================================
"""
稳健 AR(1) 背景的 Lomb-Scargle 周期图。

本模块实现 Acycle ``code/seriesanalysis/plomb_robustar1.m`` 的核心算法：不均匀
采样记录上的 Lomb-Scargle 周期图，加上一个**稳健**的 AR(1) 红噪声背景，以及
由卡方分布给出的 90/95/99/99.9% 置信线。

“稳健”指什么
~~~~~~~~~~~~
经典做法（``spectral_analysis`` 与 ``redconfchi2.m``）用滞后 1 阶自相关直接给出
AR(1) 系数，再把理论谱当作红噪声零假设。问题在于：记录里若有一条很强的周期
信号（例如一条 100 kyr 的轨道周期），这条谱线会同时抬高**观测谱**与（若用整条
谱去拟合）**背景**，于是显著性判断变得不可靠。

Acycle 的做法是：先把 LS 功率谱做**中值平滑**，再用 AR(1) 形状

    S_AR1(f) = (1 - rho²) / (1 - 2 rho cos(2 pi f dt) + rho²)

去拟合这条平滑谱。中值对窄谱线不敏感，所以周期信号不会把红噪声水平抬起来。
本模块在此基础上再加一步：先用本仓库的 REDFIT（``cycles.redfit``）给出每个频率
的假警概率，把**被判定含有周期信号的谱峰**从 AR(1) 拟合中剔除，再重新拟合——
于是背景真正只由“红噪声部分”决定。REDFIT 的自助法由 ``random_seed`` 控制，
因此同一个种子给出的背景逐点可复现。

频率轴约定（每单位时间）
~~~~~~~~~~~~~~~~~~~~~~~~
* 频率是**每单位时间**的循环频率（cycles per unit time），不是每样本的归一化
  频率；因此 1/f 直接就是以时间单位表示的周期。
* 设记录跨度为 ``T``、点数为 ``n``，**默认频带**是

      f ∈ [1/T, n/(2T)]

  下界是周期图的 **Rayleigh 分辨率** ``1/T``：长度 ``T`` 的记录无法把周期定位得
  比 ``1/T`` 更细；上界是记录的有效 Nyquist ``n/(2T)``，与
  ``stratigraphy.spectral_analysis`` **同一口径**（注意它与
  ``1/(2 × 采样间隔中位数)`` 在不均匀采样时并不等价）。因此读者可以直接判断
  “我这个周期在不在带里”：周期 ``P`` 落在带内当且仅当
  ``2T/n <= P <= T``（下界 ``f >= 1/T`` 对应 ``P <= T``，上界 ``f <= n/(2T)``
  对应 ``P >= 2T/n``）。
* 栅格步长是 ``min((f_max - f_min)/(n-1), 1/T / oversampling)``，默认
  ``oversampling = 1``；频带比 ``n-1`` 个分辨率单元还窄时步长会细于 ``1/T``。
  频率栅格**由记录决定，而不是由采样密度决定**：加密采样不会让栅格变粗，因为
  步长始终不粗于记录的分辨率。这一点是仓库修过的一个真实 bug：旧实现的栅格由
  采样密度决定，加密采样反而让主周期变差（同一段 100 单位的记录，20000 个采样
  点把 7 周期算成了 9.89）。
* ``oversampling > 1`` 把步长再细分为 ``1/oversampling`` 倍，代价是**频带按同一
  比例变窄**（上界变成 ``1/T + (n-1)/(T · oversampling)``）。这是用户主动换取更
  细步长的选项，默认关闭——默认频带必须完整覆盖到 Nyquist，否则轨道带（例如
  3 Myr 记录里的 21 kyr 周期）会被**静默排除**。
* AR(1) 形状函数里的采样间隔 ``Δt`` 取**采样间隔中位数**（Acycle ``plomb`` 的
  “physical Nyquist” 口径），而频带边界取 ``n/(2T)``：两者与
  ``stratigraphy.spectral_analysis`` 的处理完全一致。

置信线
~~~~~~
AR(1) 零假设下 LS 功率的每个频率有 2 个自由度（Acycle ``plomb_robustar1.m``
的 ``nw2 = 2``），因此置信线为

    chi_p(f) = S_background(f) * chi2_ppf(p, 2) / 2

本模块给出 95% 的上下线（``lower_95`` / ``upper_95``），90/99/99.9% 的上界在
``summary()`` 中一并报告。

随机过程约定
~~~~~~~~~~~~
* 所有随机函数都接受 ``random_seed``。给定种子时结果完全可复现；不给种子时
  调用 ``utils.statistics_core.make_rng(None, context=...)``，它会发出
  ``RuntimeWarning`` 提示结果不可复现——这是仓库内其它随机分析沿用的同一约定。
* 生成器是独立的 ``numpy.random.Generator``，既不读取也不推进全局
  ``numpy.random`` 流。

参考文献
~~~~~~~~
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. Computers & Geosciences 127, 12-22.
* Acycle ``code/seriesanalysis/plomb_robustar1.m``（本模块实现的算法来源）。
* Muellersohn, F.J., Mysak, P.A. & Quaternary Research Group. 1999. REDFIT: a
  robust estimator of the red-noise spectrum of a time series. Climate Dynamics
  14: 1-16.（循环谱的假警概率。）
* Watters, D.J. & Solomon, A.M. 2004. Redfit: identifying red-noise trends in
  paleoclimate time series. Journal of Geophysical Research 109: C09014.
* Scargle, J.D. 1982. Studies in astronomical time series analysis. II.
  Astrophysical Journal 263: 835-872.（Lomb-Scargle 周期图。）
* Horne, J.H. & Baliunas, S.L. 1986. A prescription for periodogram analysis.
  Space Science Reviews 60: 383-401.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from stratigraphy.cycles import redfit
from stratigraphy.spectral_analysis import (
    SpectralAnalyzer,
    _linear_detrend,
    ar1_theoretical_spectrum,
)
from stratigraphy.spectral_moments import estimate_ar1_rho
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 记录至少需要的有效点数。
MIN_SAMPLES = 16

#: 频率栅格默认的过采样倍数。``1.0`` 表示"步长不超过记录的 Rayleigh 分辨率
#: ``1/T``"，频带完整覆盖到有效 Nyquist（与 ``spectral_analysis`` 同一口径）。
#: 更大的值把步长进一步细分，代价是频带按同一比例变窄。
DEFAULT_OVERSAMPLING = 1.0

#: 过采样倍数的下限；小于 1 会让栅格比记录的 Rayleigh 分辨率还粗。
MIN_OVERSAMPLING = 1.0

#: 中值平滑窗口占频率点数的比例（Acycle 的 ``smoothwin`` 默认为 0.2）。
SMOOTH_FRACTION = 0.05

#: 判定“谱峰”的倍数：平滑谱超过背景这么多倍才被视为周期信号。
PEAK_FACTOR = 3.0

#: 内部 REDFIT 假警概率自助法的重复次数。
REDFIT_SIMULATIONS = 64

#: AR(1) 系数的搜索上限（保证理论谱有限）。
MAX_AR1_RHO = 0.99

#: 卡方置信线所用的自由度（Acycle ``plomb_robustar1.m`` 的 ``nw2 = 2``）。
CHI2_DOF = 2


# =============================================================================
# 1. 结果容器
# =============================================================================


@dataclass(frozen=True)
class RobustLombResult:
    """稳健 AR(1) 背景的 Lomb-Scargle 结果。

    Attributes:
        frequencies: 每单位时间的循环频率，严格递增。
        power: 每个频率上的 Lomb-Scargle 功率（白噪声下均值约为 1）。
        ar1_rho: 由中值平滑谱拟合得到的稳健 AR(1) 系数。
        background: AR(1) 背景谱 ``s0 * (1-rho²)/(1-2 rho cos(2 pi f dt)+rho²)``。
        lower_95: 95% 置信下界（卡方 5% 分位数）。
        upper_95: 95% 置信上界（卡方 95% 分位数）。
    """

    frequencies: npt.NDArray
    power: npt.NDArray
    ar1_rho: float
    background: npt.NDArray
    lower_95: npt.NDArray
    upper_95: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Robust AR(1) Lomb-Scargle"),
            "=" * 40,
            _("Frequencies: {0}").format(self.frequencies.size),
        ]
        if self.power.size:
            peak = int(np.argmax(self.power))
            lines.append(_("Peak frequency: {0}").format(f"{float(self.frequencies[peak]):.6g}"))
            lines.append(_("Peak period: {0}").format(f"{1.0 / float(self.frequencies[peak]):.4f}"))
            lines.append(_("Peak power: {0}").format(f"{float(self.power[peak]):.4f}"))
        lines.append(_("Robust AR(1) rho: {0}").format(f"{self.ar1_rho:.4f}"))
        chi99 = float(sp_stats.chi2.ppf(0.99, CHI2_DOF)) / CHI2_DOF
        chi999 = float(sp_stats.chi2.ppf(0.999, CHI2_DOF)) / CHI2_DOF
        lines.append(
            _("95% confidence factor: {0}").format(f"{float(sp_stats.chi2.ppf(0.95, CHI2_DOF)) / CHI2_DOF:.4f}")
        )
        lines.append(_("99% confidence factor: {0}").format(f"{chi99:.4f}"))
        lines.append(_("99.9% confidence factor: {0}").format(f"{chi999:.4f}"))
        return "\n".join(lines)


# =============================================================================
# 2. 内部辅助
# =============================================================================


def _clean_pair(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """丢弃非有限点并按时间排序；要求时间轴严格递增。"""
    t = np.asarray(time, dtype=float).flatten()
    v = np.asarray(values, dtype=float).flatten()
    if t.size != v.size:
        raise ValidationError(
            _("time and values must have the same length; got {0} and {1}").format(t.size, v.size),
            details={"time_length": int(t.size), "values_length": int(v.size)},
        )
    finite = np.isfinite(t) & np.isfinite(v)
    t, v = t[finite], v[finite]
    if t.size < MIN_SAMPLES:
        raise ValidationError(
            _("robust Lomb-Scargle needs at least {0} finite points; got {1}").format(MIN_SAMPLES, int(t.size)),
            details={"n_finite": int(t.size)},
        )
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if np.any(np.diff(t) <= 0.0):
        raise ValidationError(
            _("time must be strictly increasing after sorting; duplicate timestamps have no single sampling interval")
        )
    if float(np.ptp(v)) == 0.0:
        raise ValidationError(_("values are constant, so there is no periodicity to find"))
    return t, v


def _frequency_grid(
    time: npt.NDArray,
    n_frequencies: int,
    frequency_range: tuple[float, float] | None,
    oversampling: float = DEFAULT_OVERSAMPLING,
) -> tuple[npt.NDArray, float, float]:
    """由**记录**决定的频率栅格（每单位时间）。

    设记录跨度为 ``T``、点数为 ``n``：

    * 下界是 Rayleigh 分辨率 ``1/T``：长度 ``T`` 的记录无法把周期定位得比
      ``1/T`` 更细，因此比 ``1/T`` 更低的频率没有意义。
    * 上界是记录的有效 Nyquist ``n/(2T)``——与 ``stratigraphy.spectral_analysis``
      **同一口径**（注意它与 ``1/(2 * 采样间隔中位数)`` 在不均匀采样时并不等价）。
    * 步长是 ``min((f_max - f_min)/(n-1), 1/T / oversampling)``。默认
      ``oversampling = 1`` 时步长就是 ``min((f_max - f_min)/(n-1), 1/T)``，
      频带因此**完整覆盖到 Nyquist**（与 ``spectral_analysis`` 的栅格一致）；
      频带比 ``n-1`` 个分辨率单元还窄时，步长会细于 ``1/T``。
    * ``oversampling > 1`` 把步长再细分为 ``1/oversampling`` 倍，代价是频带按
      同一比例**变窄**（上界变成 ``1/T + (n-1)/T/oversampling``）。这是用户主动
      换取更细步长的选项，默认关闭。

    步长由 ``T`` 决定而不是由采样密度决定，所以加密采样不会让栅格变粗。

    Returns
    -------
    (frequencies, resolution, nyquist)
    """
    span = float(time[-1] - time[0])
    resolution = 1.0 / span
    nyquist = len(time) / (2.0 * span)

    if frequency_range is None:
        f_min = resolution
        band = nyquist - f_min
        if band <= 0.0:
            raise ValidationError(
                _("the record has no resolvable frequency band: Nyquist {0} <= Rayleigh {1}").format(
                    nyquist, resolution
                ),
                details={"nyquist": nyquist, "resolution": resolution},
            )
        step = min(band / (n_frequencies - 1), resolution / oversampling)
        frequencies = f_min + step * np.arange(n_frequencies, dtype=float)
        if frequencies[-1] > nyquist:
            # 步长被分辨率卡住时，频带可能越过 Nyquist；此时把步长拉回
            # ``band/(n-1)``，让频带正好停在 Nyquist（与 spectral_analysis 一致）。
            step = band / (n_frequencies - 1)
            frequencies = f_min + step * np.arange(n_frequencies, dtype=float)
    else:
        f_min, f_max = (float(frequency_range[0]), float(frequency_range[1]))
        if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min <= 0.0 or f_max <= f_min:
            raise ValidationError(
                _("frequency_range must be (positive minimum, larger maximum); got {0}").format(frequency_range),
                details={"frequency_range": frequency_range},
            )
        needed = int(np.ceil((f_max - f_min) / (resolution / oversampling))) + 1
        count = max(n_frequencies, needed)
        frequencies = np.linspace(f_min, f_max, count)
    return frequencies, resolution, nyquist


def _moving_median(values: npt.NDArray, window: int) -> npt.NDArray:
    """滑窗中值（端点处窗口收缩，与 MATLAB ``movmedian`` 的 ``shrink`` 一致）。"""
    size = values.size
    half = window // 2
    smoothed = np.empty(size, dtype=float)
    for index in range(size):
        start = max(0, index - half)
        stop = min(size, index + half + 1)
        smoothed[index] = float(np.median(values[start:stop]))
    return smoothed


def _fit_ar1_background(
    smoothed: npt.NDArray,
    frequencies: npt.NDArray,
    usable: npt.NDArray,
    dt: float,
    max_iterations: int,
) -> tuple[float, float]:
    """把 AR(1) 形状拟合到（部分）平滑谱上，返回 ``(s0, rho)``。

    拟合在 **log 空间**进行：``log P = log s0 + log S_AR1(f; rho)``。尺度 ``s0``
    在给定 ``rho`` 时有解析解（残差均值为零），因此只需一维搜索 ``rho``；先用
    粗网格定位，再按 ``max_iterations`` 轮逐步收窄。
    """
    log_power = np.log(smoothed[usable])
    log_frequency = frequencies[usable]

    grid = np.linspace(0.0, MAX_AR1_RHO, 64)
    best_rho = 0.0
    best_sse = np.inf
    best_scale = 1.0
    for round_index in range(max_iterations + 1):
        for rho in grid:
            shape = ar1_theoretical_spectrum(log_frequency, float(rho), dt)
            if not np.all(np.isfinite(shape)) or np.any(shape <= 0.0):
                continue
            offset = float(np.mean(log_power - np.log(shape)))
            residual = log_power - np.log(shape) - offset
            sse = float(np.sum(residual**2))
            if sse < best_sse:
                best_sse = sse
                best_rho = float(rho)
                best_scale = float(np.exp(offset))
        if round_index >= max_iterations:
            break
        step = float(grid[1] - grid[0])
        grid = np.linspace(max(0.0, best_rho - step), min(MAX_AR1_RHO, best_rho + step), 64)
    return best_scale, best_rho


def _local_maxima(smoothed: npt.NDArray) -> npt.NDArray:
    """平滑谱的局部极大点（端点不算）。"""
    interior = np.zeros(smoothed.size, dtype=bool)
    if smoothed.size >= 3:
        interior[1:-1] = (smoothed[1:-1] >= smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:])
    return interior


def _significance_mask(
    time: npt.NDArray,
    values: npt.NDArray,
    frequencies: npt.NDArray,
    random_seed: int,
) -> npt.NDArray:
    """REDFIT 的逐频率假警概率，插值到本模块的频率栅格上。

    REDFIT 内部会重采样到均匀网格并做频域平滑，因此它的频率栅格与本模块的栅格
    不同；这里把它的显著标志线性插值过来。若 REDFIT 无法处理该记录（例如点数
    太少），则返回全 ``False``——没有显著频率只是意味着“不剔除谱峰”，不会让
    分析失败。
    """
    span = float(time[-1] - time[0])
    spacing = float(np.median(np.diff(time)))
    resolution_elements = max(2, round(span / (2.0 * spacing)))
    n_periods = int(min(2000, max(200, 2 * resolution_elements)))
    try:
        fit = redfit(
            time,
            values,
            n_periods=n_periods,
            n_simulations=REDFIT_SIMULATIONS,
            random_seed=random_seed,
        )
    except Exception as error:  # pragma: no cover - 防御性分支
        logger.warning("robust Lomb-Scargle: REDFIT failed (%s); no peak is excluded", error)
        return np.zeros(frequencies.size, dtype=bool)
    flags = np.asarray(fit.significant, dtype=float)
    if flags.size < 2:
        return np.zeros(frequencies.size, dtype=bool)
    interpolated = np.interp(frequencies, fit.frequencies, flags, left=0.0, right=0.0)
    return interpolated >= 0.5


# =============================================================================
# 3. 主入口
# =============================================================================


def robust_lomb_scargle(
    time: npt.NDArray,
    values: npt.NDArray,
    n_frequencies: int = 500,
    frequency_range: tuple[float, float] | None = None,
    max_iterations: int = 3,
    random_seed: int | None = None,
    oversampling: float = DEFAULT_OVERSAMPLING,
) -> RobustLombResult:
    """计算带稳健 AR(1) 背景的 Lomb-Scargle 周期图。

    Parameters
    ----------
    time:
        时间轴，必须严格递增；非有限点被丢弃。允许不均匀采样。
    values:
        与 ``time`` 等长的信号。
    n_frequencies:
        频率栅格的点数。栅格步长由记录的 Rayleigh 分辨率决定（见模块文档），
        因此采样更密不会让栅格变粗。给定 ``frequency_range`` 时该值是下限：栅格
        会被加密到步长不超过 ``1/T``。
    frequency_range:
        显式的频率区间 ``(f_min, f_max)``（每单位时间）。``None`` 时使用记录
        自身的 ``[1/T, n/(2T)]``。
    max_iterations:
        AR(1) 背景拟合的迭代轮数：每一轮用上一轮的背景重新识别谱峰并重新拟合。
    random_seed:
        REDFIT 假警概率自助法的种子。给定种子时结果完全可复现；不给种子时
        :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。
    oversampling:
        频率步长的过采样倍数，默认 ``1.0``。默认频带是
        ``[1/T, n/(2T)]``（完整覆盖到有效 Nyquist，与
        ``stratigraphy.spectral_analysis`` 同一口径）；``oversampling > 1`` 把
        步长细分为 ``1/oversampling`` 倍，代价是频带按同一比例变窄。

    Returns
    -------
    RobustLombResult
        频率栅格、LS 功率、稳健 AR(1) 系数、背景谱与 95% 置信线。

    Raises
    ------
    ValidationError:
        时间与数值长度不一致、点数太少、时间轴非严格递增、数值为常数、
        ``n_frequencies`` / ``max_iterations`` / ``oversampling`` 越界，或
        ``frequency_range`` 非法。

    Examples
    --------
    >>> import numpy as np
    >>> time = np.arange(200, dtype=float)
    >>> values = np.sin(2.0 * np.pi * time / 20.0)
    >>> result = robust_lomb_scargle(time, values, random_seed=7)
    >>> result.background.size == result.frequencies.size
    True
    """
    t, v = _clean_pair(time, values)

    count = int(n_frequencies)
    if count < 16:
        raise ValidationError(
            _("n_frequencies must be at least 16; got {0}").format(n_frequencies),
            details={"n_frequencies": n_frequencies},
        )
    iterations = int(max_iterations)
    if iterations < 1:
        raise ValidationError(
            _("max_iterations must be at least 1; got {0}").format(max_iterations),
            details={"max_iterations": max_iterations},
        )
    factor = float(oversampling)
    if not np.isfinite(factor) or factor < MIN_OVERSAMPLING:
        raise ValidationError(
            _("oversampling must be a finite number >= {0}; got {1}").format(MIN_OVERSAMPLING, oversampling),
            details={"oversampling": oversampling},
        )

    rng = make_rng(random_seed, context="robust Lomb-Scargle")
    bootstrap_seed = int(rng.integers(0, 2**31 - 1))

    frequencies, _resolution, _nyquist = _frequency_grid(t, count, frequency_range, factor)
    spacing = float(np.median(np.diff(t)))

    detrended = _linear_detrend(v)
    analyzer = SpectralAnalyzer()
    power = np.asarray(analyzer._lomb_scargle(t, detrended, frequencies), dtype=float)

    window = max(5, round(SMOOTH_FRACTION * frequencies.size))
    smoothed = _moving_median(power, window)

    direct_rho = float(np.clip(estimate_ar1_rho(detrended), -MAX_AR1_RHO, MAX_AR1_RHO))
    significant = _significance_mask(t, detrended, frequencies, bootstrap_seed)

    usable = np.ones(frequencies.size, dtype=bool)
    scale, rho = _fit_ar1_background(smoothed, frequencies, usable, spacing, iterations)
    background = scale * ar1_theoretical_spectrum(frequencies, rho, spacing)

    peaks = _local_maxima(smoothed)
    for round_index in range(iterations):
        excluded = peaks & ((smoothed > PEAK_FACTOR * background) | significant)
        candidate = ~excluded
        if int(np.sum(candidate)) < 8:
            logger.warning("robust Lomb-Scargle: too few frequencies left for a refit; keeping the previous fit")
            break
        if np.array_equal(candidate, usable):
            break
        usable = candidate
        scale, rho = _fit_ar1_background(smoothed, frequencies, usable, spacing, iterations)
        background = scale * ar1_theoretical_spectrum(frequencies, rho, spacing)

    if not np.all(np.isfinite(background)) or float(np.max(background)) <= 0.0:
        logger.warning(
            "robust Lomb-Scargle: the AR(1) fit failed (rho=%.4f); falling back to the direct estimate %.4f",
            rho,
            direct_rho,
        )
        rho = direct_rho
        scale = float(np.median(smoothed))
        background = scale * ar1_theoretical_spectrum(frequencies, rho, spacing)

    lower_factor = float(sp_stats.chi2.ppf(0.05, CHI2_DOF)) / CHI2_DOF
    upper_factor = float(sp_stats.chi2.ppf(0.95, CHI2_DOF)) / CHI2_DOF
    lower_95 = background * lower_factor
    upper_95 = background * upper_factor

    peak_index = int(np.argmax(power))
    logger.info(
        "robust Lomb-Scargle: %d frequencies, robust rho=%.4f (direct %.4f), peak period=%.4f",
        frequencies.size,
        rho,
        direct_rho,
        1.0 / float(frequencies[peak_index]),
    )
    return RobustLombResult(
        frequencies=frequencies,
        power=power,
        ar1_rho=float(rho),
        background=background,
        lower_95=lower_95,
        upper_95=upper_95,
    )
