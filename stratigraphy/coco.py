# =============================================================================
# FILE: stratigraphy/coco.py
# =============================================================================
"""
COCO（相关系数法）与 eCOCO（演化相关系数法）地层年代学。

依据 Li, M., Kump, L.R., Hinnov, L.A., Mann, M.E., 2018, *Earth and Planetary
Science Letters* 501: 165-179：把一条地层记录（δ¹³C、伽马测井、孔隙度……）
调到天文时间轴上，办法是在一系列候选沉积速率下把记录的功率谱与天文解的
"目标谱"（target spectrum）做相关系数，取相关系数极大值对应的速率作为
"最可能沉积速率"；eCOCO 把记录沿轴切成若干窗口逐窗做 COCO，得到随时间
变化的沉积速率曲线。

数学框架
--------
记记录的坐标轴为 ``x``（下文的 ``time`` 参数），采样点数 ``n``，坐标跨度
``X``、中位采样间距 ``dx``。沉积速率假设 ``r`` 的单位是"坐标单位 / Ma"：
在该假设下，坐标 ``x`` 对应的天文时间是

    t = x / r                                        [Ma]

于是记录的采样在时间上被拉开成间距 ``dt = dx / r`` 的均匀时间轴，记录的
时间跨度变成 ``T = X / r``，功率谱的频率轴也随之变成"每 Ma"：

    Rayleigh 分辨率 = 1 / T = r / X                  [cycles/Ma]
    Nyquist 频率     = 1 / (2 dt) = r / (2 dx)       [cycles/Ma]

**频率栅格由记录本身决定，而不是由采样密度决定**：栅格上限永远不超过
Rayleigh 分辨率的 1/4（见 :func:`_spectral_grid`）。这一点是本模块最容易
出错、也最容易静默产生错值的地方——若栅格上限随采样密度升高而升高，加密
采样反而会让周期估计变差。

目标谱
------
目标谱由 Berger (1989) 的现代轨道周期表（见 :func:`berger1989_periods`）
构造：每条轨道周期在 ``f = 1000 / period_kyr``（cycles/Ma）处给一个梯形
频带（平顶 + 线性斜边），各频带按偏心率/倾角/岁差的相对权重叠加，得到
``target(f)``。调用方也可以自带目标谱（``target_spectrum`` 参数），这样日后
接入 ``stratigraphy.astronomy`` 的天文解时不必改动本模块。

相关系数与显著性
----------------
* 相关系数：记录功率谱与目标谱在共同频带上的 Pearson 相关系数（与 Acycle
  ``corrcoefsig.m`` 的口径一致）。
* 显著性：蒙特卡洛。每次速率扫描都生成与记录同长度、同 AR(1) 系数的红噪声
  替代序列（``lfilter`` 递推，与 ``stratigraphy.signal_generator`` 的做法
  相同），各自算一条功率谱并与同一目标谱求相关；p 值是
  ``(1 + #{零假设相关 >= 实测相关}) / (n_monte_carlo + 1)``。
* 不确定度：最优速率两侧"相关系数与最优值之差不超过零假设分布标准差"的
  连续速率区间，即 Acycle 自适应搜索区间的等价物，记录在
  :attr:`CocoResult.rate_confidence`。

与 Acycle 的差异
----------------
本模块是按上面的算法规格与 Li et al. (2018) 独立实现的，Acycle 是 GPL
软件，此处未翻译其 MATLAB 源码。相对 Acycle 的取舍：

* 目标谱用固定的 Berger (1989) 现代周期表构造，不依赖外部天文解数据文件；
* 频率轴一律"每 Ma"，且栅格上限由记录长度（Rayleigh 分辨率）决定；
* 坐标轴可以是不等间距的（用线性插值重采样到均匀时间轴），但重采样后的
  点数与采样间距都会做合理性检查，不合格的候选速率被跳过并记录原因。

单位提示
--------
``time`` 参数是记录的**地层坐标轴**，不是已经定年的年龄轴；沉积速率的单位
是"坐标单位 / Ma"。因此：

* 传入以米计的深度剖面时，速率就是 m/Ma，与 ``stratigraphy.correlation``
  里 m/Myr 的沉积速率口径一致；
* 传入已经定年的年龄轴（Ma）时，速率是 Ma/Ma，``r = 1`` 表示"给定的坐标轴
  本身就是天文时间轴"。

若把年龄轴当作坐标轴并期望恢复出 m/Ma 的沉积速率，需要额外给出剖面的
长度尺度，本模块不做这种换算。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter, periodogram

from config.i18n import _
from stratigraphy.cycles import _ar1_phi
from stratigraphy.spectral_analysis import _linear_detrend
from utils.exceptions import DataValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

# =============================================================================
# 参数常量
# =============================================================================

#: 参与分析的最少样本数。
MIN_SAMPLES = 32

#: 每条轨道周期带至少要采到的点数（Nyquist 判据的一半再留一倍余量）。
MIN_POINTS_PER_CYCLE = 4.0

#: 重采样后允许的最大点数；超过则该候选速率被跳过（算不动/无意义）。
MAX_RESAMPLED_POINTS = 200_000

#: 频率栅格相对 Rayleigh 分辨率的加密倍数（零填充长度 = 该倍数 × 样本数）。
PAD_FACTOR = 4

#: 目标谱频带的相对半宽（半宽 = 该系数 × 频带中心频率）。
BAND_HALF_WIDTH = 0.10

#: 频带平顶占半宽的比例：平顶内取满值，半宽处线性降到 0。
BAND_FLAT_CORE = 0.5

#: 各轨道族在目标谱中的相对权重（偏心率最强，岁差最弱）。
TARGET_BAND_WEIGHTS: dict[str, float] = {
    "eccentricity": 1.0,
    "obliquity": 0.6,
    "precession": 0.4,
}

# =============================================================================
# 轨道周期表（Berger 1989）
# =============================================================================

#: Berger (1989) 现代（0 Ma）轨道周期，单位 kyr。
#:
#: 偏心率 405/123/95 kyr 取自 Berger (1989) 轨道解（405 kyr 是长周期"超级
#: 周期"，123/95 kyr 夹住经典的"100 kyr 偏心率拍"）；倾角 41/39/37 kyr 与
#: 岁差 23/19/17 kyr 是现代轨道解在 0 Ma 的三条带，与 Acycle ``gentarget``
#: 的现代目标谱所用的周期组一致。
_BERGER1989_PERIODS_KYR: dict[str, float] = {
    "eccentricity_405kyr": 405.0,
    "eccentricity_123kyr": 123.0,
    "eccentricity_95kyr": 95.0,
    "obliquity_41kyr": 41.0,
    "obliquity_39kyr": 39.0,
    "obliquity_37kyr": 37.0,
    "precession_23kyr": 23.0,
    "precession_19kyr": 19.0,
    "precession_17kyr": 17.0,
}


def berger1989_periods() -> dict[str, float]:
    """Berger (1989) 现代轨道周期表。

    Returns
    -------
    dict
        轨道周期，键为 ``"<轨道族>_<周期>kyr"``（轨道族取 ``eccentricity`` /
        ``obliquity`` / ``precession``），值为周期（kyr）。

    Notes
    -----
    返回的是副本，调用方修改它不会影响模块内部的常量表。

    Examples
    --------
    >>> periods = berger1989_periods()
    >>> periods["obliquity_41kyr"]
    41.0
    """
    return dict(_BERGER1989_PERIODS_KYR)


def milankovitch_target_spectrum(frequencies: npt.NDArray) -> npt.NDArray:
    """构造天文驱动的目标谱 ``target(f)``。

    每条轨道周期在 ``f = 1000 / period_kyr``（cycles/Ma）处给出一个梯形
    频带：半宽为 :data:`BAND_HALF_WIDTH` × 频带中心频率，平顶占半宽的
    :data:`BAND_FLAT_CORE`，频带之间按 :data:`TARGET_BAND_WEIGHTS` 的相对
    权重叠加。

    Parameters
    ----------
    frequencies
        频率点，**单位是每单位时间（cycles/Ma）**，不是每样本。

    Returns
    -------
    ndarray
        与 ``frequencies`` 等长的目标谱功率。

    Raises
    ------
    DataValidationError
        频率数组为空、含非有限值或含负值。
    """
    f = np.asarray(frequencies, dtype=float).flatten()
    if f.size == 0:
        raise DataValidationError(_("frequencies must contain at least one value"))
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise DataValidationError(
            _("frequencies must be finite and non-negative"),
            details={"n_invalid": int(np.sum(~np.isfinite(f) | (f < 0.0)))},
        )

    target = np.zeros_like(f)
    for key, period_kyr in berger1989_periods().items():
        band = key.split("_", 1)[0]
        weight = TARGET_BAND_WEIGHTS.get(band, 1.0)
        centre = 1000.0 / float(period_kyr)
        half = BAND_HALF_WIDTH * centre
        taper = half * (1.0 - BAND_FLAT_CORE)
        shape = np.clip((half - np.abs(f - centre)) / taper, 0.0, 1.0)
        target += weight * shape
    return target


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class CocoResult:
    """COCO 扫描结果。

    Attributes:
        rates: 实际尝试的候选速率（坐标单位/Ma）。
        correlations: 每个候选速率的相关系数；被跳过的速率为 ``nan``。
        p_values: 每个候选速率的蒙特卡洛 p 值；被跳过的速率为 ``nan``。
        best_rate: 相关系数最大的候选速率。
        best_correlation: 最优速率处的相关系数。
        best_p_value: 最优速率处的 p 值。
        target_frequencies: 最优速率所用比较频带的频率点（cycles/Ma）。
        target_spectrum: 上述频率点上的目标谱功率。
        rate_confidence: 最优速率的不确定度区间，相关系数与最优值之差不超过
            零假设分布标准差的连续速率区间；无法确定时为 ``None``。
    """

    rates: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    best_rate: float
    best_correlation: float
    best_p_value: float
    target_frequencies: npt.NDArray
    target_spectrum: npt.NDArray
    rate_confidence: tuple[float, float] | None = None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("COCO Correlation-Coefficient Results"),
            "=" * 50,
            _("Candidate rates tried: {0}").format(int(self.rates.size)),
            _("Best rate: {0:.6g}").format(self.best_rate),
            _("Best correlation: {0:.4f}").format(self.best_correlation),
            _("Best p-value: {0:.4g}").format(self.best_p_value),
        ]
        if self.rate_confidence is not None:
            low, high = self.rate_confidence
            lines.append(_("Rate confidence interval: [{0:.6g}, {1:.6g}]").format(low, high))
        valid = int(np.sum(np.isfinite(self.correlations)))
        if valid < int(self.rates.size):
            lines.append(_("Rates skipped as inadmissible: {0}").format(int(self.rates.size) - valid))
        return "\n".join(lines)


@dataclass(frozen=True)
class ECocoResult:
    """eCOCO（演化 COCO）结果。

    Attributes:
        window_centres: 各窗口中心在记录坐标轴上的位置。
        best_rates: 各窗口给出的最可能沉积速率。
        correlations: 各窗口最优速率处的相关系数。
    """

    window_centres: npt.NDArray
    best_rates: npt.NDArray
    correlations: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("eCOCO Evolutionary Correlation-Coefficient Results"),
            "=" * 50,
            _("Windows: {0}").format(int(self.window_centres.size)),
        ]
        if self.window_centres.size:
            lines.append(_("Mean rate: {0:.6g}").format(float(np.nanmean(self.best_rates))))
            lines.append(
                _("Rate range: {0:.6g} - {1:.6g}").format(
                    float(np.nanmin(self.best_rates)), float(np.nanmax(self.best_rates))
                )
            )
            lines.append(_("Mean correlation: {0:.4f}").format(float(np.nanmean(self.correlations))))
        return "\n".join(lines)


# =============================================================================
# 内部工具
# =============================================================================


def _validate_record(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """校验记录：等长、有限、点数足够、坐标轴严格递增。"""
    t = np.asarray(time, dtype=float).flatten()
    v = np.asarray(values, dtype=float).flatten()
    if t.size != v.size:
        raise DataValidationError(
            _("time and values must have the same length; got {0} and {1}").format(t.size, v.size),
            details={"time_length": int(t.size), "values_length": int(v.size)},
        )
    if t.size < MIN_SAMPLES:
        raise DataValidationError(
            _("a COCO record needs at least {0} samples; got {1}").format(MIN_SAMPLES, int(t.size)),
            details={"n_samples": int(t.size), "minimum_required": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise DataValidationError(
            _("time and values must be finite"),
            details={
                "n_invalid_time": int(np.sum(~np.isfinite(t))),
                "n_invalid_values": int(np.sum(~np.isfinite(v))),
            },
        )
    if np.any(np.diff(t) <= 0.0):
        raise DataValidationError(
            _(
                "time must be strictly increasing; a non-monotonic axis has no single "
                "sedimentation rate and no single spectral resolution"
            ),
            details={"n_non_increasing": int(np.sum(np.diff(t) <= 0.0))},
        )
    if float(np.ptp(v)) == 0.0:
        raise DataValidationError(_("values are constant, so there is nothing to correlate"))
    return t, v


def _validate_rate_range(rate_range: tuple[float, float], n_rates: int) -> tuple[float, float]:
    """校验候选速率区间与扫描点数。"""
    try:
        low, high = (float(rate_range[0]), float(rate_range[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise DataValidationError(
            _("rate_range must be a (low, high) pair of positive numbers"),
            details={"rate_range": repr(rate_range)},
        ) from exc
    if not np.isfinite(low) or not np.isfinite(high):
        raise DataValidationError(
            _("rate_range must be finite"),
            details={"rate_range": [low, high]},
        )
    if low <= 0.0 or high <= 0.0:
        raise DataValidationError(
            _("rate_range must be positive; got [{0}, {1}]").format(low, high),
            details={"rate_range": [low, high]},
        )
    if high <= low:
        raise DataValidationError(
            _("rate_range must be increasing; got [{0}, {1}]").format(low, high),
            details={"rate_range": [low, high]},
        )
    if int(n_rates) < 2:
        raise DataValidationError(
            _("n_rates must be at least 2; got {0}").format(int(n_rates)),
            details={"n_rates": int(n_rates)},
        )
    return low, high


def _spectral_grid(n_samples: int) -> int:
    """零填充后的 FFT 长度。

    频率栅格是记录长度的函数：零填充到 ``PAD_FACTOR × n`` 之后，栅格间距
    是 Rayleigh 分辨率的 ``1 / PAD_FACTOR``。栅格**不会**因为采样更密而
    变粗或变细的上限——上限永远由记录长度决定。
    """
    target = max(PAD_FACTOR * int(n_samples), 64)
    return int(1 << int(np.ceil(np.log2(target))))


def _pearson_rows(matrix: npt.NDArray, vector: npt.NDArray) -> npt.NDArray:
    """``matrix`` 每行与 ``vector`` 的 Pearson 相关系数。"""
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    target = vector - float(np.mean(vector))
    numerator = centred @ target
    denominator = np.sqrt(np.sum(centred**2, axis=1) * float(np.sum(target**2)))
    return np.where(denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0)


def _resample_to_uniform_time(
    time: npt.NDArray, values: npt.NDArray, rate: float
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """把记录重采样到速率 ``rate`` 下的均匀天文时间轴。

    Returns
    -------
    (grid, resampled, dt)
        均匀时间轴（Ma）、重采样后的数值、采样间距（Ma）。
    """
    dx = float(np.median(np.diff(time)))
    span = float(time[-1] - time[0])
    dt = dx / float(rate)
    n_grid = int(np.floor(span / dx)) + 1
    grid = time[0] / float(rate) + dt * np.arange(n_grid)
    resampled = np.interp(grid, time / float(rate), values)
    return grid, resampled, dt


def _rate_correlation(
    time: npt.NDArray,
    values: npt.NDArray,
    rate: float,
    nfft: int,
    phi: float,
    n_monte_carlo: int,
    rng: np.random.Generator,
    target_freq: npt.NDArray | None,
    target_power: npt.NDArray | None,
) -> tuple[float, float, npt.NDArray, npt.NDArray, float] | None:
    """单个候选速率的相关系数、p 值与比较频带。

    Returns
    -------
    tuple 或 None
        ``(correlation, p_value, band_frequencies, band_target, null_spread)``；
        该速率不可用时返回 ``None``（采样点太少/太多，或记录长度不足以分辨
        任何一条轨道周期带）。
    """
    dx = float(np.median(np.diff(time)))
    span = float(time[-1] - time[0])
    dt = dx / float(rate)
    total = span / float(rate)

    n_grid = int(np.floor(span / dx)) + 1
    if n_grid < MIN_SAMPLES:
        logger.warning("COCO: rate %.6g skipped -- resampling leaves only %d points", float(rate), n_grid)
        return None
    if n_grid > MAX_RESAMPLED_POINTS:
        logger.warning("COCO: rate %.6g skipped -- resampling would leave %d points", float(rate), n_grid)
        return None

    if target_freq is None or target_power is None:
        band_lo = 1000.0 / max(_BERGER1989_PERIODS_KYR.values())
        band_hi = 1000.0 / min(_BERGER1989_PERIODS_KYR.values())
    else:
        band_lo = float(np.min(target_freq))
        band_hi = float(np.max(target_freq))

    nyquist = 1.0 / (2.0 * dt)
    if band_hi > 0.0 and nyquist < 2.0 * band_hi:
        logger.warning(
            "COCO: rate %.6g skipped -- the resampled axis (dt=%.6g Ma) undersamples the shortest target period",
            float(rate),
            dt,
        )
        return None

    rayleigh = 1.0 / total
    low = max(rayleigh, band_lo)
    high = min(nyquist, band_hi)
    if not high > low:
        logger.warning(
            "COCO: rate %.6g skipped -- the record spans %.6g Ma, too short to resolve any target band",
            float(rate),
            total,
        )
        return None

    _, resampled, _ = _resample_to_uniform_time(time, values, rate)
    freqs, power = periodogram(
        resampled,
        fs=1.0 / dt,
        detrend="linear",
        nfft=nfft,
        scaling="spectrum",
    )
    mask = (freqs >= low) & (freqs <= high)
    if int(np.sum(mask)) < 8:
        logger.warning("COCO: rate %.6g skipped -- fewer than 8 frequencies in the common band", float(rate))
        return None

    band_freqs = freqs[mask]
    if target_freq is None or target_power is None:
        band_target = milankovitch_target_spectrum(band_freqs)
    else:
        band_target = np.interp(band_freqs, target_freq, target_power, left=0.0, right=0.0)

    observed_power = power[mask]
    if float(np.ptp(observed_power)) == 0.0 or float(np.ptp(band_target)) == 0.0:
        logger.warning("COCO: rate %.6g skipped -- the comparison band is degenerate", float(rate))
        return None

    correlation = float(_pearson_rows(observed_power[None, :], band_target)[0])

    if n_monte_carlo <= 0:
        return correlation, float("nan"), band_freqs, band_target, float("nan")

    burn_in = int(np.ceil(10.0 / (1.0 - abs(phi))))
    noise = rng.normal(size=(n_monte_carlo, n_grid + burn_in))
    surrogates = lfilter([1.0], [1.0, -phi], noise, axis=-1)[:, burn_in:]
    _, null_power = periodogram(
        surrogates,
        fs=1.0 / dt,
        detrend="linear",
        nfft=nfft,
        scaling="spectrum",
        axis=-1,
    )
    null_correlations = _pearson_rows(null_power[:, mask], band_target)
    p_value = (1.0 + float(np.sum(null_correlations >= correlation))) / (float(n_monte_carlo) + 1.0)
    spread = float(np.std(null_correlations))
    return correlation, p_value, band_freqs, band_target, spread


def _confidence_interval(
    rates: npt.NDArray, correlations: npt.NDArray, best_index: int, spread: float
) -> tuple[float, float] | None:
    """最优速率两侧"与最优值之差不超过零假设标准差"的连续速率区间。"""
    if not np.isfinite(spread) or spread <= 0.0:
        return None
    best = float(correlations[best_index])
    threshold = best - float(spread)
    low = best_index
    while low > 0 and np.isfinite(correlations[low - 1]) and correlations[low - 1] >= threshold:
        low -= 1
    high = best_index
    while high < correlations.size - 1 and np.isfinite(correlations[high + 1]) and correlations[high + 1] >= threshold:
        high += 1
    return float(rates[low]), float(rates[high])


# =============================================================================
# COCO
# =============================================================================


def coco(
    time: npt.NDArray,
    values: npt.NDArray,
    rate_range: tuple[float, float],
    n_rates: int = 200,
    target_spectrum: tuple[npt.NDArray, npt.NDArray] | None = None,
    n_monte_carlo: int = 200,
    random_seed: int | None = None,
) -> CocoResult:
    """COCO：用功率谱-目标谱相关系数反演最可能沉积速率。

    Parameters
    ----------
    time
        记录的地层坐标轴（严格递增）。速率的单位是"坐标单位 / Ma"：传深度
        剖面（m）时速率即 m/Ma；传年龄轴（Ma）时速率是 Ma/Ma。
    values
        与 ``time`` 等长的测量序列（δ¹³C、伽马测井……）。
    rate_range
        候选速率区间 ``(low, high)``，单位同 ``rate``。
    n_rates
        扫描的候选速率个数。
    target_spectrum
        可选的外部目标谱 ``(频率, 功率)``，频率单位为 cycles/Ma。为 ``None``
        时用 :func:`milankovitch_target_spectrum` 的 Berger (1989) 目标谱。
    n_monte_carlo
        每个候选速率的蒙特卡洛替代序列条数（显著性检验）。为 0 时不计算
        p 值（结果为 ``nan``）。
    random_seed
        随机种子；``None`` 时结果不可复现（按仓库约定发出 ``RuntimeWarning``）。

    Returns
    -------
    CocoResult
        扫描结果，含最优速率、其相关系数与 p 值。

    Raises
    ------
    DataValidationError
        记录长度不匹配、点数太少、坐标轴非单调、速率区间非法等。

    Examples
    --------
    >>> import numpy as np
    >>> rate = 4.0
    >>> astro = np.linspace(0.0, 0.8, 800)
    >>> record = np.sin(2.0 * np.pi * astro / 0.02)
    >>> result = coco(rate * astro, record, rate_range=(2.0, 6.0), n_rates=40, n_monte_carlo=20,
    ...               random_seed=1)
    >>> result.best_rate > 0.0
    True
    """
    t, v = _validate_record(time, values)
    low, high = _validate_rate_range(rate_range, n_rates)
    if int(n_monte_carlo) < 0:
        raise DataValidationError(
            _("n_monte_carlo must be >= 0; got {0}").format(int(n_monte_carlo)),
            details={"n_monte_carlo": int(n_monte_carlo)},
        )

    target_freq: npt.NDArray | None = None
    target_power: npt.NDArray | None = None
    if target_spectrum is not None:
        try:
            target_freq = np.asarray(target_spectrum[0], dtype=float).flatten()
            target_power = np.asarray(target_spectrum[1], dtype=float).flatten()
        except (TypeError, ValueError, IndexError) as exc:
            raise DataValidationError(_("target_spectrum must be a (frequencies, power) pair of arrays")) from exc
        if target_freq.size != target_power.size or target_freq.size == 0:
            raise DataValidationError(
                _("target_spectrum frequencies and power must have the same non-zero length"),
                details={
                    "n_frequencies": int(target_freq.size),
                    "n_power": int(target_power.size),
                },
            )
        if not np.all(np.isfinite(target_freq)) or np.any(target_freq <= 0.0):
            raise DataValidationError(
                _("target_spectrum frequencies must be finite and positive (cycles per unit time)"),
                details={"n_invalid": int(np.sum(~np.isfinite(target_freq) | (target_freq <= 0.0)))},
            )
        order = np.argsort(target_freq)
        target_freq = target_freq[order]
        target_power = target_power[order]

    rates = np.linspace(low, high, int(n_rates))
    nfft = _spectral_grid(t.size)
    phi = _ar1_phi(_linear_detrend(v))
    rng = make_rng(random_seed, context="COCO")

    correlations = np.full(rates.size, np.nan)
    p_values = np.full(rates.size, np.nan)
    spreads = np.full(rates.size, np.nan)
    band_freqs = np.empty(0)
    band_target = np.empty(0)

    for index, rate in enumerate(rates):
        outcome = _rate_correlation(t, v, float(rate), nfft, phi, int(n_monte_carlo), rng, target_freq, target_power)
        if outcome is None:
            continue
        correlation, p_value, freqs_here, target_here, spread = outcome
        correlations[index] = correlation
        p_values[index] = p_value
        spreads[index] = spread
        band_freqs = freqs_here
        band_target = target_here

    if not np.any(np.isfinite(correlations)):
        raise DataValidationError(
            _(
                "no candidate rate in [{0}, {1}] is admissible: the record is too short or too "
                "coarse to resolve any astronomical band"
            ).format(low, high),
            details={"rate_range": [low, high], "n_rates": int(rates.size)},
        )

    best_index = int(np.nanargmax(correlations))
    best_rate = float(rates[best_index])
    best_correlation = float(correlations[best_index])
    best_p = float(p_values[best_index])
    confidence = _confidence_interval(rates, correlations, best_index, float(spreads[best_index]))

    logger.info(
        "COCO: best rate %.6g (correlation %.4f, p %.4g) over %d candidate rates",
        best_rate,
        best_correlation,
        best_p,
        int(rates.size),
    )

    return CocoResult(
        rates=rates,
        correlations=correlations,
        p_values=p_values,
        best_rate=best_rate,
        best_correlation=best_correlation,
        best_p_value=best_p,
        target_frequencies=band_freqs,
        target_spectrum=band_target,
        rate_confidence=confidence,
    )


# =============================================================================
# eCOCO
# =============================================================================


def ecoco(
    time: npt.NDArray,
    values: npt.NDArray,
    rate_range: tuple[float, float],
    n_windows: int = 10,
    n_rates: int = 100,
    n_monte_carlo: int = 100,
    random_seed: int | None = None,
) -> ECocoResult:
    """eCOCO：逐窗做 COCO，得到随位置变化的沉积速率。

    记录沿坐标轴被等分成 ``n_windows`` 个窗口，每个窗口独立扫描一次
    候选速率区间，返回每个窗口的最可能沉积速率与相关系数。

    Parameters
    ----------
    time
        记录的地层坐标轴（严格递增）。
    values
        与 ``time`` 等长的测量序列。
    rate_range
        候选速率区间 ``(low, high)``。
    n_windows
        窗口个数。
    n_rates
        每个窗口扫描的候选速率个数。
    n_monte_carlo
        每个候选速率的蒙特卡洛替代序列条数。
    random_seed
        随机种子；``None`` 时结果不可复现。

    Returns
    -------
    ECocoResult
        各窗口中心、最可能速率与相关系数。

    Raises
    ------
    DataValidationError
        记录不合法、窗口数不足或速率区间非法。
    """
    t, v = _validate_record(time, values)
    _validate_rate_range(rate_range, n_rates)
    if int(n_windows) < 2:
        raise DataValidationError(
            _("n_windows must be at least 2; got {0}").format(int(n_windows)),
            details={"n_windows": int(n_windows)},
        )

    edges = np.linspace(float(t[0]), float(t[-1]), int(n_windows) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    best_rates = np.full(centres.size, np.nan)
    correlations = np.full(centres.size, np.nan)

    for index in range(centres.size):
        mask = (t >= edges[index]) & (t <= edges[index + 1])
        window_time = t[mask]
        window_values = v[mask]
        if window_time.size < MIN_SAMPLES:
            logger.warning("eCOCO: window %d skipped -- only %d samples", index, int(window_time.size))
            continue
        try:
            window = coco(
                window_time,
                window_values,
                rate_range=rate_range,
                n_rates=n_rates,
                n_monte_carlo=n_monte_carlo,
                random_seed=random_seed,
            )
        except DataValidationError as exc:
            logger.warning("eCOCO: window %d skipped -- %s", index, exc)
            continue
        best_rates[index] = window.best_rate
        correlations[index] = window.best_correlation

    return ECocoResult(
        window_centres=centres,
        best_rates=best_rates,
        correlations=correlations,
    )
