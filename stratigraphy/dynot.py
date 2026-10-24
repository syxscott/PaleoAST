# =============================================================================
# FILE: stratigraphy/dynot.py
# =============================================================================
"""
沉积噪声模型（sedimentary noise model）与 ρ1 沉积速率比方法。

本模块实现 Li 等 (2018) *Nature Communications* 9: 1004 提出的**沉积噪声模型**
的两条思路中可以被一条普通记录直接支撑的那一部分：**噪声谱与噪声基线的比较**。

论文原文把该模型拆成两条互补的路线：

* **DYNOT**（dynamic noise after orbital tuning，轨道调谐后的动态噪声）：
  在 10³–10⁶ 年频带内，沿滑动时间窗计算**非轨道信号方差与总方差之比**，
  把该比值当作"噪声"的度量；轨道信号被扣除后剩下的部分视为与地层记录
  无关的噪声。
* **ρ1**（lag-1 autocorrelation coefficient，滞后 1 阶自相关系数）：
  直接在时间序列上计算

      ρ1 = Σ_{i=2..n} x(i)·x(i-1) / Σ_{i=2..n} x(i-1)²

  （x 为轨道调谐后的地层代用序列），用"持续性"来表征噪声：噪声越多，
  相邻样本之间的相关性越弱，ρ1 越低。

本模块的做法与边界
------------------

本模块**不做轨道调谐**（那需要外部给出的目标轨道谱），因此也不复刻 Acycle
的完整 DYNOT 流程（Acycle 的 DYNOT 还要用 Weibull 拟合采样速率分布、对随机
沉积速率重采样、滑动窗口多锥谱、蒙特卡洛分位数带）。本模块实现的是这个模型
中**不依赖轨道调谐**的核心：**估计记录自身的噪声谱，并把它的噪声功率与所选
基线噪声模型的功率相比较**。

两个公开函数
------------

* :func:`sedimentary_noise` —— 估计噪声谱、噪声功率比 ρ1、AR(1) 系数，以及
  在基线模型下的**蒙特卡洛零假设离散度**（``rho1_uncertainty``）。
* :func:`rho1_sedimentation_ratio` —— 以 **AR(1) 红噪声基线**给出 ρ1 及其
  **分块自助法（block bootstrap）不确定度**。

ρ1 在本模块中的定义（务必与论文中的 ρ1 区分）
--------------------------------------------

:attr:`DynotResult.rho1` 是**"记录在噪声频带内的噪声功率"与"所选基线模型在
同一频带内的噪声功率"之比**：

    ρ1 = mean(P_observed over noise band) / mean(P_baseline over noise band)

* ``> 1``：记录在噪声频带内携带的噪声功率**强于**所选基线模型的预期。
* ``≈ 1``：记录在噪声频带内的噪声水平与所选基线模型一致。

**噪声频带（noise band）** 是分析频带的**短周期半边**（频率栅格的上半部分，
点数与下半部分相同）。必须这样取，理由是：记录与基线模型的总方差相同（功率
谱按记录自身的方差归一化），若在**整个**频带上比平均值，两者的积分必然相等，
比值恒为 1，统计量不携带任何信息。限制到短周期半边，正好落在风暴、潮汐、
生物扰动、沉积速率不稳与测量误差这类**沉积噪声**所在的高频端；长周期半边
（低频端）承载的是持续性成分与轨道信号。

**这不是论文里的 ρ1。** 论文里的 ρ1 是**滞后 1 阶自相关系数**（见上式），是
一个*持续性*度量；本模块把它单独放在 :attr:`DynotResult.ar1_rho` 里，用的
估计量与论文公式、Acycle 的 ``rhoAR1.m`` / ``rhoAR1ML.m`` 完全一致
（Yule–Walker 形式 ``Σ x(i)x(i-1) / Σ x(i-1)²``）。

**本模块也不声称 ρ1 等于沉积速率。** 论文没有把 ρ1 定义成沉积速率，也没有
定义过"噪声功率与基线之比"这样的比值；上面这个比值是本模块为便于复用而
给出的**工程定义**，其依据是：它比较的是"这条记录在短周期端实际携带了多少
噪声功率"与"基线模型说它应该携带多少"，而这正是沉积噪声模型要回答的问题。
要判断论文意义上的"噪声"（噪声越多越"白"），请用
:attr:`DynotResult.ar1_rho`（论文的 ρ1：噪声越多越低）。

两种基线的差别（重要）
----------------------

两个基线都是**绝对**的：它们不向记录重新标定，而是"同样方差、同样（对
AR(1) 基线而言）持续性的过程应当给出的噪声谱"。

* ``noise_model="white"``：基线是**白噪声**参考谱（平的一条水平线，功率为
  1）。它与记录的持续性无关，因此 ρ1 直接回答"记录在短周期端比白噪声参考
  多（或少）了多少噪声功率"。红噪声记录把功率集中在长周期端，其 ρ1 明显
  小于 1；短周期噪声占主导的记录则明显大于 1。
* ``noise_model="ar1"``：基线是**红噪声**参考谱
  ``(1-ρ²)/(1-2ρcos(2πfΔt)+ρ²)``，其中 ρ 取自记录自身的 AR(1) 系数
  （即 :attr:`DynotResult.ar1_rho`）。它按记录的持续性做了标定，因此
  对任何 AR(1) 族的过程 ρ1 ≈ 1（实测 1.0 ± 0.1）：这是有意义的性质——
  它说明"这条记录的噪声正好等于它自己的持续性所预测的噪声"。
  由于额外叠加的白噪声会**削弱**持续性估计，叠加噪声后 ρ1 会略低于 1；
  换句话说，用 AR(1) 基线时 ρ1 < 1 表示"记录比它的持续性所暗示的更白、
  短周期噪声更多"，而 ρ1 > 1 表示"记录在短周期端携带的噪声功率多于同持续性
  的 AR(1) 模型所预测的"。

频率轴约定
~~~~~~~~~~

* 频率一律是**每单位时间的循环频率**（cycles/Ma、cycles/kyr……），不是
  "每样本"的归一化频率。功率谱的单位是"噪声背景 = 1"的无量纲功率
  （白噪声在任一频率上的期望值都是 1），与本仓库
  ``stratigraphy.spectral_analysis`` 的 Lomb–Scargle 归一化一致。
* 频率栅格的**下界是记录的 Rayleigh 分辨率** ``1/T``（T 为记录跨度），
  **上界是采样决定的 Nyquist 频率** ``1/(2Δt)``，而**栅格步长永远不比
  Rayleigh 分辨率更粗**：这一点是本仓库修过的真实 bug —— 早期版本让栅格
  由采样密度决定，同一条记录采样越密、报出的周期反而越差。现在步长取
  ``min((f_nyq - f_rayleigh)/(n-1), f_rayleigh)``，所以采样越密只会让频带
  变宽，不会让频率分辨率变差。
* 频率点数的上限是记录本身能支持的：一条跨度 T 的记录最多分辨 1/T 的周期，
  超过 Nyquist 的频率只会混叠。

参考文献
~~~~~~~~

* Li, M., Hinnov, L., Huang, C., Ogg, J., 2018. Sedimentary noise and sea
  levels linked to land–ocean water exchange and obliquity forcing.
  *Nature Communications* 9: 1004. (沉积噪声模型与 DYNOT / ρ1 两条路线。)
* Li, M., Kump, L.R., Hinnov, L.A., Mann, M.E., 2018. Cyclostratigraphic
  correlation of astronomical forcing and Earth system processes.
  *Earth and Planetary Science Letters* 501: 165–179. (Acycle 方法学。)
* Mann, M.E. & Lees, J.M., 1996. Red noise and the climate record.
  *Geophysical Research Letters* 23: 2325–2328. (AR(1) 红噪声参考谱。)
* Mudelsee, M., 2002. AR(1) 过程的谱估计与滞后 1 阶自相关估计。
  (Acycle ``rhoAR1.m`` 采用同一估计量。)
* Box, G.E.P. & Jenkins, G.M., 1976. *Time Series Analysis*. (Yule–Walker。)
* Politis, D.N. & Romano, J.P., 1994. The stationary bootstrap.
  *Journal of Time Series Analysis* 15: 297–306. (分块自助法。)
* Li, M., Hinnov, L., Kump, L., 2019. Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences* 127: 12–22.

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只沿用其公开的行为约定
（频率轴按每单位时间、AR(1) 参考谱的形状、ρ1 的 Yule–Walker 估计量），
未翻译或复制其 MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from stratigraphy.spectral_analysis import SpectralAnalyzer, ar1_theoretical_spectrum
from stratigraphy.spectral_moments import estimate_ar1_rho
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: :func:`sedimentary_noise` 至少需要的记录点数。
MIN_SERIES_POINTS = 8

#: 频率点数的下限（``n_frequencies`` 必须至少为 1）。
MIN_FREQUENCIES = 1

#: 自助法重复次数的下限（``n_bootstrap`` 至少为 2 才能估出标准差）。
MIN_BOOTSTRAP = 2

#: :func:`sedimentary_noise` 估计零假设离散度所用的蒙特卡洛次数。
NULL_SIMULATIONS = 64

#: AR(1) 参考谱允许使用的系数上下限（留出数值余量）。
MAX_ABS_RHO = 0.99

#: 支持的噪声基线模型。
NOISE_MODELS = ("ar1", "white")


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class DynotResult:
    """
    沉积噪声模型（DYNOT / ρ1）的输出。

    Attributes:
        times: 记录的时间轴（已排序、已去非有限值）。
        noise_power: 估计出的噪声谱（功率谱密度，白噪声背景为 1），长度与
            ``noise_frequencies`` 相同。
        noise_frequencies: 噪声谱的频率轴，单位**每单位时间**（cycles/Ma 等）。
        rho1: 噪声功率比 = 记录在**噪声频带**（分析频带的短周期半边）内的
            平均噪声功率 ÷ 所选基线模型在同一频带内的平均功率。``> 1`` 表示
            记录携带的噪声强于该基线。
            **注意**：这不是 Li 等 (2018) 论文里的 ρ1（滞后 1 阶自相关系数），
            也不是沉积速率；见模块 docstring。
        rho1_uncertainty: ρ1 的不确定度。:func:`sedimentary_noise` 给出的是
            基线模型下的蒙特卡洛零假设离散度，:func:`rho1_sedimentation_ratio`
            给出的是分块自助法标准误。
        ar1_rho: 记录的 AR(1) 滞后 1 阶自相关系数（Yule–Walker 估计），
            即论文意义上的 ρ1。
    """

    times: npt.NDArray
    noise_power: npt.NDArray
    noise_frequencies: npt.NDArray
    rho1: float
    rho1_uncertainty: float
    ar1_rho: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Sedimentary noise model\n"
            f"{'=' * 48}\n"
            f"Record length: {float(self.times[-1] - self.times[0]):.6g}\n"
            f"Time points: {int(self.times.size)}\n"
            f"Frequency band: {float(self.noise_frequencies[0]):.6g} .. "
            f"{float(self.noise_frequencies[-1]):.6g} (per unit time)\n"
            f"Noise power ratio (rho1): {self.rho1:.6g}\n"
            f"rho1 uncertainty: {self.rho1_uncertainty:.6g}\n"
            f"AR(1) rho: {self.ar1_rho:.6g}"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_series(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """校验时间轴与观测值，返回按时间排序的副本。"""
    try:
        axis = np.asarray(time, dtype=float).flatten()
        signal = np.asarray(values, dtype=float).flatten()
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("time and values must be numeric arrays"),
            details={"parameter": "time/values"},
            original_exception=exc,
        ) from exc
    if axis.size != signal.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(axis.size), "values_length": int(signal.size)},
        )
    if axis.size < MIN_SERIES_POINTS:
        raise ValidationError(
            _("a sedimentary noise model needs at least {0} points; got {1}").format(MIN_SERIES_POINTS, int(axis.size)),
            details={"length": int(axis.size), "minimum": MIN_SERIES_POINTS},
        )
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(signal)):
        raise ValidationError(
            _("time and values must contain only finite values"),
            details={"n_invalid": int(np.sum(~np.isfinite(axis)) + np.sum(~np.isfinite(signal)))},
        )
    order = np.argsort(axis, kind="stable")
    axis = axis[order]
    signal = signal[order]
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("time must be strictly increasing; a non-monotonic axis has no single spectral resolution"),
            details={"n_non_increasing": int(np.sum(np.diff(axis) <= 0.0))},
        )
    if float(np.ptp(signal)) == 0.0:
        raise ValidationError(
            _("values are constant, so the record carries no noise to model"),
            details={"length": int(signal.size)},
        )
    return axis, signal


def _validate_noise_model(noise_model: str) -> str:
    """校验基线模型名称。"""
    model = str(noise_model).strip().lower()
    if model not in NOISE_MODELS:
        raise ValidationError(
            _("noise_model must be one of {0}; got {1}").format(", ".join(NOISE_MODELS), noise_model),
            details={"noise_model": str(noise_model), "supported": list(NOISE_MODELS)},
        )
    return model


def _validate_n_frequencies(n_frequencies: int) -> int:
    """校验频率点数。"""
    budget = int(n_frequencies)
    if budget != n_frequencies or budget < MIN_FREQUENCIES:
        raise ValidationError(
            _("n_frequencies must be a positive integer; got {0}").format(n_frequencies),
            details={"n_frequencies": n_frequencies, "minimum": MIN_FREQUENCIES},
        )
    return budget


# =============================================================================
# Frequency axis
# =============================================================================


def _frequency_grid(
    time: npt.NDArray,
    n_frequencies: int,
    frequency_range: tuple[float, float] | None,
) -> tuple[npt.NDArray, float, float, float]:
    """
    构造"每单位时间"的频率栅格。

    Returns:
        tuple: ``(频率栅格, 平均采样间隔 Δt, Rayleigh 分辨率, Nyquist 频率)``。
    """
    span = float(time[-1] - time[0])
    spacing = float(np.median(np.diff(time)))
    rayleigh = 1.0 / span
    nyquist = 1.0 / (2.0 * spacing)
    budget = _validate_n_frequencies(n_frequencies)

    if frequency_range is None:
        if budget == 1:
            return np.array([rayleigh], dtype=float), spacing, rayleigh, nyquist
        # The grid is set by the RECORD, not by the sampling density.  The
        # Rayleigh resolution of a record of length T is 1/T, and a step
        # coarser than that would report periods the record cannot resolve.
        # Sampling the same record more densely therefore widens the band but
        # never coarsens the resolution -- the bug this rule replaces.
        step = min((nyquist - rayleigh) / (budget - 1), rayleigh)
        highest = rayleigh + (budget - 1) * step
        if highest > nyquist:
            highest = nyquist
            step = (highest - rayleigh) / (budget - 1)
        frequencies = rayleigh + np.arange(budget, dtype=float) * step
    else:
        try:
            bounds = tuple(float(bound) for bound in frequency_range)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("frequency_range must be a (min, max) pair"),
                details={"frequency_range": repr(frequency_range)},
                original_exception=exc,
            ) from exc
        if len(bounds) != 2:
            raise ValidationError(
                _("frequency_range must be a (min, max) pair"),
                details={"frequency_range": list(bounds)},
            )
        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValidationError(
                _("frequency_range must be finite"),
                details={"frequency_range": [lower, upper]},
            )
        if not 0.0 < lower < upper:
            raise ValidationError(
                _("frequency_range must satisfy 0 < min < max"),
                details={"frequency_range": [lower, upper]},
            )
        if upper > nyquist:
            raise ValidationError(
                _("frequency_range exceeds the Nyquist frequency of the record"),
                details={"upper": upper, "nyquist": nyquist},
            )
        frequencies = np.linspace(lower, upper, budget)

    return frequencies, spacing, rayleigh, nyquist


# =============================================================================
# Spectrum estimation
# =============================================================================


def _observed_spectrum(time: npt.NDArray, values: npt.NDArray, frequencies: npt.NDArray) -> npt.NDArray:
    """
    估计记录的噪声谱（Lomb–Scargle 功率谱，白噪声背景 = 1）。

    复用 :class:`stratigraphy.spectral_analysis.SpectralAnalyzer`，因此频率轴
    归一化与本仓库其它谱分析一致：白噪声在任一频率上的期望功率为 1。
    """
    analyzer = SpectralAnalyzer()
    result = analyzer.analyze(
        time,
        values,
        frequency_range=(float(frequencies[0]), float(frequencies[-1])),
        n_frequencies=int(frequencies.size),
        detrend=True,
        ar1_significance=False,
    )
    power = np.asarray(result.power, dtype=float)
    if power.size != frequencies.size or not np.all(np.isfinite(power)):
        raise ValidationError(
            _("the power spectrum could not be estimated for this record"),
            details={"n_frequencies": int(frequencies.size)},
        )
    return power


def _baseline_spectrum(frequencies: npt.NDArray, spacing: float, rho: float, noise_model: str) -> npt.NDArray:
    """
    构造噪声基线模型在给定频率栅格上的参考谱。

    * ``"white"``：平的一条水平线（白噪声参考，功率为 1）。
    * ``"ar1"``：AR(1) 红噪声参考谱 ``(1-ρ²)/(1-2ρcos(2πfΔt)+ρ²)``，其
      形状与 Acycle ``minirho.m`` 使用的传递函数一致，且保留**物理 Nyquist**
      频率（不是被截断的频带上界）—— 这正是 Acycle ``calculateRhoM.m`` 注释
      里强调的一点。
    """
    if noise_model == "white":
        return np.ones_like(frequencies)
    reference = ar1_theoretical_spectrum(frequencies, rho, spacing)
    if not np.all(np.isfinite(reference)):
        raise ValidationError(
            _("the AR(1) baseline spectrum is not finite for this record"),
            details={"rho": float(rho), "spacing": float(spacing)},
        )
    return reference


def _noise_band_start(n_bins: int) -> int:
    """
    噪声频带（短周期半边）的起始下标。

    噪声频带取频率栅格的**上半部分**（点数与下半部分相同），即短周期端。
    必须在子频带上比较：记录与基线模型的总方差相同，整条频带上的平均功率
    之比恒为 1，比值不携带任何信息。
    """
    return int(n_bins) // 2


def _band_ratio(observed: npt.NDArray, baseline: npt.NDArray) -> float:
    """噪声功率比：观测谱与基线谱在同一噪声频带内的平均功率之比。"""
    start = _noise_band_start(observed.size)
    observed_noise = observed[start:]
    baseline_noise = baseline[start:]
    baseline_mean = float(np.mean(baseline_noise))
    if not np.isfinite(baseline_mean) or baseline_mean <= 0.0:
        raise ValidationError(
            _("the baseline noise model has no power in the analysis band"),
            details={"baseline_mean": baseline_mean},
        )
    observed_mean = float(np.mean(observed_noise))
    if not np.isfinite(observed_mean):
        raise ValidationError(
            _("the observed noise spectrum has no finite mean power"),
            details={"observed_mean": observed_mean},
        )
    return observed_mean / baseline_mean


# =============================================================================
# Stochastic helpers
# =============================================================================


def _ar1_coefficient(values: npt.NDArray) -> float:
    """记录的 AR(1) 滞后 1 阶自相关系数（Yule–Walker 估计，论文的 ρ1）。"""
    rho = float(estimate_ar1_rho(values))
    return float(np.clip(rho, -MAX_ABS_RHO, MAX_ABS_RHO))


def _simulate_baseline(
    length: int,
    variance: float,
    rho: float,
    noise_model: str,
    rng: np.random.Generator,
) -> npt.NDArray:
    """按基线模型生成一条零均值代理序列（采样域上的 AR(1) 或白噪声）。"""
    if noise_model == "white":
        return rng.normal(0.0, np.sqrt(max(variance, 0.0)), size=length)
    innovation_sd = float(np.sqrt(max(variance * (1.0 - rho * rho), 0.0)))
    innovations = rng.normal(0.0, innovation_sd, size=length)
    series = np.empty(length, dtype=float)
    series[0] = rng.normal(0.0, float(np.sqrt(max(variance, 0.0))))
    for index in range(1, length):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _null_spread(
    time: npt.NDArray,
    values: npt.NDArray,
    frequencies: npt.NDArray,
    spacing: float,
    noise_model: str,
    rho: float,
    rng: np.random.Generator,
    n_simulations: int = NULL_SIMULATIONS,
) -> float:
    """
    基线模型下的蒙特卡洛零假设离散度。

    按基线模型生成代理序列，逐条计算噪声功率比，得到"记录完全服从基线模型
    时 ρ1 的抽样离散度"。它是 ρ1 估计量的采样不确定度的一个下界估计。
    """
    centred = values - float(np.mean(values))
    variance = float(np.var(centred))
    estimates: list[float] = []
    for _ in range(int(n_simulations)):
        surrogate = _simulate_baseline(centred.size, variance, rho, noise_model, rng)
        observed = _observed_spectrum(time, surrogate, frequencies)
        baseline = _baseline_spectrum(frequencies, spacing, rho, noise_model)
        estimates.append(_band_ratio(observed, baseline))
    if len(estimates) < 2:
        return float("nan")
    return float(np.std(estimates, ddof=1))


def _block_bootstrap_spread(
    time: npt.NDArray,
    values: npt.NDArray,
    frequencies: npt.NDArray,
    spacing: float,
    noise_model: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """
    分块自助法给出的 ρ1 标准误。

    记录是自相关的，普通自助法会低估方差，因此采用**移动分块**重抽
    (Politis & Romano 1994 的移动分块自助法)：把序列切成若干长度为
    ``n^(1/3)`` 的块，有放回地抽取并拼接成与原序列等长的替身，再放到**原
    来的时间轴**上重新估计 ρ1。

    注意：替身沿用原时间轴，因此对不等间隔记录而言，分块重抽只近似保持
    时间结构；其适用范围与 Acycle ``erhoAR1.m`` 的做法一致（后者先把序列
    重采样到等间隔网格上再做滑窗估计）。
    """
    length = int(values.size)
    block = max(2, round(length ** (1.0 / 3.0)))
    block = min(block, length)
    n_blocks = int(np.ceil(length / block))
    estimates: list[float] = []
    for _ in range(int(n_bootstrap)):
        starts = rng.integers(0, length - block + 1, size=n_blocks)
        surrogate = np.concatenate([values[start : start + block] for start in starts])[:length]
        observed = _observed_spectrum(time, surrogate, frequencies)
        rho = _ar1_coefficient(surrogate)
        baseline = _baseline_spectrum(frequencies, spacing, rho, noise_model)
        estimates.append(_band_ratio(observed, baseline))
    if len(estimates) < 2:
        return float("nan")
    return float(np.std(estimates, ddof=1))


# =============================================================================
# Public API
# =============================================================================


def sedimentary_noise(
    time: npt.NDArray,
    values: npt.NDArray,
    frequency_range: tuple[float, float] | None = None,
    n_frequencies: int = 500,
    noise_model: str = "ar1",
    random_seed: int | None = None,
) -> DynotResult:
    """
    估计一条地层记录的沉积噪声谱与噪声功率比 ρ1。

    与论文的差别：本函数**不做轨道调谐**，因此它给出的是"记录自身的噪声谱
    与基线噪声模型的比较"，而不是 Li 等 (2018) 完整的 DYNOT 曲线。

    ρ1 是**噪声频带**（分析频带的短周期半边）上的噪声功率比；两个基线都是
    绝对的，不向记录重新标定。详见模块 docstring。

    Parameters:
        time: 记录的时间轴，每单位时间（Ma、kyr……），严格递增。
        values: 与 ``time`` 等长的观测值（地层代用序列）。
        frequency_range: 可选的 ``(最小频率, 最大频率)``（每单位时间）。不给
            时使用默认栅格：下界为 Rayleigh 分辨率 ``1/T``，上界不超过
            Nyquist ``1/(2Δt)``，且**步长永远不比 Rayleigh 分辨率更粗**
            （采样越密只会让频带变宽，不会让分辨率变差）。
        n_frequencies: 频率栅格的点数，至少为 1。
        noise_model: 基线模型，``"ar1"``（红噪声，默认）或 ``"white"``
            （白噪声）。见模块 docstring 中两种基线的差别。
        random_seed: 蒙特卡洛零假设的随机种子。给定种子时结果完全可复现；
            不给种子时调用 ``utils.statistics_core.make_rng(None, ...)``，
            它会发出 ``RuntimeWarning`` 提示结果不可复现。

    Returns:
        DynotResult: 噪声谱、频率轴、噪声功率比 ρ1、ρ1 的零假设离散度以及
        记录的 AR(1) 系数。

    Raises:
        ValidationError: 输入非法（长度不一致、点数太少、非有限值、时间轴
            非严格递增、序列为常数）、频率范围非法、``n_frequencies`` 越界或
            ``noise_model`` 不受支持。

    Examples:
        >>> import numpy as np
        >>> axis = np.linspace(0.0, 1.0, 400)
        >>> signal = np.sin(2 * np.pi * 40 * axis) + 0.01 * np.random.default_rng(3).normal(size=400)
        >>> result = sedimentary_noise(axis, signal, noise_model="white", random_seed=1)
        >>> result.rho1 > 0.0
        True
    """
    axis, signal = _validate_series(time, values)
    model = _validate_noise_model(noise_model)
    frequencies, spacing, _rayleigh, _nyquist = _frequency_grid(axis, n_frequencies, frequency_range)

    observed = _observed_spectrum(axis, signal, frequencies)
    rho = _ar1_coefficient(signal)
    baseline = _baseline_spectrum(frequencies, spacing, rho, model)
    rho1 = _band_ratio(observed, baseline)

    rng = make_rng(random_seed, context="DYNOT sedimentary noise")
    uncertainty = _null_spread(axis, signal, frequencies, spacing, model, rho, rng)

    logger.info(
        "DYNOT: %d points, %d frequencies, rho1=%.6g, ar1_rho=%.6g",
        int(axis.size),
        int(frequencies.size),
        rho1,
        rho,
    )
    return DynotResult(
        times=axis,
        noise_power=observed,
        noise_frequencies=frequencies,
        rho1=rho1,
        rho1_uncertainty=uncertainty,
        ar1_rho=rho,
    )


def rho1_sedimentation_ratio(
    time: npt.NDArray,
    values: npt.NDArray,
    frequency_range: tuple[float, float] | None = None,
    n_frequencies: int = 500,
    n_bootstrap: int = 200,
    random_seed: int | None = None,
) -> DynotResult:
    """
    以 **AR(1) 红噪声基线**给出噪声功率比 ρ1 及其自助法不确定度。

    这是 Li 等 (2018) 沉积噪声模型里 ρ1 路线在本仓库的可实现版本：本函数
    报告的 :attr:`DynotResult.ar1_rho` 就是论文里那个滞后 1 阶自相关系数
    （噪声越多越低），而 :attr:`DynotResult.rho1` 是**噪声功率与红噪声基线
    功率之比**（噪声频带 = 分析频带的短周期半边，见模块 docstring）。两者
    都不是沉积速率。

    Parameters:
        time: 记录的时间轴，每单位时间，严格递增。
        values: 与 ``time`` 等长的观测值。
        frequency_range: 可选的 ``(最小频率, 最大频率)``（每单位时间）。
        n_frequencies: 频率栅格的点数，至少为 1。
        n_bootstrap: 分块自助法重复次数，至少为 2。
        random_seed: 自助法的随机种子。给定种子时结果完全可复现；不给种子
            时调用 ``utils.statistics_core.make_rng(None, ...)``，它会发出
            ``RuntimeWarning`` 提示结果不可复现。

    Returns:
        DynotResult: 噪声谱、频率轴、噪声功率比 ρ1、其自助法标准误，以及
        记录的 AR(1) 系数。

    Raises:
        ValidationError: 输入非法、频率范围非法、``n_frequencies`` 或
            ``n_bootstrap`` 越界。

    Examples:
        >>> import numpy as np
        >>> axis = np.linspace(0.0, 1.0, 400)
        >>> signal = np.random.default_rng(7).normal(size=400)
        >>> result = rho1_sedimentation_ratio(axis, signal, n_bootstrap=8, random_seed=2)
        >>> result.rho1 > 0.0
        True
    """
    axis, signal = _validate_series(time, values)
    replicates = int(n_bootstrap)
    if replicates != n_bootstrap or replicates < MIN_BOOTSTRAP:
        raise ValidationError(
            _("n_bootstrap must be an integer of at least {0}; got {1}").format(MIN_BOOTSTRAP, n_bootstrap),
            details={"n_bootstrap": n_bootstrap, "minimum": MIN_BOOTSTRAP},
        )
    frequencies, spacing, _rayleigh, _nyquist = _frequency_grid(axis, n_frequencies, frequency_range)

    observed = _observed_spectrum(axis, signal, frequencies)
    rho = _ar1_coefficient(signal)
    baseline = _baseline_spectrum(frequencies, spacing, rho, "ar1")
    rho1 = _band_ratio(observed, baseline)

    rng = make_rng(random_seed, context="DYNOT rho1 ratio")
    uncertainty = _block_bootstrap_spread(axis, signal, frequencies, spacing, "ar1", replicates, rng)

    logger.info(
        "DYNOT rho1: %d points, %d frequencies, rho1=%.6g +/- %.6g, ar1_rho=%.6g",
        int(axis.size),
        int(frequencies.size),
        rho1,
        uncertainty,
        rho,
    )
    return DynotResult(
        times=axis,
        noise_power=observed,
        noise_frequencies=frequencies,
        rho1=rho1,
        rho1_uncertainty=uncertainty,
        ar1_rho=rho,
    )
