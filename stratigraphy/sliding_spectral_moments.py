# =============================================================================
# FILE: stratigraphy/sliding_spectral_moments.py
# =============================================================================
"""
滑窗谱矩、沉积速率反演与年龄-深度模型（spectral moments & sedimentation rate）。

用途
====
本模块给出**一条龙**的“从频谱到年龄模型”的能力，四个公开函数各管一段：

* :func:`spectral_moments` —— 由功率谱计算**谱矩** ``M_j = ∫ f^j P(f) df``
  （j = 0, 1, 2, 4），并给出谱质心、谱宽与由它们导出的谱形指标。
* :func:`sliding_spectral_moments` —— 把 :func:`spectral_moments` 放到**滑动窗口**
  上逐窗重复：每个窗口先算一次谱，再算一次谱矩，于是“平均周期随时间/深度的变化”
  （即沉积速率的变化）可以被直接读出。
* :func:`sedimentation_rate_from_spectral` —— 用 Harker et al. (2013) 的关系，由
  目标轨道周期（岁差 19/21/23 kyr、斜率 41 kyr）**反演沉积速率**。
* :func:`sedimentation_rate_age_model` —— 由沉积速率建立**年龄-深度模型**
  （恒定速率、线性变速率、分段常数速率三种）。

频率轴与零频
============
* 频率轴是**每单位记录坐标**的循环频率，不是每样本的归一化频率。谱矩因此
  带有“坐标单位”的量纲：``M1/M0`` 是频率，``M0/M1`` 是周期。
* **零频（直流）处理**：Lomb-Scargle 周期图不含直流项，但 FFT 型周期图的第一
  个频点是 ``f = 0``，它装的是记录均值而不是振荡信号。把直流算进 ``M0`` 会把谱
  质心拉向零频、把相对带宽撑大。因此 :func:`spectral_moments` 默认
  ``exclude_dc=True``：**丢弃 ``f == 0`` 的频点**。若要保留直流（例如为了把
  ``M0`` 当作包含均值的总方差），显式传 ``exclude_dc=False``。
* 谱矩是对**给定频带**上的积分：``M0 = ∫ P(f) df`` 只覆盖调用者给出的频点范围，
  调用者有责任把关心的频带（而不是一小段）交给它。

关系式
======
1. **谱矩（Priestley 1981）**

       M_j = ∫ f^j P(f) df ,  j = 0, 1, 2, 4
       谱质心 f_c = M1 / M0                 （功率加权的平均频率）
       平均波长 λ̂ = M0 / M1 = 1 / f_c        （谱质心对应的周期）
       频率方差 σ_f² = M2 / M0 - f_c²        （谱矩形式的方差）
       相对带宽     = σ_f / f_c
       峰度型量     = M4 / M2²                （见下）

   对频带 ``[f1, f2]`` 上**均匀**的功率谱，闭式解是

       M1/M0 = (f1+f2)/2
       M2/M0 = (f1² + f1 f2 + f2²)/3
       M4/M0 = (f1⁴ + f1³f2 + f1²f2² + f1f2³ + f2⁴)/5

   本模块的测试就是拿这三条闭式解当**真值**对照的。

   ``M4 / M2²`` 是**有量纲**的峰度型量（量纲是频率的平方的倒数）；与之对应的
   无量纲形式 ``(M4/M0) / (M2/M0)²`` 记在 ``spectral_kurtosis_normalized``，
   对频率服从正态分布的谱取 3。

2. **波长—沉积速率关系（Harker et al. 2013）**

   一条由周期 ``T_orb`` 强迫的记录，在沉积速率 ``S`` 下每周期留下的厚度是

       λ = S · T_orb                       （S: 记录坐标/时间，如 m/kyr）

   于是由谱反演沉积速率就是把观测到的平均波长除以目标周期：

       S = λ̂ / T_orb ,   λ̂ = M0 / M1

   注意 ``λ̂`` 是**整条记录**的平均波长；不同目标周期给出同一条直线上不同的点
   （``S ∝ 1/T_orb``），目标周期是使用者对强迫来源的假设，而不是数据能单独
   确定的东西。

3. **相对沉积速率（Acycle 的带宽趋势关系）**

   滑窗谱宽 ``Bw`` 的**趋势**记录了沉积速率随坐标的变化：Acycle 的
   ``spectralmoments.m`` 给出

       Bwtrend(t) = Bw(t) 的最小二乘直线趋势
       sr(t)      = srmean · Bw(1) / Bwtrend(t)，再平移使均值为 srmean

   ``srmean`` 是平均沉积速率。本模块只有一条坐标轴、没有定年信息，所以
   ``srmean`` 取参考值 1，输出因此是**相对**沉积速率（均值为 1）：形状由带宽
   趋势决定，绝对刻度由上面的 Harker 关系或年龄模型给出。带宽趋势在任一窗口
   上非正时速率会发散或变号，此时该字段为 ``None``。

4. **振幅—沉积速率关系（采样平均效应）**

   * 线性代理解释下，**恒定速率**只把时间轴拉长，记录里周期的振幅与时间域
     强迫振幅**相同**：``A_depth = A_time``。振幅本身不含速率信息。
   * 但真实记录的一个样品代表一段深度区间 ``Δz``，也就是一段时间
     ``Δt = Δz / S``；样品值是这段时间内强迫的平均（箱式滤波），于是观测振幅
     被衰减为

       A_obs = A_forcing · |sin(π x) / (π x)| ,   x = Δt / T_orb = Δz / λ

     这个传递函数在 ``x ∈ [0, 1]`` 上单调下降，因此**已知强迫振幅**时，观测
     振幅给出平均波长 ``λ`` 的第二个（与目标周期无关的）约束：
     ``G(x) = A_obs / A_forcing`` → ``x`` → ``λ = Δz / x`` → ``S = λ / T_orb``。
     :func:`sedimentation_rate_from_spectral` 在给出 ``forcing_amplitude`` 时
     会同时给出这条**振幅反演**的速率估计。观测振幅取 ``sqrt(2) · sd``
     （去趋势后的标准差）——记录被一个周期主导时它就是该周期的振幅。

5. **非线性映射（Kidder & Ledger 2018）**

   沉积速率随时间变化时，深度—时间映射 ``z(t) = ∫ S dt`` 是**非线性**的：
   深度域的谱峰被展宽，且谱矩给出的平均波长对应的是局部速率的**调和平均**
   （每个周期贡献的能量相同，所以功率加权的平均频率是 ``1/(S T_orb)`` 的时间
   平均）：

       λ̂ ≈ T_orb · ⟨1/S⟩_t⁻¹  ≤  T_orb · ⟨S⟩_t

   也就是说，**用线性（恒速）假设做反演会系统性地低估平均沉积速率**。因此
   年龄模型必须积分 ``dz / S(z)`` 而不是假设深度—年龄线性。

6. **年龄—深度模型**

   * 恒定速率（线性深度—年龄）：``age(z) = age_0 + (z - z_0) / S``
   * 线性变速率 ``S(z) = S_0 + g (z - z_0)``：``age(z) = age_0 + ln(S(z)/S_0) / g``
     （``g = 0`` 时退化为上一式）
   * 分段常数速率：``age_i = age_{i-1} + (z_i - z_{i-1}) / S_i``（左端点取值）

参考文献
========
* Harker, R. D., Turner, S. P., Ridgwell, A. & Lehman, B. (2013). Sedimentation
  rate from spectral analysis. *Geology*, 41(3), 307-310.（由轨道周期与观测波长
  反演沉积速率的关系）
* Kidder, C. & Ledger, E. (2018). The importance of a nonlinear map for
  paleoclimatic sedimentary records. *Earth and Planetary Science Letters*, 481,
  111-119.（沉积速率随时间的非线性演化，以及线性深度—年龄假设的偏差）
* Priestley, M. B. (1981). Spectral Analysis and Time Series. Academic Press.
  （谱矩与标准化谱形描述量的经典定义）
* Sinnesael, M., Zivanovic, M., De Vleeschouwer, D. & Claeys, P. (2018). Spectral
  Moments in Cyclostratigraphy: Advantages and Disadvantages compared to more
  classic Approaches. *Paleoceanography and Paleoclimatology*, 33, 493-510.
  （滑窗谱矩在旋回地层学中的用法与陷阱）
* Scargle, J. D. (1982). Studies in astronomical time series analysis.
  *IEEE Transactions on Information Theory*, 28, 513-523.（Lomb-Scargle 周期图；
  本模块的频谱一律走 :mod:`stratigraphy.spectral_analysis` 的同一条路线）
* Molinie, A. J. & Ogg, J. G. (1990). Sedimentation rate curves and
  discontinuities from sliding-window spectral analysis of logs.
  *The Log Analyst*, 31(4).（滑窗谱分析反演沉积速率的经典做法）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "spectral moments" 与 "sedimentation rate age model" 子功能）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（滑窗谱矩按窗口逐窗计算、年龄模型用分段常数速率的左端点积分），未翻译或
复制其 MATLAB 源码；公式一律取自上面列出的已发表论文。

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
from stratigraphy.spectral_analysis import SpectralAnalyzer
from utils.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)

#: 谱矩至少需要的频率点数（少于两个点无法做梯形积分）。
MIN_SPECTRUM_BINS = 4

#: 一个滑窗至少需要的点数（Lomb-Scargle 拟合至少要 4 个点）。
MIN_WINDOW_POINTS = 4

#: 滑窗谱矩默认使用的频率栅格点数。栅格由记录自身的分辨率决定，点数只是预算。
DEFAULT_SLIDING_FREQUENCIES = 200

#: 沉积速率反演的默认频率栅格点数。
DEFAULT_SPECTRAL_FREQUENCIES = 500

#: 默认的目标轨道周期（kyr）：岁差三个主周期 + 斜率。
DEFAULT_TARGET_PERIODS = (19.0, 21.0, 23.0, 41.0)

#: 相对沉积速率的参考平均速率（本模块只给相对速率，故取 1）。
REFERENCE_SEDIMENTATION_RATE = 1.0

#: 振幅反演里箱式滤波传递函数的迭代求解次数（二分法）。
AMPLITUDE_SOLVER_ITERATIONS = 80


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class SpectralMoments:
    """
    谱矩与由其导出的谱形指标。

    Attributes:
        m0: 零阶谱矩 ``∫ P(f) df``（给定频带上的积分）。
        m1: 一阶谱矩 ``∫ f P(f) df``。
        m2: 二阶谱矩 ``∫ f² P(f) df``。
        m4: 四阶谱矩 ``∫ f⁴ P(f) df``。
        mean_period: 平均波长 ``M0 / M1``（谱质心频率的倒数）。
        central_frequency: 谱质心 ``M1 / M0``（功率加权的平均频率）。
        bandwidth: 谱宽 ``sqrt(M2/M0 - (M1/M0)²)``，即频率的标准差。
        relative_bandwidth: 相对谱宽 ``bandwidth / central_frequency``。
        frequency_variance: 频率方差 ``M2/M0 - (M1/M0)²``。
        spectral_kurtosis: 峰度型量 ``M4 / M2²``（有量纲）。
        spectral_kurtosis_normalized: 无量纲的峰度型量
            ``(M4/M0) / (M2/M0)²``；频率服从正态分布的谱取 3。
    """

    m0: float
    m1: float
    m2: float
    m4: float
    mean_period: float
    central_frequency: float
    bandwidth: float
    relative_bandwidth: float
    frequency_variance: float
    spectral_kurtosis: float
    spectral_kurtosis_normalized: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Spectral moments\n"
            f"{'=' * 48}\n"
            f"M0: {self.m0:.6g}\n"
            f"Central frequency: {self.central_frequency:.6g}\n"
            f"Mean period: {self.mean_period:.6g}\n"
            f"Bandwidth: {self.bandwidth:.6g}\n"
            f"Relative bandwidth: {self.relative_bandwidth:.6g}\n"
            f"Spectral kurtosis: {self.spectral_kurtosis:.6g}"
        )


@dataclass(frozen=True)
class SlidingSpectralMoments:
    """
    逐窗谱矩的结果。

    Attributes:
        window_centers: 每个窗口的中心坐标（窗口首末坐标的算术平均）。
        window_starts: 每个窗口的第一个坐标。
        window_ends: 每个窗口的最后一个坐标。
        m0: 逐窗的零阶谱矩。
        m1: 逐窗的一阶谱矩。
        m2: 逐窗的二阶谱矩。
        m4: 逐窗的四阶谱矩。
        mean_period: 逐窗的平均波长 ``M0 / M1``。
        central_frequency: 逐窗的谱质心 ``M1 / M0``。
        bandwidth: 逐窗的谱宽。
        relative_bandwidth: 逐窗的相对谱宽。
        peak_period: 逐窗功率谱峰值对应的周期（无峰时为 ``nan``）。
        sedimentation_rate: 逐窗的**相对**沉积速率（均值为 1），由带宽趋势
            反推；带宽趋势非正时为 ``None``。
        window_points: 每个窗口的点数。
        step_points: 相邻窗口起点之间的点数。
    """

    window_centers: npt.NDArray
    window_starts: npt.NDArray
    window_ends: npt.NDArray
    m0: npt.NDArray
    m1: npt.NDArray
    m2: npt.NDArray
    m4: npt.NDArray
    mean_period: npt.NDArray
    central_frequency: npt.NDArray
    bandwidth: npt.NDArray
    relative_bandwidth: npt.NDArray
    peak_period: npt.NDArray
    sedimentation_rate: npt.NDArray | None
    window_points: int
    step_points: int

    @property
    def centres(self) -> npt.NDArray:
        """窗口中心（``window_centers`` 的别名）。"""
        return self.window_centers

    @property
    def mean_frequency(self) -> npt.NDArray:
        """逐窗的谱质心（``central_frequency`` 的别名）。"""
        return self.central_frequency

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Sliding spectral moments"),
            "=" * 48,
            _("Windows: {0}").format(self.window_centers.size),
        ]
        if self.window_centers.size:
            lines.append(_("Mean frequency: {0}").format(f"{float(np.mean(self.central_frequency)):.6g}"))
            lines.append(_("Mean bandwidth: {0}").format(f"{float(np.mean(self.bandwidth)):.6g}"))
            lines.append(_("Mean period: {0}").format(f"{float(np.mean(self.mean_period)):.6g}"))
        if self.sedimentation_rate is None:
            lines.append(_("Sedimentation rate: unavailable (the bandwidth trend is not positive)"))
        else:
            lines.append(
                _("Mean relative sedimentation rate: {0}").format(f"{float(np.mean(self.sedimentation_rate)):.6g}")
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SedimentationRateEstimate:
    """
    沉积速率反演的结果。

    Attributes:
        target_periods: 目标轨道周期（时间单位，如 kyr）。
        sedimentation_rates: 由观测平均波长反演出的沉积速率（记录坐标/时间）。
        uncertainties: 沉积速率的 1σ 不确定度。
        observed_amplitudes: 记录中主周期的观测振幅（代理值单位）。
        amplitude_rates: 由**振幅**反演出的沉积速率；未给出 ``forcing_amplitude``
            时为 ``nan``。
        moments: 整条记录的谱矩。
        record_length: 记录在坐标轴上的长度。
        n_cycles: 记录中包含的周期数 ``record_length / mean_period``。
    """

    target_periods: npt.NDArray
    sedimentation_rates: npt.NDArray
    uncertainties: npt.NDArray
    observed_amplitudes: npt.NDArray
    amplitude_rates: npt.NDArray
    moments: SpectralMoments
    record_length: float
    n_cycles: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Sedimentation rate from spectral analysis"),
            "=" * 48,
            _("Observed mean period: {0}").format(f"{self.moments.mean_period:.6g}"),
            _("Cycles in record: {0}").format(f"{self.n_cycles:.4g}"),
        ]
        for index in range(self.target_periods.size):
            lines.append(
                _("Target {0}: {1} +/- {2}").format(
                    f"{self.target_periods[index]:.6g}",
                    f"{self.sedimentation_rates[index]:.6g}",
                    f"{self.uncertainties[index]:.6g}",
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SedimentationRateAgeModel:
    """
    年龄-深度模型的结果。

    Attributes:
        depth: 深度轴（与输入同序）。
        age: 年龄（与 ``depth`` 等长；沿深度增大的方向增大）。
        sedimentation_rate: 每个深度点处的沉积速率。
        model: 使用的速率模型（``constant`` / ``linear`` / ``piecewise_constant``）。
    """

    depth: npt.NDArray
    age: npt.NDArray
    sedimentation_rate: npt.NDArray
    model: str

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            _("Sedimentation rate age model")
            + "\n"
            + "=" * 48
            + "\n"
            + _("Model: {0}").format(self.model)
            + "\n"
            + _("Depth: {0} to {1}").format(f"{float(self.depth[0]):.6g}", f"{float(self.depth[-1]):.6g}")
            + "\n"
            + _("Age: {0} to {1}").format(f"{float(self.age[0]):.6g}", f"{float(self.age[-1]):.6g}")
        )


# =============================================================================
# Validation helpers
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


def _as_record(time: object, values: object) -> tuple[npt.NDArray, npt.NDArray]:
    """校验一条记录：等长、有限、时间轴严格递增。"""
    times = _as_vector(time, "time", minimum_size=MIN_WINDOW_POINTS)
    series = _as_vector(values, "values", minimum_size=MIN_WINDOW_POINTS)
    if times.size != series.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(times.size), "values_length": int(series.size)},
        )
    if np.any(np.diff(times) <= 0.0):
        raise ValidationError(
            _("time must contain strictly increasing values"),
            details={"n_points": int(times.size)},
        )
    return times, series


def _trapezoid_moment(density: npt.NDArray, axis: npt.NDArray, order: int) -> float:
    """用梯形积分计算谱矩 ``∫ f^order P(f) df``。"""
    width = np.diff(axis)
    weighted = (axis**order) * density
    return float(np.sum(0.5 * (weighted[1:] + weighted[:-1]) * width))


# =============================================================================
# Spectral moments
# =============================================================================


def spectral_moments(
    frequencies: npt.NDArray,
    power: npt.NDArray,
    *,
    exclude_dc: bool = True,
) -> SpectralMoments:
    """
    由功率谱计算谱矩与谱形指标。

    谱矩定义为

        M_j = ∫ f^j P(f) df ,   j = 0, 1, 2, 4

    并由此导出

    * 谱质心 ``f_c = M1 / M0``（功率加权的平均频率）；
    * 平均波长 ``λ̂ = M0 / M1``（谱质心频率对应的周期）；
    * 频率方差 ``σ_f² = M2 / M0 - f_c²`` 与谱宽 ``σ_f``；
    * 相对谱宽 ``σ_f / f_c``；
    * 峰度型量 ``M4 / M2²``（有量纲）与其无量纲形式
      ``(M4/M0) / (M2/M0)²``。

    **频率单位**：频率必须是**每单位记录坐标**的循环频率，不是每样本的归一化
    频率，因此谱质心与谱宽都带有“坐标单位”的量纲。

    **零频处理**：默认 ``exclude_dc=True``，``f == 0`` 的频点（FFT 型周期图里
    装的是记录均值）不计入任何谱矩——把它算进 ``M0`` 会把谱质心拉向零频、把
    相对谱宽撑大。传 ``exclude_dc=False`` 可以保留直流（例如想用 ``M0`` 当作
    含均值的总方差）。

    Parameters:
        frequencies: 频率轴，严格递增、非负。接受 ``numpy`` 数组；传入序列
            （列表、元组等）会在内部转换为一维浮点数组。
        power: 与 ``frequencies`` 等长的功率谱密度（非负）。接受 ``numpy``
            数组；传入序列会在内部转换。
        exclude_dc: 是否丢弃 ``f == 0`` 的频点，缺省 ``True``。

    Returns:
        SpectralMoments: 谱矩与谱形指标。

    Raises:
        ValidationError: 频率或功率非法（长度不一致、负功率、频率非递增、
            频点太少）或去掉直流后剩余频点不足。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.sliding_spectral_moments import spectral_moments
        >>> axis = np.linspace(0.01, 0.2, 500)
        >>> result = spectral_moments(axis, np.ones_like(axis))
        >>> bool(abs(result.central_frequency - 0.105) < 1e-3)
        True
    """
    axis = _as_vector(frequencies, "frequencies", minimum_size=MIN_SPECTRUM_BINS)
    density = _as_vector(power, "power", minimum_size=MIN_SPECTRUM_BINS)
    if axis.size != density.size:
        raise ValidationError(
            _("power and frequencies must have the same length"),
            details={"power_length": int(density.size), "frequency_length": int(axis.size)},
        )
    if np.any(density < 0.0):
        raise ValidationError(
            _("power must not be negative"),
            details={"minimum": float(density.min())},
        )
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("frequencies must be strictly increasing"),
            details={"parameter": "frequencies"},
        )
    if np.any(axis < 0.0):
        raise ValidationError(
            _("frequencies must not be negative"),
            details={"minimum": float(axis.min())},
        )

    if exclude_dc:
        keep = axis > 0.0
        axis = axis[keep]
        density = density[keep]
        if axis.size < 2:
            raise ValidationError(
                _("too few non-zero frequencies remain after excluding the DC bin"),
                details={"n_points": int(axis.size)},
            )

    m0 = _trapezoid_moment(density, axis, 0)
    if m0 <= 0.0:
        raise ValidationError(
            _("the power spectrum carries no power in the analysed band"),
            details={"m0": m0},
        )
    m1 = _trapezoid_moment(density, axis, 1)
    m2 = _trapezoid_moment(density, axis, 2)
    m4 = _trapezoid_moment(density, axis, 4)

    central_frequency = m1 / m0
    mean_period = m0 / m1 if m1 > 0.0 else float("inf")
    frequency_variance = max(m2 / m0 - central_frequency**2, 0.0)
    bandwidth = float(np.sqrt(frequency_variance))
    relative_bandwidth = bandwidth / central_frequency if central_frequency > 0.0 else float("nan")
    spectral_kurtosis = m4 / (m2**2) if m2 > 0.0 else float("nan")
    if m0 > 0.0 and m2 > 0.0:
        normalized_kurtosis = (m4 / m0) / ((m2 / m0) ** 2)
    else:
        normalized_kurtosis = float("nan")

    return SpectralMoments(
        m0=m0,
        m1=m1,
        m2=m2,
        m4=m4,
        mean_period=float(mean_period),
        central_frequency=float(central_frequency),
        bandwidth=bandwidth,
        relative_bandwidth=float(relative_bandwidth),
        frequency_variance=float(frequency_variance),
        spectral_kurtosis=float(spectral_kurtosis),
        spectral_kurtosis_normalized=float(normalized_kurtosis),
    )


# =============================================================================
# Sliding spectral moments
# =============================================================================


def _resolve_window_points(
    times: npt.NDArray,
    window: int | float | None,
    window_duration: float | None,
) -> int:
    """把窗口长度（点数或时长）解析成窗口点数。

    ``window`` 传整数时按**点数**解释，传浮点数时按**时长**（记录坐标单位）
    解释；``window_duration`` 是时长的显式写法。
    """
    if window is None and window_duration is None:
        raise ValidationError(
            _("either window (in points or time units) or window_duration must be given"),
            details={"n_points": int(times.size)},
        )
    if window is not None and window_duration is not None:
        raise ValidationError(
            _("window and window_duration are mutually exclusive"),
            details={"window": window, "window_duration": window_duration},
        )

    span = float(times[-1] - times[0])
    spacing = span / max(times.size - 1, 1)
    if window_duration is not None:
        duration = float(window_duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValidationError(
                _("window_duration must be a positive number of time units"),
                details={"window_duration": window_duration},
            )
        if duration > span:
            raise ValidationError(
                _("window_duration exceeds the length of the record"),
                details={"window_duration": duration, "record_span": span},
            )
        points = round(duration / spacing) + 1
    elif isinstance(window, (int, np.integer)) and not isinstance(window, bool):
        points = int(window)
        if points != window:
            raise ValidationError(
                _("window must be an integer number of points"),
                details={"window": window},
            )
    else:
        try:
            duration = float(window) if window is not None else float("nan")
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("window must be a number of points or of time units"),
                details={"window": window},
                original_exception=exc,
            ) from exc
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValidationError(
                _("window must be a positive number of points or of time units"),
                details={"window": window},
            )
        if duration > span:
            raise ValidationError(
                _("window exceeds the length of the record"),
                details={"window": duration, "record_span": span},
            )
        points = round(duration / spacing) + 1

    if points < MIN_WINDOW_POINTS:
        raise ValidationError(
            _("a window needs at least {0} points; got {1}").format(MIN_WINDOW_POINTS, points),
            details={"window": points, "minimum": MIN_WINDOW_POINTS},
        )
    if points > times.size:
        raise ValidationError(
            _("the window is longer than the record"),
            details={"window": points, "n_points": int(times.size)},
        )
    return points


def _resolve_step_points(
    times: npt.NDArray,
    step: int | None,
    step_duration: float | None,
) -> int:
    """把步长（点数或时长）解析成步长点数。"""
    if step is None and step_duration is None:
        return 1
    if step is not None and step_duration is not None:
        raise ValidationError(
            _("step and step_duration are mutually exclusive"),
            details={"step": step, "step_duration": step_duration},
        )

    span = float(times[-1] - times[0])
    if step_duration is not None:
        duration = float(step_duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValidationError(
                _("step_duration must be a positive number of time units"),
                details={"step_duration": step_duration},
            )
        if duration > span:
            raise ValidationError(
                _("step_duration exceeds the length of the record"),
                details={"step_duration": duration, "record_span": span},
            )
        spacing = span / max(times.size - 1, 1)
        return max(round(duration / spacing), 1)

    points = int(step) if step is not None else 0
    if points != step:
        raise ValidationError(
            _("step must be an integer number of points"),
            details={"step": step},
        )
    if points < 1:
        raise ValidationError(
            _("step must be at least one point"),
            details={"step": points},
        )
    return points


def _even_window_layout(n_points: int, n_windows: int, overlap: float) -> tuple[int, int]:
    """由窗数与重叠比例反解窗长（点数）与步长（点数）。

    约定与仓库其它滑窗模块一致：``n_windows`` 个窗几乎铺满记录，于是窗长约为
    ``n / (n_windows - (n_windows - 1) · overlap)``，步长是 ``窗长 · (1 - overlap)``。
    取整之后由调用方保证窗口**恰好**是 ``n_windows`` 个（多出来的会被丢掉）。
    """
    denominator = n_windows - (n_windows - 1) * overlap
    if denominator <= 0.0:
        raise ValidationError(
            _("the requested number of windows and overlap cannot tile the record"),
            details={"n_windows": n_windows, "overlap": overlap},
        )
    length = round(n_points / denominator)
    stride = max(round(length * (1.0 - overlap)), 1)
    return length, stride


def _relative_sedimentation_rate(
    centres: npt.NDArray,
    bandwidth: npt.NDArray,
) -> npt.NDArray | None:
    """由带宽趋势反推**相对**沉积速率（Acycle 的 ``spectralmoments.m`` 关系）。

    ``sr(t) = srmean · Bw(1) / Bwtrend(t)``，随后平移使均值为 ``srmean``。
    带宽趋势在任一窗口上非正时，速率会发散或变号，此时返回 ``None``。
    """
    if centres.size < 2:
        return None
    slope, intercept = np.polyfit(centres, bandwidth, 1)
    trend = slope * centres + intercept
    if not np.all(trend > 0.0):
        logger.warning(
            "sliding_spectral_moments: the bandwidth trend is not positive everywhere; "
            "the relative sedimentation rate is left undefined"
        )
        return None
    ratio = REFERENCE_SEDIMENTATION_RATE * bandwidth[0] / trend
    return ratio - float(np.mean(ratio)) + REFERENCE_SEDIMENTATION_RATE


def sliding_spectral_moments(
    time: npt.NDArray,
    values: npt.NDArray,
    window: int | float | None = None,
    step: int | None = None,
    *,
    window_duration: float | None = None,
    step_duration: float | None = None,
    n_windows: int | None = None,
    overlap: float = 0.5,
    frequency_range: tuple[float, float] | None = None,
    n_frequencies: int = DEFAULT_SLIDING_FREQUENCIES,
    detrend: bool = True,
) -> SlidingSpectralMoments:
    """
    滑窗谱矩：逐窗计算一次谱，再计算一次谱矩。

    谱一律走 :mod:`stratigraphy.spectral_analysis` 的 Lomb-Scargle 路线
    （:class:`~stratigraphy.spectral_analysis.SpectralAnalyzer`），因此与仓库里
    其余频谱分析共用同一套频率轴口径（每单位坐标的循环频率）与同一个零频/
    去趋势约定。本函数**不复制**那条实现，只是在每个窗口上调用它。

    **窗口长度**：``window`` 传整数时是**点数**，传浮点数时是**时长**（记录坐标
    单位）；``window_duration`` 是时长的显式写法。

    **窗口个数**：给了 ``n_windows`` 时忽略 ``window``/``step``，窗长与步长由
    ``n_windows`` 与 ``overlap`` 反解（``n_windows`` 个窗恰好铺满记录），这与
    仓库其它滑窗模块（:mod:`stratigraphy.evolutionary`）的约定一致。

    **步长**：``step`` 以点数给出（缺省 1，即逐点滑动）；``step_duration`` 以
    时间单位给出。两者同样互斥。

    **窗口中心**：窗口首、末坐标的算术平均（与 Acycle 的
    ``acycleSpectralMoments.m`` 同一约定）。

    Parameters:
        time: 记录坐标轴（严格递增）。接受 ``numpy`` 数组；传入序列（列表、
            元组等）会在内部转换为一维浮点数组。
        values: 记录数值。接受 ``numpy`` 数组；传入序列会在内部转换。
        window: 窗口点数（整数）或窗口时长（浮点数）。
        step: 相邻窗口起点之间的点数，缺省 1。
        window_duration: 窗口的时长，与 ``window`` 互斥。
        step_duration: 步长的时长，与 ``step`` 互斥。
        n_windows: 窗数（至少 2）；给了它就由 ``overlap`` 反解窗长与步长。
        overlap: 相邻窗的重叠比例，取值 ``[0, 1)``。
        frequency_range: ``(最低频率, 最高频率)``，单位是每单位记录坐标的循环
            频率；缺省时由每个窗口自身的分辨率决定。
        n_frequencies: 频率栅格的点数预算，缺省 200。
        detrend: 是否在每个窗口内先去掉线性趋势（缺省 ``True``，与
            :class:`~stratigraphy.spectral_analysis.SpectralAnalyzer` 一致）。

    Returns:
        SlidingSpectralMoments: 逐窗的谱矩、谱形指标、窗口几何，以及由带宽
        趋势反推的相对沉积速率。

    Raises:
        ValidationError: 输入长度不一致、点数太少、时间轴非严格递增、窗口长度
            非法（点数太少或超过记录长度）、步长非法，或某个窗口里没有可分析的
            信号（例如窗口内是常数）。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.sliding_spectral_moments import sliding_spectral_moments
        >>> axis = np.linspace(0.0, 200.0, 1000)
        >>> signal = np.sin(2.0 * np.pi * axis / 20.0)
        >>> result = sliding_spectral_moments(axis, signal, window=200, step=100)
        >>> bool(result.mean_period.size > 0)
        True
    """
    times, series = _as_record(time, values)
    overlap_fraction = float(overlap)
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValidationError(
            _("overlap must lie in [0, 1); got {0}").format(overlap),
            details={"overlap": overlap},
        )

    if n_windows is not None:
        windows_wanted = int(n_windows)
        if windows_wanted != n_windows or windows_wanted < 2:
            raise ValidationError(
                _("n_windows must be an integer of at least 2"),
                details={"n_windows": n_windows},
            )
        window_points, step_points = _even_window_layout(times.size, windows_wanted, overlap_fraction)
        if window_points < MIN_WINDOW_POINTS:
            raise ValidationError(
                _("a window needs at least {0} points; got {1}").format(MIN_WINDOW_POINTS, window_points),
                details={"window": window_points, "minimum": MIN_WINDOW_POINTS},
            )
        if window_points > times.size:
            raise ValidationError(
                _("the window is longer than the record"),
                details={"window": window_points, "n_points": int(times.size)},
            )
    else:
        window_points = _resolve_window_points(times, window, window_duration)
        step_points = _resolve_step_points(times, step, step_duration)

    budget = int(n_frequencies)
    if budget != n_frequencies or budget < 2:
        raise ValidationError(
            _("n_frequencies must be an integer of at least 2"),
            details={"n_frequencies": n_frequencies},
        )

    analyzer = SpectralAnalyzer()
    centers: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    m0_values: list[float] = []
    m1_values: list[float] = []
    m2_values: list[float] = []
    m4_values: list[float] = []
    mean_periods: list[float] = []
    central_frequencies: list[float] = []
    bandwidths: list[float] = []
    relative_bandwidths: list[float] = []
    peak_periods: list[float] = []

    n_starts = times.size - window_points + 1
    if n_windows is not None:
        # 恰好 n_windows 个窗：起点等距铺开，最后一个窗被夹到记录内部。
        window_starts = [min(index * step_points, times.size - window_points) for index in range(n_windows)]
    else:
        window_starts = list(range(0, n_starts, step_points))
    for start in window_starts:
        stop = start + window_points
        window_time = times[start:stop]
        window_values = series[start:stop]
        try:
            spectrum = analyzer.analyze(
                window_time,
                window_values,
                frequency_range=frequency_range,
                n_frequencies=budget,
                detrend=detrend,
                ar1_significance=False,
            )
        except ComputationError as exc:
            raise ValidationError(
                _("a sliding window carries no analysable signal"),
                details={"window_start": float(window_time[0]), "window_end": float(window_time[-1])},
                original_exception=exc,
            ) from exc

        moments = spectral_moments(spectrum.frequencies, spectrum.power)
        centers.append(0.5 * (float(window_time[0]) + float(window_time[-1])))
        starts.append(float(window_time[0]))
        ends.append(float(window_time[-1]))
        m0_values.append(moments.m0)
        m1_values.append(moments.m1)
        m2_values.append(moments.m2)
        m4_values.append(moments.m4)
        mean_periods.append(moments.mean_period)
        central_frequencies.append(moments.central_frequency)
        bandwidths.append(moments.bandwidth)
        relative_bandwidths.append(moments.relative_bandwidth)
        peak_periods.append(float(spectrum.peak_period) if spectrum.peak_period is not None else float("nan"))

    center_array = np.asarray(centers, dtype=float)
    bandwidth_array = np.asarray(bandwidths, dtype=float)
    sedimentation_rate = _relative_sedimentation_rate(center_array, bandwidth_array)

    logger.info(
        "Sliding spectral moments complete: %d windows of %d points, step %d",
        center_array.size,
        window_points,
        step_points,
    )
    return SlidingSpectralMoments(
        window_centers=center_array,
        window_starts=np.asarray(starts, dtype=float),
        window_ends=np.asarray(ends, dtype=float),
        m0=np.asarray(m0_values, dtype=float),
        m1=np.asarray(m1_values, dtype=float),
        m2=np.asarray(m2_values, dtype=float),
        m4=np.asarray(m4_values, dtype=float),
        mean_period=np.asarray(mean_periods, dtype=float),
        central_frequency=np.asarray(central_frequencies, dtype=float),
        bandwidth=bandwidth_array,
        relative_bandwidth=np.asarray(relative_bandwidths, dtype=float),
        peak_period=np.asarray(peak_periods, dtype=float),
        sedimentation_rate=sedimentation_rate,
        window_points=window_points,
        step_points=step_points,
    )


# =============================================================================
# Sedimentation rate from the spectrum
# =============================================================================


def _box_filter_gain(x: float) -> float:
    """箱式（平均）滤波的传递函数 ``|sin(pi x) / (pi x)|``。"""
    if abs(x) < 1e-12:
        return 1.0
    return float(abs(np.sin(np.pi * x) / (np.pi * x)))


def _solve_box_filter_argument(gain: float) -> float:
    """由箱式滤波增益反解 ``x``（``gain = |sin(pi x)/(pi x)|``，``x ∈ [0, 1]``）。

    传递函数在 ``[0, 1]`` 上单调下降，所以二分法收敛到唯一解。
    """
    low, high = 0.0, 1.0
    for iteration in range(AMPLITUDE_SOLVER_ITERATIONS):
        middle = 0.5 * (low + high)
        if _box_filter_gain(middle) > gain:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sedimentation_rate_from_spectral(
    time: npt.NDArray,
    values: npt.NDArray,
    target_periods: npt.NDArray | None = None,
    *,
    frequency_range: tuple[float, float] | None = None,
    n_frequencies: int = DEFAULT_SPECTRAL_FREQUENCIES,
    detrend: bool = True,
    forcing_amplitude: float | None = None,
) -> SedimentationRateEstimate:
    """
    由谱反演沉积速率（Harker et al. 2013）。

    **记录坐标**：本函数把 ``time`` 当作**记录坐标轴**（地层学里就是深度），
    ``target_periods`` 当作**时间单位**的轨道周期（如 kyr）。返回的沉积速率单位
    是“记录坐标/时间”（如 m/kyr）。

    **关系式**：由周期 ``T_orb`` 强迫的记录，在沉积速率 ``S`` 下每周期留下的
    厚度是 ``λ = S · T_orb``，因此

        S = λ̂ / T_orb ,   λ̂ = M0 / M1

    ``λ̂`` 是整条记录的**平均波长**（谱矩给出的谱质心波长）。不同目标周期给出
    同一条直线上不同的点（``S ∝ 1/T_orb``）：目标周期是使用者对强迫来源的假设，
    而不是数据能单独确定的东西。

    **振幅关系**：线性代理解释下恒定速率只把时间轴拉长，周期振幅本身不含速率
    信息；但真实样品代表一段深度区间 ``Δz``（即一段时间 ``Δt = Δz / S``），
    样品值是这段时间内强迫的平均，于是观测振幅被箱式滤波衰减

        A_obs = A_forcing · |sin(pi x) / (pi x)| ,  x = Δz / (S · T_orb) = Δz / λ̂

    给出 ``forcing_amplitude``（时间域强迫振幅）时，本函数用这条关系反解出平均
    波长并给出**第二条**沉积速率估计（``amplitude_rates``）；未给出时该列为
    ``nan``。观测振幅取 ``sqrt(2) · sd``（去趋势后的标准差）：记录被**一个**
    周期主导时它就是该周期的振幅；记录里混有多个成分时它偏大，因此这条估计
    只在“目标周期支配整条记录”时可用。

    **不确定度**：记录里的每个周期贡献近似相等的能量，沉积速率在整条记录上
    的平均值因此带有抽样误差。记相对谱宽 ``ρ = σ_f / f_c``（谱峰展宽的相对量，
    也就是局部速率的相对散布）、记录中的周期数 ``N = L / λ̂``，则

        σ_S = S · ρ / sqrt(N)

    这是一个**近似**（把记录切成 N 段准独立速率的标准误），不是严格的置信区间。

    **非线性偏差（Kidder & Ledger 2018）**：当沉积速率随时间变化时，谱矩给出
    的平均波长对应的是局部速率的**调和平均**，因此上面的反演在变速记录上会
    系统性地**低估**平均速率；需要绝对年龄时应改用
    :func:`sedimentation_rate_age_model` 积分 ``dz / S(z)``。

    Parameters:
        time: 记录坐标轴（严格递增）。接受 ``numpy`` 数组；传入序列（列表、
            元组等）会在内部转换为一维浮点数组。
        values: 记录数值。接受 ``numpy`` 数组；传入序列会在内部转换。
        target_periods: 目标轨道周期；缺省为 ``(19, 21, 23, 41)`` kyr。接受
            ``numpy`` 数组；传入序列（列表、元组等）会在内部转换。
        frequency_range: ``(最低频率, 最高频率)``，单位是每单位记录坐标的循环
            频率。
        n_frequencies: 频率栅格的点数预算，缺省 500。
        detrend: 是否先去掉线性趋势，缺省 ``True``。
        forcing_amplitude: 时间域强迫的振幅（可选项，用于振幅反演）。

    Returns:
        SedimentationRateEstimate: 每个目标周期对应的沉积速率、不确定度、观测
        振幅，以及整条记录的谱矩。

    Raises:
        ValidationError: 输入非法（长度不一致、点数太少、时间轴非严格递增、
            目标周期非正或非有限）或记录里没有可分析的信号。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.sliding_spectral_moments import sedimentation_rate_from_spectral
        >>> depth = np.linspace(0.0, 100.0, 2000)
        >>> signal = np.cos(2.0 * np.pi * depth / 2.0)
        >>> result = sedimentation_rate_from_spectral(depth, signal, [20.0])
        >>> bool(abs(result.sedimentation_rates[0] - 0.1) < 0.01)
        True
    """
    times, series = _as_record(time, values)

    if target_periods is None:
        periods = np.asarray(DEFAULT_TARGET_PERIODS, dtype=float)
    else:
        periods = _as_vector(target_periods, "target_periods", minimum_size=1)
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValidationError(
            _("target_periods must be finite and positive"),
            details={"target_periods": periods.tolist()},
        )

    budget = int(n_frequencies)
    if budget != n_frequencies or budget < 2:
        raise ValidationError(
            _("n_frequencies must be an integer of at least 2"),
            details={"n_frequencies": n_frequencies},
        )

    analyzer = SpectralAnalyzer()
    try:
        spectrum = analyzer.analyze(
            times,
            series,
            frequency_range=frequency_range,
            n_frequencies=budget,
            detrend=detrend,
            ar1_significance=False,
        )
    except ComputationError as exc:
        raise ValidationError(
            _("the record carries no analysable signal"),
            details={"n_points": int(times.size)},
            original_exception=exc,
        ) from exc

    moments = spectral_moments(spectrum.frequencies, spectrum.power)
    mean_period = moments.mean_period
    record_length = float(times[-1] - times[0])
    n_cycles = record_length / mean_period if mean_period > 0.0 else float("nan")

    rates = mean_period / periods
    if np.isfinite(n_cycles) and n_cycles > 0.0:
        uncertainties = np.abs(rates) * moments.relative_bandwidth / np.sqrt(n_cycles)
    else:
        uncertainties = np.full(rates.shape, np.nan)

    # 观测振幅：记录被一个周期主导时，它的振幅就是 ``sqrt(2) * sd``（正弦的
    # 方差是振幅平方的一半）。这个估计量比 Lomb-Scargle 的峰值功率稳得多：
    # 后者在采样稀疏时会被接近退化的基函数放大好几个数量级。
    detrended = np.asarray(detrend_series(times, series, method="linear").detrended, dtype=float)
    signal_sd = float(np.std(detrended))
    amplitude = signal_sd * float(np.sqrt(2.0))

    if forcing_amplitude is None:
        amplitude_rates = np.full(rates.shape, np.nan)
    else:
        forcing = float(forcing_amplitude)
        if not np.isfinite(forcing) or forcing <= 0.0:
            raise ValidationError(
                _("forcing_amplitude must be a positive number"),
                details={"forcing_amplitude": forcing_amplitude},
            )
        gain = amplitude / forcing if forcing > 0.0 else 0.0
        if gain > 1.0 or gain <= 0.0:
            # 观测振幅不可能超过强迫振幅；超出说明模型或数据不自洽。
            amplitude_rates = np.full(rates.shape, np.nan)
        else:
            argument = _solve_box_filter_argument(gain)
            sample_spacing = float(np.median(np.diff(times)))
            if argument <= 0.0:
                amplitude_period = float("inf")
            else:
                amplitude_period = sample_spacing / argument
            amplitude_rates = amplitude_period / periods

    logger.info(
        "Sedimentation rate from spectrum: mean period %.6g over %.4g cycles, %d target periods",
        mean_period,
        n_cycles,
        int(periods.size),
    )
    return SedimentationRateEstimate(
        target_periods=periods,
        sedimentation_rates=rates,
        uncertainties=uncertainties,
        observed_amplitudes=np.full(rates.shape, amplitude),
        amplitude_rates=amplitude_rates,
        moments=moments,
        record_length=record_length,
        n_cycles=float(n_cycles),
    )


# =============================================================================
# Age model
# =============================================================================


def sedimentation_rate_age_model(
    depth: npt.NDArray,
    sedimentation_rate: npt.NDArray,
    *,
    age_start: float = 0.0,
    model: str = "auto",
    rate_gradient: float | None = None,
) -> SedimentationRateAgeModel:
    """
    由沉积速率建立年龄-深度模型。

    三种速率模型：

    * ``constant``（恒定速率，深度—年龄线性）::

          age(z) = age_start + (z - z_0) / S

    * ``linear``（速率随深度线性变化 ``S(z) = S_0 + g (z - z_0)``）::

          age(z) = age_start + ln(S(z) / S_0) / g          (g != 0)
          age(z) = age_start + (z - z_0) / S_0              (g == 0)

      这是**非线性**的深度—年龄映射（Kidder & Ledger 2018）：沉积速率变化时，
      线性假设会系统性地给出错误的年龄。

    * ``piecewise_constant``（逐点速率，左端点积分）::

          age_i = age_{i-1} + (z_i - z_{i-1}) / S_i

      与 Acycle 的 ``acycleSedimentationRateAgeModel`` 同一口径：区间用**左端点**
      的速率，最后一行的速率只被校验、不参与积分。

    ``model="auto"`` 时按输入形状判断：标量速率 → ``constant``；与 ``depth``
    等长的数组 → ``piecewise_constant``；其它长度一律报错并要求显式指定。

    Parameters:
        depth: 深度轴，严格单调（递增或递减均可），至少 2 个点。接受 ``numpy``
            数组；传入序列（列表、元组等）会在内部转换为一维浮点数组。
        sedimentation_rate: 恒定速率（标量）或逐点速率（数组）。接受 ``numpy``
            数组；传入序列（列表、元组等）或标量会在内部转换（标量变成 0 维
            数组，按恒定速率处理）。
        age_start: 第一个深度点的年龄，缺省 0。
        model: ``auto`` / ``constant`` / ``linear`` / ``piecewise_constant``。
        rate_gradient: ``linear`` 模型里的速率梯度 ``dS/dz``；缺省 0。

    Returns:
        SedimentationRateAgeModel: 深度、年龄、逐点速率与所用模型。

    Raises:
        ValidationError: 深度轴非单调或点数太少、速率非正、模型名未知、
            速率数组长度与深度轴不符。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.sliding_spectral_moments import sedimentation_rate_age_model
        >>> depths = np.linspace(0.0, 10.0, 11)
        >>> model = sedimentation_rate_age_model(depths, 0.5)
        >>> bool(abs(model.age[-1] - 20.0) < 1e-12)
        True
    """
    depths = _as_vector(depth, "depth", minimum_size=2)
    spacing = np.diff(depths)
    if np.all(spacing > 0.0):
        increasing = True
    elif np.all(spacing < 0.0):
        increasing = False
    else:
        raise ValidationError(
            _("depth must be strictly monotonic"),
            details={"n_points": int(depths.size)},
        )

    try:
        rate_array = np.asarray(sedimentation_rate, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("sedimentation_rate must be numeric"),
            details={"parameter": "sedimentation_rate"},
            original_exception=exc,
        ) from exc

    resolved = str(model).lower()
    if resolved not in {"auto", "constant", "linear", "piecewise_constant"}:
        raise ValidationError(
            _("model must be auto, constant, linear or piecewise_constant"),
            details={"model": model},
        )

    if resolved == "auto":
        if rate_array.ndim == 0:
            resolved = "constant"
        elif rate_array.size == depths.size:
            resolved = "piecewise_constant"
        else:
            raise ValidationError(
                _("the rate array does not match the depth axis; give model explicitly"),
                details={"rate_length": int(rate_array.size), "depth_length": int(depths.size)},
            )

    if resolved == "piecewise_constant":
        rates = _as_vector(rate_array, "sedimentation_rate", minimum_size=2)
        if rates.size != depths.size:
            raise ValidationError(
                _("the rate array must have one entry per depth point"),
                details={"rate_length": int(rates.size), "depth_length": int(depths.size)},
            )
        if np.any(rates <= 0.0):
            raise ValidationError(
                _("every sedimentation rate must be positive"),
                details={"minimum": float(rates.min())},
            )
        working_depth = depths if increasing else depths[::-1]
        working_rates = rates if increasing else rates[::-1]
        increments = np.diff(working_depth) / working_rates[:-1]
        ages = np.concatenate(([float(age_start)], float(age_start) + np.cumsum(increments)))
        if not increasing:
            ages = ages[::-1]
            working_depth = working_depth[::-1]
            working_rates = working_rates[::-1]
        return SedimentationRateAgeModel(
            depth=depths.copy(),
            age=ages,
            sedimentation_rate=working_rates.copy(),
            model=resolved,
        )

    if rate_array.size != 1:
        raise ValidationError(
            _("a constant or linear rate model needs a single rate value"),
            details={"rate_length": int(rate_array.size)},
        )
    base_rate = float(rate_array.reshape(-1)[0])
    if not np.isfinite(base_rate) or base_rate <= 0.0:
        raise ValidationError(
            _("the sedimentation rate must be a positive number"),
            details={"sedimentation_rate": base_rate},
        )

    gradient = 0.0 if rate_gradient is None else float(rate_gradient)
    if not np.isfinite(gradient):
        raise ValidationError(
            _("rate_gradient must be a finite number"),
            details={"rate_gradient": rate_gradient},
        )

    working_depth = depths if increasing else depths[::-1]
    offset = working_depth - working_depth[0]
    if resolved == "constant":
        gradient = 0.0
    rates = base_rate + gradient * offset
    if np.any(rates <= 0.0):
        raise ValidationError(
            _("the sedimentation rate must stay positive over the whole depth range"),
            details={"minimum": float(rates.min())},
        )

    if gradient == 0.0:
        ages = float(age_start) + offset / base_rate
    else:
        ages = float(age_start) + np.log(rates / base_rate) / gradient

    if not increasing:
        ages = ages[::-1]
        rates = rates[::-1]
    return SedimentationRateAgeModel(
        depth=depths.copy(),
        age=ages,
        sedimentation_rate=rates,
        model=resolved,
    )
