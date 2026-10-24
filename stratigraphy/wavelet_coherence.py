# =============================================================================
# FILE: stratigraphy/wavelet_coherence.py
# =============================================================================
"""
小波相干分析族：交叉小波变换（XWT）、小波相干（WTC）与小波显著性。

用途
====
判断两条地层记录（例如 δ¹⁸O 与 δ¹³C、粒度与粉砂含量、两条测年曲线）是不是
被**同一个外部强迫**（通常是天文强迫）驱动。本模块提供四个层次的分析：

* :func:`wavelet_transform` —— **连续小波变换（CWT）**，给出时频平面上的复
  小波系数；
* :func:`cross_wavelet_transform` —— **交叉小波变换（XWT）**，
  ``W_xy = W_x * conj(W_y)``，以及逐尺度相位；
* :func:`wavelet_coherence` —— **小波相干（WTC）**，给出幅度平方相干性
  ``R²``、相位差与 95% 显著性临界值曲线；
* :func:`wavelet_significance` / :func:`wavelet_coherence_significance` ——
  以 **AR(1) 蒙特卡洛背景**给出单序列谱与 WTC 的临界值。

估计量的构造
============
1. **小波基。** 两种母小波，均按 Torrence & Compo (1998) 的定义在**傅里叶
   空间**离散化（与该文的 ``wave_bases.m`` 同一口径）：

   * Morlet（``w0 = 6``）：``psi(eta) = pi^-0.4 * exp(i w0 eta) * exp(-eta^2/2)``，
     其幅度谱 ``exp(-(s k - w0)^2 / 2)`` 在 ``s k = w0`` 处取峰值；
   * Mexican Hat（DOG m=2，二阶高斯导数）：``psi(eta) = (1 - eta^2) exp(-eta^2/2)``，
     幅度谱正比于 ``(s k)^2 exp(-(s k)^2 / 2)``。

   小波系数的归一化与 Fourier 因子**必须成对使用**：本模块采用
   Torrence & Compo (1998) 的"总能量正比于尺度"归一化
   （``norm = sqrt(s * k1) * pi^-0.25 * sqrt(N)``），并配以同一文献表 1 的
   Fourier 因子。两者是自洽的：**响应峰值所在尺度会被还原成真实周期**
   （见 ``tests/golden/test_wavelet_coherence.py`` 的周期映射断言）。

2. **尺度 → 频率/周期的映射（Fourier 因子）。** 尺度以**样本**为单位给出，
   频率轴按 Fourier 因子换算：

   * Morlet：``Fourier 因子 = 4 * pi / (w0 + sqrt(2 + w0^2))``；
   * Mexican Hat：``Fourier 因子 = 2 * pi * sqrt(2 / (2 m + 1))``。

   写成任务书要求的形式即 ``freq = wn / scale``、``周期 = 1 / freq``，其中

   * ``wn = 1 / Fourier 因子``（Morlet 为 ``(w0 + sqrt(2 + w0^2)) / (4 pi)``
     ≈ 0.9680 周期/单位尺度；Mexican Hat 为 ``sqrt(2 m + 1) / (2 pi)``
     ≈ 0.2517 周期/单位尺度）；
   * ``freq`` 以**每样本循环频率**给出；乘以 ``1/dt`` 得到每单位时间的循环
     频率，``周期 = 1 / freq``（时间单位）。

   频率轴与周期轴都是**每单位时间**的口径（与
   :mod:`stratigraphy.coherence` 一致），因此周期轴可以直接与地层记录的
   时间单位（年、ka、米……）对照。

3. **相干性的定义。** 先在**时间维**与**尺度维**做平滑，再由平滑后的谱给出

       S_xx = smooth(|W_x|^2 / s),  S_yy = smooth(|W_y|^2 / s)
       S_xy = smooth(W_x * conj(W_y) / s)
       R^2  = |S_xy|^2 / (S_xx * S_yy)

   除以尺度 ``s`` 是 Grinsted et al. (2004) 的做法：它把各尺度的白噪声本底
   拉平，使相干性在不同尺度之间可比。

4. **平滑的口径与系数（本模块采用的具体约定）。**

   * **时间维**：高斯窗，窗的标准差 = ``0.6 × 尺度``（**以样本计**），
     即窗宽与尺度成正比（Torrence & Compo 2008 / Grinsted et al. 2004 的
     "时间维按尺度比例平滑"）。本模块取系数 0.6；Grinsted et al. (2004) 的
     实现用系数 1.0，差别只影响相干性曲线的锐度，不影响 ``R²`` 的定义。
   * **尺度维**：高斯窗，作用在 ``log2(尺度)`` 坐标上，标准差
     ``sigma = 0.6 / 2 = 0.3`` 个倍频程（"0.6"取为窗口在倍频程上的全宽，
     高斯取其半宽）。Grinsted et al. (2004) 用的是 0.6 八度的**箱式**窗，
     本模块用等宽的高斯窗替代，两者都只做尺度维的平滑，结论对平滑核的形状
     不敏感。

5. **相位约定（重要）。** 交叉小波一律取 ``W_xy = W_x * conj(W_y)``（与
   Grinsted et al. 2004 与 Acycle 的 ``wcoherence`` 一致），因此

       phase = arg(W_xy) = 第一条序列的相位 − 第二条序列的相位

   * 第二条序列相对第一条**滞后** Δt（``y(t) = x(t - Δt)``）时，
     ``phase = +2 * pi * f * Δt``；
   * 第二条序列相对第一条**提前** Δt（``y(t) = x(t + Δt)``）时，
     ``phase = -2 * pi * f * Δt``。

   相干性极低（低于 ``phase_threshold``，缺省 0.5）时相位没有意义，本模块
   返回 ``NaN`` 并在结果里保留该阈值。

**怎么读 WTC 的峰值周期**：``R²`` 是比值。共享信号很强时它会在信号尺度
**附近饱和**（``R² → 1``），实测在真实周期 ±10% 的范围内只差 0.003，
因此"时间平均 R² 曲线"的极大值**不是**一个良态的周期估计量（换一个噪声
实现，极大值可以移动 10%）。要找共享周期，应读**交叉小波功率**
``|W_xy|``（``WaveletCoherenceResult.cross_wavelet``）或**小波谱**
（``WaveletTransformResult.coefficients``）的峰：两者都在真实周期的 1%
以内（实测 0.3%）。

时间轴
======
本模块只支持**均匀**时间轴：给出 ``time`` 时要求严格递增**且**等间距
（否则抛 ``ValidationError``）；只给 ``dt`` 时按等间距处理；两者都不给时
按 ``dt = 1``（样本即时间单位）处理。地层记录常见的**不等间距**采样需要
先重采样，本模块不做插值（插值会改变噪声的统计性质，显著性检验随之失真）。

显著性
======
* :func:`wavelet_significance` 与 :func:`wavelet_coherence_significance`
  都用 **蒙特卡洛**：按观测序列的 **AR(1)** 系数（朴素估计量
  ``rho = sum x[n] x[n-1] / sum x[n-1]^2``）生成替代序列，重复 ``N`` 次，
  取 ``level``（缺省 0.95）分位数作为临界值。
* 单序列显著性比较的是**时间平均谱**（只在锥形影响域 COI 之外的点上平均）。
* WTC 显著性按 Grinsted et al. (2004) §3：生成**两条独立**的 AR(1) 序列，
  计算它们的 WTC，把每个尺度上所有有效时间点的 ``R²`` 汇总后取 95% 分位。
* 所有蒙特卡洛函数都接受 ``seed`` 参数，**同种子逐位可复现**。

参考文献
========
* Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross
  wavelet transform and wavelet coherence to geophysical time series.
  *Nonlinear Processes in Geophysics*, 11(5), 561-566.（XWT/WTC 的定义、
  相位约定与蒙特卡洛显著性）
* Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis.
  *Bulletin of the American Meteorological Society*, 79(1), 61-78.（母小波
  定义、Fourier 因子、锥形影响域）
* Torrence, C. & Compo, G. P. (2008). Interdecadal changes in the ENSO-Monsoon
  system. *Journal of Climate*, 21(11), 2448-2462.（时间维与尺度维的平滑）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "wavelet coherence" 子功能）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（互谱取 ``W_x * conj(W_y)``、相位箭头语义、AR(1) 蒙特卡洛临界值），未翻译
或复制其 MATLAB 源码；算法本身按上述已发表论文的定义自行实现。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from config.i18n import _
from stratigraphy.detrending import detrend_series
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: Morlet 母小波的中央角频率 w0（Torrence & Compo 1998）。
MORLET_OMEGA0 = 6.0

#: Mexican Hat（DOG）的阶数 m。
MEXICAN_HAT_ORDER = 2

#: 时间维平滑的高斯窗系数：窗的标准差 = 本系数 × 尺度（以样本计）。
TIME_SMOOTHING_FACTOR = 0.6

#: 尺度维平滑的高斯窗系数（单位：倍频程，即 log2(尺度) 坐标上的全宽）。
SCALE_SMOOTHING_OCTAVES = 0.6

#: 相干性低于该阈值时，相位返回 NaN（相位在该处没有意义）。
PHASE_COHERENCE_THRESHOLD = 0.5

#: 允许的最少样本数。
MIN_SAMPLES = 16

#: 允许的最少尺度数。
MIN_SCALES = 2

#: 蒙特卡洛重复次数的缺省值。
DEFAULT_MONTE_CARLO = 200

#: 显著性水平的缺省值。
DEFAULT_LEVEL = 0.95

#: 小波谱平滑后用于相干性分母的下限保护（相对于谱自身的量级）。
COHERENCE_FLOOR = 1.0e-300


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class WaveletTransformResult:
    """
    连续小波变换的结果。

    Attributes:
        coefficients: 复小波系数，形状 ``(尺度数, 时间点数)``。
        scales: 尺度轴（**以样本计**），严格递增。
        periods: 周期轴（每单位时间），与 ``scales`` 一一对应。
        frequencies: 频率轴（每单位时间的循环频率），``= 1 / periods``。
        times: 时间轴（每单位时间）。
        cone_of_influence: 每个时间点上的**有效最大周期**（时间单位）；
            周期超过该值的点受边界效应影响。
        dt: 采样间隔（每单位时间）。
    """

    coefficients: npt.NDArray
    scales: npt.NDArray
    periods: npt.NDArray
    frequencies: npt.NDArray
    times: npt.NDArray
    cone_of_influence: npt.NDArray
    dt: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Wavelet transform"),
            "=" * 48,
            _("Scales: {0}").format(self.scales.size),
            _("Period range: {0} to {1}").format(f"{self.periods.min():.6g}", f"{self.periods.max():.6g}"),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CrossWaveletResult:
    """
    交叉小波变换（XWT）的结果。

    Attributes:
        cross_wavelet: 复交叉小波矩阵 ``W_x * conj(W_y)``，形状
            ``(尺度数, 时间点数)``。
        phase: 逐尺度相位（弧度），``arg(W_x * conj(W_y))``，即
            **第一条序列的相位减去第二条序列的相位**。
        scales: 尺度轴（以样本计）。
        periods: 周期轴（每单位时间）。
        frequencies: 频率轴（每单位时间的循环频率）。
        times: 时间轴（每单位时间）。
        cone_of_influence: 每个时间点上的有效最大周期（时间单位）。
        dt: 采样间隔（每单位时间）。
    """

    cross_wavelet: npt.NDArray
    phase: npt.NDArray
    scales: npt.NDArray
    periods: npt.NDArray
    frequencies: npt.NDArray
    times: npt.NDArray
    cone_of_influence: npt.NDArray
    dt: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Cross wavelet transform"),
            "=" * 48,
            _("Scales: {0}").format(self.scales.size),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class WaveletCoherenceResult:
    """
    小波相干（WTC）的结果。

    Attributes:
        coherence: 幅度平方相干性 ``R²``，取值 ``0..1``。
        phase: 相位差（弧度），``arg(W_x * conj(W_y))``。相干性低于
            ``phase_threshold`` 的点上为 ``NaN``（相位在那里没有意义）。
        significance: 95% 显著性临界值曲线（每个尺度一个值），未计算时为
            ``None``。
        cross_wavelet: 未平滑的复交叉小波矩阵 ``W_x * conj(W_y)``。
        scales: 尺度轴（以样本计）。
        periods: 周期轴（每单位时间）。
        frequencies: 频率轴（每单位时间的循环频率）。
        times: 时间轴（每单位时间）。
        cone_of_influence: 每个时间点上的有效最大周期（时间单位）。
        phase_threshold: 相位被置为 ``NaN`` 的相干性阈值。
        dt: 采样间隔（每单位时间）。
    """

    coherence: npt.NDArray
    phase: npt.NDArray
    significance: npt.NDArray | None
    cross_wavelet: npt.NDArray
    scales: npt.NDArray
    periods: npt.NDArray
    frequencies: npt.NDArray
    times: npt.NDArray
    cone_of_influence: npt.NDArray
    phase_threshold: float
    dt: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Wavelet coherence"),
            "=" * 48,
            _("Scales: {0}").format(self.scales.size),
        ]
        if self.coherence.size:
            # 相干性是 ``(尺度数, 时间点数)`` 的矩阵：``argmax`` 给出的是**展平
            # 后**的下标，直接拿它去索引周期轴（长度是尺度数）会越界。展平下标
            # 除以时间点数才回到尺度下标。
            flat = int(np.argmax(self.coherence))
            scale_index = flat // self.coherence.shape[1]
            lines.append(
                _("Peak coherence: {0} at period {1}").format(
                    f"{self.coherence.ravel()[flat]:.4f}", f"{self.periods[scale_index]:.6g}"
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class WaveletSignificanceResult:
    """
    单序列小波显著性的结果。

    Attributes:
        spectrum: 观测序列的**时间平均小波谱**（每个尺度一个值），只在锥形
            影响域之外的点上平均。
        critical_values: 蒙特卡洛临界值（每个尺度一个值），即 ``level``
            分位数。
        ar1_coefficient: 观测序列的 AR(1) 系数（背景模型）。
        n_monte_carlo: 蒙特卡洛重复次数。
        seed: 随机种子（用于复现）。
        scales: 尺度轴（以样本计）。
        periods: 周期轴（每单位时间）。
        frequencies: 频率轴（每单位时间的循环频率）。
    """

    spectrum: npt.NDArray
    critical_values: npt.NDArray
    ar1_coefficient: float
    n_monte_carlo: int
    seed: int
    scales: npt.NDArray
    periods: npt.NDArray
    frequencies: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Wavelet significance"),
            "=" * 48,
            _("Scales: {0}").format(self.scales.size),
            _("Monte Carlo repetitions: {0}").format(self.n_monte_carlo),
            _("Seed: {0}").format(self.seed),
            _("AR(1) coefficient: {0}").format(f"{self.ar1_coefficient:.6g}"),
        ]
        if self.critical_values.size:
            lines.append(
                _("Critical values: {0}").format(
                    f"{float(np.min(self.critical_values)):.6g} to {float(np.max(self.critical_values)):.6g}"
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class WaveletCoherenceSignificanceResult:
    """
    WTC 蒙特卡洛显著性的结果。

    Attributes:
        critical_values: 临界值曲线（每个尺度一个值），即 AR(1) 背景下
            ``R²`` 的 ``level`` 分位数。
        ar1_coefficients: 两条观测序列的 AR(1) 系数。
        n_monte_carlo: 蒙特卡洛重复次数。
        seed: 随机种子（用于复现）。
        scales: 尺度轴（以样本计）。
        periods: 周期轴（每单位时间）。
        frequencies: 频率轴（每单位时间的循环频率）。
    """

    critical_values: npt.NDArray
    ar1_coefficients: tuple[float, float]
    n_monte_carlo: int
    seed: int
    scales: npt.NDArray
    periods: npt.NDArray
    frequencies: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Wavelet coherence significance"),
            "=" * 48,
            _("Scales: {0}").format(self.scales.size),
            _("Monte Carlo repetitions: {0}").format(self.n_monte_carlo),
            _("Seed: {0}").format(self.seed),
            _("AR(1) coefficients: {0}").format(
                f"{float(self.ar1_coefficients[0]):.6g}, {float(self.ar1_coefficients[1]):.6g}"
            ),
        ]
        if self.critical_values.size:
            lines.append(
                _("Critical values: {0}").format(
                    f"{float(np.min(self.critical_values)):.6g} to {float(np.max(self.critical_values)):.6g}"
                )
            )
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


def _validate_scales(scales: object) -> npt.NDArray:
    """检查尺度轴：正数、有限、严格递增。"""
    try:
        values = np.asarray(scales, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("scales must be numeric"),
            details={"parameter": "scales"},
            original_exception=exc,
        ) from exc
    if values.ndim != 1:
        raise ValidationError(
            _("scales must be a one-dimensional array"),
            details={"parameter": "scales", "ndim": int(values.ndim)},
        )
    if values.size < MIN_SCALES:
        raise ValidationError(
            _("scales needs at least {0} entries; got {1}").format(MIN_SCALES, int(values.size)),
            details={"parameter": "scales", "length": int(values.size), "minimum": MIN_SCALES},
        )
    if not np.all(np.isfinite(values)):
        raise ValidationError(
            _("scales must contain only finite values"),
            details={"parameter": "scales"},
        )
    if np.any(values <= 0.0):
        raise ValidationError(
            _("scales must be positive"),
            details={"parameter": "scales", "minimum": float(values.min())},
        )
    if np.any(np.diff(values) <= 0.0):
        raise ValidationError(
            _("scales must be strictly increasing"),
            details={"parameter": "scales"},
        )
    return values


def _resolve_wavelet(wavelet: object) -> str:
    """把母小波名称归一化为 ``"morlet"`` 或 ``"mexican_hat"``。"""
    name = str(wavelet).strip().lower().replace("-", "_").replace(" ", "_")
    if name == "morlet":
        return "morlet"
    if name in {"mexican_hat", "mexicanhat", "dog", "mexican_hat_wavelet"}:
        return "mexican_hat"
    raise ValidationError(
        _("wavelet must be 'morlet' or 'mexican_hat'"),
        details={"parameter": "wavelet", "wavelet": str(wavelet)},
    )


def _resolve_time_axis(
    time: object | None,
    dt: object | None,
    n_samples: int,
) -> tuple[npt.NDArray, float]:
    """解析时间轴，返回 ``(时间值, 采样间隔)``。

    只支持均匀时间轴：给 ``time`` 时要求严格递增且等间距；给 ``dt`` 时按
    等间距处理；两者都不给时按 ``dt = 1`` 处理。
    """
    if time is not None:
        times = _as_vector(time, "time", minimum_size=MIN_SAMPLES)
        if times.size != n_samples:
            raise ValidationError(
                _("time and the data series must have the same length"),
                details={"time_length": int(times.size), "series_length": int(n_samples)},
            )
        gaps = np.diff(times)
        if np.any(gaps <= 0.0):
            raise ValidationError(
                _("time must contain strictly increasing values"),
                details={"n_points": int(times.size)},
            )
        spacing = float(np.mean(gaps))
        tolerance = max(abs(spacing) * 1.0e-6, 1.0e-12)
        if not np.all(np.abs(gaps - spacing) <= tolerance):
            raise ValidationError(
                _("only uniformly spaced time axes are supported"),
                details={
                    "n_points": int(times.size),
                    "spacing": spacing,
                    "minimum_gap": float(gaps.min()),
                    "maximum_gap": float(gaps.max()),
                },
            )
        return times, spacing
    if dt is None:
        return np.arange(n_samples, dtype=float), 1.0
    try:
        spacing = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("dt must be a positive number"),
            details={"parameter": "dt"},
            original_exception=exc,
        ) from exc
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValidationError(
            _("dt must be a positive number"),
            details={"parameter": "dt", "dt": dt},
        )
    return np.arange(n_samples, dtype=float) * spacing, spacing


def _fourier_factor(wavelet: str, omega0: float) -> float:
    """Fourier 因子：``周期（样本） = Fourier 因子 × 尺度（样本）``。

    取自 Torrence & Compo (1998) 的表 1（其 ``wave_bases.m`` 中的
    ``fourier_factor``），与该文的归一化口径配对使用。
    """
    if wavelet == "morlet":
        return 4.0 * math.pi / (omega0 + math.sqrt(2.0 + omega0 * omega0))
    order = float(MEXICAN_HAT_ORDER)
    return 2.0 * math.pi * math.sqrt(2.0 / (2.0 * order + 1.0))


def _period_axis(scales: npt.NDArray, fourier_factor: float, dt: float) -> tuple[npt.NDArray, npt.NDArray]:
    """由尺度轴算出周期轴（每单位时间）与频率轴（每单位时间的循环频率）。"""
    frequencies = (1.0 / fourier_factor) / scales / dt
    periods = 1.0 / frequencies
    return periods, frequencies


def _cone_of_influence(n_samples: int, dt: float, fourier_factor: float) -> npt.NDArray:
    """锥形影响域：每个时间点上的有效最大周期（时间单位）。

    距离最近的边界 ``d`` 个样本处，尺度 ``s`` 的小波支撑（``sqrt(2) * s``）
    已经触到边界，因此该点上的有效最大周期为
    ``Fourier 因子 * d * dt / sqrt(2)``（Grinsted et al. 2004 的口径）。
    """
    distances = np.minimum(np.arange(n_samples), np.arange(n_samples)[::-1]) + 1.0
    return fourier_factor * distances * dt / math.sqrt(2.0)


def _daughter_wavelet(
    wavelet: str,
    omega0: float,
    scale: float,
    k: npt.NDArray,
    n_pad: int,
) -> npt.NDArray:
    """母小波在傅里叶空间的离散形式（Torrence & Compo 1998 的口径）。"""
    if wavelet == "morlet":
        amplitude = np.exp(-0.5 * (scale * k - omega0) ** 2) * (k > 0.0)
        norm = math.sqrt(scale * k[1]) * math.pi ** (-0.25) * math.sqrt(n_pad)
        return norm * amplitude
    order = float(MEXICAN_HAT_ORDER)
    amplitude = (scale * k) ** order * np.exp(-0.5 * (scale * k) ** 2)
    norm = math.sqrt(scale * k[1] / math.gamma(order + 0.5)) * math.sqrt(n_pad)
    return norm * amplitude


def _cwt(values: npt.NDArray, scales: npt.NDArray, wavelet: str, omega0: float) -> npt.NDArray:
    """连续小波变换（傅里叶空间实现）。

    Returns:
        npt.NDArray: 复系数矩阵，形状 ``(尺度数, 时间点数)``。
    """
    n_samples = int(values.size)
    n_pad = 1 << math.ceil(math.log2(max(n_samples, 2)))
    spectrum = np.fft.fft(values, n_pad)
    k = 2.0 * math.pi * np.fft.fftfreq(n_pad)
    coefficients = np.empty((scales.size, n_samples), dtype=complex)
    for index, scale in enumerate(scales):
        daughter = _daughter_wavelet(wavelet, omega0, float(scale), k, n_pad)
        coefficients[index] = np.fft.ifft(spectrum * daughter)[:n_samples]
    return coefficients


def _smooth_time(coefficients: npt.NDArray, scales: npt.NDArray, factor: float) -> npt.NDArray:
    """时间维平滑：高斯窗，标准差 = ``factor × 尺度``（以样本计）。"""
    n_samples = int(coefficients.shape[1])
    n_pad = 1 << math.ceil(math.log2(max(n_samples, 2)))
    k = 2.0 * math.pi * np.fft.fftfreq(n_pad)
    smoothed = np.empty((scales.size, n_samples), dtype=complex)
    for index, scale in enumerate(scales):
        sigma = factor * float(scale)
        kernel = np.exp(-0.5 * (sigma * k) ** 2)
        smoothed[index] = np.fft.ifft(np.fft.fft(coefficients[index], n_pad) * kernel)[:n_samples]
    return smoothed


def _smooth_scale(coefficients: npt.NDArray, scales: npt.NDArray, octaves: float) -> npt.NDArray:
    """尺度维平滑：高斯窗，作用在 ``log2(尺度)`` 坐标上。

    ``octaves`` 是窗口在倍频程上的**全宽**，高斯标准差取其一半。
    """
    sigma = 0.5 * float(octaves)
    log_scales = np.log2(scales)
    smoothed = np.empty_like(coefficients)
    for index in range(scales.size):
        offsets = (log_scales - log_scales[index]) / sigma
        weights = np.exp(-0.5 * offsets**2)
        weights /= weights.sum()
        smoothed[index] = weights @ coefficients
    return smoothed


def _prepare_series(
    values: npt.NDArray,
    times: npt.NDArray,
    name: str,
    detrend: bool,
) -> npt.NDArray:
    """去均值（可选去线性趋势），得到参与小波变换的序列。"""
    if detrend:
        detrended = np.asarray(detrend_series(times, values, method="linear").detrended, dtype=float)
        if not np.all(np.isfinite(detrended)):
            raise ValidationError(
                _("{0} must contain only finite values").format(name),
                details={"parameter": name},
            )
        return detrended
    return values - float(np.mean(values))


def _coherence_from_transforms(
    wave_x: npt.NDArray,
    wave_y: npt.NDArray,
    scales: npt.NDArray,
    time_factor: float,
    scale_octaves: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """由两条小波变换算出相干性与相位。

    Returns:
        tuple: ``(R², 相位)``。
    """
    inverse_scale = (1.0 / scales)[:, None]
    smoothed_xx = _smooth_scale(
        _smooth_time(np.abs(wave_x) ** 2 * inverse_scale, scales, time_factor), scales, scale_octaves
    )
    smoothed_yy = _smooth_scale(
        _smooth_time(np.abs(wave_y) ** 2 * inverse_scale, scales, time_factor), scales, scale_octaves
    )
    smoothed_xy = _smooth_scale(
        _smooth_time(wave_x * np.conj(wave_y) * inverse_scale, scales, time_factor), scales, scale_octaves
    )

    denominator = smoothed_xx.real * smoothed_yy.real
    # 分母下限保护：白噪声在极小尺度上的谱值可以任意接近 0。
    positive = denominator[denominator > 0.0]
    floor = COHERENCE_FLOOR
    if positive.size:
        floor = max(COHERENCE_FLOOR, float(positive.max()) * COHERENCE_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.abs(smoothed_xy) ** 2 / np.maximum(denominator, floor)
    coherence = np.where(np.isfinite(coherence), np.clip(coherence, 0.0, 1.0), 0.0)
    return coherence, np.angle(smoothed_xy)


def _ar1_coefficient(values: npt.NDArray) -> float:
    """AR(1) 系数的朴素估计量（Grinsted et al. 2004 的 ``ar1nv`` 口径）。"""
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered[:-1], centered[:-1]))
    if denominator <= 0.0:
        return 0.0
    rho = float(np.dot(centered[1:], centered[:-1])) / denominator
    return float(np.clip(rho, -0.99, 0.99))


def _ar1_surrogate(rng: np.random.Generator, n_samples: int, rho: float, variance: float) -> npt.NDArray:
    """生成一条 AR(1) 替代序列（方差与观测序列一致）。"""
    noise = rng.standard_normal(n_samples)
    series = lfilter([1.0], [1.0, -rho], noise)
    series = np.asarray(series, dtype=float)
    series -= float(np.mean(series))
    current = float(np.var(series))
    if current > 0.0:
        series *= math.sqrt(max(variance, 0.0) / current)
    return series


def _valid_mask(periods: npt.NDArray, cone_of_influence: npt.NDArray) -> npt.NDArray:
    """锥形影响域之外的布尔掩码，形状 ``(尺度数, 时间点数)``。"""
    return periods[:, None] <= cone_of_influence[None, :]


def _global_spectrum(power: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """时间平均的小波谱（只在有效点上平均）。"""
    spectrum = np.full(power.shape[0], np.nan, dtype=float)
    for index in range(power.shape[0]):
        selected = power[index][mask[index]]
        if selected.size:
            spectrum[index] = float(np.mean(selected))
    return spectrum


def _quantile_of_spectra(samples: list[npt.NDArray], level: float) -> npt.NDArray:
    """把每次重复的**逐尺度谱值**汇总成每个尺度的分位数。"""
    if not samples:
        raise ValidationError(
            _("the Monte Carlo run produced no samples"),
            details={"n_samples": 0},
        )
    stacked = np.vstack([np.asarray(sample, dtype=float) for sample in samples])
    critical = np.full(stacked.shape[1], np.nan, dtype=float)
    for index in range(stacked.shape[1]):
        column = stacked[:, index]
        column = column[np.isfinite(column)]
        if column.size:
            critical[index] = float(np.quantile(column, level))
    return critical


def _quantile_of_pointwise(samples: list[npt.NDArray], level: float) -> npt.NDArray:
    """把每次重复的**逐点统计量**按尺度汇总成每个尺度的分位数。

    每个样本是 ``(尺度数, 时间点数)`` 的矩阵：同一尺度上所有有效时间点、
    所有重复的取值汇总成一个分布，再取 ``level`` 分位数
    （Grinsted et al. 2004 §3 的做法）。
    """
    if not samples:
        raise ValidationError(
            _("the Monte Carlo run produced no samples"),
            details={"n_samples": 0},
        )
    critical = np.full(samples[0].shape[0], np.nan, dtype=float)
    for index in range(critical.size):
        pooled = np.concatenate([sample[index] for sample in samples])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            critical[index] = float(np.quantile(pooled, level))
    return critical


# =============================================================================
# Public interface
# =============================================================================


def wavelet_transform(
    values: npt.NDArray,
    scales: npt.NDArray,
    wavelet: str = "morlet",
    *,
    time: npt.NDArray | None = None,
    dt: float | None = None,
    omega0: float = MORLET_OMEGA0,
    detrend: bool = True,
) -> WaveletTransformResult:
    """连续小波变换（CWT）。

    母小波在**傅里叶空间**离散化（Torrence & Compo 1998 的口径），尺度以
    **样本**为单位给出；频率轴由 Fourier 因子换算：

        freq（每样本） = wn / scale，  wn = 1 / Fourier 因子
        周期（时间单位） = Fourier 因子 × scale（样本） × dt

    Fourier 因子：Morlet 为 ``4 pi / (w0 + sqrt(2 + w0^2))``，Mexican Hat
    （DOG m=2）为 ``2 pi sqrt(2 / 5)``。它与本模块的归一化口径配对使用，
    保证响应峰值所在尺度能还原出真实周期。

    Parameters:
        values: 一维有限序列（接受数组；标量或序列会在内部转换成数组）。
        scales: 尺度轴（**以样本计**），正数、有限、严格递增，至少 2 个
            （接受数组；标量或序列会在内部转换成数组）。
        wavelet: ``"morlet"``（缺省，w0 = 6）或 ``"mexican_hat"``。
        time: 时间轴（每单位时间，接受数组）。必须严格递增**且等间距**；
            不等间距会抛 ``ValidationError``（本模块不做重采样）。
        dt: 采样间隔（每单位时间，标量）。``time`` 缺省时使用；两者都缺省时
            按 ``dt = 1`` 处理。
        omega0: Morlet 的中央角频率 w0，缺省 6。
        detrend: 是否先去掉线性趋势（缺省 True），避免趋势泄漏到低频。

    Returns:
        WaveletTransformResult: 复系数、尺度轴、周期轴、频率轴与锥形影响域。

    Raises:
        ValidationError: 输入长度不合法、含非有限值、点数太少、尺度轴非法、
            时间轴非严格递增或不等间距、``dt`` 非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.wavelet_coherence import wavelet_transform
        >>> time = np.linspace(0.0, 200.0, 1000)
        >>> signal = np.sin(2.0 * np.pi * time / 20.0)
        >>> result = wavelet_transform(signal, np.arange(4.0, 40.0, 0.5), time=time)
        >>> bool(result.coefficients.shape[0] == result.periods.size)
        True
    """
    series = _as_vector(values, "values", minimum_size=MIN_SAMPLES)
    scale_values = _validate_scales(scales)
    wavelet_name = _resolve_wavelet(wavelet)
    times, spacing = _resolve_time_axis(time, dt, int(series.size))
    prepared = _prepare_series(series, times, "values", detrend)
    coefficients = _cwt(prepared, scale_values, wavelet_name, float(omega0))
    factor = _fourier_factor(wavelet_name, float(omega0))
    periods, frequencies = _period_axis(scale_values, factor, spacing)
    return WaveletTransformResult(
        coefficients=coefficients,
        scales=scale_values,
        periods=periods,
        frequencies=frequencies,
        times=times,
        cone_of_influence=_cone_of_influence(int(series.size), spacing, factor),
        dt=spacing,
    )


def cross_wavelet_transform(
    values_x: npt.NDArray,
    values_y: npt.NDArray,
    scales: npt.NDArray,
    wavelet: str = "morlet",
    *,
    time: npt.NDArray | None = None,
    dt: float | None = None,
    omega0: float = MORLET_OMEGA0,
    detrend: bool = True,
) -> CrossWaveletResult:
    """交叉小波变换（XWT）。

    交叉小波矩阵取 ``W_xy = W_x * conj(W_y)``（Grinsted et al. 2004 与 Acycle
    的 ``wcoherence`` 一致），逐尺度相位为 ``arg(W_xy)``，即**第一条序列的
    相位减去第二条序列的相位**：第二条序列滞后 Δt 时相位为
    ``+2 pi f Δt``，提前 Δt 时为 ``-2 pi f Δt``。

    Parameters:
        values_x: 第一条序列（接受数组；标量或序列会在内部转换成数组）。
        values_y: 第二条序列（必须与第一条等长）。
        scales: 尺度轴（以样本计，接受数组；标量或序列会在内部转换成数组）。
        wavelet: ``"morlet"`` 或 ``"mexican_hat"``。
        time: 时间轴（每单位时间，接受数组），必须严格递增且等间距。
        dt: 采样间隔（每单位时间，标量）。
        omega0: Morlet 的中央角频率 w0。
        detrend: 是否先去掉线性趋势。

    Returns:
        CrossWaveletResult: 复交叉小波矩阵、逐尺度相位与坐标轴。

    Raises:
        ValidationError: 两条序列长度不一致、点数太少、含非有限值、时间轴
            非法等。
    """
    series_x = _as_vector(values_x, "values_x", minimum_size=MIN_SAMPLES)
    series_y = _as_vector(values_y, "values_y", minimum_size=MIN_SAMPLES)
    if series_x.size != series_y.size:
        raise ValidationError(
            _("the two series must have the same length"),
            details={"first_length": int(series_x.size), "second_length": int(series_y.size)},
        )
    scale_values = _validate_scales(scales)
    wavelet_name = _resolve_wavelet(wavelet)
    times, spacing = _resolve_time_axis(time, dt, int(series_x.size))
    prepared_x = _prepare_series(series_x, times, "values_x", detrend)
    prepared_y = _prepare_series(series_y, times, "values_y", detrend)
    wave_x = _cwt(prepared_x, scale_values, wavelet_name, float(omega0))
    wave_y = _cwt(prepared_y, scale_values, wavelet_name, float(omega0))
    cross = wave_x * np.conj(wave_y)
    factor = _fourier_factor(wavelet_name, float(omega0))
    periods, frequencies = _period_axis(scale_values, factor, spacing)
    return CrossWaveletResult(
        cross_wavelet=cross,
        phase=np.angle(cross),
        scales=scale_values,
        periods=periods,
        frequencies=frequencies,
        times=times,
        cone_of_influence=_cone_of_influence(int(series_x.size), spacing, factor),
        dt=spacing,
    )


def wavelet_coherence(
    values_x: npt.NDArray,
    values_y: npt.NDArray,
    scales: npt.NDArray,
    wavelet: str = "morlet",
    *,
    time: npt.NDArray | None = None,
    dt: float | None = None,
    omega0: float = MORLET_OMEGA0,
    detrend: bool = True,
    significance: bool = True,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
    phase_threshold: float = PHASE_COHERENCE_THRESHOLD,
    time_smoothing: float = TIME_SMOOTHING_FACTOR,
    scale_smoothing: float = SCALE_SMOOTHING_OCTAVES,
) -> WaveletCoherenceResult:
    """小波相干（WTC）。

    相干性由**平滑后**的自谱与互谱给出（Grinsted et al. 2004）：

        S_xx = smooth(|W_x|^2 / s),  S_yy = smooth(|W_y|^2 / s)
        S_xy = smooth(W_x * conj(W_y) / s)
        R^2  = |S_xy|^2 / (S_xx * S_yy)

    平滑口径：**时间维**用与尺度成正比的高斯窗（标准差
    ``time_smoothing × 尺度``，以样本计，缺省 0.6）；**尺度维**用作用在
    ``log2(尺度)`` 上的高斯窗（全宽 ``scale_smoothing`` 个倍频程，缺省
    0.6，高斯标准差取其一半）。分母带下限保护，避免 0/0。

    **相位约定**：``phase = arg(W_x * conj(W_y))``，即第一条序列的相位减去
    第二条序列的相位；第二条序列滞后 Δt 时为 ``+2 pi f Δt``，提前 Δt 时为
    ``-2 pi f Δt``。相干性低于 ``phase_threshold`` 的点上相位为 ``NaN``
    （相位在那里没有意义）。

    Parameters:
        values_x: 第一条序列（接受数组；标量或序列会在内部转换成数组）。
        values_y: 第二条序列（必须与第一条等长）。
        scales: 尺度轴（以样本计，接受数组；标量或序列会在内部转换成数组）。
        wavelet: ``"morlet"`` 或 ``"mexican_hat"``。
        time: 时间轴（每单位时间，接受数组），必须严格递增且等间距。
        dt: 采样间隔（每单位时间，标量）。
        omega0: Morlet 的中央角频率 w0。
        detrend: 是否先去掉线性趋势。
        significance: 是否计算 95% 蒙特卡洛临界值曲线（缺省 True）。
        n_monte_carlo: 蒙特卡洛重复次数。
        seed: 随机种子（可复现）。
        level: 显著性水平。
        phase_threshold: 相位被置为 ``NaN`` 的相干性阈值。
        time_smoothing: 时间维高斯窗系数（乘以尺度）。
        scale_smoothing: 尺度维高斯窗全宽（倍频程）。

    Returns:
        WaveletCoherenceResult: 相干性、相位、临界值曲线与坐标轴。

    Raises:
        ValidationError: 输入非法（长度不一致、点数太少、含非有限值、时间轴
            非严格递增或不等间距、尺度轴非法、蒙特卡洛次数非法等）。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.wavelet_coherence import wavelet_coherence
        >>> time = np.linspace(0.0, 200.0, 1000)
        >>> signal = np.sin(2.0 * np.pi * time / 20.0)
        >>> result = wavelet_coherence(
        ...     signal, signal, np.arange(4.0, 40.0, 0.5), time=time, significance=False
        ... )
        >>> bool(result.coherence.max() > 0.9)
        True
    """
    series_x = _as_vector(values_x, "values_x", minimum_size=MIN_SAMPLES)
    series_y = _as_vector(values_y, "values_y", minimum_size=MIN_SAMPLES)
    if series_x.size != series_y.size:
        raise ValidationError(
            _("the two series must have the same length"),
            details={"first_length": int(series_x.size), "second_length": int(series_y.size)},
        )
    scale_values = _validate_scales(scales)
    wavelet_name = _resolve_wavelet(wavelet)
    times, spacing = _resolve_time_axis(time, dt, int(series_x.size))
    prepared_x = _prepare_series(series_x, times, "values_x", detrend)
    prepared_y = _prepare_series(series_y, times, "values_y", detrend)
    wave_x = _cwt(prepared_x, scale_values, wavelet_name, float(omega0))
    wave_y = _cwt(prepared_y, scale_values, wavelet_name, float(omega0))
    coherence, phase = _coherence_from_transforms(
        wave_x, wave_y, scale_values, float(time_smoothing), float(scale_smoothing)
    )
    factor = _fourier_factor(wavelet_name, float(omega0))
    periods, frequencies = _period_axis(scale_values, factor, spacing)
    cone = _cone_of_influence(int(series_x.size), spacing, factor)

    threshold = float(phase_threshold)
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValidationError(
            _("phase_threshold must lie between 0 and 1"),
            details={"phase_threshold": phase_threshold},
        )
    phase = np.where(coherence < threshold, np.nan, phase)

    critical: npt.NDArray | None = None
    if significance:
        critical = wavelet_coherence_significance(
            series_x,
            series_y,
            scale_values,
            wavelet_name,
            dt=spacing,
            omega0=float(omega0),
            n_monte_carlo=n_monte_carlo,
            seed=seed,
            level=level,
            time_smoothing=float(time_smoothing),
            scale_smoothing=float(scale_smoothing),
        ).critical_values

    return WaveletCoherenceResult(
        coherence=coherence,
        phase=phase,
        significance=critical,
        cross_wavelet=wave_x * np.conj(wave_y),
        scales=scale_values,
        periods=periods,
        frequencies=frequencies,
        times=times,
        cone_of_influence=cone,
        phase_threshold=threshold,
        dt=spacing,
    )


def wavelet_significance(
    values: npt.NDArray,
    scales: npt.NDArray,
    wavelet: str = "morlet",
    *,
    time: npt.NDArray | None = None,
    dt: float | None = None,
    omega0: float = MORLET_OMEGA0,
    detrend: bool = True,
    ar1: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
) -> WaveletSignificanceResult:
    """单序列的小波显著性（AR(1) 蒙特卡洛背景）。

    零假设是"该序列没有显著的小波谱峰"，背景由与观测序列同方差、同 AR(1)
    系数的替代序列给出。对每次重复计算**时间平均小波谱**（只在锥形影响域
    之外的点上平均），逐尺度取 ``level`` 分位数作为临界值。

    Parameters:
        values: 一维有限序列（接受数组；标量或序列会在内部转换成数组）。
        scales: 尺度轴（以样本计，接受数组；标量或序列会在内部转换成数组）。
        wavelet: ``"morlet"`` 或 ``"mexican_hat"``。
        time: 时间轴（每单位时间，接受数组），必须严格递增且等间距。
        dt: 采样间隔（每单位时间，标量）。
        omega0: Morlet 的中央角频率 w0。
        detrend: 是否先去掉线性趋势。
        ar1: AR(1) 系数；缺省时由观测序列估计（朴素估计量）。
        n_monte_carlo: 蒙特卡洛重复次数，至少 1。
        seed: 随机种子（可复现）。
        level: 显著性水平。

    Returns:
        WaveletSignificanceResult: 观测谱、逐尺度临界值、AR(1) 系数与坐标轴。

    Raises:
        ValidationError: 输入非法、蒙特卡洛次数非法、显著性水平非法。
    """
    series = _as_vector(values, "values", minimum_size=MIN_SAMPLES)
    scale_values = _validate_scales(scales)
    wavelet_name = _resolve_wavelet(wavelet)
    times, spacing = _resolve_time_axis(time, dt, int(series.size))
    repetitions = _validate_monte_carlo(n_monte_carlo)
    confidence = _validate_level(level)

    prepared = _prepare_series(series, times, "values", detrend)
    wave = _cwt(prepared, scale_values, wavelet_name, float(omega0))
    factor = _fourier_factor(wavelet_name, float(omega0))
    periods, frequencies = _period_axis(scale_values, factor, spacing)
    cone = _cone_of_influence(int(series.size), spacing, factor)
    mask = _valid_mask(periods, cone)

    variance = float(np.var(prepared))
    rho = _validate_ar1(ar1) if ar1 is not None else _ar1_coefficient(prepared)
    generator = np.random.default_rng(int(seed))
    samples: list[npt.NDArray] = []
    for _repetition in range(repetitions):
        surrogate = _ar1_surrogate(generator, int(series.size), rho, variance)
        surrogate_wave = _cwt(surrogate, scale_values, wavelet_name, float(omega0))
        samples.append(_global_spectrum(np.abs(surrogate_wave) ** 2, mask))

    return WaveletSignificanceResult(
        spectrum=_global_spectrum(np.abs(wave) ** 2, mask),
        critical_values=_quantile_of_spectra(samples, confidence),
        ar1_coefficient=rho,
        n_monte_carlo=repetitions,
        seed=int(seed),
        scales=scale_values,
        periods=periods,
        frequencies=frequencies,
    )


def wavelet_coherence_significance(
    values_x: npt.NDArray,
    values_y: npt.NDArray,
    scales: npt.NDArray,
    wavelet: str = "morlet",
    *,
    time: npt.NDArray | None = None,
    dt: float | None = None,
    omega0: float = MORLET_OMEGA0,
    detrend: bool = True,
    ar1_x: float | None = None,
    ar1_y: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
    time_smoothing: float = TIME_SMOOTHING_FACTOR,
    scale_smoothing: float = SCALE_SMOOTHING_OCTAVES,
) -> WaveletCoherenceSignificanceResult:
    """WTC 的蒙特卡洛临界值（Grinsted et al. 2004 §3）。

    每次重复生成**两条独立**的 AR(1) 序列（系数与方差分别取自两条观测
    序列），计算它们的 WTC，把每个尺度上所有有效时间点的 ``R²`` 汇总后
    取 ``level`` 分位数。

    Parameters:
        values_x: 第一条序列（接受数组；标量或序列会在内部转换成数组）。
        values_y: 第二条序列（必须与第一条等长）。
        scales: 尺度轴（以样本计，接受数组；标量或序列会在内部转换成数组）。
        wavelet: ``"morlet"`` 或 ``"mexican_hat"``。
        time: 时间轴（每单位时间，接受数组），必须严格递增且等间距。
        dt: 采样间隔（每单位时间，标量）。
        omega0: Morlet 的中央角频率 w0。
        detrend: 是否先去掉线性趋势。
        ar1_x: 第一条序列的 AR(1) 系数（缺省时估计）。
        ar1_y: 第二条序列的 AR(1) 系数（缺省时估计）。
        n_monte_carlo: 蒙特卡洛重复次数，至少 1。
        seed: 随机种子（可复现）。
        level: 显著性水平。
        time_smoothing: 时间维高斯窗系数（乘以尺度）。
        scale_smoothing: 尺度维高斯窗全宽（倍频程）。

    Returns:
        WaveletCoherenceSignificanceResult: 逐尺度临界值、两个 AR(1) 系数与
        坐标轴。

    Raises:
        ValidationError: 输入非法、蒙特卡洛次数非法、显著性水平非法。
    """
    series_x = _as_vector(values_x, "values_x", minimum_size=MIN_SAMPLES)
    series_y = _as_vector(values_y, "values_y", minimum_size=MIN_SAMPLES)
    if series_x.size != series_y.size:
        raise ValidationError(
            _("the two series must have the same length"),
            details={"first_length": int(series_x.size), "second_length": int(series_y.size)},
        )
    scale_values = _validate_scales(scales)
    wavelet_name = _resolve_wavelet(wavelet)
    times, spacing = _resolve_time_axis(time, dt, int(series_x.size))
    repetitions = _validate_monte_carlo(n_monte_carlo)
    confidence = _validate_level(level)

    prepared_x = _prepare_series(series_x, times, "values_x", detrend)
    prepared_y = _prepare_series(series_y, times, "values_y", detrend)
    rho_x = _validate_ar1(ar1_x) if ar1_x is not None else _ar1_coefficient(prepared_x)
    rho_y = _validate_ar1(ar1_y) if ar1_y is not None else _ar1_coefficient(prepared_y)
    variance_x = float(np.var(prepared_x))
    variance_y = float(np.var(prepared_y))

    factor = _fourier_factor(wavelet_name, float(omega0))
    periods, frequencies = _period_axis(scale_values, factor, spacing)
    cone = _cone_of_influence(int(series_x.size), spacing, factor)
    mask = _valid_mask(periods, cone)

    generator = np.random.default_rng(int(seed))
    samples: list[npt.NDArray] = []
    for _repetition in range(repetitions):
        surrogate_x = _ar1_surrogate(generator, int(series_x.size), rho_x, variance_x)
        surrogate_y = _ar1_surrogate(generator, int(series_y.size), rho_y, variance_y)
        wave_x = _cwt(surrogate_x, scale_values, wavelet_name, float(omega0))
        wave_y = _cwt(surrogate_y, scale_values, wavelet_name, float(omega0))
        coherence, _phase = _coherence_from_transforms(
            wave_x, wave_y, scale_values, float(time_smoothing), float(scale_smoothing)
        )
        masked = np.where(mask, coherence, np.nan)
        samples.append(masked)

    return WaveletCoherenceSignificanceResult(
        critical_values=_quantile_of_pointwise(samples, confidence),
        ar1_coefficients=(rho_x, rho_y),
        n_monte_carlo=repetitions,
        seed=int(seed),
        scales=scale_values,
        periods=periods,
        frequencies=frequencies,
    )


def _validate_monte_carlo(n_monte_carlo: object) -> int:
    """检查蒙特卡洛重复次数。"""
    try:
        repetitions = int(n_monte_carlo)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("n_monte_carlo must be a positive integer"),
            details={"parameter": "n_monte_carlo"},
            original_exception=exc,
        ) from exc
    if repetitions != n_monte_carlo or repetitions < 1:
        raise ValidationError(
            _("n_monte_carlo must be a positive integer"),
            details={"parameter": "n_monte_carlo", "n_monte_carlo": n_monte_carlo},
        )
    return repetitions


def _validate_level(level: object) -> float:
    """检查显著性水平。"""
    try:
        confidence = float(level)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("level must lie strictly between 0 and 1"),
            details={"parameter": "level"},
            original_exception=exc,
        ) from exc
    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValidationError(
            _("level must lie strictly between 0 and 1"),
            details={"parameter": "level", "level": level},
        )
    return confidence


def _validate_ar1(ar1: object) -> float:
    """检查用户给定的 AR(1) 系数。"""
    try:
        rho = float(ar1)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("the AR(1) coefficient must be a number"),
            details={"parameter": "ar1"},
            original_exception=exc,
        ) from exc
    if not math.isfinite(rho) or not (-1.0 < rho < 1.0):
        raise ValidationError(
            _("the AR(1) coefficient must lie strictly between -1 and 1"),
            details={"parameter": "ar1", "ar1": ar1},
        )
    return rho
