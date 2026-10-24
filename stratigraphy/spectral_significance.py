# =============================================================================
# FILE: stratigraphy/spectral_significance.py
# =============================================================================
"""
显著性判据：SWA 谱背景、Benjamini–Hochberg FDR 与贝叶斯周期图。

本模块给出同一问题的几个层次，它们共用同一条谱，因此结论可以互相印证：

* :func:`swa_background` —— **非参数地估计谱背景**（SWA，"smoothed window
  averages"）。背景先由"窗口平均 + 线性插值/外推"在对数谱上铺出，高频端再
  用**线性**或**二次**最小二乘尾拟合替换；两种变体按整条谱上的 RMSE
  （与 R² 等价）自动择优。背景在白噪声段收敛到白噪声水平，这正是显著性
  检验所需要的零假设水平。
* :func:`weedon_periodogram` —— 产出**自由度已知**的谱（默认 8 自由度）及其
  SWA 背景，让用户一次调用就能走完"背景 → FDR"的流程。
* :func:`swa_fdr` —— 以背景为基准，给出每个频率的 p 值，再用
  **Benjamini–Hochberg (1995)** 的错误发现率（FDR）程序挑出显著频率。
* :func:`bayesian_periodogram` —— Lomb–Scargle 周期图配合**贝叶斯因子**，
  给出"某个频率是真实周期"的后验概率。

适用域
~~~~~~
SWA 背景是为**红噪声**设计的非参数方法（Weedon 2019, 2020, 2022）：它**不
适用于白噪声**（滞后 1 阶自相关为 0），也**不适用于幂律噪声**（滞后 1 阶
自相关为 1）。这两类噪声各自有自己的零假设模型，不该用非参数的谱背景。

SWA 背景的做法
~~~~~~~~~~~~~~
1. 取 ``log10`` 功率谱；
2. 用若干个**等宽窗口**对谱做平均，窗口中心之间线性插值、两端按最后一段
   斜率线性外推，得到一条平滑的对数谱包络；
3. 窗口宽度在候选集合中择优：先最小化包络中"非降步"的个数（背景应当是
   单调不增的），再最小化 RMSE；
4. 高频端（默认最高 25% 的频率）改用**线性**或**二次**最小二乘拟合的尾
   模型，两种变体按整条谱上的 RMSE 择优（等价于选 R² 更高者）。

背景与 p 值：自由度契约
~~~~~~~~~~~~~~~~~~~~~~~
p 值由 ``p(f) = sf(dof * power(f) / background(f), dof)`` 得到，**只有当
``power`` 恰好是每频点自由度为 ``dof`` 的周期图时，这个 p 值才成立**：

* :func:`weedon_periodogram` 的谱每点是 ``n_segments`` 段**互不重叠**的
  Lomb–Scargle 周期图的平均，每段按自身方差归一化，于是每点服从
  ``chi2(2 * n_segments) / (2 * n_segments)``，自由度 ``2 * n_segments``
  （默认 8，与 Weedon 的 ``specswafdr.m`` 用三次 Hanning 加窗得到的自由度
  相同）；
* 仓库里现成的、未加窗未平均的
  :class:`stratigraphy.spectral_analysis.SpectralAnalyzer` 周期图每点只有
  **2** 个自由度（``chi2(2) / 2``），必须配 ``degrees_of_freedom=2``。

把 8 自由度的临界值套到 2 自由度的谱上，p 值会系统性偏小：实测在
AR(1) 红噪声上（``alpha = 0.05``），逐点假阳性率从 9% 膨胀到 17%–19%，
BH 拒绝的频率占比从 0% 涨到 25% 左右。

再对全部频率做 Benjamini–Hochberg 过程：把 p 值升序排列，取满足
``p_(k) <= alpha * k / m`` 的最大 ``k``，前 ``k`` 个频率判为显著。注意 BH 是
"阶梯向上"过程：零假设下它要么一个都不拒、要么拒绝一整块，因此**单次实现
的拒绝个数**不是水平的良好诊断量，逐点超越率（``p <= alpha`` 的占比）才是
自由度契约直接控制的量。

贝叶斯周期图
~~~~~~~~~~~~
把序列标准化（去趋势 + AR(1) 预白化）后，Lomb–Scargle 周期图给出归一化
功率 ``u(f)``（白噪声下均值为 1）。若频率 ``f`` 上存在幅度为 ``A``（以序列
标准差为单位）的正弦信号，它对 ``u(f)`` 的贡献是 ``z = n * A^2 / 4``，而
噪声部分仍是均值 1 的指数分布。于是

    H0 (无信号): p0(u) = exp(-u)
    H1 (有信号): p1(u | z) = exp(-(u - z)) * 1[u >= z]

似然比 ``p1 / p0 = exp(z) * 1[u >= z]`` 对幅度先验（半正态）积分就是**贝叶斯
因子**；本模块用蒙特卡洛（``n_iterations`` 次抽样）估计这个积分，再把各频
率的贝叶斯因子归一化成后验概率。

方法依据
~~~~~~~~
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate:
  a practical and powerful approach to multiple testing. JRSS B 57, 289-300.
* Weedon, G. P. (2020). Confirmed detection of Palaeogene and Jurassic
  orbitally-forced sedimentary cycles in the depth domain using false discovery
  rates and Bayesian probability spectra. Boletin de la Sociedad Geologica
  Mexicana 131, 207-230.
* Weedon, G. P. (2022). Problems with the current practice of spectral analysis
  in cyclostratigraphy: avoiding false positive detection of regular
  cyclicity. Earth-Science Reviews 235, 104261.
* Foreman-Mackey, D. (2016). Time-domain and Fourier-domain methods for the
  characterization of periodic processes. JOSS 1, e00241.
* Gregory, P. C. (2005). Bayesian analysis of astronomical time series.
  PhD thesis, University of Oxford.
* Mitrofanov, A. & Čekan, P. (2017). Spectral distortion of the test noise.
  Solid Earth 8, 565-571.

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考其公开的行为约定与
文献中的公式，未翻译或复制其 MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.special import logsumexp

from config.i18n import _
from stratigraphy.spectral_analysis import SpectralAnalyzer, _linear_detrend
from stratigraphy.spectral_moments import estimate_ar1_rho
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

#: SWA 背景所需的最小频率点数：至少要放下三个完整的平均窗口。
MIN_SPECTRUM_POINTS = 16

#: 平均窗口的最小宽度（奇数）。
MIN_WINDOW_WIDTH = 5

#: 平均窗口的最大宽度（奇数）。
MAX_WINDOW_WIDTH = 99

#: 尾拟合区占整条频率轴的比例（从高频端往低频方向取）。
TAIL_FRACTION = 0.25

#: 尾拟合区至少包含的点数（二次拟合至少需要 3 个互不相同的频率）。
MIN_TAIL_POINTS = 6

#: FDR 检验所用的卡方自由度（对应 3 次 Hanning 加窗的周期图，Acycle 的取法）。
FDR_DEGREES_OF_FREEDOM = 8.0

#: 未加窗、未平均的 Lomb–Scargle 周期图每个频点的自由度。
PLAIN_PERIODOGRAM_DEGREES_OF_FREEDOM = 2.0

#: 8 自由度谱的默认分段数：每段贡献 2 个实自由度（``2 * n_segments``）。
WEEDON_SEGMENTS = 4

#: 每个 Welch 分段至少需要的样本数。
MIN_SEGMENT_SAMPLES = 32

#: 贝叶斯周期图中信号幅度先验的尺度（以序列标准差为单位）。
AMPLITUDE_PRIOR_SCALE = 1.0

#: 贝叶斯周期图所需的最小频率数。
MIN_FREQUENCIES = 8

#: 贝叶斯周期图所需的最小样本数。
MIN_SERIES_SAMPLES = 16


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class SWAResult:
    """
    SWA 谱背景的输出。

    Attributes:
        swa: 背景功率谱，与输入频率轴等长，量纲与输入功率相同。
        variant: 选中的尾模型：``"linear"``、``"quadratic"`` 或 ``"auto"``
            择优后的实际取值。
        r_squared: 背景与观测对数谱的拟合优度（决定系数）。
    """

    swa: npt.NDArray
    variant: str
    r_squared: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "SWA spectral background\n"
            f"{'=' * 48}\n"
            f"Variant: {self.variant}\n"
            f"R^2 (log spectrum): {self.r_squared:.6f}\n"
            f"Background level (median): {float(np.median(self.swa)):.6g}\n"
            f"Points: {self.swa.size}"
        )


@dataclass(frozen=True)
class FDRResult:
    """
    Benjamini–Hochberg FDR 检验的输出。

    Attributes:
        p_values: 每个频率的 p 值（白噪声零假设下由功率比换算）。
        significant: 布尔掩码，标记被 FDR 判定为显著的频率。
        threshold: 临界 p 值阈值，即被选中集合所在的 BH 临界值
            ``alpha * k / m``；没有任何频率显著时为 0。
    """

    p_values: npt.NDArray
    significant: npt.NDArray
    threshold: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Benjamini-Hochberg FDR\n"
            f"{'=' * 48}\n"
            f"Frequencies tested: {self.p_values.size}\n"
            f"Significant: {int(np.sum(self.significant))}\n"
            f"Critical p-value threshold: {self.threshold:.6g}\n"
            f"Smallest p-value: {float(np.min(self.p_values)):.6g}"
        )


@dataclass(frozen=True)
class WeedonPeriodogramResult:
    """
    8 自由度谱及其 SWA 背景。

    Attributes:
        frequencies: 频率轴（每单位时间的循环频率），步长为分段分辨率。
        power: 功率谱：``n_segments`` 段独立 Lomb–Scargle 周期图的平均，
            白噪声下每点服从 ``chi2(2 * n_segments) / (2 * n_segments)``。
        swa: 由 :func:`swa_background` 得到的背景谱，可直接喂给
            :func:`swa_fdr`。
        degrees_of_freedom: 该谱每个频点的自由度 ``2 * n_segments``，
            必须原样传给 :func:`swa_fdr` 的 ``degrees_of_freedom``。
    """

    frequencies: npt.NDArray
    power: npt.NDArray
    swa: npt.NDArray
    degrees_of_freedom: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Weedon-style 8-dof periodogram\n"
            f"{'=' * 48}\n"
            f"Frequencies: {self.frequencies.size}\n"
            f"Degrees of freedom: {self.degrees_of_freedom:.0f}\n"
            f"Frequency range: {self.frequencies[0]:.6g} - {self.frequencies[-1]:.6g}\n"
            f"Median power: {float(np.median(self.power)):.6g}"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _as_spectrum(
    frequencies: Sequence[float] | npt.NDArray,
    power: Sequence[float] | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """校验频率轴与功率谱，返回一维有限数组。"""
    axis = validate_data_array(
        frequencies, allow_nan=False, allow_inf=False, min_values=MIN_SPECTRUM_POINTS, name="frequencies"
    ).flatten()
    density = validate_data_array(
        power, allow_nan=False, allow_inf=False, min_values=MIN_SPECTRUM_POINTS, name="power"
    ).flatten()
    if axis.size != density.size:
        raise ValidationError(
            _("frequencies and power must have the same length"),
            details={"frequencies": int(axis.size), "power": int(density.size)},
        )
    if np.any(axis <= 0.0):
        raise ValidationError(
            _("frequencies must be strictly positive"),
            details={"minimum": float(np.min(axis))},
        )
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("frequencies must be strictly increasing"),
            details={"parameter": "frequencies"},
        )
    if np.any(density < 0.0):
        raise ValidationError(
            _("power must not be negative"),
            details={"minimum": float(np.min(density))},
        )
    if float(np.sum(density)) <= 0.0:
        raise ValidationError(
            _("the power spectrum is all zero"),
            details={"length": int(density.size)},
        )
    return axis, density


def _prepare_series(
    time: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """校验时间轴与序列，返回按时间排序的有限数组。"""
    axis = validate_data_array(
        time, allow_nan=False, allow_inf=False, min_values=MIN_SERIES_SAMPLES, name="time"
    ).flatten()
    series = validate_data_array(
        values, allow_nan=False, allow_inf=False, min_values=MIN_SERIES_SAMPLES, name="values"
    ).flatten()
    if axis.size != series.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time": int(axis.size), "values": int(series.size)},
        )
    order = np.argsort(axis, kind="stable")
    axis = axis[order]
    series = series[order]
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("time must be strictly increasing"),
            details={"parameter": "time"},
        )
    if float(np.ptp(series)) == 0.0:
        raise ValidationError(
            _("the series is constant, so it has no periodicity to find"),
            details={"n_points": int(series.size)},
        )
    return axis, series


# =============================================================================
# SWA spectral background
# =============================================================================


def _window_average_background(log_power: npt.NDArray, width: int) -> npt.NDArray:
    """
    对数谱的窗口平均背景：窗口中心之间线性插值，两端按邻段斜率线性外推。

    Parameters:
        log_power: 对数谱。
        width: 窗口宽度（点数）。

    Returns:
        npt.NDArray: 与输入等长的对数谱背景。
    """
    count = log_power.size // width
    averages = np.array(
        [float(np.mean(log_power[index * width : (index + 1) * width])) for index in range(count)],
        dtype=float,
    )
    centres = (np.arange(count, dtype=float) + 0.5) * width - 0.5
    index = np.arange(log_power.size, dtype=float)
    background = np.interp(index, centres, averages)
    slope_low = (averages[1] - averages[0]) / (centres[1] - centres[0])
    slope_high = (averages[-1] - averages[-2]) / (centres[-1] - centres[-2])
    low = index < centres[0]
    background[low] = averages[0] + slope_low * (index[low] - centres[0])
    high = index > centres[-1]
    background[high] = averages[-1] + slope_high * (index[high] - centres[-1])
    return background


def _select_window_width(log_power: npt.NDArray) -> int:
    """
    择优选出平均窗口的宽度。

    判据与 Acycle 的 ``specswa`` 一致：先最小化背景中"非降步"的个数（背景
    应当是单调不增的），再最小化 RMSE。宽度必须是奇数，且至少放下三个窗口。
    """
    limit = min(MAX_WINDOW_WIDTH, log_power.size // 3)
    best_width = MIN_WINDOW_WIDTH
    best_key: tuple[int, float] | None = None
    width = MIN_WINDOW_WIDTH
    while width <= limit:
        background = _window_average_background(log_power, width)
        reversals = int(np.sum(np.diff(background) >= 0.0))
        rmse = float(np.sqrt(np.mean((log_power - background) ** 2)))
        key = (reversals, rmse)
        if best_key is None or key < best_key:
            best_key = key
            best_width = width
        width += 2
    return best_width


def _tail_indices(n_points: int) -> npt.NDArray:
    """尾拟合区的下标：从高频端往低频方向取。"""
    count = max(MIN_TAIL_POINTS, round(TAIL_FRACTION * n_points))
    count = min(count, n_points)
    return np.arange(n_points - count, n_points, dtype=int)


def _tail_fit(frequencies: npt.NDArray, log_power: npt.NDArray, degree: int) -> npt.NDArray:
    """尾区的最小二乘多项式拟合（线性或二次）。"""
    try:
        coefficients = np.polyfit(frequencies, log_power, degree)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - 病态输入的兜底
        return np.full_like(log_power, float(np.mean(log_power)))
    return np.polyval(coefficients, frequencies)


def _fit_quality(observed: npt.NDArray, background: npt.NDArray) -> tuple[float, float]:
    """返回背景拟合的 ``(RMSE, R^2)``（在对数谱上计算）。"""
    residual = observed - background
    ss_residual = float(np.sum(residual**2))
    ss_total = float(np.sum((observed - float(np.mean(observed))) ** 2))
    rmse = float(np.sqrt(ss_residual / observed.size))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else 1.0
    return rmse, r_squared


def swa_background(
    frequencies: Sequence[float] | npt.NDArray,
    power: Sequence[float] | npt.NDArray,
    variant: str = "auto",
) -> SWAResult:
    """
    估计谱的 SWA 背景（白噪声背景）。

    背景先由对数谱的窗口平均铺出，高频端再用线性或二次尾拟合替换；两种
    变体按整条谱上的 RMSE（与 R² 等价）择优。

    适用域
    ~~~~~~
    SWA 背景是为**红噪声**设计的非参数方法（Weedon 2019, 2020, 2022）：
    它**不适用于白噪声**（滞后 1 阶自相关为 0），也**不适用于幂律噪声**
    （滞后 1 阶自相关为 1）——这两类噪声分别有自己的零假设模型，不该用
    非参数的谱背景。

    背景不携带自由度信息：它只是一个水平估计。用它做显著性检验时，
    **输入功率谱的自由度必须与 :func:`swa_fdr` 的 ``degrees_of_freedom``
    一致**（见 :func:`weedon_periodogram` 与 :func:`swa_fdr` 的自由度契约）。

    Parameters:
        frequencies: 频率轴（每单位时间的循环频率），严格为正且严格递增，
            至少 :data:`MIN_SPECTRUM_POINTS` 个点。
        power: 与频率轴等长的功率谱，非负且不全为零。
        variant: ``"auto"``（自动择优）、``"linear"``（线性尾拟合）或
            ``"quadratic"``（二次尾拟合）。

    Returns:
        SWAResult: 背景谱、实际使用的变体与拟合优度 R²。

    Raises:
        ValidationError: 输入非法（点数太少、频率非递增、功率为负或全零）
            或 ``variant`` 不在允许的取值之内。

    Examples:
        >>> import numpy as np
        >>> axis = np.linspace(0.05, 1.0, 64)
        >>> result = swa_background(axis, np.full_like(axis, 2.0))
        >>> bool(np.allclose(result.swa, 2.0, rtol=1e-6))
        True
    """
    if variant not in ("auto", "linear", "quadratic"):
        raise ValidationError(
            _("variant must be one of 'auto', 'linear' or 'quadratic'; got {0}").format(variant),
            details={"variant": variant},
        )
    axis, density = _as_spectrum(frequencies, power)

    floor = float(np.max(density)) * 1e-12
    log_power = np.log10(np.maximum(density, floor))

    width = _select_window_width(log_power)
    base = _window_average_background(log_power, width)
    tail = _tail_indices(axis.size)

    backgrounds: dict[str, npt.NDArray] = {}
    for name, degree in (("linear", 1), ("quadratic", 2)):
        candidate = base.copy()
        candidate[tail] = _tail_fit(axis[tail], log_power[tail], degree)
        backgrounds[name] = candidate

    quality = {name: _fit_quality(log_power, background) for name, background in backgrounds.items()}
    if variant == "auto":
        chosen = min(quality, key=lambda name: (quality[name][0], name == "quadratic"))
    else:
        chosen = variant
    rmse, r_squared = quality[chosen]
    logger.info(
        "swa_background: window=%d, variant=%s, rmse=%.6g, r^2=%.6f",
        width,
        chosen,
        rmse,
        r_squared,
    )
    return SWAResult(swa=np.power(10.0, backgrounds[chosen]), variant=chosen, r_squared=r_squared)


# =============================================================================
# False discovery rate
# =============================================================================


def swa_fdr(
    frequencies: Sequence[float] | npt.NDArray,
    power: Sequence[float] | npt.NDArray,
    swa: Sequence[float] | npt.NDArray,
    alpha: float = 0.05,
    degrees_of_freedom: float = FDR_DEGREES_OF_FREEDOM,
) -> FDRResult:
    """
    以 SWA 背景为基准做 Benjamini–Hochberg FDR 检验。

    自由度契约
    ~~~~~~~~~~
    p 值由 ``p = sf(dof * power / swa, dof)`` 得到，**只有在 ``power`` 恰好是
    每频点自由度为 ``dof`` 的周期图时，这个 p 值才成立**：

    * :func:`weedon_periodogram` 给出的谱每点是 ``n_segments`` 段独立
      Lomb–Scargle 周期图的平均，自由度为 ``2 * n_segments``（默认 8），
      应当用 ``degrees_of_freedom=8``；
    * 仓库里现成的、未加窗未平均的 :class:`stratigraphy.spectral_analysis.SpectralAnalyzer`
      周期图每点只有 **2** 个自由度（``chi2(2) / 2``），必须用
      ``degrees_of_freedom=2``。若把 8 自由度的临界值套到 2 自由度的谱上，
      p 值会系统性偏小，实测的"显著频率占比"会从名义的 5% 膨胀到 20% 以上。

    适用域
    ~~~~~~
    SWA 背景是为**红噪声**设计的非参数方法（Weedon 2019, 2020, 2022）：
    它不适用于白噪声（滞后 1 阶自相关为 0），也不适用于幂律噪声
    （滞后 1 阶自相关为 1）。对这两类噪声，应当改用与噪声模型匹配的
    零假设（AR(1) 理论谱等），而不是 SWA 背景。

    Parameters:
        frequencies: 频率轴（每单位时间的循环频率）。
        power: 与频率轴等长的功率谱，**必须是 ``degrees_of_freedom`` 自由度的
            周期图**（见上面的自由度契约）。
        swa: 与频率轴等长的背景谱（通常来自 :func:`swa_background`），必须
            为正。
        alpha: FDR 水平，取值在 ``(0, 1)`` 之间。
        degrees_of_freedom: 输入谱每频点的自由度，默认 8（Acycle 的取值）。

    Returns:
        FDRResult: p 值向量、显著频率的布尔掩码与临界 p 值阈值。

    Raises:
        ValidationError: 输入非法或 ``alpha`` 不在 ``(0, 1)`` 内。

    Notes:
        Acycle 在 BH 之前还用 Hopkins et al. (2002) 的方法按频率分辨率折算
        独立检验的个数；本模块实现的是**未修正的** Benjamini–Hochberg 过程，
        检验族是全部频率，因此给出的显著集合在密集采样的谱上偏保守。

        Benjamini–Hochberg 是"阶梯向上"过程：在零假设下它要么一个都不拒绝，
        要么拒绝一整块，因此**单次实现的拒绝个数**并不是检验水平的良好
        诊断量。逐点的超越率（``p <= alpha`` 的频率占比）才是自由度契约
        直接控制的量。
    """
    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValidationError(
            _("alpha must lie in (0, 1); got {0}").format(alpha),
            details={"alpha": alpha},
        )
    dof = float(degrees_of_freedom)
    if not np.isfinite(dof) or dof <= 0.0:
        raise ValidationError(
            _("degrees_of_freedom must be a positive number; got {0}").format(degrees_of_freedom),
            details={"degrees_of_freedom": degrees_of_freedom},
        )
    axis, density = _as_spectrum(frequencies, power)
    background = validate_data_array(
        swa, allow_nan=False, allow_inf=False, min_values=MIN_SPECTRUM_POINTS, name="swa"
    ).flatten()
    if background.size != axis.size:
        raise ValidationError(
            _("swa must have the same length as the spectrum"),
            details={"swa": int(background.size), "frequencies": int(axis.size)},
        )
    if np.any(background <= 0.0):
        raise ValidationError(
            _("the background spectrum must be strictly positive"),
            details={"minimum": float(np.min(background))},
        )

    ratio = density / background
    p_values = np.asarray(stats.chi2.sf(dof * ratio, dof), dtype=float)

    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    count = sorted_p.size
    critical = level * np.arange(1, count + 1, dtype=float) / count
    below = sorted_p <= critical

    significant = np.zeros(count, dtype=bool)
    threshold = 0.0
    if bool(np.any(below)):
        cutoff = int(np.max(np.nonzero(below)[0])) + 1
        significant[order[:cutoff]] = True
        threshold = float(critical[cutoff - 1])
    logger.info(
        "swa_fdr: tested %d frequencies, %d significant at alpha=%.3g (dof=%.3g)",
        count,
        int(np.sum(significant)),
        level,
        dof,
    )
    return FDRResult(p_values=p_values, significant=significant, threshold=threshold)


# =============================================================================
# Weedon-style 8 degrees-of-freedom periodogram
# =============================================================================


def _segment_bounds(axis: npt.NDArray, n_segments: int) -> list[tuple[int, int]]:
    """把记录按**时间**切成等时长的若干段（不等间距采样也不会让某段特别密）。"""
    start_time = float(axis[0])
    span = float(axis[-1] - axis[0])
    bounds: list[tuple[int, int]] = []
    for index in range(n_segments):
        lower = start_time + span * index / n_segments
        upper = start_time + span * (index + 1) / n_segments
        first = int(np.searchsorted(axis, lower, side="left"))
        last = int(np.searchsorted(axis, upper, side="left"))
        if index == n_segments - 1:
            last = axis.size
        if last <= first:
            last = min(first + 1, axis.size)
        bounds.append((first, last))
    return bounds


def weedon_periodogram(
    time: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
    n_frequencies: int = 512,
    frequency_range: tuple[float, float] | None = None,
    n_segments: int = WEEDON_SEGMENTS,
    detrend: bool = True,
) -> WeedonPeriodogramResult:
    """
    产出可直接进入 FDR 流程的**多自由度**谱及其 SWA 背景。

    构造
    ~~~~
    1. 线性去趋势；
    2. 把记录按时间切成 ``n_segments`` 段（互不重叠，因此各段的谱估计互相
       独立）；
    3. 每段做一次 Lomb–Scargle 周期图，按**该段自身的方差**归一化（于是每
       点的分布是 ``chi2(2) / 2``）；
    4. 把各段的谱平均。每点因此是 ``n_segments`` 个独立的 ``chi2(2) / 2``
       的平均，服从 ``chi2(2 * n_segments) / (2 * n_segments)``，自由度
       ``2 * n_segments``（默认 8）。

    这正是 Weedon 的 ``specswafdr.m`` 所假定的自由度（该程序用三次 Hanning
    加窗得到 8 自由度）。本模块不翻译其 MATLAB 源码，而是用仓库已有的
    Lomb–Scargle 约定（与 :mod:`stratigraphy.coherence` 的 Welch 分段一致）
    独立实现同一条契约。

    频率栅格
    ~~~~~~~~
    步长取**分段分辨率**（瑞利频率 ``1 / 分段时长``），因此相邻频点近似
    互相独立，BH 过程把每个频点当作一次独立检验才站得住。代价是分辨率
    比整条记录的瑞利分辨率粗 ``n_segments`` 倍。

    Returns:
        WeedonPeriodogramResult: 频率、功率、SWA 背景与自由度。背景与自由度
        都要原样传给 :func:`swa_fdr`。

    Raises:
        ValidationError: 输入非法、分段数或频率点数越界、记录放不下所要求
            的分段。

    Examples:
        >>> import numpy as np
        >>> axis = np.arange(2048.0)
        >>> series = np.sin(2.0 * np.pi * axis / 32.0)
        >>> result = weedon_periodogram(axis, series)
        >>> result.degrees_of_freedom
        8.0
    """
    segments = int(n_segments)
    if segments < 2:
        raise ValidationError(
            _("n_segments must be at least 2; got {0}").format(n_segments),
            details={"n_segments": n_segments},
        )
    budget = int(n_frequencies)
    if budget < MIN_FREQUENCIES:
        raise ValidationError(
            _("n_frequencies must be at least {0}; got {1}").format(MIN_FREQUENCIES, n_frequencies),
            details={"n_frequencies": n_frequencies},
        )

    axis, series = _prepare_series(time, values)
    if detrend:
        series = _linear_detrend(series)
    bounds = _segment_bounds(axis, segments)
    for first, last in bounds:
        if last - first < MIN_SEGMENT_SAMPLES:
            raise ValidationError(
                _("each segment needs at least {0} samples; got {1}").format(MIN_SEGMENT_SAMPLES, last - first),
                details={"n_segments": segments, "segment_samples": last - first},
            )

    span = float(axis[-1] - axis[0])
    step = float(np.median(np.diff(axis)))
    nyquist = 0.5 / step
    resolution = 1.0 / (span / segments)
    if frequency_range is None:
        available = int(np.floor((nyquist - resolution) / resolution)) + 1
        count = max(MIN_FREQUENCIES, min(budget, max(available, MIN_FREQUENCIES)))
        frequencies = resolution + np.arange(count, dtype=float) * resolution
    else:
        low, high = (float(frequency_range[0]), float(frequency_range[1]))
        if not (0.0 < low < high):
            raise ValidationError(
                _("frequency_range must satisfy 0 < minimum < maximum; got {0}").format(frequency_range),
                details={"frequency_range": frequency_range},
            )
        frequencies = np.linspace(low, high, budget)

    analyzer = SpectralAnalyzer()
    segment_powers = []
    for first, last in bounds:
        segment_power = analyzer._lomb_scargle(axis[first:last], series[first:last], frequencies)
        segment_powers.append(np.asarray(segment_power, dtype=float))
    power = np.mean(segment_powers, axis=0)

    background = swa_background(frequencies, power).swa
    dof = 2.0 * segments
    logger.info(
        "weedon_periodogram: %d segments, dof=%.0f, %d frequencies",
        segments,
        dof,
        frequencies.size,
    )
    return WeedonPeriodogramResult(
        frequencies=frequencies,
        power=power,
        swa=background,
        degrees_of_freedom=dof,
    )


# =============================================================================
# Bayesian periodogram
# =============================================================================


def _frequency_grid(
    axis: npt.NDArray,
    n_frequencies: int,
    frequency_range: tuple[float, float] | None,
) -> npt.NDArray:
    """贝叶斯周期图的频率栅格。"""
    span = float(axis[-1] - axis[0])
    step = float(np.median(np.diff(axis)))
    nyquist = 0.5 / step
    if frequency_range is None:
        minimum = 1.0 / span
        maximum = nyquist
        if maximum <= minimum:
            raise ValidationError(
                _("the record is too short to resolve a single cycle"),
                details={"span": span, "nyquist": nyquist},
            )
        return np.geomspace(minimum, maximum, n_frequencies)
    low, high = (float(frequency_range[0]), float(frequency_range[1]))
    if not (0.0 < low < high):
        raise ValidationError(
            _("frequency_range must satisfy 0 < minimum < maximum; got {0}").format(frequency_range),
            details={"frequency_range": frequency_range},
        )
    return np.linspace(low, high, n_frequencies)


def bayesian_periodogram(
    time: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
    n_frequencies: int = 500,
    frequency_range: tuple[float, float] | None = None,
    n_iterations: int = 200,
    random_seed: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    贝叶斯周期图：Lomb–Scargle 周期图 + 贝叶斯因子给出各频率的后验概率。

    序列先做线性去趋势，再按 AR(1) 系数预白化（使红噪声背景变白，白噪声零
    假设才成立），最后标准化到单位方差。周期图给出归一化功率 ``u(f)``，每个
    频率的贝叶斯因子由幅度先验（半正态）下的似然比积分**蒙特卡洛**估计，
    归一化后即为后验概率。

    Parameters:
        time: 时间轴（每单位时间的循环频率与之对应），严格递增。
        values: 序列值。
        n_frequencies: 频率栅格的点数。
        frequency_range: ``(最小频率, 最大频率)``；缺省时从瑞利频率
            ``1 / 记录跨度`` 到奈奎斯特频率按几何间隔取点。
        n_iterations: 贝叶斯因子蒙特卡洛估计的抽样次数。
        random_seed: 随机种子；给出时结果完全可复现，缺省时用
            :func:`utils.statistics_core.make_rng` 现场抽取并发出
            ``RuntimeWarning``。

    Returns:
        tuple: ``(频率, 功率, 后验概率)``。功率的尺度是"白噪声均值为 1"，
        后验概率在频率栅格上求和为 1。

    Raises:
        ValidationError: 输入非法、点数太少或频率栅格参数越界。

    Examples:
        >>> import numpy as np
        >>> axis = np.arange(400.0)
        >>> series = np.sin(2.0 * np.pi * axis / 20.0)
        >>> frequencies, power, posterior = bayesian_periodogram(
        ...     axis, series, n_frequencies=64, random_seed=3
        ... )
        >>> bool(posterior[int(np.argmax(posterior))] > 0.5)
        True
    """
    iterations = int(n_iterations)
    if iterations < 1:
        raise ValidationError(
            _("n_iterations must be at least 1; got {0}").format(n_iterations),
            details={"n_iterations": n_iterations},
        )
    budget = int(n_frequencies)
    if budget < MIN_FREQUENCIES:
        raise ValidationError(
            _("n_frequencies must be at least {0}; got {1}").format(MIN_FREQUENCIES, n_frequencies),
            details={"n_frequencies": n_frequencies},
        )

    axis, series = _prepare_series(time, values)
    frequencies = _frequency_grid(axis, budget, frequency_range)

    detrended = _linear_detrend(series)
    rho = float(estimate_ar1_rho(detrended))
    if rho > 0.0:
        # 预白化：x(t) - rho * x(t+1)，时间轴同步截断。
        whitened = detrended[:-1] - rho * detrended[1:]
        sample_times = axis[:-1]
    else:
        whitened = detrended
        sample_times = axis
    variance = float(np.var(whitened))
    if variance <= 0.0 or whitened.size < MIN_SERIES_SAMPLES:
        raise ValidationError(
            _("the series carries no variance after detrending and pre-whitening"),
            details={"n_points": int(whitened.size)},
        )
    normalized = (whitened - float(np.mean(whitened))) / np.sqrt(variance)

    analyzer = SpectralAnalyzer()
    power = np.asarray(analyzer._lomb_scargle(sample_times, normalized, frequencies), dtype=float)

    rng = make_rng(random_seed, context="Bayesian periodogram")
    amplitudes = np.abs(rng.normal(0.0, AMPLITUDE_PRIOR_SCALE, size=iterations))
    signal_power = normalized.size * amplitudes**2 / 4.0

    # 似然比 p1(u | z) / p0(u) = exp(z) * 1[u >= z]。
    log_ratios = np.full((iterations, frequencies.size), -np.inf, dtype=float)
    for index, signal in enumerate(signal_power):
        active = power >= signal
        log_ratios[index, active] = signal
    log_bayes = logsumexp(log_ratios, axis=0) - np.log(iterations)
    posterior = np.exp(log_bayes - logsumexp(log_bayes))

    peak = int(np.argmax(posterior))
    logger.info(
        "bayesian_periodogram: peak frequency=%.6g, posterior=%.4f, rho=%.3f",
        frequencies[peak],
        posterior[peak],
        rho,
    )
    return frequencies, power, posterior
