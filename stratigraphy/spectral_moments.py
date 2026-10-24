# =============================================================================
# FILE: stratigraphy/spectral_moments.py
# =============================================================================
"""
谱矩（spectral moments）、谱形描述量与 AR(1) 持续性估计。

本模块提供三组互相配合的函数：

* :func:`spectral_moments` —— 由功率谱计算谱矩
  ``m_k = ∫ f^k P(f) df``（k = 0, 1, ..., ``max_order``），并由矩导出谱质心、
  谱宽、谱偏度与谱峰度；同时给出该谱的 AR(1) 拟合系数 ``ar1_rho`` 与保守
  上界 ``min_rho``。
* :func:`estimate_ar1_rho` —— AR(1) 的滞后 1 阶自相关（Yule–Walker 估计）。
* :func:`minimum_rho` —— “能解释观测序列的最小 AR(1) 系数”。

频率轴约定
~~~~~~~~~~
* 频率必须是**每单位时间**的循环频率（cycles per unit time），而不是每样本
  的归一化频率；谱矩的积分因此具有“每单位时间”的量纲。
* 频率必须严格为正且严格递增（本模块不处理 DC 分量）。
* 功率 ``P(f)`` 是功率谱密度（密度 / 频率），因此 ``∫ P df`` 得到方差。

谱形描述量
~~~~~~~~~~
由谱矩导出的四个描述量都基于频率分布的（标准化）中心矩：

* 谱质心 ``f_c = m_1 / m_0``：功率加权的平均频率；
* 谱宽 ``sigma_f = sqrt(m_2 / m_0 - f_c^2)``：频率的标准差；
* 谱偏度 ``gamma = mu_3 / sigma_f^3``：频率分布的偏度；
* 谱峰度 ``kappa = mu_4 / sigma_f^4 - 3``：频率分布的**超额**峰度
  （与 ``scipy.stats.kurtosis`` 的默认约定一致；正态分布取 0）。

“最小 rho”的含义
~~~~~~~~~~~~~~~~
:func:`minimum_rho` 以及 :func:`spectral_moments` 返回的 ``min_rho`` 都取
**保守上界**：在给定置信水平下，数据允许的 AR(1) 系数的最大值。它不会低于
数据真实的 AR(1) 持续性，因此用它做 AR(1) 滤波时，红噪声一定被解释掉
（宁可略微过一点，也不要把红噪声留在序列里）。与之相对的 ``ar1_rho`` 是点估计。

随机过程约定
~~~~~~~~~~~~
* 所有随机函数都接受 ``random_seed``。给定种子时结果完全可复现；不给种子时
  调用 ``utils.statistics_core.make_rng(None, context=...)``，它会发出
  ``RuntimeWarning`` 提示结果不可复现——这是仓库内其它随机分析沿用的同一约定。
* 生成器都是独立的 ``numpy.random.Generator``，既不读取也不推进全局
  ``numpy.random`` 流，因此不会扰动调用方自己的随机数据。

参考文献
~~~~~~~~
* Priestley, M. B. (1981). Spectral Analysis and its Applications.
  Academic Press.（谱矩与标准化谱形描述量的经典定义）
* Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and
  Control. Holden & Day.（AR(1) 的 Yule–Walker 估计与创新理论）
* Mudelsee, M. (2002). AR(1) 过程的谱估计（tau 函数方法）与滞后 1 阶自相关
  估计。（Acycle 的 ``rhoAR1.m`` 采用同一估计量）
* Sinnesael, M., Zivanovic, M., De Vleeschouwer, D. & Claeys, P. (2018).
  Spectral Moments in Cyclostratigraphy: Advantages and Disadvantages compared
  to more classic Approaches. Paleoceanography and Paleoclimatology 33, 493-510.
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. Computers & Geosciences 127, 12-22.

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考其公开的行为约定
（频率轴按每单位时间、AR(1) 理论谱的形状、rho 的网格搜索加局部精化），
未翻译或复制其 MATLAB 源码。

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
from scipy.signal import lfilter

from config.i18n import _
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 谱矩阶数的上限（数值稳定性与内存考虑）。
MAX_MOMENT_ORDER = 8

#: 自助法（参数自助）重复次数，用于标定 AR(1) 系数估计量的抽样标准差。
BOOTSTRAP_REPLICATES = 200

#: AR(1) 系数搜索与拟合时允许的最大取值（留出数值余量）。
MAX_AR1_RHO = 0.999

#: :func:`spectral_moments` 至少需要的频率点数。
MIN_SPECTRUM_BINS = 4

#: :func:`estimate_ar1_rho` 至少需要的样本点数。
MIN_AR1_SAMPLES = 4

#: :func:`minimum_rho` 至少需要的样本点数（检验与自助法需要足够长度）。
MIN_ROO_SAMPLES = 16


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class SpectralMomentsResult:
    """
    谱矩计算的输出。

    Attributes:
        moments: 谱矩 ``m_0, m_1, ..., m_max_order``，其中
            ``m_k = ∫ f^k P(f) df``。
        orders: 与 ``moments`` 对应的矩阶（``0 .. max_order``）。
        centroid: 谱质心（功率加权的平均频率）。
        bandwidth: 谱宽（频率的标准差）。
        skewness: 谱偏度；矩阶不足时为 ``nan``。
        kurtosis: 谱峰度（超额峰度）；矩阶不足时为 ``nan``。
        ar1_rho: 观测谱的 AR(1) 拟合系数（点估计）。
        min_rho: AR(1) 系数的保守上界，用于滤波时保证红噪声被完全解释。
    """

    moments: npt.NDArray
    orders: npt.NDArray
    centroid: float
    bandwidth: float
    skewness: float
    kurtosis: float
    ar1_rho: float
    min_rho: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Spectral moments\n"
            f"{'=' * 48}\n"
            f"Centroid: {self.centroid:.6g}\n"
            f"Bandwidth: {self.bandwidth:.6g}\n"
            f"Skewness: {self.skewness:.6g}\n"
            f"Kurtosis: {self.kurtosis:.6g}\n"
            f"AR(1) rho: {self.ar1_rho:.6g}\n"
            f"Minimum rho: {self.min_rho:.6g}"
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


def _yule_walker_rho(values: npt.NDArray) -> float:
    """
    AR(1) 的滞后 1 阶自相关（Yule–Walker / Mudelsee 2002 形式）。

    估计量为 ``sum(x_t * x_{t-1}) / sum(x_{t-1}^2)``，其中 ``x`` 已去均值。
    """
    centred = values - values.mean()
    lagged = centred[:-1]
    leading = centred[1:]
    denominator = float(np.dot(lagged, lagged))
    if denominator <= 0.0:
        raise ValidationError(
            _("a constant series has no defined AR(1) coefficient"),
            details={"length": int(values.size)},
        )
    rho = float(np.dot(leading, lagged) / denominator)
    return float(np.clip(rho, -1.0, 1.0))


# =============================================================================
# Spectral moments
# =============================================================================


def _trapezoid_moments(power: npt.NDArray, frequencies: npt.NDArray, max_order: int) -> npt.NDArray:
    """用梯形积分计算谱矩 ``m_k = ∫ f^k P(f) df``。"""
    width = np.diff(frequencies)
    moments = np.empty(max_order + 1, dtype=float)
    for order in range(max_order + 1):
        weighted = (frequencies**order) * power
        moments[order] = float(np.sum(0.5 * (weighted[1:] + weighted[:-1]) * width))
    return moments


def _central_moments(power: npt.NDArray, frequencies: npt.NDArray, centre: float, max_order: int) -> npt.NDArray:
    """用梯形积分计算频率分布关于 ``centre`` 的中心矩。"""
    width = np.diff(frequencies)
    centred = frequencies - centre
    moments = np.empty(max_order + 1, dtype=float)
    for order in range(max_order + 1):
        weighted = (centred**order) * power
        moments[order] = float(np.sum(0.5 * (weighted[1:] + weighted[:-1]) * width))
    return moments


def _ar1_shape(frequencies: npt.NDArray, rho: float) -> npt.NDArray:
    """
    AR(1) 理论谱的形状函数（归一化到 DC 处）。

    频率轴按 DC 到 Nyquist 归一化：``u = f / f_nyquist``，于是 AR(1) 的
    功率谱形状为 ``(1 - rho^2) / (1 - 2 rho cos(pi u) + rho^2)``。
    """
    normalised = frequencies / frequencies[-1]
    denominator = 1.0 - 2.0 * rho * np.cos(np.pi * normalised) + rho * rho
    return (1.0 - rho * rho) / denominator


def _ar1_log_misfit(log_power: npt.NDArray, frequencies: npt.NDArray, rho: float) -> tuple[float, float]:
    """
    给定 rho 的 AR(1) 谱在 log 空间与观测谱的失配。

    Returns:
        tuple: ``(失配平方和, 最优尺度 s0 的自然对数)``。
    """
    log_shape = np.log(_ar1_shape(frequencies, rho))
    residual = log_power - log_shape
    offset = float(residual.mean())
    distance = float(np.sum((residual - offset) ** 2))
    return distance, offset


def _fit_ar1_spectrum(
    power: npt.NDArray, frequencies: npt.NDArray, *, n_grid: int = 64, n_refine: int = 3
) -> tuple[float, float, float]:
    """
    在 log 空间把 AR(1) 理论谱拟合到观测谱（网格搜索 + 局部精化）。

    Returns:
        tuple: ``(最佳 rho, 对应的尺度 s0, 最小失配平方和)``。
    """
    floor = float(np.max(power)) * 1e-12
    log_power = np.log(np.maximum(power, floor))
    grid = np.linspace(0.0, MAX_AR1_RHO, n_grid)
    best_rho = 0.0
    best_distance = np.inf
    best_offset = 0.0
    for _ in range(n_refine + 1):
        for rho in grid:
            distance, offset = _ar1_log_misfit(log_power, frequencies, float(rho))
            if distance < best_distance:
                best_distance = distance
                best_offset = offset
                best_rho = float(rho)
        step = float(grid[1] - grid[0])
        grid = np.linspace(max(0.0, best_rho - step), min(MAX_AR1_RHO, best_rho + step), n_grid)
    return best_rho, float(np.exp(best_offset)), float(best_distance)


def _spectral_rho_upper_limit(power: npt.NDArray, frequencies: npt.NDArray, distance_min: float, alpha: float) -> float:
    """
    由谱拟合的失配曲线给出 AR(1) 系数的可接受区间上界。

    以 log 空间失配的嵌套模型 F 检验近似：若最优拟合的失配为 ``D_min``，
    则系数 rho 可接受当且仅当 ``D(rho) <= D_min * F(1, n - 2)``。
    """
    floor = float(np.max(power)) * 1e-12
    log_power = np.log(np.maximum(power, floor))
    grid = np.linspace(0.0, MAX_AR1_RHO, 200)
    distances = np.array([_ar1_log_misfit(log_power, frequencies, float(rho))[0] for rho in grid])
    degrees_of_freedom = max(int(power.size) - 2, 1)
    threshold = distance_min * float(stats.f.ppf(1.0 - alpha, 1, degrees_of_freedom))
    acceptable = grid[distances <= threshold]
    if acceptable.size == 0:
        return float(MAX_AR1_RHO)
    return float(acceptable.max())


def spectral_moments(
    power: Sequence[float] | npt.NDArray,
    frequencies: Sequence[float] | npt.NDArray,
    max_order: int = 4,
) -> SpectralMomentsResult:
    """
    由功率谱计算谱矩与谱形描述量。

    谱矩定义为 ``m_k = ∫ f^k P(f) df``（k = 0, 1, ..., ``max_order``），
    频率轴必须以**每单位时间**给出（不是每样本），因此谱质心与谱宽都带有
    “每单位时间”的量纲。

    Parameters:
        power: 功率谱密度（密度 / 频率），非负，一维。
        frequencies: 与 ``power`` 等长的频率轴（每单位时间的循环频率），
            严格为正且严格递增。
        max_order: 最高谱矩阶数；谱偏度需要 3 阶、谱峰度需要 4 阶矩，
            因此 ``max_order < 4`` 时相应描述量为 ``nan``。

    Returns:
        SpectralMomentsResult: 谱矩、谱形描述量以及谱的 AR(1) 系数。
        其中 ``min_rho`` 是按 5% 显著性水平给出的 AR(1) 系数保守上界，
        ``ar1_rho`` 是同一条谱的 AR(1) 点估计。

    Raises:
        ValidationError: 功率或频率非法（全零谱、负功率、非正频率、频率
            非递增、点数太少）或 ``max_order`` 越界。

    Examples:
        >>> import numpy as np
        >>> f = np.linspace(0.01, 1.0, 200)
        >>> p = np.full_like(f, 0.5)
        >>> result = spectral_moments(p, f)
        >>> result.centroid > 0.0
        True
    """
    order = int(max_order)
    if order != max_order or order < 1 or order > MAX_MOMENT_ORDER:
        raise ValidationError(
            _("max_order must be an integer between 1 and {0}").format(MAX_MOMENT_ORDER),
            details={"max_order": max_order},
        )

    density = _as_vector(power, "power", minimum_size=MIN_SPECTRUM_BINS)
    axis = _as_vector(frequencies, "frequencies", minimum_size=MIN_SPECTRUM_BINS)
    if density.size != axis.size:
        raise ValidationError(
            _("power and frequencies must have the same length"),
            details={"power_length": int(density.size), "frequency_length": int(axis.size)},
        )
    if np.any(density < 0.0):
        raise ValidationError(
            _("power must not be negative"),
            details={"minimum": float(density.min())},
        )
    if np.any(axis <= 0.0):
        raise ValidationError(
            _("frequencies must be strictly positive"),
            details={"minimum": float(axis.min())},
        )
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("frequencies must be strictly increasing"),
            details={"parameter": "frequencies"},
        )
    if float(np.sum(density)) <= 0.0:
        raise ValidationError(
            _("the power spectrum is all zero"),
            details={"length": int(density.size)},
        )

    moments = _trapezoid_moments(density, axis, order)
    total_power = float(moments[0])
    centroid = float(moments[1] / total_power)

    centred = _central_moments(density, axis, centroid, order)
    variance = max(float(centred[2]) / total_power, 0.0)
    bandwidth = float(np.sqrt(variance))

    if bandwidth > 0.0 and order >= 3:
        skewness = float((centred[3] / total_power) / bandwidth**3)
    else:
        skewness = float("nan")
    if bandwidth > 0.0 and order >= 4:
        kurtosis = float((centred[4] / total_power) / bandwidth**4 - 3.0)
    else:
        kurtosis = float("nan")

    rho_best, _s0, distance_min = _fit_ar1_spectrum(density, axis)
    rho_upper = _spectral_rho_upper_limit(density, axis, distance_min, 0.05)

    return SpectralMomentsResult(
        moments=moments,
        orders=np.arange(order + 1, dtype=int),
        centroid=centroid,
        bandwidth=bandwidth,
        skewness=skewness,
        kurtosis=kurtosis,
        ar1_rho=rho_best,
        min_rho=rho_upper,
    )


# =============================================================================
# AR(1) persistence
# =============================================================================


def estimate_ar1_rho(values: Sequence[float] | npt.NDArray) -> float:
    """
    估计 AR(1) 的滞后 1 阶自相关。

    估计量为 Yule–Walker 形式（Mudelsee 2002）::

        rho = sum(x_t * x_{t-1}) / sum(x_{t-1}^2)

    其中 ``x`` 已去均值。对平稳 AR(1) 过程，该估计量一致且几乎无偏。

    Parameters:
        values: 一维有限序列，至少 4 个点。

    Returns:
        float: 滞后 1 阶自相关，取值在 ``[-1, 1]``。

    Raises:
        ValidationError: 输入不是一维有限数组、点数太少或序列为常数。

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(7)
        >>> series = np.zeros(500)
        >>> for index in range(1, series.size):
        ...     series[index] = 0.5 * series[index - 1] + rng.normal()
        >>> abs(estimate_ar1_rho(series) - 0.5) < 0.15
        True
    """
    array = _as_vector(values, "values", minimum_size=MIN_AR1_SAMPLES)
    return _yule_walker_rho(array)


def _bootstrap_ar1_sd(
    values: npt.NDArray, rho: float, rng: np.random.Generator, replicates: int = BOOTSTRAP_REPLICATES
) -> float:
    """
    参数自助法：标定 Yule–Walker 估计量的抽样标准差。

    以 ``rho`` 为真值、以观测方差为尺度模拟 AR(1) 序列，重复计算估计量，
    其标准差即为 ``rho`` 的抽样标准差。
    """
    size = int(values.size)
    variance = float(np.var(values))
    bounded = float(np.clip(rho, -0.99, 0.99))
    innovation_sd = float(np.sqrt(max(variance * (1.0 - bounded * bounded), 0.0)))
    estimates: list[float] = []
    chunk = max(1, int(1.0e6 / max(size, 1)))
    remaining = int(replicates)
    while remaining > 0:
        batch = min(chunk, remaining)
        remaining -= batch
        innovations = rng.normal(0.0, innovation_sd, size=(batch, size))
        series = lfilter([1.0], [1.0, -bounded], innovations, axis=1)
        series[:, 0] = rng.normal(0.0, float(np.sqrt(max(variance, 0.0))), size=batch)
        centred = series - series.mean(axis=1, keepdims=True)
        numerator = np.sum(centred[:, 1:] * centred[:, :-1], axis=1)
        denominator = np.sum(centred[:, :-1] ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimates.extend(np.asarray(numerator / denominator, dtype=float).tolist())
    finite = [value for value in estimates if np.isfinite(value)]
    if len(finite) < 2:
        return float(np.sqrt(max(1.0 - bounded * bounded, 0.0)) / np.sqrt(size))
    return float(np.std(finite, ddof=1))


def _residual_redness_statistic(values: npt.NDArray, rho: float) -> float:
    """
    AR(1) 滤波残差的标准化滞后 1 阶自相关。

    残差为 ``r_t = x_t - rho * x_{t-1}``；统计量 ``sqrt(n) * r1 / r0`` 在
    残差为白噪声时服从标准正态分布，显著为正即表示残差仍然偏红。
    """
    centred = values - values.mean()
    residual = centred[1:] - rho * centred[:-1]
    residual = residual - residual.mean()
    denominator = float(np.dot(residual, residual))
    if denominator <= 0.0:
        return float("-inf")
    lag_one = float(np.dot(residual[1:], residual[:-1]))
    return float(np.sqrt(residual.size) * lag_one / denominator)


def _redness_critical_value(
    length: int, alpha: float, rng: np.random.Generator, replicates: int = BOOTSTRAP_REPLICATES
) -> float:
    """
    残差偏红统计量的零假设临界值（单侧，水平 ``alpha``）。

    零假设是“残差为白噪声”，其分布与候选 rho 无关，因此只需标定一次。
    """
    size = max(int(length), 2)
    chunk = max(1, int(1.0e6 / size))
    values: list[float] = []
    remaining = int(replicates)
    while remaining > 0:
        batch = min(chunk, remaining)
        remaining -= batch
        draws = rng.normal(0.0, 1.0, size=(batch, size))
        for row in draws:
            values.append(_residual_redness_statistic(row, 0.0))
    if not values:
        return float(stats.norm.ppf(1.0 - alpha))
    return float(np.quantile(np.asarray(values, dtype=float), 1.0 - alpha))


def minimum_rho(
    values: Sequence[float] | npt.NDArray,
    n_candidates: int = 200,
    alpha: float = 0.05,
    random_seed: int | None = None,
) -> float:
    """
    “能解释观测序列的最小 AR(1) 系数”。

    做法（单边、保守）：

    1. 用 Yule–Walker 估计量得到 AR(1) 系数点估计 ``phi_hat``；
    2. 用参数自助法标定该估计量的抽样标准差 ``sd``，取
       ``rho_ref = phi_hat + z_{1 - alpha/2} * sd``——即 AR(1) 持续性的
       **保守上界**（在置信水平 ``1 - alpha`` 下数据允许的最大系数）；
    3. 在 ``[0, 0.999]`` 上取 ``n_candidates`` 个候选 rho，对每个候选做
       单边检验：把候选代入 AR(1) 滤波后，残差若仍显著偏红（备择假设：
       数据比候选模型更红），该候选即不可接受；
    4. 在“检验通过”且“不低于保守上界”的候选中取最小者。

    之所以取上界而不是下界：AR(1) 滤波的作用是**清除**红噪声，系数取得太小
    会把红噪声留在序列里（残差仍偏红），取得略大一点只是多滤掉一点点高频，
    因此按上界取是保守的选择。

    Parameters:
        values: 一维有限序列，至少 16 个点。
        n_candidates: 候选 rho 的个数，至少 2。
        alpha: 显著性水平，取值在 ``(0, 1)``。
        random_seed: 随机种子；``None`` 时结果不可复现，
            :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。

    Returns:
        float: 保守的最小 AR(1) 系数，取值在 ``[0, 0.999]``。

    Raises:
        ValidationError: 输入非法、``n_candidates`` 或 ``alpha`` 越界。

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(11)
        >>> series = np.zeros(2000)
        >>> for index in range(1, series.size):
        ...     series[index] = 0.6 * series[index - 1] + rng.normal()
        >>> minimum_rho(series, random_seed=3) >= 0.6
        True
    """
    array = _as_vector(values, "values", minimum_size=MIN_ROO_SAMPLES)
    candidates = int(n_candidates)
    if candidates != n_candidates or candidates < 2:
        raise ValidationError(
            _("n_candidates must be an integer of at least 2"),
            details={"n_candidates": n_candidates},
        )
    if not (0.0 < float(alpha) < 1.0):
        raise ValidationError(
            _("alpha must lie strictly between 0 and 1"),
            details={"alpha": float(alpha)},
        )

    rng = make_rng(random_seed, context="Minimum AR(1) rho")
    phi_hat = _yule_walker_rho(array)
    standard_error = _bootstrap_ar1_sd(array, phi_hat, rng)
    margin = float(stats.norm.ppf(1.0 - float(alpha) / 2.0)) * standard_error
    rho_reference = float(np.clip(phi_hat + margin, 0.0, MAX_AR1_RHO))

    critical = _redness_critical_value(array.size, float(alpha), rng)
    grid = np.linspace(0.0, MAX_AR1_RHO, candidates)
    acceptable = [float(rho) for rho in grid if _residual_redness_statistic(array, float(rho)) <= critical]
    selected = [rho for rho in acceptable if rho >= rho_reference]
    if not selected:
        selected = [float(rho) for rho in grid if rho >= rho_reference]
    if not selected:
        return float(MAX_AR1_RHO)
    return float(min(selected))
