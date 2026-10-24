# =============================================================================
# FILE: stratigraphy/detrending.py
# =============================================================================
"""
去趋势（detrending）：把一条记录里的慢变趋势与高频成分分开。

频谱分析之前通常要先去掉趋势，否则一条缓慢上升的曲线会把能量漏到最低
频率上，污染红噪声拟合和周期显著性判断（Mann & Lees 1996）。Acycle
（https://github.com/mingsongli/acycle）把这件事做成了一个带 9 种趋势模型
的工具，本模块实现这 9 种模型的**完整选项集**：

============  =========================================================
方法          趋势是怎么算出来的
============  =========================================================
``mean``      常数趋势，取整条记录的均值
``linear``    1 阶最小二乘多项式
``poly2``     2 阶最小二乘多项式
``poly``      N 阶最小二乘多项式（N 由 ``order`` 给出，默认 3）
``lowess``    局部线性加权回归（Cleveland 1979）， tricube 权重
``rlowess``   稳健版 LOWESS（对离群点用 bisquare 重加权）
``loess``     局部二次加权回归
``rloess``    稳健版 LOESS
``eemd``      EEMD 分解的**残差**（最后一个 IMF）作趋势
============  =========================================================

与 Acycle 的关系
~~~~~~~~~~~~~~~~
Acycle 的 ``code/guicode/detrending.m`` 里 ``updateTrendFigure()`` 对每个
被勾选的模型做同一件事：算出趋势 ``l``，然后存 ``detr.<model> = datay - l``。
本模块沿用这个约定：

    **去趋势 = 原序列 - 趋势**（逐点成立，见 :func:`detrend_series`）。

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：多项式部分用
``numpy.polynomial`` 的最小二乘，局部回归部分按 Cleveland (1979) 与
``locfit`` 的公开算法自行编写，EEMD 部分直接复用本仓库
:mod:`stratigraphy.emd`。没有翻译或复制 Acycle 的 MATLAB 源码。

窗口语义（重要）
~~~~~~~~~~~~~~~~
Acycle 的去趋势界面把平滑窗口存成 ``smooth_win``（默认 **0.35**），并同时
显示它在**数据坐标**里的长度 ``prewhiten_win = xrange * smooth_win``，其中
``xrange = max(datax) - min(datax)``。也就是说，Acycle 的窗口长度是**记录
时间跨度的百分比**，不是样本数。

本模块的 ``span`` 参数采用同一口径：

    局部窗口的**总宽度** = ``span * (t_max - t_min)``（时间坐标里的长度），
    半宽是它的一半；每个样本点用它左右各半宽以内的数据做加权回归。

因此 ``span=0.35`` 与 Acycle 的默认值含义一致：窗口覆盖整条记录时间跨度的
35 %。均匀采样时它大致等价于"35 % 的样本点"，但时间轴不均匀时（本项目的
常态）只有时间口径才是有意义的，所以局部回归直接在时间坐标上做，而不是
用 ``statsmodels`` 那种"取最近的若干个点"的窗口。

（Acycle 在这一点上自身并不完全一致：它的界面把窗口同时显示成"数据坐标
里的长度"（``prewhiten_win = xrange * smooth_win``）和"百分比"，但它调用
MATLAB ``smooth`` 时传的是那个百分比，而 MATLAB 对**向量**输入把 ``span``
解释成**点数**的比例。本模块按 Acycle 界面与任务规格所声明的口径实现，
即"记录跨度的百分比"，并在这里写明，免得下游误以为它是点数。）

与本仓库其它平滑代码的关系
~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`stratigraphy.smoothing` 里的移动平均把跨度定义成**样本数**，而且是
移动平均而不是局部回归；本模块要的是局部多项式回归（LOWESS / LOESS）以及
Acycle 那套"窗口 = 记录跨度的百分比"的口径，两者都不是同一个东西，所以
这里没有复用它，也没有重复它的代码。

参考文献
~~~~~~~~
* Cleveland, W.S. 1979. Robust locally weighted regression and smoothing
  scatterplots. *Journal of the American Statistical Association* 74: 829-836.
* Cleveland, W.S., Cleveland, W.S., McRae, J.E. & Terpenning, I.J. 1990.
  STL: A seasonal-trend decomposition procedure based on loess.
  *Journal of Official Statistics* 6: 3-73.
* Wu, Z. & Huang, N.E. 2009. A study on the characteristics of white noise
  using the empirical mode decomposition method. *Advances in Data Science
  and Engineering* 1: 1-41.
* Mann, M.E. & Lees, J.M. 1996. Significance tests for fossil cycles.
  *Paleoceanography* 11: 545-567.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from stratigraphy.emd import eemd
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 支持的趋势模型，顺序与 Acycle 去趋势界面的模型列表一致。
DETRENDING_METHODS: tuple[str, ...] = (
    "mean",
    "linear",
    "poly2",
    "poly",
    "lowess",
    "rlowess",
    "loess",
    "rloess",
    "eemd",
)

#: 局部回归使用的模型（其余模型没有"窗口"可言）。
LOCAL_METHODS: tuple[str, ...] = ("lowess", "rlowess", "loess", "rloess")

#: 默认是稳健版（抗离群点）的模型。
ROBUST_METHODS: tuple[str, ...] = ("rlowess", "rloess")

#: 默认平滑窗口占记录时间跨度的比例，与 Acycle 的 ``smooth_win`` 一致。
DEFAULT_SPAN = 0.35

#: ``poly`` 方法的默认多项式阶数，与 Acycle 的 ``poly_order`` 一致。
DEFAULT_POLY_ORDER = 3

#: 去趋势至少需要的采样点数。
MIN_SAMPLES = 4

#: 稳健局部回归的重加权迭代次数（Cleveland 1979 的做法）。
ROBUST_ITERATIONS = 3

#: 稳健权重的截断尺度系数：``6 * median(|residual|)``。
ROBUST_SCALE_FACTOR = 6.0

#: 判定稳健尺度退化的相对阈值：中位绝对残差小于最大残差的这个比例时，
#: 中位数已经退化成浮点舍入量级，改用平均绝对残差。
DEGENERATE_SCALE_RATIO = 1e-9


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class DetrendResult:
    """去趋势结果。

    Attributes:
        times: 时间轴，与输入等长。
        trend: 趋势项，与输入等长。
        detrended: 去趋势后的序列，**逐点等于** ``values - trend``。
        method: 实际使用的方法名（见 :data:`DETRENDING_METHODS`）。
    """

    times: npt.NDArray
    trend: npt.NDArray
    detrended: npt.NDArray
    method: str

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [
            _("Detrending"),
            "=" * 50,
            _("Method: {0}").format(self.method),
            _("Samples: {0}").format(int(self.times.size)),
            _("Trend min: {0}").format(f"{float(np.min(self.trend)):.6g}"),
            _("Trend max: {0}").format(f"{float(np.max(self.trend)):.6g}"),
            _("Detrended mean: {0}").format(f"{float(np.mean(self.detrended)):.6g}"),
            _("Detrended sd: {0}").format(f"{float(np.std(self.detrended)):.6g}"),
        ]
        return "\n".join(lines)


# =============================================================================
# 内部工具
# =============================================================================


def _as_pair(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """把输入整理成两条等长、有限的一维序列。

    Args:
        time: 时间轴。
        values: 数值序列。

    Returns:
        ``(time, values)``，都是 ``float64`` 一维数组。

    Raises:
        ValidationError: 输入不是一维、长度不一致、点数太少，或含有非有限值。
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.ndim != 1:
        raise ValidationError(
            _("time and values must be one-dimensional arrays"),
            details={"time_ndim": int(t.ndim), "values_ndim": int(v.ndim)},
        )
    if t.size != v.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(t.size), "values_length": int(v.size)},
        )
    if t.size < MIN_SAMPLES:
        raise ValidationError(
            _("detrending needs at least {0} samples; got {1}").format(MIN_SAMPLES, int(t.size)),
            details={"n_samples": int(t.size), "minimum": MIN_SAMPLES},
        )
    if not bool(np.all(np.isfinite(t))):
        raise ValidationError(
            _("time must contain only finite values"),
            details={"n_nonfinite": int(np.count_nonzero(~np.isfinite(t)))},
        )
    if not bool(np.all(np.isfinite(v))):
        raise ValidationError(
            _("values must contain only finite values"),
            details={"n_nonfinite": int(np.count_nonzero(~np.isfinite(v)))},
        )
    return t, v


def _tricube(u: npt.NDArray) -> npt.NDArray:
    """Tricube 权重核 ``(1 - |u|^3)^3``（|u| < 1 时非零）。"""
    weights = np.zeros_like(u)
    inside = np.abs(u) < 1.0
    weights[inside] = (1.0 - np.abs(u[inside]) ** 3) ** 3
    return weights


def _bisquare(residuals: npt.NDArray, scale: float) -> npt.NDArray:
    """Bisquare 稳健权重 ``(1 - (r/s)^2)^2``（|r| >= s 时为 0）。"""
    if scale <= 0.0:
        return np.ones_like(residuals)
    scaled = residuals / scale
    weights = np.zeros_like(scaled)
    inside = np.abs(scaled) < 1.0
    weights[inside] = (1.0 - scaled[inside] ** 2) ** 2
    return weights


def _local_polynomial_value(
    u: npt.NDArray,
    y: npt.NDArray,
    weights: npt.NDArray,
    degree: int,
) -> float:
    """在 ``u = 0`` 处求一次加权局部多项式拟合的值。

    基函数取 ``[1, u, u^2, ...]``，其中 ``u`` 是**以拟合点为中心、已按
    半宽归一化**的坐标，因此 ``|u| <= 1``，正规方程的条件数良好。拟合值
    就是常数项。

    Args:
        u: 归一化后的坐标（以拟合点为中心）。
        y: 对应的数值。
        weights: 每个点的权重。
        degree: 多项式阶数（1 = 局部线性，2 = 局部二次）。

    Returns:
        拟合值；权重退化时退化为加权均值。
    """
    if weights.size == 0 or float(np.sum(weights)) <= 0.0:
        return float("nan")

    powers = [np.ones_like(u)]
    for power in range(1, 2 * degree + 1):
        powers.append(powers[power - 1] * u)

    system = np.empty((degree + 1, degree + 1), dtype=float)
    right = np.empty(degree + 1, dtype=float)
    for row in range(degree + 1):
        for column in range(degree + 1):
            system[row, column] = float(np.sum(weights * powers[row + column]))
        right[row] = float(np.sum(weights * powers[row] * y))

    try:
        coefficients = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        # 局部窗口退化（点数太少或共线）：退化成加权均值。
        return float(np.sum(weights * y) / np.sum(weights))
    return float(coefficients[0])


def _local_regression(
    t: npt.NDArray,
    v: npt.NDArray,
    span: float,
    degree: int,
    robust: bool,
) -> npt.NDArray:
    """局部加权回归（LOWESS / LOESS 及其稳健版）。

    窗口是**时间坐标里的长度**：总宽度 ``span * (t_max - t_min)``，半宽是它
    的一半。权重是 tricube 核；稳健版再乘上由残差推出的 bisquare 权重，
    并迭代 :data:`ROBUST_ITERATIONS` 次。

    Args:
        t: 时间轴。
        v: 数值序列。
        span: 窗口宽度占记录时间跨度的比例，(0, 1]。
        degree: 局部多项式阶数（1 = LOWESS，2 = LOESS）。
        robust: 是否启用稳健重加权。

    Returns:
        趋势数组，与输入等长。
    """
    n_samples = int(t.size)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    total_span = t_max - t_min
    if total_span <= 0.0:
        raise ValidationError(
            _("time must span a non-zero range for local regression"),
            details={"t_min": t_min, "t_max": t_max},
        )
    half_width = 0.5 * span * total_span

    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    v_sorted = v[order]

    trend = np.empty(n_samples, dtype=float)
    robust_weights = np.ones(n_samples, dtype=float)
    iterations = ROBUST_ITERATIONS if robust else 1
    scale: float | None = None

    for iteration in range(iterations):
        for index in range(n_samples):
            centre = float(t[index])
            left = int(np.searchsorted(t_sorted, centre - half_width, side="left"))
            right = int(np.searchsorted(t_sorted, centre + half_width, side="right"))
            if right <= left:
                # 窗口里一个点都没有（时间轴有重复或间距异常）：退化成最近点。
                nearest = int(np.argmin(np.abs(t_sorted - centre)))
                trend[index] = float(v_sorted[nearest])
                continue
            window = order[left:right]
            u = (t[window] - centre) / half_width
            weights = _tricube(u) * robust_weights[window]
            value = _local_polynomial_value(u, v[window], weights, degree)
            if not np.isfinite(value):
                nearest = int(np.argmin(np.abs(t_sorted - centre)))
                value = float(v_sorted[nearest])
            trend[index] = value

        if robust and scale is None:
            # 稳健尺度：先用中位绝对残差（MAD）。
            #
            # 有一条退化路径必须绕开：一条拟合得很干净的记录，残差几乎全是
            # 浮点舍入量级（1e-15），中位数也是这个量级。此时 ``6 * MAD``
            # 会小到**舍入误差本身**都超过它，于是几乎所有点的稳健权重都
            # 变成 0，局部拟合直接垮掉（比不稳健还糟）。所以当中位数相对
            # 最大残差可以忽略时，退化为平均绝对残差。
            absolute = np.abs(v - trend)
            largest = float(np.max(absolute))
            statistic = float(np.median(absolute))
            if statistic <= DEGENERATE_SCALE_RATIO * largest:
                statistic = float(np.mean(absolute))
            scale = ROBUST_SCALE_FACTOR * statistic

        if robust and scale is not None and scale > 0.0:
            robust_weights = _bisquare(v - trend, scale)

    return trend


def _polynomial_trend(t: npt.NDArray, v: npt.NDArray, degree: int) -> npt.NDArray:
    """最小二乘多项式趋势。

    Args:
        t: 时间轴。
        v: 数值序列。
        degree: 多项式阶数。

    Returns:
        趋势数组，与输入等长。

    Raises:
        ValidationError: 阶数过大（点数不够）或拟合数值上失败。
    """
    if degree + 1 > t.size:
        raise ValidationError(
            _("a polynomial of order {0} needs at least {1} samples; got {2}").format(
                int(degree), int(degree + 1), int(t.size)
            ),
            details={"order": int(degree), "n_samples": int(t.size)},
        )
    try:
        coefficients = np.polyfit(t, v, degree)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValidationError(
            _("polynomial trend fit failed"),
            details={"order": int(degree)},
            original_exception=exc,
        ) from exc
    return np.polyval(coefficients, t)


def _resolve_robust(method: str, robust: bool | None) -> bool:
    """确定是否启用稳健重加权。

    ``robust`` 为 ``None`` 时按方法名决定：``rlowess`` / ``rloess`` 稳健，
    其余不稳健。
    """
    if robust is None:
        return method in ROBUST_METHODS
    return bool(robust)


# =============================================================================
# 公开接口
# =============================================================================


def available_detrending_methods() -> tuple[str, ...]:
    """列出本模块支持的趋势模型。

    Returns:
        方法名元组，见 :data:`DETRENDING_METHODS`。
    """
    return DETRENDING_METHODS


def detrend_series(
    time: npt.NDArray,
    values: npt.NDArray,
    method: str = "linear",
    order: int = 3,
    span: float | None = None,
    robust: bool | None = None,
    n_trials: int = 50,
    random_seed: int | None = None,
) -> DetrendResult:
    """对一条记录去趋势。

    **去趋势 = 原序列 - 趋势**：返回值里的 ``detrended`` 逐点等于
    ``values - trend``（与 Acycle ``detrending.m`` 中
    ``detr.<model> = datay - trend`` 的定义一致）。

    各方法的趋势定义：

    * ``"mean"``：常值趋势，等于整条记录的均值。
    * ``"linear"``：1 阶最小二乘多项式。
    * ``"poly2"``：2 阶最小二乘多项式。
    * ``"poly"``：``order`` 阶最小二乘多项式（``order >= 1``）。
    * ``"lowess"``：局部线性加权回归（Cleveland 1979）。
    * ``"rlowess"``：稳健版 LOWESS。
    * ``"loess"``：局部二次加权回归。
    * ``"rloess"``：稳健版 LOESS。
    * ``"eemd"``：EEMD 分解的残差（最后一个 IMF）作趋势。

    **窗口语义**：``span`` 是**记录时间跨度的百分比**（与 Acycle 的
    ``smooth_win`` 一致，默认 0.35），不是样本数。局部窗口的总宽度为
    ``span * (t_max - t_min)``（时间坐标长度），半宽是它的一半。

    Args:
        time: 时间轴，一维、有限、长度不少于 :data:`MIN_SAMPLES`。
        values: 数值序列，与 ``time`` 等长且全部有限。
        method: 趋势模型，见 :data:`DETRENDING_METHODS`。
        order: ``"poly"`` 方法的多项式阶数，必须 ``>= 1``。
        span: 局部回归窗口占记录时间跨度的比例，``(0, 1]``。``None`` 时
            用 :data:`DEFAULT_SPAN`（0.35）。
        robust: 是否启用稳健重加权。``None`` 时按方法名决定
            （``rlowess`` / ``rloess`` 稳健，其余不稳健）；对非局部回归
            方法该参数不起作用。
        n_trials: ``"eemd"`` 方法的 EEMD 集合平均试验次数。
        random_seed: ``"eemd"`` 方法的随机种子。``None`` 时结果不可复现，
            :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。
            其余方法是确定性的，不使用随机数。

    Returns:
        DetrendResult: 包含时间轴、趋势与去趋势序列。

    Raises:
        ValidationError: 输入不合法、方法未知、阶数小于 1、窗口比例不在
            ``(0, 1]``，或时间轴含有非有限值。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.detrending import detrend_series
        >>> t = np.arange(0.0, 100.0, 1.0)
        >>> y = 3.0 * t + 1.0
        >>> result = detrend_series(t, y, method="linear")
        >>> bool(np.allclose(result.detrended, 0.0))
        True
    """
    t, v = _as_pair(time, values)

    if not isinstance(method, str):
        raise ValidationError(
            _("method must be a string"),
            details={"method": str(method)},
        )
    name = method.strip().lower()
    if name not in DETRENDING_METHODS:
        raise ValidationError(
            _("method must be one of {0}; got '{1}'").format(", ".join(DETRENDING_METHODS), method),
            details={"method": method, "valid_methods": list(DETRENDING_METHODS)},
        )

    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise ValidationError(
            _("order must be an integer"),
            details={"order": str(order)},
        )
    order = int(order)
    if order < 1:
        raise ValidationError(
            _("order must be at least 1; got {0}").format(order),
            details={"order": order},
        )

    if span is None:
        span = DEFAULT_SPAN
    try:
        span = float(span)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("span must be a number"),
            details={"span": str(span)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(span) or not 0.0 < span <= 1.0:
        raise ValidationError(
            _("span must lie in (0, 1]; got {0}").format(span),
            details={"span": span},
        )

    if not isinstance(n_trials, (int, np.integer)) or isinstance(n_trials, bool):
        raise ValidationError(
            _("n_trials must be an integer"),
            details={"n_trials": str(n_trials)},
        )
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValidationError(
            _("n_trials must be at least 1; got {0}").format(n_trials),
            details={"n_trials": n_trials},
        )

    use_robust = _resolve_robust(name, robust)

    if name == "mean":
        trend = np.full(v.size, float(np.mean(v)), dtype=float)
    elif name == "linear":
        trend = _polynomial_trend(t, v, 1)
    elif name == "poly2":
        trend = _polynomial_trend(t, v, 2)
    elif name == "poly":
        trend = _polynomial_trend(t, v, order)
    elif name in LOCAL_METHODS:
        degree = 2 if name in ("loess", "rloess") else 1
        trend = _local_regression(t, v, span=span, degree=degree, robust=use_robust)
    else:
        # "eemd"：EEMD 的残差（最后一个 IMF）作趋势。
        decomposition = eemd(v, n_trials=n_trials, random_seed=random_seed, times=t)
        trend = np.asarray(decomposition.trend, dtype=float)
        if trend.size != v.size:
            raise ValidationError(
                _("EEMD residual length does not match the input"),
                details={"trend_length": int(trend.size), "values_length": int(v.size)},
            )

    detrended = v - trend
    return DetrendResult(times=t, trend=trend, detrended=detrended, method=name)
