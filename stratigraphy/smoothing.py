# =============================================================================
# FILE: stratigraphy/smoothing.py
# =============================================================================
"""
平滑与**最优跨度自动选择**（Acycle 的 "Bootstrap smoothing / best span" 那一套）。

手工挑一个平滑窗口是主观的: 窗口太窄, 噪声没被压下去; 窗口太宽, 真实的
阶跃与旋回被抹平。本模块提供两条互补的能力:

* :func:`moving_average_smooth` —— 固定跨度的中心移动平均 (边界用缩短窗口);
* :func:`optimal_span_smooth` —— **在候选跨度里自动挑一个**, 并用**块自助**
  (moving block bootstrap) 给出置信带。

判据
----
跨度选择支持两种判据, 都在候选跨度上取最小值:

* ``"gcv"`` —— **广义交叉验证** (generalized cross-validation, Golub et al.
  1979; Craven et al. 1978):

      GCV(m) = [ RSS(m) / n ] / [ 1 - tr(H_m) / n ] ** 2

  其中 ``H_m`` 是跨度为 ``m`` 的平滑算子的 hat 矩阵。分母是"用数据本身拟合
  所付出的自由度代价": 跨度越大, ``tr(H_m)`` 越大, 拟合越"奢侈", GCV 越大。
  GCV 平衡"残差变小"与"自由度被吃掉"两件事, 选出**预测误差最小**的跨度,
  而不是残差最小的跨度。

* ``"aic"`` —— **Akaike 信息准则** (高斯误差、方差未知的模型形式):

      AIC(m) = n * ln( RSS(m) / n ) + 2 * tr(H_m)

  (省略了与参数无关的常数项, 因为只有相对大小有意义。) AIC 以**拟合优度**
  为主, 用每参数 2 的惩罚压住"一味加长窗口"的倾向; 它比 GCV 更倾向于短跨度
  (AIC 惩罚是常数 2, GCV 的分母惩罚随 tr(H)/n 增长更快)。

两种判据都用**有效自由度** ``tr(H_m)`` 而不是"参数个数": 移动平均是一个
**线性**平滑算子, 它的有效参数个数是 hat 矩阵迹。对跨度为 ``m`` 的中心移动
平均, 窗口内的权重都是 ``1/len(window)``, 所以 ``H[i, i] = 1 / len(window_i)``,
于是

    tr(H_m) = sum_i 1 / len(window_i)

—— 内部点约为 ``n / m``, 边界点(窗口被截短)略大。

置信带
------
置信带用**块自助** (moving block bootstrap; Künsch 1989) 而非普通自助。
理由: 地质记录里相邻样本是**相关**的 (沉积速率、仪器漂移、自相关噪声都会让
相邻点成串地偏高或偏低), 普通自助把样本当作独立同分布, 会把这条相关性丢掉,
从而**低估**不确定度。块自助抽的是**连续的一段**样本, 因此把相邻样本之间
的相关结构保留了下来。

具体做法是**残差块自助**:

1. 用选中的跨度平滑原序列, 得 ``ŷ``; 残差 ``r = y - ŷ`` 去均值;
2. 每次重复把 ``r`` 按长度 ``L`` 的**连续块**有放回地重排 (移动块自助),
   ``L`` 由残差自相关自动估计 (见 :func:`_estimate_block_length`);
3. 因为移动平均是**线性**算子, 重抽残差后的平滑曲线等于
   ``ŷ + S(r*)`` (``S`` 为同一窗口的移动平均), 所以每次重复只重算一次移动
   平均;
4. 取各重复的 2.5 % / 97.5 % 分位数作为 ``lower`` / ``upper``。

与 Acycle 的关系
----------------
Acycle 的 ``smoothbestSpan.m`` / ``smoothci.m`` / ``SmoothBootGUI.m`` 做的是同一
件事: 在 5 %–95 % 的跨度范围内按 GCV(带惩罚项) 选一个跨度, 再用 bootstrap
给出置信带。Acycle 是 GPL 许可的软件, 本模块**没有翻译**它的 MATLAB 源码, 而是
按上述定义与文献 (Golub et al. 1979; Craven et al. 1978; Akaike 1974;
Künsch 1989) 独立实现; 差别在于本模块用块自助而非普通自助, 并且把跨度定义
为**样本数**而不是记录长度的百分比。

参考文献
--------
* Golub, G.H., Heath, M.T., Wahba, G. & clair, G., 1979. Generalized
  cross-validation as a method for choosing a good ridge parameter.
  *Numerische Mathematik*, 36: 447-473.
* Craven, P. & Wahba, G., 1978. Smoothing noisy data with spline functions.
  *Numerische Mathematik*, 31: 377-404.
* Akaike, H., 1974. A new look at the statistical model identification.
  *IEEE Transactions on Automatic Control*, 19: 716-723.
* Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary
  observations. *Annals of Statistics*, 17: 1217-1241.
* Li, M., Hinnov, L. & Kump, L., 2019. Acycle: time-series analysis software
  for paleoclimate projects and education. *Computers & Geosciences*.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

__all__ = [
    "SmoothingResult",
    "moving_average_smooth",
    "optimal_span_smooth",
]

# =============================================================================
# 常量
# =============================================================================

#: 跨度选择支持的判据。
SUPPORTED_CRITERIA = ("gcv", "aic")

#: 残差自相关跌破该阈值 (|acf| < 0.05) 时, 认为噪声已经"变白", 用该滞后作为
#: 块自助的块长。
BLOCK_LENGTH_ACF_THRESHOLD = 0.05

#: 置信带的分位数 (百分位)。
CI_PERCENTILES = (2.5, 97.5)

#: ``max_span`` 缺省时, 允许的最大跨度占记录长度的比例 (上限即"平滑掉的东西
#: 不超过记录的五分之一")。
DEFAULT_MAX_SPAN_FRACTION = 0.2

#: 记录至少要有这么多个样本, 跨度选择才有意义。
MIN_SAMPLES = 4

#: 移动平均窗口的最小样本数 (奇数窗口才有对称中心)。
MIN_SPAN = 3


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class SmoothingResult:
    """
    最优跨度平滑的结果。

    Attributes:
        times: 时间轴, 与输入等长 (原样返回)。
        smoothed: 选中的跨度下平滑曲线。
        span: 选中的跨度 (**样本数**)。
        lower: bootstrap 置信带下界 (2.5 % 分位数)。
        upper: bootstrap 置信带上界 (97.5 % 分位数)。
        criterion: 选跨度用的判据 (``"gcv"`` 或 ``"aic"``)。
    """

    times: npt.NDArray
    smoothed: npt.NDArray
    span: float
    lower: npt.NDArray
    upper: npt.NDArray
    criterion: str = "gcv"

    def summary(self) -> str:
        """
        结果的文字摘要。

        Returns:
            多行字符串, 含判据、跨度、残差标准差与置信带宽度。
        """
        smoothed = np.asarray(self.smoothed, dtype=float)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        width = upper - lower
        return "\n".join(
            [
                f"Optimal-span smoothing ({self.criterion})",
                f"  samples      : {smoothed.size}",
                f"  selected span: {self.span:g} samples",
                f"  band width   : mean {float(width.mean()):.4g}, max {float(width.max()):.4g}",
                f"  band order   : lower <= smoothed <= upper: "
                f"{bool(np.all(lower <= smoothed + 1e-12) and np.all(smoothed <= upper + 1e-12))}",
            ]
        )


# =============================================================================
# 内部工具
# =============================================================================


def _window_bounds(n: int, half: int) -> tuple[npt.NDArray, npt.NDArray]:
    """
    中心移动平均的窗口边界 (边界处窗口被截短)。

    Parameters:
        n: 样本数。
        half: 窗口半宽。

    Returns:
        ``(start, stop)`` 两个整型数组, 窗口为 ``[start[i], stop[i])``。
    """
    index = np.arange(n)
    start = np.maximum(index - half, 0)
    stop = np.minimum(index + half + 1, n)
    return start, stop


def _moving_average(values: npt.NDArray, start: npt.NDArray, stop: npt.NDArray) -> npt.NDArray:
    """
    用累积和在线性时间内求移动平均 (窗口长度由 ``start``/``stop`` 给出)。

    Parameters:
        values: 输入序列。
        start: 窗口起点 (含)。
        stop: 窗口终点 (不含)。

    Returns:
        与 ``values`` 等长的移动平均。
    """
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    totals = cumulative[stop] - cumulative[start]
    lengths = (stop - start).astype(float)
    return totals / lengths


def _moving_average_blocks(values: npt.NDArray, start: npt.NDArray, stop: npt.NDArray) -> npt.NDArray:
    """
    对一批等长序列同时求移动平均。

    每条序列用**自己**的窗口 (窗口位置相对于该序列的开头), 所以这里要把
    窗口下标平移到各条序列在拼接数组中的实际位置。

    Parameters:
        values: 形状 ``(n_series, n)`` 的序列批。
        start: 窗口起点 (相对各序列开头, 形状 ``(n,)``)。
        stop: 窗口终点 (相对各序列开头, 形状 ``(n,)``)。

    Returns:
        形状 ``(n_series, n)`` 的移动平均。
    """
    n_series, n = values.shape
    cumulative = np.concatenate(([0.0], np.cumsum(values.ravel())))
    offsets = (np.arange(n_series) * n)[:, None]
    totals = cumulative[offsets + stop[None, :]] - cumulative[offsets + start[None, :]]
    lengths = (stop - start).astype(float)
    return totals / lengths[None, :]


def _effective_df(start: npt.NDArray, stop: npt.NDArray) -> float:
    """
    移动平均算子的有效自由度 ``tr(H)``。

    窗口内权重都是 ``1/len(window)``, 故 ``H[i, i] = 1/len(window_i)``。
    """
    lengths = (stop - start).astype(float)
    return float(np.sum(1.0 / lengths))


def _estimate_block_length(residuals: npt.NDArray) -> int:
    """
    由残差自相关估计块自助的块长。

    取残差自相关**首次**跌破 ``0.05`` 的滞后作为块长 —— 那正是"噪声已经变白"
    的地方; 若在搜索范围内一直不跌破, 退回常用的 ``n ** (1/3)`` 规则。

    Parameters:
        residuals: 去均值后的残差序列。

    Returns:
        块长 (样本数), 至少 1, 至多记录长度的一半。
    """
    n = int(residuals.size)
    if n < 3:
        return 1

    centered = residuals - residuals.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 1

    max_lag = min(n - 1, max(10, n // 4))
    lags = np.arange(1, max_lag + 1)
    autocorrelations = np.array(
        [float(np.dot(centered[:-lag], centered[lag:])) / denominator for lag in lags],
        dtype=float,
    )
    below = np.nonzero(np.abs(autocorrelations) < BLOCK_LENGTH_ACF_THRESHOLD)[0]
    if below.size:
        length = int(lags[below[0]])
    else:
        length = int(np.ceil(n ** (1.0 / 3.0)))
    return int(np.clip(length, 1, max(1, n // 2)))


def _block_bootstrap_series(
    residuals: npt.NDArray, block_length: int, n_bootstrap: int, rng: np.random.Generator
) -> npt.NDArray:
    """
    移动块自助: 生成 ``n_bootstrap`` 条重排后的残差序列。

    Parameters:
        residuals: 去均值后的残差序列 (长度 ``n``)。
        block_length: 块长 (样本数)。
        n_bootstrap: 重复次数。
        rng: 随机数发生器。

    Returns:
        形状 ``(n_bootstrap, n)`` 的数组, 每行是一条重排后的残差序列。
    """
    n = int(residuals.size)
    length = int(max(1, min(block_length, n)))
    n_blocks = int(np.ceil(n / length))
    max_start = n - length
    draws = rng.integers(0, max_start + 1, size=(n_bootstrap, n_blocks))
    offsets = np.arange(length)
    indices = (draws[:, :, None] + offsets[None, None, :]).reshape(n_bootstrap, n_blocks * length)[:, :n]
    return residuals[indices]


def _criterion_value(rss: float, n: int, effective_df: float, criterion: str) -> float:
    """
    给定跨度算出判据值 (越小越好)。

    Parameters:
        rss: 残差平方和。
        n: 样本数。
        effective_df: 有效自由度 ``tr(H)``。
        criterion: ``"gcv"`` 或 ``"aic"``。

    Returns:
        判据值; 退化情形返回 ``inf``。
    """
    if rss <= 0.0:
        rss = np.finfo(float).tiny
    if criterion == "gcv":
        denominator = 1.0 - effective_df / n
        if denominator <= 0.0:
            return float("inf")
        return (rss / n) / (denominator**2)
    # AIC: 高斯误差、方差未知; 省略与参数无关的常数项。
    return float(n * np.log(rss / n) + 2.0 * effective_df)


def _validate_series(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    校验时间轴与数值序列。

    Returns:
        ``(time, values)`` 两个 float 数组。
    """
    times = np.atleast_1d(np.asarray(time, dtype=float))
    series = np.atleast_1d(np.asarray(values, dtype=float))
    if times.shape != series.shape:
        raise ValidationError(
            "time and values must have the same length",
            details={"time_length": int(times.size), "values_length": int(series.size)},
        )
    if times.size == 0:
        raise ValidationError("time and values must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValidationError("time must contain finite values only")
    if not np.all(np.isfinite(series)):
        raise ValidationError("values must contain finite values only")
    return times, series


# =============================================================================
# 公开接口
# =============================================================================


def moving_average_smooth(time: npt.NDArray, values: npt.NDArray, span: int) -> npt.NDArray:
    """
    固定跨度的中心移动平均。

    窗口按**样本序号**定义 (跨度 = 样本数), 不按时间定义: 本仓库的记录
    (例如 LR04 栈) 采样间隔并不均匀, 用样本数表达跨度才与"跨度"的单位一致。

    边界处窗口被截短 (即用可用的邻居求平均), 因此输出与输入等长, 且不会
    在两端出现人为的阶跃。

    Parameters:
        time: 时间轴, 只用于校验与保持输出对齐。
        values: 数值序列。
        span: 窗口宽度 (样本数), 至少 1。

    Returns:
        与 ``values`` 等长的平滑序列。

    Raises:
        ValidationError: 若长度不一致、含非有限值, 或 ``span`` 非法。
    """
    times, series = _validate_series(time, values)

    width = int(span)
    if width < 1:
        raise ValidationError(f"span must be a positive integer, got {span!r}")
    if width > times.size:
        raise ValidationError(
            "span must not exceed the number of samples",
            details={"span": width, "samples": int(times.size)},
        )

    half = width // 2
    start, stop = _window_bounds(times.size, half)
    return _moving_average(series, start, stop)


def optimal_span_smooth(
    time: npt.NDArray,
    values: npt.NDArray,
    min_span: int = 3,
    max_span: int | None = None,
    n_candidates: int = 20,
    criterion: str = "gcv",
    n_bootstrap: int = 200,
    random_seed: int | None = None,
) -> SmoothingResult:
    """
    自动选择平滑跨度, 并给出 bootstrap 置信带。

    在 ``[min_span, max_span]`` 之间取 ``n_candidates`` 个候选跨度, 对每个
    跨度算出判据值 (:func:`_criterion_value`), 取判据最小者; 然后用**块自助**
    给出该跨度下平滑曲线的 95 % 置信带。

    Parameters:
        time: 时间轴 (原样返回在结果里)。
        values: 数值序列。
        min_span: 最小候选跨度 (样本数), 至少 3 (跨度小于 3 的"平滑"没有意义)。
        max_span: 最大候选跨度 (样本数); 缺省为记录长度的 20 % (向上取整,
            且至少 ``min_span + 1``)。
        n_candidates: 候选跨度的个数, 至少 2。
        criterion: 选跨度的判据, ``"gcv"`` (默认) 或 ``"aic"``。两者的含义
            见模块 docstring。
        n_bootstrap: 自助重复次数, 至少 1。
        random_seed: 随机种子。给出种子时结果可复现; 不给种子时结果不可复现,
            :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。

    Returns:
        :class:`SmoothingResult`, 其中 ``span`` 是选中的跨度 (样本数),
        ``lower``/``upper`` 是 bootstrap 置信带。

    Raises:
        ValidationError: 若点数太少、``min_span`` < 3、``max_span`` <=
            ``min_span``、``n_candidates`` < 2、判据未知, 或输入不合法。
    """
    times, series = _validate_series(time, values)
    n = int(times.size)

    if n < MIN_SAMPLES:
        raise ValidationError(
            "Too few samples for optimal-span smoothing",
            details={"samples": n, "minimum": MIN_SAMPLES},
        )

    lower_bound = int(min_span)
    if lower_bound < MIN_SPAN:
        raise ValidationError(
            f"min_span must be at least {MIN_SPAN}, got {min_span!r}",
            details={"min_span": lower_bound},
        )
    if lower_bound >= n:
        raise ValidationError(
            "min_span must be smaller than the number of samples",
            details={"min_span": lower_bound, "samples": n},
        )

    if max_span is None:
        upper_bound = max(lower_bound + 1, round(DEFAULT_MAX_SPAN_FRACTION * n))
        upper_bound = min(upper_bound, n)
    else:
        upper_bound = int(max_span)
        if upper_bound <= lower_bound:
            raise ValidationError(
                "max_span must be greater than min_span",
                details={"min_span": lower_bound, "max_span": upper_bound},
            )
        if upper_bound > n:
            raise ValidationError(
                "max_span must not exceed the number of samples",
                details={"max_span": upper_bound, "samples": n},
            )

    candidates = int(n_candidates)
    if candidates < 2:
        raise ValidationError(
            f"n_candidates must be at least 2, got {n_candidates!r}",
            details={"n_candidates": candidates},
        )

    key = str(criterion).strip().lower()
    if key not in SUPPORTED_CRITERIA:
        raise ValidationError(
            f"Unknown smoothing criterion: {criterion!r}",
            details={"known": list(SUPPORTED_CRITERIA)},
        )

    repeats = int(n_bootstrap)
    if repeats < 1:
        raise ValidationError(
            f"n_bootstrap must be at least 1, got {n_bootstrap!r}",
            details={"n_bootstrap": repeats},
        )

    # ---- 候选跨度与判据 -----------------------------------------------------
    raw_spans = np.linspace(lower_bound, upper_bound, candidates)
    span_grid = np.unique(np.round(raw_spans).astype(int))
    span_grid = span_grid[(span_grid >= lower_bound) & (span_grid <= upper_bound)]
    if span_grid.size < 2:
        # 候选跨度去重后只剩一个 (例如 min_span 与 max_span 相邻), 补上另一个端点。
        span_grid = np.unique(np.array([lower_bound, upper_bound], dtype=int))

    scores: list[tuple[int, float]] = []
    for span in span_grid:
        half = int(span) // 2
        start, stop = _window_bounds(n, half)
        fitted = _moving_average(series, start, stop)
        residuals = series - fitted
        rss = float(np.dot(residuals, residuals))
        score = _criterion_value(rss, n, _effective_df(start, stop), key)
        scores.append((int(span), score))

    # 取判据最小者; 并列时取较短的跨度 (更保守, 不抹掉细节)。
    best_span, best_score = min(scores, key=lambda item: (item[1], item[0]))
    logger.debug("optimal-span smoothing: criterion=%s best span=%d score=%g", key, best_span, best_score)

    half = best_span // 2
    start, stop = _window_bounds(n, half)
    fitted = _moving_average(series, start, stop)

    # ---- 块自助置信带 -------------------------------------------------------
    rng = make_rng(random_seed, context="optimal-span smoothing")
    residuals = series - fitted
    residuals = residuals - residuals.mean()
    block_length = _estimate_block_length(residuals)
    resampled = _block_bootstrap_series(residuals, block_length, repeats, rng)

    # 移动平均是线性算子, 故 S(y + r*) = S(y) + S(r*): 只需对重抽残差再平滑一次。
    replicates = fitted[None, :] + _moving_average_blocks(resampled, start, stop)
    lower = np.percentile(replicates, CI_PERCENTILES[0], axis=0)
    upper = np.percentile(replicates, CI_PERCENTILES[1], axis=0)

    logger.debug("optimal-span smoothing: block length=%d, %d bootstrap replicates", block_length, repeats)

    return SmoothingResult(
        times=times,
        smoothed=fitted,
        span=float(best_span),
        lower=lower,
        upper=upper,
        criterion=key,
    )
