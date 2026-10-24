# =============================================================================
# FILE: stratigraphy/evolutionary_rho.py
# =============================================================================
"""
演化 ρ1（evolutionary rho1）：沿时间轴滑动的滞后 1 阶自相关。

本模块实现 Acycle ``code/seriesanalysis/acycleEvolutionaryRho1.m`` 的核心算法
（"确定性单次运行"部分），用于回答一个具体问题：**这条记录的自相关性是随时间
变化的吗？** 岩芯记录里常见的现象是：某一层段（如浊流沉积、碳酸盐台地）几乎
没有噪声，而另一层段（如陆源碎屑输入强烈的段）噪声极大。演化 ρ1 把这种"局部
信噪比"的变化量化成一条随时间滑动的 ρ1 曲线。

算法
~~~~
1. **线性重采样**：记录先按固定步长 ``sampling_interval`` 线性插值到一张均匀
   网格上。网格以记录起点为锚点，格点数为 ``floor(跨度/dt) + 1``（当跨度恰好是
   步长的整数倍时保留末端点），这与 Acycle ``acycleInterpolateSeries`` 的
   "release-stable integer-index" 网格一致。
2. **滑窗**：窗口点数 ``K = round(window / sampling_interval)``，窗从网格起点
   逐点右移，因此共有 ``N - K + 1`` 个窗。
3. **逐窗独立去趋势**：每个窗在**居中的样本序号基**上单独做一次最小二乘直线
   去趋势（基为 ``i - (K+1)/2``），去趋势后再减去窗内均值得到 ``z``。逐窗去趋势
   是关键：整条记录只做一次去趋势会让窗内残差携带窗间趋势的差异，逐窗去趋势
   保证每个 ρ1 都只反映**该窗自身**的短程持续性。
4. **ρ1 估计**：沿用 Acycle ``rhoAR1ML`` 的约定

       rho = Σ z(2:) · z(1:-1) / Σ z(1:-1)²

   即 Yule–Walker 的滞后 1 阶估计量（分子用后一时刻、分母用前一时刻）。
5. **窗中心**：取窗内第一个与最后一个网格坐标的中点，而不是由请求的物理窗口
   长度反推——后者在非整数倍窗口下会偏离实际覆盖的区间。

不确定性
~~~~~~~~
``uncertainty`` 是每窗 ρ1 的抽样标准差，由**移动块自助法**（moving-block
bootstrap）给出：以块长 ``round(N^(1/3))`` 从重采样后的序列中循环抽取连续块，
拼成等长的替代序列，对每个替代序列重算整条滑窗 ρ1 轨迹，再对每个窗取标准差。
块自助保留了块内的自相关结构，因此给出的不确定度比假定独立同分布的自助更
诚实；标准差仍然偏小（块自助不重建长程结构），文档中如实说明。

随机过程约定
~~~~~~~~~~~~
* 所有随机函数都接受 ``random_seed``。给定种子时结果完全可复现；不给种子时
  调用 ``utils.statistics_core.make_rng(None, context=...)``，它会发出
  ``RuntimeWarning`` 提示结果不可复现——这是仓库内其它随机分析沿用的同一约定。
* 生成器是独立的 ``numpy.random.Generator``，既不读取也不推进全局
  ``numpy.random`` 流，因此不会扰动调用方自己的随机数据。

参考文献
~~~~~~~~
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. Computers & Geosciences 127, 12-22.
* Acycle ``code/seriesanalysis/acycleEvolutionaryRho1.m`` 与 ``rhoAR1.m``
  （本模块实现的算法来源）。
* Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and
  Control. Holden & Day.（Yule–Walker 滞后 1 阶估计量。）
* Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap.
  Journal of the American Statistical Association 89: 1303-1313.（块自助法。）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 记录至少需要的有效点数。
MIN_SAMPLES = 8

#: 未给出 ``window`` 时，窗口长度占记录跨度的比例。
DEFAULT_WINDOW_FRACTION = 0.2

#: 单次滑窗计算允许的最大工作量（窗口数 × 窗口点数）；与 Acycle 的
#: ``maximumWindowWork`` 同量级，用于挡住病态的大记录。
MAX_WINDOW_WORK = 10_000_000

#: 自助法允许的最大工作量（自助次数 × 重采样点数 × 窗口点数）。
MAX_BOOTSTRAP_WORK = 200_000_000


# =============================================================================
# 1. 结果容器
# =============================================================================


@dataclass(frozen=True)
class EvolutionaryRhoResult:
    """演化 ρ1 的结果。

    Attributes:
        window_centres: 每个滑窗的时间中心（与 ``time`` 同一量纲），严格递增。
        rho: 每个滑窗的滞后 1 阶自相关 ``Σz(2:)·z(1:-1) / Σz(1:-1)²``。
        uncertainty: 每个滑窗 ρ1 的自助标准差（同一量纲，随机过程的抽样误差）。
    """

    window_centres: npt.NDArray
    rho: npt.NDArray
    uncertainty: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Evolutionary rho1"),
            "=" * 40,
            _("Windows: {0}").format(self.window_centres.size),
        ]
        if self.rho.size:
            lines.append(
                _("Centre range: {0} .. {1}").format(f"{self.window_centres[0]:.4g}", f"{self.window_centres[-1]:.4g}")
            )
            lines.append(_("Mean rho: {0}").format(f"{float(np.mean(self.rho)):.4f}"))
            lines.append(_("Min rho: {0}").format(f"{float(np.min(self.rho)):.4f}"))
            lines.append(_("Max rho: {0}").format(f"{float(np.max(self.rho)):.4f}"))
        if self.uncertainty.size:
            lines.append(_("Mean uncertainty: {0}").format(f"{float(np.mean(self.uncertainty)):.4f}"))
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
            _("evolutionary rho1 needs at least {0} finite points; got {1}").format(MIN_SAMPLES, int(t.size)),
            details={"n_finite": int(t.size)},
        )
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if np.any(np.diff(t) <= 0.0):
        raise ValidationError(
            _("time must be strictly increasing after sorting; duplicate timestamps have no single sampling interval")
        )
    return t, v


def _resample_uniform(
    time: npt.NDArray, values: npt.NDArray, sampling_interval: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """把记录线性插值到以起点为锚点的均匀网格。

    网格点数为 ``floor(跨度/dt) + 1``；当跨度几乎是步长的整数倍时保留末端点
    （Acycle ``acycleInterpolateSeries`` 的整数索引网格策略）。网格绝不越过记录
    的末端，因此不会外推。
    """
    span = float(time[-1] - time[0])
    raw_intervals = span / sampling_interval
    nearest = float(round(raw_intervals))
    tolerance = 8.0 * float(np.finfo(float).eps) * max(1.0, abs(raw_intervals))
    if abs(raw_intervals - nearest) <= tolerance:
        intervals = int(nearest)
    else:
        intervals = int(np.floor(raw_intervals))
    if intervals < 1:
        raise ValidationError(
            _("the sampling interval is longer than the record's span; nothing to resample"),
            details={"sampling_interval": float(sampling_interval), "span": span},
        )
    grid = time[0] + sampling_interval * np.arange(intervals + 1, dtype=float)
    resampled = np.interp(grid, time, values)
    return grid, resampled


def _window_rho(values: npt.NDArray, window_points: int) -> npt.NDArray:
    """每个滑窗独立去趋势后的滞后 1 阶自相关。

    去趋势在居中的样本序号基 ``i - (K-1)/2`` 上做最小二乘直线拟合，随后减去
    窗内均值；ρ1 沿用 Acycle ``rhoAR1ML`` 的分子/分母约定。

    基向量必须是**居中**的（元素之和为零）。Acycle 用的是 1 起始的样本序号
    ``sampleIndex - (K+1)/2``，换成 0 起始的下标就是 ``i - (K-1)/2``，它恰好
    居中。用一个不居中的基（例如 ``i - (K+1)/2``）会同时改变拟合的能量项
    ``sum(basis**2)``，斜率因此被按 ``sum(basis_c**2) / (sum(basis_c**2) + K)``
    的比例低估（K = 20 时约 2.9%），去趋势不彻底：一条完全线性的记录会剩下
    一道斜坡，而斜坡的滞后 1 阶自相关接近 1，看上去像极强的红噪声。

    退化窗口（残差相对窗内散布可以忽略不计，例如一段完全线性的趋势）返回
    ``0.0`` 而不是舍入噪声之比。Acycle 在同样的情形下会把两个舍入量相除，
    得到的数在 ``-1`` 与 ``1`` 之间随机取值；那是噪声，不是自相关。本模块
    在这里与 Acycle **有意不同**，并且把这个差别写在这里。
    """
    windows = np.lib.stride_tricks.sliding_window_view(values, window_points)
    centred = windows - np.mean(windows, axis=1, keepdims=True)
    basis = np.arange(window_points, dtype=float) - (window_points - 1) / 2.0
    basis_energy = float(np.sum(basis**2))
    slope = (centred @ basis) / basis_energy
    detrended = centred - slope[:, None] * basis
    z = detrended - np.mean(detrended, axis=1, keepdims=True)
    lagged = z[:, :-1]
    leading = z[:, 1:]
    denominator = np.sum(lagged**2, axis=1)
    numerator = np.sum(leading * lagged, axis=1)
    # A window whose residual is negligible against its own spread carries
    # nothing to correlate; the ratio there is round-off over round-off.
    residual_energy = np.sum(z**2, axis=1)
    window_energy = np.sum(centred**2, axis=1)
    degenerate = (window_energy > 0.0) & (residual_energy <= 1e-12 * window_energy)
    usable = (denominator > 0.0) & ~degenerate
    rho = np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)
    return rho


def _block_bootstrap_indices(size: int, block_length: int, rng: np.random.Generator) -> npt.NDArray:
    """移动块自助的抽样下标（循环取块，等长返回）。"""
    n_blocks = int(np.ceil(size / block_length))
    starts = rng.integers(0, size, size=n_blocks)
    offsets = np.arange(block_length, dtype=int)
    index = (starts[:, None] + offsets[None, :]) % size
    return index.reshape(-1)[:size]


# =============================================================================
# 3. 主入口
# =============================================================================


def evolutionary_rho1(
    time: npt.NDArray,
    values: npt.NDArray,
    window: float | None = None,
    sampling_interval: float | None = None,
    n_bootstrap: int = 200,
    random_seed: int | None = None,
) -> EvolutionaryRhoResult:
    """计算演化 ρ1（滑窗滞后 1 阶自相关）。

    Parameters
    ----------
    time:
        时间轴，必须严格递增；非有限点被丢弃。
    values:
        与 ``time`` 等长的信号。
    window:
        窗口长度（与 ``time`` 同量纲）。``None`` 时取记录跨度的
        ``DEFAULT_WINDOW_FRACTION``（20%）。窗口点数为
        ``K = round(window / sampling_interval)``。
    sampling_interval:
        重采样网格的步长（与 ``time`` 同量纲）。``None`` 时取原始记录的
        采样间隔中位数。
    n_bootstrap:
        自助重复次数，用于估计每个窗 ρ1 的抽样标准差。``0`` 表示不给出
        不确定度（``uncertainty`` 全为 ``nan``）。
    random_seed:
        自助法的种子。给定种子时结果完全可复现；不给种子时
        :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。

    Returns
    -------
    EvolutionaryRhoResult
        窗中心、逐窗 ρ1 与逐窗不确定度。

    Raises
    ------
    ValidationError:
        时间与数值长度不一致、点数太少、时间轴非严格递增、步长或窗口非法
        （``K < 3`` 或 ``K`` 超过重采样点数），或自助工作量超过上限。

    Examples
    --------
    >>> import numpy as np
    >>> time = np.arange(200, dtype=float)
    >>> values = np.sin(time / 5.0)
    >>> result = evolutionary_rho1(time, values, window=50.0, n_bootstrap=0)
    >>> result.rho.size == result.window_centres.size
    True
    """
    t, v = _clean_pair(time, values)
    span = float(t[-1] - t[0])

    if sampling_interval is None:
        interval = float(np.median(np.diff(t)))
    else:
        interval = float(sampling_interval)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValidationError(
            _("sampling_interval must be a finite positive number; got {0}").format(sampling_interval),
            details={"sampling_interval": sampling_interval},
        )
    if interval > span:
        raise ValidationError(
            _("sampling_interval must not exceed the record's span; got {0} > {1}").format(interval, span),
            details={"sampling_interval": interval, "span": span},
        )

    if window is None:
        window_length = DEFAULT_WINDOW_FRACTION * span
    else:
        window_length = float(window)
    if not np.isfinite(window_length) or window_length <= 0.0:
        raise ValidationError(
            _("window must be a finite positive number; got {0}").format(window),
            details={"window": window},
        )

    grid, series = _resample_uniform(t, v, interval)
    window_points = round(window_length / interval)
    if window_points < 3:
        raise ValidationError(
            _("round(window / sampling_interval) must be at least 3 points; got {0}").format(window_points),
            details={"window_points": window_points, "window": window_length},
        )
    if window_points > grid.size:
        raise ValidationError(
            _("the window of {0} points exceeds the {1} resampled points").format(window_points, int(grid.size)),
            details={"window_points": window_points, "n_resampled": int(grid.size)},
        )

    n_windows = grid.size - window_points + 1
    if n_windows * window_points > MAX_WINDOW_WORK:
        raise ValidationError(
            _(
                "the sliding window would need {0} operations, above the limit of {1}; "
                "use a longer window or a coarser sampling interval"
            ).format(int(n_windows * window_points), MAX_WINDOW_WORK),
            details={"window_work": int(n_windows * window_points), "limit": MAX_WINDOW_WORK},
        )

    rho = _window_rho(series, window_points)
    first = grid[:n_windows]
    last = grid[window_points - 1 :]
    window_centres = first + (last - first) / 2.0

    replicates = int(n_bootstrap)
    if replicates < 0:
        raise ValidationError(
            _("n_bootstrap must be >= 0; got {0}").format(n_bootstrap),
            details={"n_bootstrap": n_bootstrap},
        )
    if replicates == 0:
        uncertainty = np.full(rho.shape, np.nan, dtype=float)
    else:
        if replicates * series.size * window_points > MAX_BOOTSTRAP_WORK:
            raise ValidationError(
                _(
                    "the bootstrap would need {0} operations, above the limit of {1}; "
                    "reduce n_bootstrap or use a coarser sampling interval"
                ).format(int(replicates * series.size * window_points), MAX_BOOTSTRAP_WORK),
                details={
                    "bootstrap_work": int(replicates * series.size * window_points),
                    "limit": MAX_BOOTSTRAP_WORK,
                },
            )
        rng = make_rng(random_seed, context="evolutionary rho1")
        block_length = max(2, round(series.size ** (1.0 / 3.0)))
        trajectories = np.empty((replicates, n_windows), dtype=float)
        for replicate in range(replicates):
            index = _block_bootstrap_indices(series.size, block_length, rng)
            trajectories[replicate] = _window_rho(series[index], window_points)
        uncertainty = np.std(trajectories, axis=0, ddof=1)

    logger.info(
        "evolutionary rho1: %d resampled points, K=%d, %d windows, mean rho=%.4f",
        int(grid.size),
        window_points,
        int(n_windows),
        float(np.mean(rho)) if rho.size else float("nan"),
    )
    return EvolutionaryRhoResult(window_centres=window_centres, rho=rho, uncertainty=uncertainty)
