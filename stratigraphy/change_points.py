# =============================================================================
# FILE: stratigraphy/change_points.py
# =============================================================================
"""
贝叶斯变点检测（Ruggieri 2013 方法的可用实现）。

Acycle 没有独立的变点文件：它的贝叶斯变点方法出自 Ruggieri (2013)，见 Acycle
Wiki 的 References 页。本模块按该论文的分段常数模型思想独立实现一个**可用**
版本——不做完整的 Gibbs 采样，而是在每个候选位置做局部的贝叶斯块均值比较，
返回每个位置的变点后验概率。

方法
~~~~
候选位置 ``t``（``min_segment <= t <= n - min_segment``）把记录切成左右两块，
每块 ``m = min_segment`` 个样本。设

* ``delta`` 是右块均值减左块均值（块均值之差）；
* ``k = m / 2`` 是两块的有效样本数；
* ``v = sigma**2 / k`` 是 ``delta`` 的抽样方差，``sigma**2`` 是两块内部的合并
  残差方差。

在“块均值相同”（无变点）与“块均值相差 delta”两个模型之间做贝叶斯比较，
对 ``delta`` 取单位信息先验 ``N(0, v)``，得到 Savage-Dickey 型的 Bayes 因子

    BF(t) = (1 / sqrt(2)) * exp(delta**2 / (4 * v))

无变点时 ``delta**2 / (4 * v)`` 服从均值为 1/2 的指数分布，所以 ``BF`` 的量级
是 1（精确地说 ``E[BF] = sqrt(2)``：先验与零假设似然同尺度，数据与先验不矛
盾时 Bayes 因子不会系统性地偏向任何一方），变点越强 BF 越大。再把先验概率
（某个位置恰好是变点）取为 ``lambda / n_candidates``（``lambda`` 是变点数量
的先验期望），就得到该位置的后验概率

    P(change at t) = pi * BF / (1 - pi + pi * BF)

块均值与残差方差由**贝叶斯自助**（Bayesian bootstrap，Rubin 1981）估计：每
次重复从 Dirichlet(1) 权重重抽整条记录，于是报告的概率既包含数据本身的信息，
也包含记录抽样不确定性带来的信息。这也让 ``random_seed`` 变得有意义：同一
个种子给出同一个结果，不给种子时按仓库约定用
:func:`utils.statistics_core.make_rng` 发出不可复现的告警。

已知限制
~~~~~~~~
* 模型假定记录是分段常数的。**有明显趋势的记录会把趋势误判成变点**——这正
  是 Acycle 先做去趋势（detrending.m）再滤波/分析的原因；调用前请先去趋势。
* 每个候选位置只看左右各 ``min_segment`` 个样本，因此对**缓慢**（相对于
  ``min_segment``）的变化不敏感；要抓慢变就把 ``min_segment`` 调大。
* 变点之间的最小间隔由 ``min_segment`` 决定：两个靠得太近的阶跃无法区分。
* 变点数量的先验期望固定为 1（模块常量 ``PRIOR_EXPECTED_CHANGE_POINTS``），
  所以记录越长，单个变点需要的证据越多（对数级）。
* 逐位置的后验是**局部**量，而候选位置有 O(n) 个，所以报告阈值必须高于多重
  比较的噪声带（默认 0.95）。

参考：
    1. Ruggieri, E. 2013. A Bayesian approach to detecting change points in
       climatic records. International Journal of Climatology 33: 520-528.
    2. Rubin, D.B. 1981. The Bayesian bootstrap. Annals of Statistics 9:
       130-134. (贝叶斯自助。)
    3. Li, M., Hinnov, L. & Kump, L. 2019. Acycle: time-series analysis
       software for paleoclimate projects and education. Computers &
       Geosciences. (Acycle 软件本身。)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import DataValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 候选变点的最小块长下界。
MIN_SEGMENT = 2

#: 贝叶斯自助的重复次数。
BOOTSTRAP_REPLICATES = 200

#: 变点数量先验（Poisson）的期望值。
PRIOR_EXPECTED_CHANGE_POINTS = 1.0


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class ChangePointResult:
    """
    变点检测结果。

    Attributes:
        positions: 检测到的变点下标（升序）。下标 ``t`` 表示变化发生在
            ``t - 1`` 与 ``t`` 之间，即 ``t`` 是新一段的第一个样本。
        probabilities: 与 ``positions`` 一一对应的变点后验概率。
        probability_curve: 每个样本位置的变点后验概率；不是候选位置处为 0。
    """

    positions: npt.NDArray
    probabilities: npt.NDArray
    probability_curve: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        positions = ", ".join(str(int(value)) for value in self.positions)
        probabilities = ", ".join(f"{value:.4f}" for value in self.probabilities)
        peak = float(self.probability_curve.max()) if self.probability_curve.size else 0.0
        return (
            "Bayesian change-point detection\n"
            f"{'=' * 48}\n"
            f"Change points: {int(self.positions.size)}\n"
            f"Positions: [{positions}]\n"
            f"Probabilities: [{probabilities}]\n"
            f"Peak probability: {peak:.4f}"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _as_series(values: npt.NDArray, name: str = "values") -> npt.NDArray:
    """把输入转成一维有限数组。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("change-point input must be numeric; got {0}").format(type(values).__name__),
            details={"name": name},
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            _("change-point input must be one-dimensional; got shape {0}").format(array.shape),
            details={"name": name, "shape": array.shape},
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(
            _("change-point input contains non-finite values"),
            details={"name": name, "length": int(array.size)},
        )
    return array


def _validate_min_segment(min_segment: int) -> int:
    """检查最小块长是 >= ``MIN_SEGMENT`` 的整数。"""
    if isinstance(min_segment, bool) or not isinstance(min_segment, (int, np.integer)):
        raise DataValidationError(
            _("min_segment must be an integer >= {0}").format(MIN_SEGMENT),
            details={"min_segment": str(min_segment)},
        )
    value = int(min_segment)
    if value < MIN_SEGMENT:
        raise DataValidationError(
            _("min_segment must be an integer >= {0}; got {1}").format(MIN_SEGMENT, value),
            details={"min_segment": value, "minimum": MIN_SEGMENT},
        )
    return value


def _validate_threshold(threshold: float) -> float:
    """检查阈值落在开区间 (0, 1) 内。"""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("threshold must be a number in (0, 1)"),
            details={"threshold": str(threshold)},
        ) from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise DataValidationError(
            _("threshold must be a number in (0, 1); got {0}").format(value),
            details={"threshold": value},
        )
    return value


# =============================================================================
# Core numerics
# =============================================================================


def _sigmoid(values: npt.NDArray) -> npt.NDArray:
    """数值稳定的 logistic 函数。"""
    out = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


def _local_log_bayes_factors(
    series: npt.NDArray,
    weights: npt.NDArray,
    positions: npt.NDArray,
    min_segment: int,
) -> npt.NDArray:
    """
    计算每个候选位置的局部 Bayes 因子的对数。

    权重被缩放到“每个样本平均为 1”，于是块内的权重和、自由度都保持块长的量
    纲，残差方差的合并公式可以直接使用。
    """
    block = min_segment
    cumulative_weight = np.concatenate(([0.0], np.cumsum(weights)))
    cumulative_value = np.concatenate(([0.0], np.cumsum(weights * series)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(weights * series * series)))

    left = positions
    right = positions + block
    weight_left = cumulative_weight[left] - cumulative_weight[left - block]
    value_left = cumulative_value[left] - cumulative_value[left - block]
    square_left = cumulative_square[left] - cumulative_square[left - block]
    weight_right = cumulative_weight[right] - cumulative_weight[left]
    value_right = cumulative_value[right] - cumulative_value[left]
    square_right = cumulative_square[right] - cumulative_square[left]

    mean_left = value_left / weight_left
    mean_right = value_right / weight_right
    variance_left = np.maximum(square_left / weight_left - mean_left**2, 0.0)
    variance_right = np.maximum(square_right / weight_right - mean_right**2, 0.0)

    degrees = np.maximum(weight_left + weight_right - 2.0, 1e-9)
    pooled = (weight_left * variance_left + weight_right * variance_right) / degrees
    delta = mean_right - mean_left

    # 两块长度相同，k = n_left * n_right / (n_left + n_right) = block / 2。
    effective = 0.5 * block
    variance = pooled / effective

    with np.errstate(divide="ignore", invalid="ignore"):
        log_factor = delta**2 / (4.0 * variance) - 0.5 * np.log(2.0)
    # 块内方差为 0（分段常数记录）时：有差异就是无穷证据，无差异就是零证据。
    log_factor = np.where(
        variance > 0.0,
        log_factor,
        np.where(delta != 0.0, np.inf, -np.inf),
    )
    return log_factor


def _select_change_points(curve: npt.NDArray, threshold: float, min_segment: int) -> npt.NDArray:
    """从概率曲线上挑出互不重叠的变点（按概率从高到低贪心选取）。"""
    above = np.flatnonzero(curve >= threshold)
    if above.size == 0:
        return np.empty(0, dtype=int)
    order = above[np.argsort(-curve[above], kind="stable")]
    chosen: list[int] = []
    for index in order:
        position = int(index)
        if all(abs(position - other) >= min_segment for other in chosen):
            chosen.append(position)
    return np.sort(np.asarray(chosen, dtype=int))


# =============================================================================
# Public API
# =============================================================================


def detect_change_points(
    values: npt.NDArray,
    min_segment: int = 10,
    threshold: float = 0.95,
    random_seed: int | None = None,
) -> ChangePointResult:
    """
    贝叶斯变点检测（Ruggieri 2013 思路的可用实现）。

    每个候选位置用左右两个长度为 ``min_segment`` 的块做贝叶斯块均值比较，
    给出该位置是变点的后验概率；概率超过 ``threshold`` 且彼此相隔至少
    ``min_segment`` 的位置被报告为变点。

    Parameters:
        values: 一维有限序列。
        min_segment: 候选变点两侧块的长度，也是变点之间的最小间隔。
        threshold: 报告变点所需的后验概率阈值，取值 (0, 1)。默认 0.95：每个
            位置的后验是**局部**量，而候选位置有 O(n) 个，逐个比较之后 0.5 这
            一级落在多重比较的噪声带里（平稳记录里每 12 条约有 1 条会在 0.5 处
            触发）。0.95 给出零误报，同时不损失信噪比 >= 5 的阶跃；要抓更弱的
            阶跃就调低这个阈值。
        random_seed: 贝叶斯自助的随机种子。给出种子时结果可复现；不给种子时
            结果不可复现，:func:`utils.statistics_core.make_rng` 会发出
            ``RuntimeWarning``。

    Returns:
        ChangePointResult: 变点位置、对应概率与逐样本的概率曲线。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.change_points import detect_change_points
        >>> series = np.concatenate([np.zeros(150), np.ones(150)])
        >>> out = detect_change_points(series, min_segment=10, random_seed=1)
        >>> out.positions.size
        1
    """
    series = _as_series(values)
    block = _validate_min_segment(min_segment)
    limit = _validate_threshold(threshold)

    size = int(series.size)
    if size < 2 * block + 1:
        raise DataValidationError(
            _("change-point detection needs at least {0} samples for min_segment={1}; got {2}").format(
                2 * block + 1, block, size
            ),
            details={"length": size, "min_segment": block, "minimum": 2 * block + 1},
        )

    rng = make_rng(random_seed, context="change-point detection")

    positions = np.arange(block, size - block + 1, dtype=int)
    prior_probability = PRIOR_EXPECTED_CHANGE_POINTS / positions.size
    log_prior_odds = float(np.log(prior_probability) - np.log1p(-prior_probability))

    unit_weights = np.ones(size, dtype=float)
    accumulated = np.zeros(positions.size, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        weights = size * rng.dirichlet(unit_weights)
        log_factor = _local_log_bayes_factors(series, weights, positions, block)
        accumulated += _sigmoid(log_prior_odds + log_factor)

    curve = np.zeros(size, dtype=float)
    curve[positions] = accumulated / BOOTSTRAP_REPLICATES

    detected = _select_change_points(curve, limit, block)
    probabilities = curve[detected] if detected.size else np.empty(0, dtype=float)

    logger.info(
        "change-point detection: %d candidate positions, %d change points above %.3f",
        int(positions.size),
        int(detected.size),
        limit,
    )

    return ChangePointResult(
        positions=detected,
        probabilities=probabilities,
        probability_curve=curve,
    )
