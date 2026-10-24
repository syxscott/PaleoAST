# =============================================================================
# FILE: stratigraphy/recurrence.py
# =============================================================================
"""
递归图与递归定量分析（recurrence plot & recurrence quantification analysis, RQA）。

用途
====
把一条地层记录（δ¹⁸O、粒度、磁化率……）变成一张**递归图**（recurrence plot），
并给出它的定量指标 **DET / MDL / LAM / LAMIN / ENTROPY**。这套指标回答的是
"这条记录在多大程度上重复访问了相同的状态"：

* **DET**（determinism）—— 复发点里有多大比例落在对角线结构上；
* **MDL**（mean diagonal length）—— 对角线的平均长度；
* **LAM**（laminarity）—— 垂直线的平均长度；
* **LAMIN**（laminarity without the Theiler window）—— 不施加 Theiler 窗时的
  垂直线平均长度；
* **ENTROPY** —— 垂直线长度分布的 Shannon 熵。

古环境研究里常用 DET 与 LAMIN 这一对：周期性的轨道线（旋回）会产生长的
对角线（DET 高），而"停在某个状态里"的阶段会产生长的垂直线（LAMIN 高）。

定义
====
**递归矩阵**：设轨迹为 ``x_1 .. x_N``（一维序列，或 ``N x d`` 的相空间点列），
两点距离取某个范数 ``d(x_i, x_j)``，则

    R(i, j) = 1  当且仅当  d(x_i, x_j) <= radius

半径 ``radius`` 决定"多近才算重复"。本模块支持三种距离度量：``"euclidean"``
（2-范数，默认）、``"manhattan"``（1-范数）、``"chebyshev"``（无穷范数）。
注意：**一维输入时三种范数给出完全相同的距离**（``|x_i - x_j|``），要区分它们
必须传入多维的相空间点列。

**Theiler 窗**：Theiler 窗是用来剔除"平凡邻近"（trivial neighbours）的带状
区域，即主对角线附近 ``|i - j| <= theiler_window`` 的那些点对。**只有
``|i - j| > theiler_window`` 的点对才计入 DET / MDL / LAM**。
本实现的口径：

* ``theiler_window = 0``（缺省）：**不剔除任何点对**（主对角线本身也参与统计）；
* ``theiler_window > 0``：剔除 ``|i - j| <= theiler_window`` 的整条带状区域
  （含主对角线），剩下的点对才参与统计。被剔除的点在矩阵里被置为 0，因此
  它们同时会**切断**穿过该带的线结构。

**线结构与最小长度阈值**：对角线结构是沿对角线方向的极大连续复发点段，
垂直线结构是沿列方向的极大连续复发点段。长度小于 ``min_line_length``
（``v_min``，缺省 2，Marwan et al. 2007 的缺省值）的线**不计入**任何指标，
但它们仍然是复发点，仍然进入 DET 的分母。

**各指标的确切定义**（分析矩阵 = 施加 Theiler 窗之后的矩阵；未加窗的原始
矩阵称为完整矩阵）：

* **DET** = 分析矩阵中所有长度 ``>= v_min`` 的对角线长度之和 / 分析矩阵的
  复发点总数。分母是**复发点总数**（不是"落在对角线上的点数"），所以
  ``DET`` 落在 ``[0, 1]``。
* **MDL** = 分析矩阵中所有长度 ``>= v_min`` 的对角线长度的平均值。
  （没有这样的对角线时取 0.0）。
* **LAM** = 分析矩阵中所有长度 ``>= v_min`` 的垂直线长度的平均值
  （没有这样的垂直线时取 0.0）。
* **LAMIN** = **完整矩阵**（不施加 Theiler 窗）中所有长度 ``>= v_min`` 的
  垂直线长度的平均值，即 Marwan et al. (2007) 所说的 "laminarity without
  considering the Theiler window"。
* **ENTROPY** = 分析矩阵中垂直线长度分布的 Shannon 熵（以 2 为底，单位 bit）：

      H = -sum_v p_v * log2(p_v),   p_v = n_v / sum_v n_v

  其中 ``n_v`` 是长度为 ``v`` 的垂直线条数，``v`` 只取 ``>= v_min`` 的长度。

因为 Theiler 窗会把垂直线**切断**（完整矩阵里的一条长垂直线在分析矩阵里被
拆成若干更短的片段，甚至短到 ``v_min`` 以下而被丢弃），而 LAMIN 用的是
**没有被切断**的完整矩阵，所以一般有 ``LAM <= LAMIN``；``theiler_window = 0``
时两者相等。

参数选择
========
* **半径**：给 ``radius``（绝对距离）或 ``radius_fraction``（相对半径，占
  **最大点对距离**的比例，取值 ``[0, 1]``），两者必须且只能给一个。
  ``radius_fraction`` 特别适合"不关心量纲"的地层记录：它随记录的幅度自适应。
  返回的 ``radii`` 是**实际采用**的半径（``radius_fraction`` 会被换算成绝对值）。
* **Theiler 窗**：对于含有轨道线（旋回）的地层记录，窗宽至少要盖住"相邻
  旋回"之间的距离，否则平凡邻近会把 DET 抬得很高。窗宽为 0 意味着不做任何
  剔除，此时主对角线（N 个平凡点）也进入统计，DET 至少为 ``1/N``。
* **最小线长**：``min_line_length = 2`` 是 Marwan et al. (2007) 的缺省值；
  想统计"只有一个点的线"可以设成 1。

与 Acycle 的差异
===============
Acycle（Li, Hinnov & Kump 2019，GPL-3.0）的 "recurrence plot" 只画图，
"DET" 用滑动窗口逐窗计算。本文件是**独立实现**，没有翻译其 MATLAB 源码，
只参考了公开的行为约定（同样的距离矩阵与阈值语义）。具体差异：

* Theiler 窗的边界：Acycle 的 ``crp_pdist`` 保留 ``|i - j| >= w``（``w = 1``
  时只剔除主对角线）；本实现剔除 ``|i - j| <= w``，与 Marwan et al. (2007)
  的 Theiler 窗定义一致（``w = 0`` 两者都不剔除）。
* Acycle 的 DET 用 ``lmin`` 过滤对角线长度，分母是窗口内的复发点总数；
  本实现的 DET 定义与之**一致**（分子是长度 ``>= v_min`` 的对角线长度之和，
  分母是分析矩阵的复发点总数）。
* 本实现额外给出 MDL / LAM / LAMIN / ENTROPY 与多种距离度量。

参考文献
========
* Eckmann, J.-P., Oliffson Kamphorst, S. & Ruelle, D. (1987). Recurrence Plots
  for Dynamical Systems. *Europhysics Letters*, 4(9), 417-427.（递归图与递归
  点的原始定义）
* Marwan, N., Kurths, J. & Herrmann, W. (2007). Recurrence plots for the
  analysis of complex systems. *Physics Reports*, 439(1-2), 1-74.（RQA 各指标
  的定义、Theiler 窗、最小线长 ``v_min``、度量选择）
* Small, M. (2012). *Nonlinear Time Series Analysis.* Cambridge University
  Press.（RQA 的标准参考）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "recurrence plot" 子功能；本模块为独立实现）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 递归图至少需要的点数（少于三个点没有任何线结构可言）。
MIN_POINTS = 3

#: 线结构（对角线、垂直线）的最小长度阈值 ``v_min``（Marwan et al. 2007 的缺省值）。
DEFAULT_MIN_LINE_LENGTH = 2

#: 支持的距离度量。
SUPPORTED_METRICS = ("euclidean", "manhattan", "chebyshev")


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class RecurrenceQuantification:
    """
    递归定量分析（RQA）的各项指标。

    Attributes:
        radii: 实际采用的半径（生成该递归矩阵所用的距离阈值）。
        det: 确定性（determinism）—— 落在对角线结构上的复发点比例，取值
            ``[0, 1]``。
        mdl: 平均对角线长度（mean diagonal length）。
        lam: 层流性（laminarity）—— Theiler 窗生效时的垂直线平均长度。
        laminarity: 不施加 Theiler 窗时的垂直线平均长度（Marwan et al. 2007
            的 "laminarity without considering the Theiler window"，通常记作
            LAMIN）。
        entropy: 垂直线长度分布的 Shannon 熵（以 2 为底，单位 bit）。
        recurrence_rate: 分析矩阵的复发点比例（``RR``）。
        theiler_window: 使用的 Theiler 窗宽（``0`` 表示不剔除任何点对）。
        min_line_length: 线结构的最小长度阈值 ``v_min``。
        n_points: 轨迹的点数。
        n_recurrence_points: 分析矩阵中的复发点总数（DET 的分母）。
        n_diagonal_lines: 计入统计的对角线条数（长度 ``>= v_min``）。
        n_vertical_lines: 计入统计的垂直线条数（长度 ``>= v_min``，Theiler 窗
            生效时的分析矩阵）。
    """

    radii: float
    det: float
    mdl: float
    lam: float
    laminarity: float
    entropy: float
    recurrence_rate: float
    theiler_window: int
    min_line_length: int
    n_points: int
    n_recurrence_points: int
    n_diagonal_lines: int
    n_vertical_lines: int


@dataclass(frozen=True)
class RecurrencePlotResult:
    """
    递归图及其定量指标。

    Attributes:
        matrix: 递归矩阵（``bool`` 的 ``N x N`` 数组），``matrix[i, j]`` 为
            ``True`` 表示 ``x_i`` 与 ``x_j`` 的距离不超过半径。主对角线恒为
            ``True``（自身距离为 0）。
        radii: 实际采用的半径。
        det: 确定性（determinism），取值 ``[0, 1]``。
        mdl: 平均对角线长度。
        lam: 层流性（laminarity）—— Theiler 窗生效时的垂直线平均长度。
        laminarity: 不施加 Theiler 窗时的垂直线平均长度（LAMIN）。
        entropy: 垂直线长度分布的 Shannon 熵（bit）。
        recurrence_rate: 复发点比例（``RR``）。
        metric: 使用的距离度量名称。
        theiler_window: 使用的 Theiler 窗宽。
        min_line_length: 线结构的最小长度阈值 ``v_min``。
        n_points: 轨迹的点数。
        n_recurrence_points: 分析矩阵中的复发点总数。
        n_diagonal_lines: 计入统计的对角线条数。
        n_vertical_lines: 计入统计的垂直线条数。
    """

    matrix: npt.NDArray
    radii: float
    det: float
    mdl: float
    lam: float
    laminarity: float
    entropy: float
    recurrence_rate: float
    metric: str
    theiler_window: int
    min_line_length: int
    n_points: int
    n_recurrence_points: int
    n_diagonal_lines: int
    n_vertical_lines: int

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Recurrence plot"),
            "=" * 48,
            _("Points: {0}").format(self.n_points),
            _("Radius: {0}").format(f"{self.radii:.6g}"),
            _("Distance metric: {0}").format(self.metric),
            _("Theiler window: {0}").format(self.theiler_window),
            _("Recurrence rate: {0}").format(f"{self.recurrence_rate:.4f}"),
            _("DET: {0}").format(f"{self.det:.4f}"),
            _("MDL: {0}").format(f"{self.mdl:.4f}"),
            _("LAM: {0}").format(f"{self.lam:.4f}"),
            _("LAMIN: {0}").format(f"{self.laminarity:.4f}"),
            _("ENTROPY: {0}").format(f"{self.entropy:.4f}"),
        ]
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================


def _as_trajectory(values: object, name: str, *, minimum_size: int) -> npt.NDArray:
    """把输入转换为一维或二维的有限浮点数组，并检查点数下限。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if array.ndim not in (1, 2):
        raise ValidationError(
            _("{0} must be a one- or two-dimensional array").format(name),
            details={"parameter": name, "ndim": int(array.ndim)},
        )
    if array.shape[0] < minimum_size:
        raise ValidationError(
            _("{0} needs at least {1} points; got {2}").format(name, minimum_size, int(array.shape[0])),
            details={"parameter": name, "length": int(array.shape[0]), "minimum": minimum_size},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


def _as_binary_matrix(matrix: object, *, minimum_size: int) -> npt.NDArray:
    """把递归矩阵转换成 ``bool`` 数组，并检查形状。"""
    try:
        array = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("the recurrence matrix must be an array of zeros and ones"),
            details={"parameter": "matrix"},
            original_exception=exc,
        ) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValidationError(
            _("the recurrence matrix must be square"),
            details={"shape": [int(array.shape[0]), int(array.shape[1])] if array.ndim == 2 else int(array.ndim)},
        )
    if array.shape[0] < minimum_size:
        raise ValidationError(
            _("the recurrence matrix needs at least {0} points; got {1}").format(minimum_size, int(array.shape[0])),
            details={"length": int(array.shape[0]), "minimum": minimum_size},
        )
    return np.asarray(array, dtype=bool)


def _validate_metric(metric: object) -> str:
    """校验并归一化距离度量名称。"""
    if not isinstance(metric, str):
        raise ValidationError(
            _("metric must be one of: {0}").format(", ".join(SUPPORTED_METRICS)),
            details={"metric": repr(metric)},
        )
    name = metric.strip().lower()
    if name not in SUPPORTED_METRICS:
        raise ValidationError(
            _("metric must be one of: {0}").format(", ".join(SUPPORTED_METRICS)),
            details={"metric": metric},
        )
    return name


def _validate_theiler_window(theiler_window: object) -> int:
    """校验 Theiler 窗宽（非负整数）。"""
    window = int(theiler_window)
    if window != theiler_window or window < 0:
        raise ValidationError(
            _("theiler_window must be a non-negative integer"),
            details={"theiler_window": theiler_window},
        )
    return window


def _validate_min_line_length(min_line_length: object) -> int:
    """校验线结构的最小长度阈值 ``v_min``（正整数）。"""
    threshold = int(min_line_length)
    if threshold != min_line_length or threshold < 1:
        raise ValidationError(
            _("min_line_length must be a positive integer"),
            details={"min_line_length": min_line_length},
        )
    return threshold


def _validate_radius(radius: object) -> float:
    """校验绝对半径（非负有限数）。"""
    try:
        value = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("radius must be a non-negative number"),
            details={"radius": repr(radius)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValidationError(
            _("radius must be a non-negative number"),
            details={"radius": radius},
        )
    return value


def _validate_radius_fraction(radius_fraction: object) -> float:
    """校验相对半径（``[0, 1]`` 之间的有限数）。"""
    try:
        value = float(radius_fraction)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("radius_fraction must lie between 0 and 1"),
            details={"radius_fraction": repr(radius_fraction)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValidationError(
            _("radius_fraction must lie between 0 and 1"),
            details={"radius_fraction": radius_fraction},
        )
    return value


def _single_radius(radii: object) -> float:
    """把 ``radii`` 归一化成单个半径。

    ``radii`` 可以是一个数，也可以是只含一个元素的序列。多个半径无法从
    已经有的递归矩阵里区分出来，因此直接报错。
    """
    if isinstance(radii, Sequence) and not isinstance(radii, (str, bytes)):
        items = list(radii)
        if len(items) != 1:
            raise ValidationError(
                _("radii must contain exactly one radius; use recurrence_plot() to sweep a range of radii"),
                details={"n_radii": len(items)},
            )
        return _validate_radius(items[0])
    return _validate_radius(radii)


def _distance_matrix(trajectory: npt.NDArray, metric: str) -> npt.NDArray:
    """计算点对距离矩阵（``N x N``）。

    全部用 numpy 向量化，不写 Python 双重循环，所以 ``N = 2000`` 时的
    ``4 x 10^6`` 个距离也只是几十 MB 的临时数组。
    """
    points = trajectory if trajectory.ndim == 2 else trajectory[:, None]
    difference = points[:, None, :] - points[None, :, :]
    if metric == "manhattan":
        return np.abs(difference).sum(axis=-1)
    if metric == "chebyshev":
        return np.abs(difference).max(axis=-1)
    return np.sqrt(np.square(difference).sum(axis=-1))


def _apply_theiler_window(matrix: npt.NDArray, theiler_window: int) -> npt.NDArray:
    """剔除 Theiler 窗内的点对（``|i - j| <= theiler_window``）。

    窗宽为 0 时原样返回（不剔除任何点对）。
    """
    if theiler_window <= 0:
        return matrix
    index = np.arange(matrix.shape[0], dtype=np.int32)
    separation = np.abs(index[:, None] - index[None, :])
    return np.asarray(matrix & (separation > theiler_window))


def _run_lengths(flags: npt.NDArray) -> npt.NDArray:
    """返回布尔序列中所有极大连续 ``True`` 段的长度。"""
    if flags.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(
        (
            np.zeros(1, dtype=np.int8),
            np.asarray(flags, dtype=np.int8),
            np.zeros(1, dtype=np.int8),
        )
    )
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return (ends - starts).astype(np.int64)


def _diagonal_line_lengths(matrix: npt.NDArray) -> npt.NDArray:
    """沿所有对角线（含主对角线）统计极大连续复发点段的长度。"""
    size = matrix.shape[0]
    lengths = [_run_lengths(np.diagonal(matrix, offset=offset)) for offset in range(-(size - 1), size)]
    if not lengths:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(lengths)


def _vertical_line_lengths(matrix: npt.NDArray) -> npt.NDArray:
    """沿所有列统计极大连续复发点段的长度。"""
    size = matrix.shape[0]
    lengths = [_run_lengths(matrix[:, column]) for column in range(size)]
    if not lengths:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(lengths)


def _mean_length(lengths: npt.NDArray) -> float:
    """线段长度的平均值（整数求和后只做一次除法，保证可精确复现）。"""
    if lengths.size == 0:
        return 0.0
    return float(np.sum(lengths, dtype=np.int64)) / float(lengths.size)


def _shannon_entropy(lengths: npt.NDArray) -> float:
    """线段长度分布的 Shannon 熵（以 2 为底，单位 bit）。"""
    if lengths.size == 0:
        return 0.0
    counts = np.bincount(lengths)
    probabilities = counts[counts > 0].astype(float) / float(lengths.size)
    # ``+ 0.0`` 把 ``-0.0`` 归一成 ``0.0``（单一长度分布的熵恰为 0）。
    return float(-np.sum(probabilities * np.log2(probabilities))) + 0.0


# =============================================================================
# Public interface
# =============================================================================


def recurrence_quantification(
    matrix: npt.NDArray,
    radii: float | Sequence[float],
    theiler_window: int = 0,
    *,
    min_line_length: int = DEFAULT_MIN_LINE_LENGTH,
) -> RecurrenceQuantification:
    """对已有的递归矩阵计算 RQA 指标（DET / MDL / LAM / LAMIN / ENTROPY）。

    这个函数不重建递归矩阵，因此可以用来**重算**指标（换一套 Theiler 窗或
    ``v_min``）或者对外部来源的递归矩阵做定量分析。

    **计入哪些点对**（与 :func:`recurrence_plot` 完全一致）：

    * Theiler 窗 ``theiler_window > 0`` 时，``|i - j| <= theiler_window`` 的
      点对被剔除（并且它们会切断穿过该带的线结构）；``theiler_window = 0``
      时不剔除任何点对。
    * DET / MDL / LAM / ENTROPY 都在**分析矩阵**（剔除之后的矩阵）上计算；
      LAMIN 在**完整矩阵**上计算。

    **线结构的最小长度阈值**：长度 ``< min_line_length`` 的对角线与垂直线
    不计入任何指标，但它们仍然是复发点，仍然进入 DET 的分母。

    **指标的定义**：

    * ``det`` = （长度 ``>= v_min`` 的对角线长度之和）/ 分析矩阵的复发点总数；
    * ``mdl`` = 长度 ``>= v_min`` 的对角线长度的平均值；
    * ``lam`` = 分析矩阵中长度 ``>= v_min`` 的垂直线长度的平均值；
    * ``laminarity`` = 完整矩阵中长度 ``>= v_min`` 的垂直线长度的平均值
      （LAMIN，Marwan et al. 2007 的 "laminarity without considering the
      Theiler window"）；
    * ``entropy`` = 分析矩阵中垂直线长度分布的 Shannon 熵（以 2 为底）。

    Parameters:
        matrix: 递归矩阵（``N x N``，元素为 ``0/1`` 或 ``bool``）。
        radii: 生成该矩阵时使用的半径。可以是一个数，也可以是只含一个元素的
            序列。半径本身不参与指标的计算，只作为记录随结果返回。
        theiler_window: Theiler 窗宽（非负整数），缺省 0（不剔除）。
        min_line_length: 线结构的最小长度阈值 ``v_min``，缺省 2。

    Returns:
        RecurrenceQuantification: 各项 RQA 指标。

    Raises:
        ValidationError: 矩阵不是方阵、点数太少、窗宽或最小线长非法，或
            ``radii`` 不是单个半径。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.recurrence import recurrence_quantification
        >>> matrix = np.eye(5, dtype=bool)
        >>> result = recurrence_quantification(matrix, radii=0.1)
        >>> bool(result.det > 0.0)
        True
    """
    binary = _as_binary_matrix(matrix, minimum_size=MIN_POINTS)
    window = _validate_theiler_window(theiler_window)
    threshold = _validate_min_line_length(min_line_length)
    radius = _single_radius(radii)

    analysis = _apply_theiler_window(binary, window)

    diagonal_lengths = _diagonal_line_lengths(analysis)
    vertical_lengths = _vertical_line_lengths(analysis)
    full_vertical_lengths = _vertical_line_lengths(binary)

    kept_diagonal = diagonal_lengths[diagonal_lengths >= threshold]
    kept_vertical = vertical_lengths[vertical_lengths >= threshold]
    kept_full_vertical = full_vertical_lengths[full_vertical_lengths >= threshold]

    n_recurrence = int(np.count_nonzero(analysis))
    det = float(np.sum(kept_diagonal, dtype=np.int64)) / float(n_recurrence) if n_recurrence else 0.0

    return RecurrenceQuantification(
        radii=radius,
        det=det,
        mdl=_mean_length(kept_diagonal),
        lam=_mean_length(kept_vertical),
        laminarity=_mean_length(kept_full_vertical),
        entropy=_shannon_entropy(kept_vertical),
        recurrence_rate=n_recurrence / float(binary.shape[0] ** 2),
        theiler_window=window,
        min_line_length=threshold,
        n_points=int(binary.shape[0]),
        n_recurrence_points=n_recurrence,
        n_diagonal_lines=int(kept_diagonal.size),
        n_vertical_lines=int(kept_vertical.size),
    )


def recurrence_plot(
    values: npt.NDArray,
    radius: float | None = None,
    radius_fraction: float | None = None,
    metric: str = "euclidean",
    theiler_window: int = 0,
    *,
    min_line_length: int = DEFAULT_MIN_LINE_LENGTH,
) -> RecurrencePlotResult:
    """计算递归图及其定量指标（RQA）。

    递归矩阵的定义、Theiler 窗的语义与各指标的确切公式见
    :func:`recurrence_quantification` 与本模块的文档字符串。

    **半径的选择**：``radius`` 与 ``radius_fraction`` 必须且只能给一个。
    ``radius_fraction`` 是"占最大点对距离的比例"（取值 ``[0, 1]``），它随记录
    的量纲自适应，适合不关心绝对尺度的地层记录。返回的 ``radii`` 是**实际
    采用**的绝对半径。

    **Theiler 窗**：``theiler_window = 0``（缺省）不剔除任何点对；
    ``theiler_window > 0`` 时 ``|i - j| <= theiler_window`` 的点对被剔除，
    只有 ``|i - j| > theiler_window`` 的点对计入 DET / MDL / LAM。

    Parameters:
        values: 轨迹。可以是一维序列（``N``），也可以是二维的相空间点列
            （``N x d``）。二维输入时三种距离度量才会给出不同结果。
        radius: 绝对半径（非负数），与 ``radius_fraction`` 二选一。
        radius_fraction: 相对半径（占最大点对距离的比例，``[0, 1]``），与
            ``radius`` 二选一。
        metric: 距离度量，``"euclidean"``（默认）、``"manhattan"`` 或
            ``"chebyshev"``。
        theiler_window: Theiler 窗宽（非负整数），缺省 0。
        min_line_length: 线结构的最小长度阈值 ``v_min``，缺省 2。

    Returns:
        RecurrencePlotResult: 递归矩阵、实际半径与各项 RQA 指标。

    Raises:
        ValidationError: 点数太少（少于 3）、含 NaN/Inf、半径与相对半径都没给
            或都给了、半径非法、距离度量未知、Theiler 窗或最小线长非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.recurrence import recurrence_plot
        >>> values = np.sin(np.linspace(0.0, 20.0 * np.pi, 200))
        >>> result = recurrence_plot(values, radius_fraction=0.1)
        >>> bool(result.det > 0.5)
        True
    """
    metric_name = _validate_metric(metric)
    window = _validate_theiler_window(theiler_window)
    threshold = _validate_min_line_length(min_line_length)
    trajectory = _as_trajectory(values, "values", minimum_size=MIN_POINTS)

    if (radius is None) == (radius_fraction is None):
        raise ValidationError(
            _("provide exactly one of radius and radius_fraction"),
            details={"radius": radius, "radius_fraction": radius_fraction},
        )

    distances = _distance_matrix(trajectory, metric_name)

    if radius is not None:
        actual_radius = _validate_radius(radius)
    else:
        fraction = _validate_radius_fraction(radius_fraction)
        actual_radius = fraction * float(np.max(distances))

    matrix = np.asarray(distances <= actual_radius)

    quantification = recurrence_quantification(
        matrix,
        actual_radius,
        window,
        min_line_length=threshold,
    )

    logger.info(
        "Recurrence plot complete: %d points, radius %.6g, metric %s, DET %.4f, LAM %.4f",
        quantification.n_points,
        actual_radius,
        metric_name,
        quantification.det,
        quantification.lam,
    )

    return RecurrencePlotResult(
        matrix=matrix,
        radii=actual_radius,
        det=quantification.det,
        mdl=quantification.mdl,
        lam=quantification.lam,
        laminarity=quantification.laminarity,
        entropy=quantification.entropy,
        recurrence_rate=quantification.recurrence_rate,
        metric=metric_name,
        theiler_window=window,
        min_line_length=threshold,
        n_points=quantification.n_points,
        n_recurrence_points=quantification.n_recurrence_points,
        n_diagonal_lines=quantification.n_diagonal_lines,
        n_vertical_lines=quantification.n_vertical_lines,
    )
