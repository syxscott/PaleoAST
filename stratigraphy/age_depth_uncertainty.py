# =============================================================================
# FILE: stratigraphy/age_depth_uncertainty.py
# =============================================================================
"""
年龄-深度不确定度模型（Undatable 风格的蒙特卡洛实现）。

用途
====
把**带不确定度的定年点**（例如有测量误差的 ¹⁴C 年龄、或有 ±σ 的磁性地层
年龄）合成为一个**年龄-深度不确定度模型**，给出

* 1–99 百分位的年龄-深度密度云（``age_depth_percentiles``）；
* 1σ / 2σ 的年龄-深度带（``sigma1_age_depth`` / ``sigma2_age_depth``）；
* 中位与平均年龄-深度曲线；
* 沉积速率的 1–99 百分位（``sedimentation_rate_percentiles``）。

这是 Acycle 的 *Timeseries → Undatable* 子功能的独立实现。Acycle 以
GPL-3.0 发布，因此本文件**不是**其 MATLAB 源码的逐行翻译：算法按下列
已发表论文的描述重新实现，只在"语义对齐"的层面参考 Acycle 的公开行为
（锚点排序、倒转剔除、自助丢点、中间点扰动、两轮锚点外推、百分位统计）。

算法
====
1. **年龄 PDF。** 每个定年点由（深度、年龄、年龄 1σ）定义，其年龄 PDF 是
   均值为该年龄、标准差为该 1σ 的**正态分布**。不确定性为零的点是退化
   正态（点质量）。
2. **蒙特卡洛抽样。** 每次模拟对每个定年点各抽一个年龄，得到一组
   年龄-深度点。
3. **自助（bootstrap）。** 每次模拟按比例**随机丢掉**一部分定年点（其余
   点照常使用），剩余点重新参与后续计算；这让模型不依赖于每一个点都
   "必须正确"。
4. **倒转剔除。** 年龄必须随深度递增。相邻两点出现"年龄不增"（倒转或
   重复）时，删掉其中**较浅**的那个点，反复迭代直到整条序列年龄随深度
   严格递增。删完后不足两点的模拟作废。
5. **中间点插入。** 每对相邻锚点之间插入一个**高斯扰动的中间点**，把沉积
   速率的不确定度沿剖面传播开；中间点的年龄被截断在两端锚点年龄之间。
6. **两轮锚点外推。** 第一轮跑完后，用第一轮的中位曲线在**最上/最下三个
   锚点窗口**上做最小二乘直线拟合，向外推出一个顶锚点和一个底锚点（它们
   的 1σ 取该窗口内 1σ 带宽度的一半再取平均），把它们与原有锚点合并后
   **再跑一轮**；最终结果取第二轮。
7. **统计。** 把每条模拟曲线在输出深度栅格上线性插值，取 1–99 百分位、
   中位数、均值、1σ/2σ 带，以及沉积速率的百分位。

与论文的差异（务必阅读）
==========================
本实现是 **Undatable 的简化复现**，不是逐位等价的移植。以下每一处都
**有意偏离**，偏离的原因与后果如下：

* **只支持正态年龄误差，不做 ¹⁴C 校准。** Acycle 的 ``udmakepdfs.m`` 对
  ¹⁴C 年龄调用 ``udmatcal``，用校准曲线把放射性碳龄转成日历年龄的概率
  分布（因此年龄分布是**非对称**、甚至多峰的）。本实现**没有**内置校准
  曲线：调用方必须传入**已经校准**的日历年龄及其 1σ，模型按正态分布抽样。
  依赖 ¹⁴C 校准曲线的研究必须另行校准后再调用。
* **不传播深度不确定度。** Acycle 的深度 PDF 有三种形态（区间均匀、点
  质量、±误差截断正态），并据此把"深度本身有误差"传播进模型。本接口
  的深度是**精确值**（严格递增），没有深度不确定度通道。
* **自助的比例口径不同。** Acycle 的 ``bootpc`` 是百分数（0..100），且
  每个定年点有一个独立的"是否参与自助"开关（``dateboot``）。本实现的
  ``bootpc`` 是**比例**（0..1，不含 1），且**所有**定年点都参与自助。
  语义本身与 Acycle 一致：每次模拟随机丢掉 ``round(bootpc * n)`` 个点。
* **中间点的扰动尺度不同。** Acycle（``udrun.m`` 第 4 步）把中间点的
  年龄与深度都以 ``xfactor * |Δ|`` 为标准差做高斯扰动（年龄和深度各自
  独立），并拒绝越出两点区间的抽样。本实现只扰动**年龄**，标准差取
  ``xfactor * (σ_i + σ_{i+1}) / 2``（锚点自身不确定度的传播），中间点的
  深度取段中点。原因是：Acycle 的 ``xfactor * |Δ|`` 尺度与锚点不确定度
  无关，于是一段"年龄完全精确"的剖面也会凭空产生一条有宽度的云；本实现
  让不确定度为零的剖面**塌缩**成锚点之间的分段线性插值，宽度只由锚点
  不确定度驱动。代价是：本实现中间点云的宽度**不是** Acycle 的"SAR 不
  确定度因子"，而是与锚点 1σ 成正比的量。
* **输出深度栅格不同。** Acycle 在 1 个深度单位的等间距栅格上汇总。本实
  现的栅格 = 锚点深度 + 每对相邻锚点之间等分插入的
  ``DEPTH_POINTS_PER_SEGMENT`` 个内点。栅格随锚点集合变化，两轮之间也
  会变化（第二轮多了两个外推锚点）。
* **沉积速率的约定不同。** Acycle 的 SAR 是"深度 / 时间"（m/ky）。本实现
  的 :func:`sedimentation_rate_from_age_depth` 返回 **d(age)/d(depth)**（年
  龄单位 / 深度单位），即 Acycle SAR 的**倒数**；这样"年龄随深度的变化
  率"与"沉积速率"在一次调用里保持同一套单位约定，取倒数即可得到 m/ky。
* **倒转剔除的批量删除。** Acycle 把违规的两点同时置 NaN 并重排（每一
  轮全部违规点同时删除）。本实现用等价的"同轮删除所有违规对的较浅点、
  再迭代"规则，结果与 Acycle 一致（保留较深的点）。
* **两轮结构无条件启用。** Acycle 只在"存在深度不确定度"时才做锚点外推
  的第二轮。本实现由 ``anchor_mode`` 显式控制（``"extrapolate"`` 默认，
  ``"none"`` 只跑一轮）。

参考文献
========
* Lougheed, C. M. & Obrochta, G. W. (2019). Uncertainty in age-depth models
  and its impact on paleoclimate reconstructions. *Paleoceanography and
  Paleoclimatology*, 34(7), 122-133. doi:10.1029/2018PA003457.
  （Undatable 的算法论文：蒙特卡洛年龄-深度模型、锚点外推与不确定度传播。）
* Lougheed, C. M. & Obrochta, G. W. (2016). Undatable: A software package
  for objective age-depth modeling. *Journal of Open Research Software*, 4(1),
  e42. doi:10.5334/jors.2016.42.（软件论文。）
* Bronk Ramsey, C. (2008). Radiocarbon calibration: individual ages from
  radiocarbon years. *Radiocarbon*, 50(4), 1303-1315.（¹⁴C 定年校准原理；
  本模块**不**实现校准，只使用校准后的年龄。）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "Undatable" 子功能。）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定，
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
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_AGE_BINS",
    "DEPTH_POINTS_PER_SEGMENT",
    "MIN_POINTS",
    "MIN_SIMULATIONS",
    "AgeDepthDensity",
    "AgeDepthUncertaintyResult",
    "age_depth_density",
    "sedimentation_rate_from_age_depth",
    "undatable_age_depth_model",
]

#: 一次年龄-深度模型至少需要的定年点数：两点才能定一条年龄-深度曲线。
MIN_POINTS = 2

#: 允许的最少蒙特卡洛次数。
MIN_SIMULATIONS = 1

#: 自助比例的上界（不含）：自助会丢掉定年点，丢光就没有曲线可言。
MAX_BOOTPC = 1.0

#: 相邻锚点之间等分插入的深度点数（不含两端锚点）。
DEPTH_POINTS_PER_SEGMENT = 9

#: 密度云的默认年龄分箱数。
DEFAULT_AGE_BINS = 64

#: 报告的百分位（1..99）。
PERCENTILE_GRID = np.arange(1, 100, dtype=float)

#: 正态分布 1σ 区间的百分位边界（erf(1/sqrt(2)) = 0.6826894921）。
SIGMA1_LOW_PERCENTILE = 15.8655257
SIGMA1_HIGH_PERCENTILE = 84.1344743

#: 正态分布 2σ 区间的百分位边界（erf(2/sqrt(2)) = 0.9544997361）。
SIGMA2_LOW_PERCENTILE = 2.2750131
SIGMA2_HIGH_PERCENTILE = 97.7249869

#: 中间点年龄被拒绝重抽的最大次数（超过则截断到锚点区间）。
_MAX_RESAMPLE = 64

#: 支持的锚点外推模式。
ANCHOR_MODES = ("none", "extrapolate")


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class AgeDepthDensity:
    """年龄-深度密度云（供绘图用）。

    Attributes:
        density: 密度值，形状 ``(n_depth_bins, n_age_bins)``；所有格子的
            密度之和为 1（是**概率质量**，不是"计数"）。
        depth_edges: 深度分箱的边界，长度 ``n_depth_bins + 1``。
        age_edges: 年龄分箱的边界，长度 ``n_age_bins + 1``。
    """

    density: npt.NDArray
    depth_edges: npt.NDArray
    age_edges: npt.NDArray

    @property
    def depth_centers(self) -> npt.NDArray:
        """深度分箱的中心。"""
        return 0.5 * (self.depth_edges[:-1] + self.depth_edges[1:])

    @property
    def age_centers(self) -> npt.NDArray:
        """年龄分箱的中心。"""
        return 0.5 * (self.age_edges[:-1] + self.age_edges[1:])


@dataclass(frozen=True)
class AgeDepthUncertaintyResult:
    """年龄-深度不确定度模型的结果。

    Attributes:
        depths: 输出深度栅格（升序，m）。栅格 = 最终一轮使用的锚点深度 +
            每对相邻锚点之间等分插入的内点。
        age_depth_percentiles: 1–99 百分位的年龄带，形状
            ``(99, n_depth)``；第 ``k`` 行是第 ``k`` 百分位（``k`` 从 1
            起）。
        median_age_depth: 中位年龄-深度曲线，形状 ``(n_depth,)``。
        mean_age_depth: 平均年龄-深度曲线，形状 ``(n_depth,)``。
        sigma1_age_depth: 1σ 带，形状 ``(2, n_depth)``；第 0 行是下界
            （15.87 百分位），第 1 行是上界（84.13 百分位）。
        sigma2_age_depth: 2σ 带，形状 ``(2, n_depth)``；第 0 行是下界
            （2.28 百分位），第 1 行是上界（97.72 百分位）。
        sedimentation_rate_percentiles: 沉积速率（d(age)/d(depth)）的
            1–99 百分位，形状 ``(99, n_depth)``。
        median_sedimentation_rate: 沉积速率的中位数，形状 ``(n_depth,)``。
        age_samples: 保留下来的蒙特卡洛年龄样本，形状
            ``(n_depth, n_accepted)``；每一列是一次模拟。
        n_simulations: 请求的蒙特卡洛次数。
        n_accepted: 通过倒转剔除后剩余的模拟数（每一次模拟给出一条曲线）。
    """

    depths: npt.NDArray
    age_depth_percentiles: npt.NDArray
    median_age_depth: npt.NDArray
    mean_age_depth: npt.NDArray
    sigma1_age_depth: npt.NDArray
    sigma2_age_depth: npt.NDArray
    sedimentation_rate_percentiles: npt.NDArray
    median_sedimentation_rate: npt.NDArray
    age_samples: npt.NDArray
    n_simulations: int
    n_accepted: int

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Age-depth uncertainty model"),
            "=" * 48,
            _("Depth points: {0}").format(self.depths.size),
            _("Simulations: {0}").format(self.n_simulations),
            _("Accepted simulations: {0}").format(self.n_accepted),
        ]
        if self.depths.size:
            lines.append(
                _("Median age range: {0} to {1}").format(
                    f"{self.median_age_depth[0]:.6g}", f"{self.median_age_depth[-1]:.6g}"
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


def _validate_uncertainties(uncertainties: object | None, size: int) -> npt.NDArray:
    """校验年龄不确定度：缺省为全零，一维、非负、长度与定年点一致。"""
    if uncertainties is None:
        return np.zeros(size, dtype=float)
    values = _as_vector(uncertainties, "age_uncertainties", minimum_size=1)
    if values.size != size:
        raise ValidationError(
            _("age_uncertainties must have the same length as the dates"),
            details={"age_uncertainties_length": int(values.size), "dates": int(size)},
        )
    if np.any(values < 0.0):
        raise ValidationError(
            _("age_uncertainties must not be negative"),
            details={"minimum": float(np.min(values))},
        )
    return values


def _validate_simulation_count(nsim: object) -> int:
    """校验蒙特卡洛次数：必须是 >= 1 的整数。"""
    try:
        count = int(nsim)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("nsim must be an integer of at least {0}").format(MIN_SIMULATIONS),
            details={"nsim": repr(nsim)},
            original_exception=exc,
        ) from exc
    if count != nsim or count < MIN_SIMULATIONS:  # type: ignore[comparison-overlap]
        raise ValidationError(
            _("nsim must be an integer of at least {0}").format(MIN_SIMULATIONS),
            details={"nsim": repr(nsim), "minimum": MIN_SIMULATIONS},
        )
    return count


def _validate_bootpc(bootpc: object) -> float:
    """校验自助比例：必须是 ``0 <= bootpc < 1`` 的有限数。

    Acycle 用的是**百分数**（0..100）；本实现用**比例**（0..1）。
    """
    try:
        fraction = float(bootpc)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("bootpc must be a fraction between 0 and 1"),
            details={"bootpc": repr(bootpc)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(fraction) or not (0.0 <= fraction < MAX_BOOTPC):
        raise ValidationError(
            _("bootpc must be a fraction between 0 and 1"),
            details={"bootpc": fraction},
        )
    return fraction


def _validate_xfactor(xfactor: object) -> float:
    """校验中间点扰动因子：必须是非负有限数。"""
    try:
        factor = float(xfactor)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("xfactor must be a non-negative number"),
            details={"xfactor": repr(xfactor)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(factor) or factor < 0.0:
        raise ValidationError(
            _("xfactor must be a non-negative number"),
            details={"xfactor": factor},
        )
    return factor


def _validate_anchor_mode(anchor_mode: object) -> str:
    """校验锚点外推模式。"""
    mode = str(anchor_mode)
    if mode not in ANCHOR_MODES:
        raise ValidationError(
            _("anchor_mode must be one of: {0}").format(", ".join(ANCHOR_MODES)),
            details={"anchor_mode": mode, "supported": list(ANCHOR_MODES)},
        )
    return mode


def _build_grid(anchor_depths: npt.NDArray) -> npt.NDArray:
    """由锚点深度构造输出栅格：锚点深度 + 每段等分插入的内点（升序）。"""
    pieces = [anchor_depths]
    for index in range(anchor_depths.size - 1):
        lower = anchor_depths[index]
        upper = anchor_depths[index + 1]
        pieces.append(np.linspace(lower, upper, DEPTH_POINTS_PER_SEGMENT + 2)[1:-1])
    return np.unique(np.concatenate(pieces))


def _remove_age_reversals(
    depths: npt.NDArray,
    ages: npt.NDArray,
    sigmas: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """剔除年龄随深度倒转（或重复）的点。

    规则与 Acycle（``udrun.m`` 第 3 步）等价：年龄必须随深度**严格递增**；
    每轮把"年龄不增"的相邻对里**较浅**的那个点删掉，迭代到整条序列单调。
    """
    keep = np.ones(depths.size, dtype=bool)
    while True:
        valid = np.flatnonzero(keep)
        if valid.size < 2:
            break
        valid_ages = ages[valid]
        violations = np.flatnonzero(valid_ages[1:] <= valid_ages[:-1])
        if violations.size == 0:
            break
        keep[valid[violations]] = False
    return depths[keep], ages[keep], sigmas[keep]


def _insert_intermediate_points(
    depths: npt.NDArray,
    ages: npt.NDArray,
    sigmas: npt.NDArray,
    xfactor: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray, npt.NDArray]:
    """在相邻锚点之间插入高斯扰动的中间点。

    见模块 docstring 的"与论文的差异"：年龄扰动的标准差是
    ``xfactor * (σ_i + σ_{i+1}) / 2``，中间点深度取段中点，年龄被截断在
    两端锚点年龄之间。
    """
    n_segments = depths.size - 1
    node_depths = np.empty(2 * depths.size - 1, dtype=float)
    node_ages = np.empty(2 * depths.size - 1, dtype=float)
    node_depths[0::2] = depths
    node_ages[0::2] = ages
    if n_segments < 1 or xfactor <= 0.0:
        node_depths[1::2] = 0.5 * (depths[:-1] + depths[1:])
        node_ages[1::2] = 0.5 * (ages[:-1] + ages[1:])
        return node_depths, node_ages

    low = np.minimum(ages[:-1], ages[1:])
    high = np.maximum(ages[:-1], ages[1:])
    middle = 0.5 * (ages[:-1] + ages[1:])
    spread = xfactor * 0.5 * (sigmas[:-1] + sigmas[1:])
    noise = rng.standard_normal(n_segments)
    candidates = middle + spread * noise
    for _attempt in range(_MAX_RESAMPLE):
        outside = (candidates < low) | (candidates > high)
        if not np.any(outside):
            break
        noise = rng.standard_normal(n_segments)
        candidates = middle + spread * noise
    candidates = np.clip(candidates, low, high)

    node_depths[1::2] = 0.5 * (depths[:-1] + depths[1:])
    node_ages[1::2] = candidates
    return node_depths, node_ages


def _interpolate_curve(
    node_depths: npt.NDArray,
    node_ages: npt.NDArray,
    grid: npt.NDArray,
) -> npt.NDArray:
    """把一条年龄-深度曲线线性插值到输出栅格上。

    曲线范围之外的栅格点记为 NaN（与 Acycle 的 ``interp1`` 行为一致），
    这些点不参与该深度的百分位统计。
    """
    values = np.interp(grid, node_depths, node_ages)
    outside = (grid < node_depths[0]) | (grid > node_depths[-1])
    if np.any(outside):
        values = values.copy()
        values[outside] = np.nan
    return values


def _bootstrap_masks(
    n_simulations: int,
    n_anchors: int,
    pool: npt.NDArray,
    bootpc: float,
    rng: np.random.Generator,
) -> npt.NDArray:
    """生成自助掩码：每次模拟随机丢掉 ``round(bootpc * |pool|)`` 个锚点。

    Acycle 的做法是"每次模拟随机保留 ``nkeep`` 个可自助的定年点"，其余
    本轮不用——语义上就是**随机丢点**，而不是有放回重抽。
    """
    masks = np.zeros((n_simulations, n_anchors), dtype=bool)
    drop_count = int(np.floor(bootpc * pool.size + 0.5))
    if drop_count <= 0 or pool.size <= drop_count:
        return masks
    noise = rng.random((n_simulations, pool.size))
    chosen = np.argpartition(noise, drop_count - 1, axis=1)[:, :drop_count]
    rows = np.repeat(np.arange(n_simulations), drop_count)
    columns = pool[chosen].ravel()
    masks[rows, columns] = True
    return masks


def _run_round(
    anchor_depths: npt.NDArray,
    anchor_ages: npt.NDArray,
    anchor_sigmas: npt.NDArray,
    grid: npt.NDArray,
    n_simulations: int,
    bootpc: float,
    xfactor: float,
    rng: np.random.Generator,
    bootstrap_pool: npt.NDArray | None = None,
) -> npt.NDArray:
    """跑一轮蒙特卡洛年龄-深度模拟。

    Returns:
        形状 ``(n_grid, n_accepted)`` 的年龄样本矩阵。
    """
    n_anchors = anchor_depths.size
    if bootstrap_pool is None:
        bootstrap_pool = np.arange(n_anchors)

    sampled = anchor_ages[:, None] + anchor_sigmas[:, None] * rng.standard_normal((n_anchors, n_simulations))
    dropped = _bootstrap_masks(n_simulations, n_anchors, bootstrap_pool, bootpc, rng)

    samples = np.full((grid.size, n_simulations), np.nan, dtype=float)
    accepted = 0
    for simulation in range(n_simulations):
        keep = ~dropped[simulation]
        depths = anchor_depths[keep]
        ages = sampled[:, simulation][keep]
        sigmas = anchor_sigmas[keep]
        depths, ages, sigmas = _remove_age_reversals(depths, ages, sigmas)
        if depths.size < 2:
            continue
        node_depths, node_ages = _insert_intermediate_points(depths, ages, sigmas, xfactor, rng)
        samples[:, accepted] = _interpolate_curve(node_depths, node_ages, grid)
        accepted += 1
    return samples[:, :accepted]


def _summarise_samples(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """把 ``(n_depths, n_simulations)`` 的样本压成 1–99 百分位与均值。

    每个深度上只统计**有限值**（自助丢点可能让某些模拟在该深度超出曲线
    范围）。
    """
    n_depths = samples.shape[0]
    percentiles = np.empty((PERCENTILE_GRID.size, n_depths), dtype=float)
    means = np.empty(n_depths, dtype=float)
    for index in range(n_depths):
        column = samples[index, :]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            percentiles[:, index] = np.nan
            means[index] = np.nan
        else:
            percentiles[:, index] = np.percentile(finite, PERCENTILE_GRID)
            means[index] = float(np.mean(finite))
    return percentiles, means


def _percentile_band(percentiles: npt.NDArray, low: float, high: float) -> npt.NDArray:
    """从 1–99 百分位表里插值出 ``[低百分位, 高百分位]`` 的带。"""
    band = np.empty((2, percentiles.shape[1]), dtype=float)
    for index in range(percentiles.shape[1]):
        column = percentiles[:, index]
        band[0, index] = np.interp(low, PERCENTILE_GRID, column)
        band[1, index] = np.interp(high, PERCENTILE_GRID, column)
    return band


def _extrapolate_anchors(
    grid: npt.NDArray,
    percentiles: npt.NDArray,
    band_low: npt.NDArray,
    band_high: npt.NDArray,
    anchor_depths: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray] | None:
    """由第一轮的年龄-深度模型外推顶/底锚点。

    做法与 Acycle 的 ``udanchors.m`` 同构：在最上/最下**三个锚点窗口**上
    对中位曲线做最小二乘直线拟合，把直线外推到窗口之外一个"两锚点间距"
    的位置，得到顶/底锚点的深度与年龄；锚点的 1σ 取该窗口内 1σ 带宽度
    的一半再对窗口平均。

    Returns:
        ``(顶锚点深度, 顶锚点年龄, 顶锚点 1σ, 底锚点深度, 底锚点年龄,
        底锚点 1σ)``，无法外推时返回 ``None``。
    """
    n_anchors = anchor_depths.size
    if n_anchors < 2:
        return None
    median = percentiles[49]

    if n_anchors >= 3:
        top_span = (grid >= anchor_depths[0]) & (grid <= anchor_depths[2])
        bottom_span = (grid >= anchor_depths[-3]) & (grid <= anchor_depths[-1])
        top_spacing = anchor_depths[2] - anchor_depths[0]
        bottom_spacing = anchor_depths[-1] - anchor_depths[-3]
    else:
        top_span = (grid >= anchor_depths[0]) & (grid <= anchor_depths[-1])
        bottom_span = top_span
        top_spacing = anchor_depths[-1] - anchor_depths[0]
        bottom_spacing = top_spacing
    if np.count_nonzero(top_span) < 2 or np.count_nonzero(bottom_span) < 2:
        return None

    top_slope, top_intercept = np.polyfit(grid[top_span], median[top_span], 1)
    bottom_slope, bottom_intercept = np.polyfit(grid[bottom_span], median[bottom_span], 1)

    top_depth = anchor_depths[0] - top_spacing
    bottom_depth = anchor_depths[-1] + bottom_spacing
    top_age = float(top_slope * top_depth + top_intercept)
    bottom_age = float(bottom_slope * bottom_depth + bottom_intercept)
    top_sigma = float(np.mean(band_high[top_span] - band_low[top_span]) / 2.0)
    bottom_sigma = float(np.mean(band_high[bottom_span] - band_low[bottom_span]) / 2.0)

    # 外推锚点必须与相邻锚点保持年龄序，否则第二轮会制造出人为的倒转。
    if not top_age < median[0] or not bottom_age > median[-1]:
        logger.info("age-depth: extrapolated anchors are not age-ordered; keeping the first round only")
        return None
    if not np.isfinite(top_sigma) or not np.isfinite(bottom_sigma):
        return None
    top_sigma = max(top_sigma, 0.0)
    bottom_sigma = max(bottom_sigma, 0.0)
    return (
        np.array([top_depth]),
        np.array([top_age]),
        np.array([top_sigma]),
        np.array([bottom_depth]),
        np.array([bottom_age]),
        np.array([bottom_sigma]),
    )


# =============================================================================
# Public interface
# =============================================================================


def sedimentation_rate_from_age_depth(depths: object, ages: object) -> npt.NDArray:
    """由一条年龄-深度曲线算沉积速率。

    返回的是 **d(age)/d(depth)**（年龄单位 / 深度单位），即"年龄随深度的
    变化率"。它与经典沉积速率（SAR = 深度 / 时间）互为倒数：SAR 的单位是
    m/ky，本函数返回 ky/m。

    .. note::
       Acycle 的 ``udsummary.m`` 把 SAR 定义为 ``diff(depth)/diff(age/1000)``
       （m/ky）。本模块统一采用 d(age)/d(depth)，取倒数即得 Acycle 的口径。

    Parameters:
        depths: 深度（严格递增，至少 2 个点）。
        ages: 与 ``depths`` 一一对应的年龄。

    Returns:
        形状 ``(len(depths),)`` 的数组：每个深度处的 d(age)/d(depth)。
        深度不均匀时用 ``numpy.gradient`` 的非均匀差分。

    Raises:
        ValidationError: 输入不是一维、长度不足、长度不一致、含 NaN 或深度
            非严格递增。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.age_depth_uncertainty import sedimentation_rate_from_age_depth
        >>> float(sedimentation_rate_from_age_depth([0.0, 10.0], [1000.0, 1250.0])[0])
        25.0
    """
    depth_array = _as_vector(depths, "depths", minimum_size=2)
    age_array = _as_vector(ages, "ages", minimum_size=2)
    if depth_array.size != age_array.size:
        raise ValidationError(
            _("depths and ages must have the same length"),
            details={"depths_length": int(depth_array.size), "ages_length": int(age_array.size)},
        )
    if np.any(np.diff(depth_array) <= 0.0):
        raise ValidationError(
            _("depths must contain strictly increasing values"),
            details={"n_points": int(depth_array.size)},
        )
    return np.gradient(age_array, depth_array)


def age_depth_density(
    ages: object,
    ages_sorted: object,
    *,
    n_age_bins: int = DEFAULT_AGE_BINS,
    age_range: tuple[float, float] | None = None,
) -> AgeDepthDensity:
    """把蒙特卡洛样本投影成年龄-深度密度云（供绘图用）。

    Parameters:
        ages: 蒙特卡洛年龄样本，形状 ``(n_simulations, n_depths)``——
            每一行是一次模拟，每一列是一个输出深度上的年龄。
        ages_sorted: 这些样本所在的**深度轴**（升序），长度 ``n_depths``。
            （参数名沿用上游接口的叫法，但它承载的是深度：密度云需要
            （深度、年龄）两个坐标，年龄坐标由 ``ages`` 自身给出。）
        n_age_bins: 年龄方向的分箱数，至少 2。
        age_range: 年龄方向的分箱范围 ``(最小, 最大)``；缺省时取样本的
            最小/最大值。

    Returns:
        AgeDepthDensity: 密度云（密度之和为 1）与两个方向的分箱边界。

    Raises:
        ValidationError: 样本不是二维、样本数或深度轴长度非法、深度轴非
            严格递增、分箱数越界。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.age_depth_uncertainty import age_depth_density
        >>> samples = np.array([[10.0, 20.0], [11.0, 19.0], [9.0, 21.0]])
        >>> cloud = age_depth_density(samples, np.array([0.0, 1.0]))
        >>> bool(abs(cloud.density.sum() - 1.0) < 1e-12)
        True
    """
    try:
        samples = np.asarray(ages, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("ages must be numeric"),
            details={"parameter": "ages"},
            original_exception=exc,
        ) from exc
    if samples.ndim != 2:
        raise ValidationError(
            _("ages must be a two-dimensional array of shape (n_simulations, n_depths)"),
            details={"ndim": int(samples.ndim)},
        )
    if samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValidationError(
            _("ages must contain at least one sample at one depth"),
            details={"shape": list(samples.shape)},
        )
    if not np.all(np.isfinite(samples)):
        raise ValidationError(
            _("ages must contain only finite values"),
            details={"parameter": "ages"},
        )
    depths = _as_vector(ages_sorted, "ages_sorted", minimum_size=1)
    if depths.size != samples.shape[1]:
        raise ValidationError(
            _("ages_sorted must have one depth per sample column"),
            details={"ages_sorted_length": int(depths.size), "n_depths": int(samples.shape[1])},
        )
    if depths.size > 1 and np.any(np.diff(depths) <= 0.0):
        raise ValidationError(
            _("ages_sorted must contain strictly increasing values"),
            details={"n_points": int(depths.size)},
        )
    bins = int(n_age_bins)
    if bins != n_age_bins or bins < 2:
        raise ValidationError(
            _("n_age_bins must be an integer of at least 2"),
            details={"n_age_bins": n_age_bins},
        )
    if age_range is None:
        lower = float(np.min(samples))
        upper = float(np.max(samples))
    else:
        try:
            lower = float(age_range[0])
            upper = float(age_range[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValidationError(
                _("age_range must be a pair of numbers"),
                details={"age_range": repr(age_range)},
                original_exception=exc,
            ) from exc
        if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
            raise ValidationError(
                _("age_range must satisfy minimum < maximum"),
                details={"age_range": [lower, upper]},
            )
    if upper <= lower:
        upper = lower + 1.0

    age_edges = np.linspace(lower, upper, bins + 1)
    depth_edges = np.concatenate((depths, depths[-1:] + 1.0))
    # 每个深度对应 ``n_simulations`` 个年龄样本，展开成一一对应的坐标对。
    x_values = np.tile(depths, samples.shape[0])
    counts = np.histogram2d(x_values, samples.ravel(), bins=(depth_edges, age_edges))[0]
    total = counts.sum()
    density = counts / total if total > 0 else counts
    return AgeDepthDensity(density=density, depth_edges=depth_edges, age_edges=age_edges)


def undatable_age_depth_model(
    depths: npt.NDArray,
    ages: npt.NDArray,
    age_uncertainties: npt.NDArray | None = None,
    nsim: int = 1000,
    bootpc: float = 0.1,
    xfactor: float = 5,
    anchor_mode: str = "extrapolate",
    seed: int | None = None,
) -> AgeDepthUncertaintyResult:
    """年龄-深度不确定度模型（Undatable 风格的蒙特卡洛实现）。

    输入是若干**带不确定度的定年点**（深度、年龄、年龄 1σ）；输出是 1–99
    百分位的年龄-深度密度云、1σ/2σ 带、中位与平均曲线，以及沉积速率的
    百分位。

    算法与**简化之处**见模块 docstring；要点：

    * 年龄 PDF 是正态分布（均值 = 给定年龄，标准差 = 给定 1σ）。**不做 ¹⁴C
      校准**——请传入已经校准的日历年龄。
    * 深度是精确值，**不传播深度不确定度**。
    * 自助按比例**随机丢掉**定年点；``bootpc`` 是比例（0..1），不是 Acycle
      的百分数。
    * 中间点只扰动**年龄**，扰动尺度是锚点 1σ 的倍数；因此不确定度为零的
      剖面会塌缩成锚点之间的分段线性插值。
    * ``anchor_mode="extrapolate"``（默认）会在第一轮之后外推顶/底锚点再跑
      一轮；``"none"`` 只跑一轮。

    Parameters:
        depths: 定年点深度（严格递增，至少 2 个点，m）。
        ages: 与 ``depths`` 一一对应的年龄（**已校准**的日历年龄）。
        age_uncertainties: 与 ``depths`` 一一对应的年龄 1σ；缺省时全部为 0
            （即"年龄完全确定"）。
        nsim: 蒙特卡洛次数，至少 1。
        bootpc: 自助比例，``0 <= bootpc < 1``；每次模拟随机丢掉
            ``round(bootpc * n)`` 个定年点。
        xfactor: 中间点年龄扰动的标准差倍数（相对于锚点 1σ）。
        anchor_mode: ``"extrapolate"``（两轮）或 ``"none"``（单轮）。
        seed: 随机种子。给定种子时结果完全可复现；不给种子时调用
            ``utils.statistics_core.make_rng(None, context=...)``，它会发出
            ``RuntimeWarning`` 提示结果不可复现——这是仓库内其它随机分析
            沿用的同一约定。

    Returns:
        AgeDepthUncertaintyResult: 年龄-深度百分位带、1σ/2σ 带、中位/平均
        曲线、沉积速率百分位与模拟计数。

    Raises:
        ValidationError: 深度非严格递增、长度不一致、含 NaN、点数太少、
            ``nsim``/``bootpc``/``xfactor`` 非法、锚点模式未知，或全部模拟
            都被倒转剔除。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.age_depth_uncertainty import undatable_age_depth_model
        >>> model = undatable_age_depth_model(
        ...     [0.0, 50.0, 100.0],
        ...     [0.0, 500.0, 1000.0],
        ...     [10.0, 10.0, 10.0],
        ...     nsim=200,
        ...     seed=7,
        ... )
        >>> model.age_depth_percentiles.shape[0]
        99
    """
    depth_array = _as_vector(depths, "depths", minimum_size=MIN_POINTS)
    age_array = _as_vector(ages, "ages", minimum_size=MIN_POINTS)
    if depth_array.size != age_array.size:
        raise ValidationError(
            _("depths and ages must have the same length"),
            details={"depths_length": int(depth_array.size), "ages_length": int(age_array.size)},
        )
    if np.any(np.diff(depth_array) <= 0.0):
        raise ValidationError(
            _("depths must contain strictly increasing values"),
            details={"n_points": int(depth_array.size)},
        )
    sigma_array = _validate_uncertainties(age_uncertainties, depth_array.size)
    simulations = _validate_simulation_count(nsim)
    bootstrap_fraction = _validate_bootpc(bootpc)
    spread_factor = _validate_xfactor(xfactor)
    mode = _validate_anchor_mode(anchor_mode)

    rng = make_rng(seed, context="undatable age-depth model")

    # ---- 第一轮 ------------------------------------------------------------
    grid = _build_grid(depth_array)
    samples = _run_round(
        depth_array,
        age_array,
        sigma_array,
        grid,
        simulations,
        bootstrap_fraction,
        spread_factor,
        rng,
    )
    if samples.shape[1] == 0:
        raise ValidationError(
            _("no age-depth simulation survived the age-reversal filter"),
            details={"n_simulations": simulations, "hint": "check the order of the dates"},
        )
    percentiles, means = _summarise_samples(samples)
    band_low = _percentile_band(percentiles, SIGMA1_LOW_PERCENTILE, SIGMA1_HIGH_PERCENTILE)[0]
    band_high = _percentile_band(percentiles, SIGMA1_LOW_PERCENTILE, SIGMA1_HIGH_PERCENTILE)[1]

    # ---- 第二轮：外推锚点后重跑 --------------------------------------------
    final_grid = grid
    final_samples = samples
    final_percentiles = percentiles
    final_means = means
    if mode == "extrapolate":
        anchors = _extrapolate_anchors(grid, percentiles, band_low, band_high, depth_array)
        if anchors is not None:
            top_depth, top_age, top_sigma, bottom_depth, bottom_age, bottom_sigma = anchors
            second_depths = np.concatenate((top_depth, depth_array, bottom_depth))
            second_ages = np.concatenate((top_age, age_array, bottom_age))
            second_sigmas = np.concatenate((top_sigma, sigma_array, bottom_sigma))
            second_grid = _build_grid(second_depths)
            # 外推锚点不参与自助（Acycle 的 udanchors.m 同样把它们排除在外）。
            pool = np.arange(1, second_depths.size - 1)
            second_samples = _run_round(
                second_depths,
                second_ages,
                second_sigmas,
                second_grid,
                simulations,
                bootstrap_fraction,
                spread_factor,
                rng,
                bootstrap_pool=pool,
            )
            if second_samples.shape[1] == 0:
                logger.info("age-depth: the anchored round produced no usable simulation; keeping the first round")
            else:
                final_grid = second_grid
                final_samples = second_samples
                final_percentiles, final_means = _summarise_samples(second_samples)

    # ---- 汇总 --------------------------------------------------------------
    sigma1 = _percentile_band(final_percentiles, SIGMA1_LOW_PERCENTILE, SIGMA1_HIGH_PERCENTILE)
    sigma2 = _percentile_band(final_percentiles, SIGMA2_LOW_PERCENTILE, SIGMA2_HIGH_PERCENTILE)
    median = final_percentiles[49]

    rates = np.gradient(final_samples, final_grid, axis=0)
    rate_percentiles, _rate_means = _summarise_samples(rates)
    median_rate = rate_percentiles[49]

    logger.info(
        "age-depth: %d simulations, %d accepted, %d depth points",
        simulations,
        final_samples.shape[1],
        final_grid.size,
    )

    return AgeDepthUncertaintyResult(
        depths=final_grid,
        age_depth_percentiles=final_percentiles,
        median_age_depth=median,
        mean_age_depth=final_means,
        sigma1_age_depth=sigma1,
        sigma2_age_depth=sigma2,
        sedimentation_rate_percentiles=rate_percentiles,
        median_sedimentation_rate=median_rate,
        age_samples=final_samples,
        n_simulations=simulations,
        n_accepted=int(final_samples.shape[1]),
    )
