# =============================================================================
# FILE: stratigraphy/radiocarbon.py
# =============================================================================
"""
¹⁴C 定年校准（MatCal 风格的概率法）。

用途
====
把实验室给出的**常规 ¹⁴C 年龄**（conventional radiocarbon age，单位 ¹⁴C yr BP）
及其 1σ 误差，用一条**校准曲线**（IntCal20 / Marine20 / SHCal20 …）转换成
**日历年龄的后验概率分布**，并给出

* 逐年（或逐曲线节点）的**后验日历年龄概率密度**；
* **最高后验密度（HPD）区间**（默认 68.2% 与 95.4%）；
* **中位年龄**与**众数年龄**；
* 水库年龄 ``R(t)`` 与 ``ΔR`` 校正，以及 ``ΔR`` 不确定度向日历年龄不确定度的
  传播。

这是 Acycle 的 *MatCal* 子功能（``udmatcal.m``，其算法源自 Lougheed &
Obrochta 2016 的 MatCal）的**独立实现**。Acycle 以 GPL-3.0 发布，因此本文件
**不是**其 MATLAB 源码的逐行翻译：算法按下列已发表论文/方法的描述重新实现，
只在"语义对齐"的层面参考 Acycle 的公开行为（曲线文件格式、储库校正的施加
方向、HPD 区间的构造方式、逐年插值）。

提供的四个能力
--------------
* :func:`read_calibration_curve` —— 把 IntCal 的 ``.14c`` 曲线文件读成
  ``(radiocarbon_age, calendar_age, sigma)`` 三个数组；
* :func:`calibrate_radiocarbon` —— 校准一个 ¹⁴C 年龄；
* :func:`calibrate_radiocarbon_batch` —— 复用同一条曲线校准多个定年点；
* :class:`RadiocarbonCalibrationAnalyzer` —— 供参数表单驱动的**门面**：表格
  给两列（¹⁴C 年龄与其 1σ），曲线由使用者给出**文件路径**。

方法
====
记号
----
* ``R_meas``：实验室测得的 ¹⁴C 年龄（¹⁴C yr BP），``σ_R``：其 1σ；
* ``R(t)``：水库年龄（¹⁴C yr），``ΔR``：局部储库偏移（¹⁴C yr），
  ``σ_ΔR``：``ΔR`` 的 1σ；
* ``μ(t)``：校准曲线在日历年龄 ``t`` 处给出的 ¹⁴C 年龄（¹⁴C yr BP），
  ``σ_c(t)``：该点 ¹⁴C 不确定度（曲线第三列；曲线也可以不带 σ）。

1. **储库校正。** 与 Acycle ``udmatcal.m``（``labdet = labdet - resage;
   laberr = sqrt(laberr^2 + reserr^2)``）一致，校正量以**减量**施加在 ¹⁴C 年龄
   上::

       R_corr   = R_meas - (R(t) + ΔR)
       σ_R,eff² = σ_R² + σ_ΔR²

   在**单调递增**的校准曲线上，这意味着施加正的 ``R(t)`` 或 ``ΔR`` 会让
   校准出的日历年龄**变年轻**：储库效应使样品的 ¹⁴C 年龄偏老，减掉这部分之后
   查曲线得到的是更年轻的日历年龄。这是本模块与 Acycle **完全一致**的约定。

2. **似然。** 在日历年龄 ``t`` 处，曲线给出的 ¹⁴C 年龄是 ``μ(t)``，于是

       L(t) = N( R_corr ; μ(t), σ_R,eff² + σ_c(t)² )

   —— 曲线的 σ **逐点**参与传播（不是加在 ¹⁴C 误差上的常数）。

   ``ΔR`` 的传播是**解析**的而非近似的：``ΔR`` 取高斯先验时，对 ``ΔR`` 的
   边缘化等于两个高斯的卷积，方差相加，因此 ``σ_ΔR²`` 直接并入上式的方差。

3. **先验。** 日历年龄上的**均匀先验**（在整个曲线覆盖的日历年龄范围内），
   与 Acycle 的 ``matcal`` 以及 Bronk Ramsey (2008) 的概率法一致：

       p(t) ∝ L(t) × 1

4. **栅格。** 曲线被**线性插值到逐年（整数年）的日历年龄栅格**上（σ_c 同样
   线性插值），后验在这个栅格上求值并归一化。这与 Acycle 的 ``interpres = 1``
   （年度插值）一致，也让 HPD 区间的边界落在整年上。

5. **HPD 区间。** 区间由**后验密度**给出，而不是等宽区间：把栅格点按密度
   升序排列并累加概率，取"累积概率达到 ``1 - level``"之后的所有点，即密度
   的**超水平集**（superlevel set）；该集合的概率质量 ≥ 请求的置信水平。再按
   日历年龄把该集合切成若干**连续段**（段与段之间是密度低谷，多峰后验因此
   会给出多个区间）。这与 Acycle ``udmatcal.m`` 的做法一致。

6. **中位与众数。** 中位年龄是后验累积概率越过 0.5 的那个栅格点；众数年龄
   是后验密度最大的那个栅格点。

曲线的来源与格式
================
**本仓库不附带任何校准曲线文件。** 曲线由**使用者提供**，本模块只提供

* :func:`read_calibration_curve`：把 IntCal 的 ``.14c`` 文本格式读成
  ``(radiocarbon_age, calendar_age, sigma)`` 三个数组；
* :func:`calibrate_radiocarbon` / :func:`calibrate_radiocarbon_batch`：接受
  调用者给出的三列数组（或上面那个三元组）。

曲线文件格式（IntCal/Marine/SHCal 的 ``.14c``）
----------------------------------------------
* **逗号分隔**的纯文本，每行 5 列：
  ``cal BP, 14C age, 14C error, Delta R, Delta R error``；
* 表头 **11 行**（IntCal20 / Marine20 / SHCal20 / IntCal13 / … 的约定；
  IntCal98 / Marine98 的表头是 18 行，可用 ``header_lines`` 参数指定）；
* 文件里的日历年龄由**老到新**排列，Acycle 用 ``flipud`` 把它翻成升序后再
  插值。本模块读取后**显式按日历年龄升序排序**，因此两种排列都能用；
* 第 4、5 列（``ΔR`` 及其误差）**不被本模块使用**：曲线文件里的 ``ΔR`` 是
  "这条曲线自身的平均储库效应"，而用户要施加的是**相对于该平均的局部偏移**，
  必须由调用方通过 ``delta_r`` 给出。读到非零 ``ΔR`` 列时本模块会记一条
  警告。

曲线的来源（**使用者需要自行下载**）
------------------------------------
* IntCal20（大气）：Reimer, P. J. et al. (2020). IntCal20 and Marine20
  radiocarbon age calibration. *Radiocarbon*, 62(4), 725-757.
  曲线文件：https://intcal.org/ （``IntCal20.14c``）
* Marine20（海洋，同一文献）；SHCal20：Hogg, A. G. et al. (2020).
  *Radiocarbon*, 62(4), 758-769.

曲线文件有版权与体积问题，因此**不随本仓库分发**；请从上述官方来源取得，
并用 :func:`read_calibration_curve` 读取。

与 Acycle ``udmatcal.m`` 的差异（务必阅读）
============================================
* **不内置曲线、不下载曲线。** Acycle 按名字（``'IntCal20'`` 等）从工作目录
  读 ``.14c`` 文件；本模块要求调用者把曲线传进来。
* **在 ¹⁴C 年龄空间做似然。** Acurve 在 **F¹⁴C 空间**做似然
  （``exp(-R/8033)``），其年度归一化的后验相当于在日历年龄上使用先验
  ``p(t) ∝ exp(μ(t)/8033)``；本模块按任务约定在 **¹⁴C 年龄空间**做似然
  （均匀先验）。两者相差一个随日历年龄缓慢变化的因子（每 100 ¹⁴C 年约
  1.2%），在真实的不确定度量级（σ_cal ≲ 500 年）下对中位数、众数与 HPD 边界
  的影响可以忽略，但**并非逐位相同**。
* **不实现 F¹⁴C 输入模式。** Acycle 的 ``dettype='f14c'`` 允许直接输入
  F¹⁴C 与 ``δ¹⁴R``；本模块只接受常规 ¹⁴C 年龄（¹⁴C yr BP）。
* **不实现 BCE/CE 输出与画图。** Acycle 可以把结果换算成 BCE/CE 并画出
  概率曲线、原始数据与校准曲线；本模块只返回数值结果（画图属于视图层）。
* **HPD 区间的端点**取栅格节点（Acurve 同）；区间质量 ≥ 请求的置信水平
  （边界节点被整体计入，故可能略高于名义值）。
* **储库年龄与 ΔR 分开给出。** Acurve 只有一个 ``resage``/``reserr`` 参数对
  （大气曲线上是 ``R(t)``，海洋曲线上是 ``ΔR``）；本模块把两者都暴露出来，
  并按 ``R(t) + ΔR`` 一起施加。
* **不做 4σ 越界截断告警之外的任何自动外推**：曲线覆盖范围之外的后验一律为
  零（网格只覆盖曲线范围），并在后验未在两端衰减到零时记一条警告。

**没有实现**的部分：曲线的下载与内置、Outlier 模型（Christen & Litton 的
离群点项）、贝叶斯年代学层面的先验（``p(t) ∝ 1`` 的均匀先验之外）、F¹⁴C
输入模式、BCE/CE 年龄输出、以及任何绘图功能。

参考文献
========
* Christen, J. A. & Litton, C. D. (1995). Bayesian calibration of radiocarbon
  dates. *Radiocarbon*, 37(2), 211-218.（贝叶斯校准的框架：似然 × 先验、
  HPD 区间的构造。）
* Bronk Ramsey, C. (2008). Radiocarbon calibration: individual ages from
  radiocarbon years. *Radiocarbon*, 50(4), 1303-1315.（概率法的推导：
  单个年龄的后验密度。）
* Reimer, P. J. et al. (2020). IntCal20 and Marine20 radiocarbon age
  calibration. *Radiocarbon*, 62(4), 725-757.（IntCal20 / Marine20 曲线与
  曲线不确定度的传播。）
* Hogg, A. G. et al. (2020). SHCal20 Southern Hemisphere calibration curve
  for radiocarbon ages. *Radiocarbon*, 62(4), 758-769.（SHCal20 曲线。）
* Lougheed, B. C. & Obrochta, S. P. (2016). MatCal: open source Bayesian ¹⁴C
  age calibration in Matlab. *Journal of Open Research Software*, 4(1), e42.
  doi:10.5334/jors.130.（Acycle 的 MatCal 所实现的算法。）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "MatCal" 子功能。）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定，
未翻译或复制其 MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIDENCE_LEVELS",
    "DEFAULT_HEADER_LINES",
    "LEGACY_HEADER_LINES",
    "HpdInterval",
    "RadiocarbonCalibration",
    "RadiocarbonCalibrationAnalyzer",
    "calibrate_radiocarbon",
    "calibrate_radiocarbon_batch",
    "read_calibration_curve",
]

# =============================================================================
# 常量
# =============================================================================

#: 默认的置信水平：68.2%（1σ）与 95.4%（2σ）。
DEFAULT_CONFIDENCE_LEVELS: tuple[float, float] = (0.6827, 0.9545)

#: 校准曲线文件（``.14c``）的表头行数。IntCal20 / Marine20 / SHCal20 /
#: IntCal13 / Marine13 / SHCal13 / IntCal09 / … 的约定都是 11 行。
DEFAULT_HEADER_LINES = 11

#: IntCal98 / Marine98 的表头行数（Acycle 对这两条曲线用的是 18 行）。
LEGACY_HEADER_LINES = 18

#: 曲线至少需要的节点数。
MIN_CURVE_POINTS = 2

#: 后验在曲线两端没有衰减到零时的告警阈值（与 Acycle 的判据一致）。
EDGE_DENSITY_WARNING = 1.0e-6


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class HpdInterval:
    """一个最高后验密度（HPD）区间。

    Attributes:
        lower: 区间下端（日历年龄，cal yr BP）。
        upper: 区间上端（日历年龄，cal yr BP）。
        probability: 该区间覆盖的后验概率质量。
    """

    lower: float
    upper: float
    probability: float

    @property
    def width(self) -> float:
        """区间宽度（年）。"""
        return float(self.upper - self.lower)


@dataclass(frozen=True)
class RadiocarbonCalibration:
    """一次 ¹⁴C 定年校准的结果。

    Attributes:
        lab_age: 输入的实验室 ¹⁴C 年龄（¹⁴C yr BP）。
        lab_error: 输入的 ¹⁴C 年龄 1σ（¹⁴C yr）。
        corrected_lab_age: 施加水库年龄与 ΔR 之后的 ¹⁴C 年龄
            （``lab_age - (reservoir_age + delta_r)``）。
        total_lab_error: 参与似然的 ¹⁴C 年龄总 1σ
            （``sqrt(lab_error**2 + delta_r_sigma**2)``；曲线的 σ 逐点加入，
            不计入这个数）。
        calendar_ages: 后验的日历年龄栅格（逐年；曲线跨度不足一年时退化为
            曲线节点本身）。
        probability_density: 每个日历年龄栅格点上的后验概率质量；对逐年栅格
            它同时就是"每日历年的概率密度"。全部栅格点求和为 1。
        hpd_intervals: 置信水平 → 该水平上的 HPD 区间（可能有多段）。
        confidence_levels: 使用的置信水平（与 ``hpd_intervals`` 的键一致）。
        median_age: 中位日历年龄（cal yr BP）。
        mode_age: 众数日历年龄（cal yr BP），即后验密度最大的栅格点。
        reservoir_age: 使用的水库年龄 ``R(t)``（¹⁴C yr）；未给出时为 0。
        delta_r: 使用的局部储库偏移 ``ΔR``（¹⁴C yr）。
        delta_r_sigma: ``ΔR`` 的 1σ（¹⁴C yr）。
        curve_sigma: 额外叠加到曲线 σ 上的常数（¹⁴C yr）；未给出时为 0。
        seed: 调用时给出的随机种子。**本算法在给定输入下是确定性的**（高斯
            假设下 ΔR 的边缘化可解析完成），因此种子只被记录而不改变数值；
            它保留下来是为了与仓库其它分析的接口一致。
    """

    lab_age: float
    lab_error: float
    corrected_lab_age: float
    total_lab_error: float
    calendar_ages: npt.NDArray
    probability_density: npt.NDArray
    hpd_intervals: dict[float, tuple[HpdInterval, ...]]
    confidence_levels: tuple[float, ...]
    median_age: float
    mode_age: float
    reservoir_age: float
    delta_r: float
    delta_r_sigma: float
    curve_sigma: float
    seed: int | None = None

    def hpd(self, level: float) -> tuple[HpdInterval, ...]:
        """取某个置信水平上的 HPD 区间。

        Parameters:
            level: 置信水平，例如 ``0.6827``。

        Returns:
            tuple: 该水平上的 HPD 区间（可能有多段）。

        Raises:
            KeyError: 该置信水平没有被计算过。
        """
        if level in self.hpd_intervals:
            return self.hpd_intervals[level]
        for key, intervals in self.hpd_intervals.items():
            if abs(key - level) <= 1.0e-12:
                return intervals
        raise KeyError(
            _("no HPD intervals for the confidence level {0}").format(level),
        )

    def hpd_bounds(self, level: float) -> tuple[float, float]:
        """某个置信水平上 HPD 区间的外包络 ``(下端, 上端)``。

        多峰后验可能给出多个区间，外包络把它们括在一起；需要逐段区间时用
        :meth:`hpd`。
        """
        intervals = self.hpd(level)
        if not intervals:
            raise KeyError(
                _("no HPD intervals for the confidence level {0}").format(level),
            )
        lower = min(interval.lower for interval in intervals)
        upper = max(interval.upper for interval in intervals)
        return float(lower), float(upper)

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Radiocarbon calibration"),
            "=" * 48,
            _("14C age: {0} +/- {1}").format(f"{self.lab_age:.0f}", f"{self.lab_error:.0f}"),
            _("Reservoir correction: R(t) = {0}, Delta R = {1} +/- {2}").format(
                f"{self.reservoir_age:.0f}", f"{self.delta_r:.0f}", f"{self.delta_r_sigma:.0f}"
            ),
            _("Median calibrated age: {0} cal yr BP").format(f"{self.median_age:.0f}"),
            _("Mode calibrated age: {0} cal yr BP").format(f"{self.mode_age:.0f}"),
        ]
        for level in self.confidence_levels:
            lower, upper = self.hpd_bounds(level)
            lines.append(_("HPD {0:.1%}: {1} to {2} cal yr BP").format(level, f"{lower:.0f}", f"{upper:.0f}"))
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================


def _as_finite_float(value: object, name: str) -> float:
    """把输入转成一个有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be a number").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if not np.isfinite(number):
        raise ValidationError(
            _("{0} must be finite").format(name),
            details={"parameter": name},
        )
    return number


def _prepare_curve(curve: object) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """把用户给出的校准曲线整理成三个按日历年龄升序排列的数组。

    接受两种形式：

    * ``(n, 3)`` 的数组，列顺序为 ``[radiocarbon_age, calendar_age, sigma]``；
    * 三个等长一维数组组成的元组/列表（:func:`read_calibration_curve` 的返回
      值）。

    Returns:
        tuple: ``(calendar_age, radiocarbon_age, sigma)``，均按日历年龄升序。
    """
    if isinstance(curve, (tuple, list)) and len(curve) == 3 and all(np.ndim(item) == 1 for item in curve):
        try:
            array = np.column_stack([np.asarray(item, dtype=float) for item in curve])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("the calibration curve must be numeric"),
                details={"parameter": "curve"},
                original_exception=exc,
            ) from exc
    else:
        try:
            array = np.asarray(curve, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("the calibration curve must be numeric"),
                details={"parameter": "curve"},
                original_exception=exc,
            ) from exc

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValidationError(
            _("the calibration curve must have three columns: 14C age, calendar age and sigma"),
            details={"shape": [int(size) for size in array.shape]},
        )
    if array.shape[0] < MIN_CURVE_POINTS:
        raise ValidationError(
            _("the calibration curve needs at least {0} points; got {1}").format(MIN_CURVE_POINTS, int(array.shape[0])),
            details={"length": int(array.shape[0]), "minimum": MIN_CURVE_POINTS},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("the calibration curve must contain only finite values"),
            details={"parameter": "curve"},
        )

    sigma = array[:, 2]
    if np.any(sigma < 0.0):
        raise ValidationError(
            _("the calibration curve uncertainties must not be negative"),
            details={"parameter": "curve"},
        )

    order = np.argsort(array[:, 1], kind="stable")
    calendar = array[order, 1]
    radiocarbon = array[order, 0]
    ordered_sigma = sigma[order]
    if float(calendar[-1]) <= float(calendar[0]):
        raise ValidationError(
            _("the calibration curve must span more than one calendar year"),
            details={"calendar_minimum": float(calendar[0]), "calendar_maximum": float(calendar[-1])},
        )
    return calendar, radiocarbon, ordered_sigma


def _annual_grid(
    calendar: npt.NDArray,
    radiocarbon: npt.NDArray,
    sigma: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """把曲线线性插值到逐年（整数年）的日历年龄栅格上。

    与 Acycle ``udmatcal.m`` 的年度插值（``interpres = 1``）一致：曲线与它的
    σ 都在 ¹⁴C 年龄/误差空间里线性插值。曲线跨度不足一年时退化为曲线节点
    本身（此时栅格不等距，概率按节点计数而不是按年）。
    """
    low = float(np.ceil(calendar[0]))
    high = float(np.floor(calendar[-1]))
    if high - low < 1.0:
        return calendar, radiocarbon, sigma
    grid = np.arange(low, high + 1.0)
    return grid, np.interp(grid, calendar, radiocarbon), np.interp(grid, calendar, sigma)


def _posterior(
    corrected_lab_age: float,
    total_lab_error: float,
    curve_radiocarbon: npt.NDArray,
    curve_sigma: npt.NDArray,
    extra_sigma: float,
) -> npt.NDArray:
    """后验概率质量（逐栅格点，未归一化）。

    ``L(t) = N(R_corr ; μ(t), σ_lab² + σ_c(t)² + σ_extra²)``，先验取日历年龄
    上的均匀先验（对归一化后的结果没有影响）。
    """
    variance = total_lab_error**2 + curve_sigma**2 + extra_sigma**2
    residual = corrected_lab_age - curve_radiocarbon
    log_likelihood = -0.5 * residual**2 / variance - 0.5 * np.log(2.0 * np.pi * variance)
    # 减去最大值只是为了数值稳定：exp 不会下溢成 0，归一化后与未归一化等价。
    weights = np.exp(log_likelihood - float(np.max(log_likelihood)))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValidationError(
            _("the calibration produced an empty posterior"),
            details={"corrected_lab_age": float(corrected_lab_age)},
        )
    return weights / total


def _hpd_intervals(
    calendar: npt.NDArray,
    density: npt.NDArray,
    level: float,
) -> tuple[HpdInterval, ...]:
    """由后验密度构造某个置信水平上的 HPD 区间。

    做法与 Acycle ``udmatcal.m`` 一致：按密度**升序**累加概率，取累积概率
    达到 ``1 - level`` 之后的所有栅格点（密度的超水平集，其质量 ≥
    ``level``），再按日历年龄把该集合切成连续段。
    """
    order = np.argsort(density, kind="stable")
    cumulative = np.cumsum(density[order])
    kept = order[cumulative >= (1.0 - level)]
    if kept.size == 0:  # pragma: no cover - 概率和为 1 时不会发生
        return ()

    by_age = kept[np.argsort(calendar[kept], kind="stable")]
    ages = calendar[by_age]
    if calendar.size > 1:
        spacing = np.diff(calendar)
        typical = float(np.median(spacing))
    else:  # pragma: no cover - 曲线至少两个点
        typical = 1.0
    # 段与段的分界：日历年龄的间断明显大于栅格步长（年度栅格上即 ≥ 2 年）。
    gap_limit = max(1.5, 1.5 * typical)

    breaks = np.flatnonzero(np.diff(ages) > gap_limit) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [ages.size]))

    intervals: list[HpdInterval] = []
    for start, stop in zip(starts, stops, strict=True):
        block = by_age[start:stop]
        intervals.append(
            HpdInterval(
                lower=float(calendar[block[0]]),
                upper=float(calendar[block[-1]]),
                probability=float(np.sum(density[block])),
            )
        )
    return tuple(intervals)


def _broadcast(value: object, count: int, name: str) -> list[float]:
    """把标量或逐点序列广播成 ``count`` 个数。"""
    if np.ndim(value) == 0:
        return [_as_finite_float(value, name)] * count
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size != count:
        raise ValidationError(
            _("{0} must be a scalar or one value per date").format(name),
            details={"parameter": name, "length": int(array.size), "expected": count},
        )
    return [_as_finite_float(item, name) for item in array]


def _validate_confidence(confidence: object) -> tuple[float, ...]:
    """校验置信水平：每个都必须落在开区间 ``(0, 1)`` 内。"""
    if confidence is None:
        raise ValidationError(
            _("the confidence levels must be given"),
            details={"parameter": "confidence"},
        )
    if np.ndim(confidence) == 0:
        raise ValidationError(
            _("the confidence levels must be a sequence of numbers"),
            details={"parameter": "confidence"},
        )
    try:
        levels = [float(item) for item in confidence]  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("the confidence levels must be a sequence of numbers"),
            details={"parameter": "confidence"},
            original_exception=exc,
        ) from exc
    if not levels:
        raise ValidationError(
            _("at least one confidence level is required"),
            details={"parameter": "confidence"},
        )
    unique: list[float] = []
    for level in levels:
        if not np.isfinite(level) or not (0.0 < level < 1.0):
            raise ValidationError(
                _("the confidence levels must lie strictly between 0 and 1"),
                details={"parameter": "confidence", "level": level},
            )
        if level not in unique:
            unique.append(level)
    return tuple(unique)


# =============================================================================
# Public interface
# =============================================================================


def read_calibration_curve(
    path: str | Path,
    *,
    header_lines: int = DEFAULT_HEADER_LINES,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """读取 IntCal 风格的 ``.14c`` 校准曲线文件。

    文件格式：逗号分隔的纯文本，每行 5 列
    ``cal BP, 14C age, 14C error, Delta R, Delta R error``，前面有 11 行表头
    （IntCal98 / Marine98 是 18 行）。文件里的日历年龄由老到新排列，本函数
    读入后**按日历年龄升序排序**，因此两种排列都能用。

    第 4、5 列（``ΔR`` 及其误差）**不被本模块使用**——曲线文件里的 ``ΔR``
    是该曲线自身的平均储库效应，而使用者要施加的是相对于该平均的**局部
    偏移**，必须通过 :func:`calibrate_radiocarbon` 的 ``delta_r`` 参数给出。
    读到非零的 ``ΔR`` 列时会记一条警告。

    .. note::
       **本仓库不附带任何校准曲线文件。** 请自行从 https://intcal.org/ 下载
       ``IntCal20.14c`` / ``Marine20.14c`` / ``SHCal20.14c`` 等。

    Parameters:
        path: 曲线文件路径。
        header_lines: 表头行数，缺省 11（IntCal20/Marine20/SHCal20 的约定）。

    Returns:
        tuple: ``(radiocarbon_age, calendar_age, sigma)`` 三个数组，均按日历
        年龄升序排列。

    Raises:
        ValidationError: 文件不存在、读不出来、表头行数与文件内容不符、某一行
            不是数字、列数不对、数据行不足两行或含有非有限值。

    Examples:
        >>> from stratigraphy.radiocarbon import read_calibration_curve
        >>> r14, cal, sigma = read_calibration_curve("IntCal20.14c")  # doctest: +SKIP
        >>> r14.size == cal.size == sigma.size  # doctest: +SKIP
        True
    """
    if not isinstance(header_lines, (int, np.integer)) or isinstance(header_lines, bool):
        raise ValidationError(
            _("header_lines must be an integer"),
            details={"parameter": "header_lines"},
        )
    header = int(header_lines)
    if header < 0:
        raise ValidationError(
            _("header_lines must not be negative"),
            details={"parameter": "header_lines", "header_lines": header},
        )

    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(
            _("the calibration curve file does not exist"),
            details={"path": str(file_path)},
            original_exception=exc,
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(
            _("the calibration curve file could not be read"),
            details={"path": str(file_path)},
            original_exception=exc,
        ) from exc

    lines = text.splitlines()
    if len(lines) < header + 1:
        raise ValidationError(
            _("the calibration curve file has fewer lines than the header needs"),
            details={"path": str(file_path), "lines": len(lines), "header_lines": header},
        )

    rows: list[list[float]] = []
    delta_r_seen = 0.0
    for raw in lines[header:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 1:
            # 没有逗号时退回按空白切分（制表符/空格分隔的曲线）。
            fields = line.split()
        if len(fields) < 3:
            raise ValidationError(
                _("a calibration curve row needs at least three columns"),
                details={"path": str(file_path), "row": line},
            )
        if len(fields) > 5:
            raise ValidationError(
                _("a calibration curve row has too many columns"),
                details={"path": str(file_path), "row": line, "columns": len(fields)},
            )
        try:
            values = [float(item) for item in fields[:3]]
        except ValueError as exc:
            raise ValidationError(
                _("a calibration curve row is not numeric"),
                details={"path": str(file_path), "row": line},
                original_exception=exc,
            ) from exc
        if len(fields) >= 4:
            try:
                delta_r_seen = max(delta_r_seen, abs(float(fields[3])))
            except ValueError:
                delta_r_seen = 0.0
        rows.append(values)

    if len(rows) < MIN_CURVE_POINTS:
        raise ValidationError(
            _("the calibration curve file has fewer than {0} data rows").format(MIN_CURVE_POINTS),
            details={"path": str(file_path), "rows": len(rows)},
        )

    table = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(table)):
        raise ValidationError(
            _("the calibration curve file must contain only finite values"),
            details={"path": str(file_path)},
        )
    if np.any(table[:, 2] < 0.0):
        raise ValidationError(
            _("the calibration curve uncertainties must not be negative"),
            details={"path": str(file_path)},
        )

    order = np.argsort(table[:, 0], kind="stable")
    calendar = table[order, 0]
    radiocarbon = table[order, 1]
    sigma = table[order, 2]
    if float(calendar[-1]) <= float(calendar[0]):
        raise ValidationError(
            _("the calibration curve must span more than one calendar year"),
            details={"path": str(file_path)},
        )

    if delta_r_seen > 0.0:
        logger.warning(
            "%s carries a non-zero Delta R column; it is ignored. The local offset "
            "must be passed as 'delta_r' to the calibration.",
            file_path.name,
        )

    return radiocarbon, calendar, sigma


def calibrate_radiocarbon(
    lab_det: float,
    lab_err: float,
    curve: object,
    *,
    confidence: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    reservoir_age: float | None = None,
    delta_r: float = 0.0,
    delta_r_sigma: float = 0.0,
    curve_sigma: float | None = None,
    seed: int | None = None,
) -> RadiocarbonCalibration:
    """把一个 ¹⁴C 年龄校准成日历年龄的后验分布。

    后验为

        p(t) ∝ N( R_meas - (R(t) + ΔR) ; μ(t), σ_lab² + σ_ΔR² + σ_c(t)² )

    其中 ``μ(t)`` 与 ``σ_c(t)`` 是校准曲线在日历年龄 ``t`` 处的 ¹⁴C 年龄与
    不确定度；先验在曲线覆盖的日历年龄范围内取**均匀先验**。

    **储库校正的方向**：``R(t)`` 与 ``ΔR`` 都以**减量**施加在 ¹⁴C 年龄上
    （与 Acycle ``udmatcal.m`` 的 ``labdet = labdet - resage`` 一致）。因此在
    单调递增的校准曲线上，施加正的储库校正会让校准出的日历年龄**变年轻**
    ——储库效应使样品的 ¹⁴C 年龄偏老，减掉它之后查曲线得到的是更年轻的日历
    年龄。

    **ΔR 的不确定度**以平方和并入似然的方差（``σ_ΔR²``）；在 ``ΔR`` 取高斯
    先验的假设下这是**精确**的（两个高斯的卷积仍然是高斯）。

    Parameters:
        lab_det: 实验室 ¹⁴C 年龄（¹⁴C yr BP），必须非负。
        lab_err: 该年龄的 1σ（¹⁴C yr），必须为正。
        curve: 校准曲线。``(n, 3)`` 的数组，列为
            ``[radiocarbon_age, calendar_age, sigma]``；也可以是
            :func:`read_calibration_curve` 返回的三元组。
        confidence: 置信水平序列，每个都必须落在开区间 ``(0, 1)`` 内；缺省
            ``(0.6827, 0.9545)``。
        reservoir_age: 水库年龄 ``R(t)``（¹⁴C yr），缺省 0。用大气曲线时它
            代表局地大气/淡水储库偏移，用海洋曲线时它是储库年龄本身。
        delta_r: 局部储库偏移 ``ΔR``（¹⁴C yr），缺省 0。它是相对于**曲线
            自身平均储库效应**的差值（例如 Marine20 的平均海洋储库年龄）。
        delta_r_sigma: ``ΔR`` 的 1σ（¹⁴C yr），必须非负；它会传播到日历
            年龄的不确定度里。
        curve_sigma: 额外叠加到曲线自身 σ 上的常数（¹⁴C yr），缺省 0。
        seed: 随机种子。**本算法是确定性的**，种子只被记录进结果、不改变
            数值；保留它是为了与仓库其它分析的接口一致。

    Returns:
        RadiocarbonCalibration: 后验密度、HPD 区间、中位与众数年龄，以及
        全部输入参数。

    Raises:
        ValidationError: ¹⁴C 年龄为负或非有限、误差非正、曲线为空/含非有限
            值/列数不对、置信水平不在 ``(0, 1)`` 内、``delta_r_sigma`` 为负、
            其它参数非有限。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.radiocarbon import calibrate_radiocarbon
        >>> calendar = np.arange(0.0, 1001.0)
        >>> curve = np.column_stack([calendar - 137.0, calendar, np.full(calendar.size, 2.0)])
        >>> result = calibrate_radiocarbon(500.0, 20.0, curve)
        >>> bool(abs(result.mode_age - 637.0) < 2.0)
        True
    """
    age = _as_finite_float(lab_det, "lab_det")
    if age < 0.0:
        raise ValidationError(
            _("the 14C age must not be negative"),
            details={"parameter": "lab_det", "lab_det": age},
        )
    error = _as_finite_float(lab_err, "lab_err")
    if error <= 0.0:
        raise ValidationError(
            _("the 14C age uncertainty must be positive"),
            details={"parameter": "lab_err", "lab_err": error},
        )

    levels = _validate_confidence(confidence)

    reservoir = 0.0 if reservoir_age is None else _as_finite_float(reservoir_age, "reservoir_age")
    offset = _as_finite_float(delta_r, "delta_r")
    offset_sigma = _as_finite_float(delta_r_sigma, "delta_r_sigma")
    if offset_sigma < 0.0:
        raise ValidationError(
            _("the Delta R uncertainty must not be negative"),
            details={"parameter": "delta_r_sigma", "delta_r_sigma": offset_sigma},
        )
    extra_sigma = 0.0 if curve_sigma is None else _as_finite_float(curve_sigma, "curve_sigma")
    if extra_sigma < 0.0:
        raise ValidationError(
            _("the extra curve uncertainty must not be negative"),
            details={"parameter": "curve_sigma", "curve_sigma": extra_sigma},
        )

    calendar, radiocarbon, sigma = _prepare_curve(curve)
    grid, grid_radiocarbon, grid_sigma = _annual_grid(calendar, radiocarbon, sigma)

    corrected_age = age - (reservoir + offset)
    total_error = float(np.sqrt(error**2 + offset_sigma**2))
    density = _posterior(corrected_age, total_error, grid_radiocarbon, grid_sigma, extra_sigma)

    cumulative = np.cumsum(density)
    median_index = int(np.argmin(np.abs(cumulative - 0.5)))
    mode_index = int(np.argmax(density))

    intervals = {level: _hpd_intervals(grid, density, level) for level in levels}

    edge = max(float(density[0]), float(density[-1]))
    if edge > EDGE_DENSITY_WARNING:
        logger.warning(
            "the calibrated probability does not decay to zero at the ends of the "
            "calibration curve; the 14C age may lie outside the curve's range"
        )

    return RadiocarbonCalibration(
        lab_age=age,
        lab_error=error,
        corrected_lab_age=float(corrected_age),
        total_lab_error=total_error,
        calendar_ages=grid,
        probability_density=density,
        hpd_intervals=intervals,
        confidence_levels=levels,
        median_age=float(grid[median_index]),
        mode_age=float(grid[mode_index]),
        reservoir_age=reservoir,
        delta_r=offset,
        delta_r_sigma=offset_sigma,
        curve_sigma=extra_sigma,
        seed=seed,
    )


def calibrate_radiocarbon_batch(
    lab_dets: Sequence[float],
    lab_errs: Sequence[float],
    curve: object,
    *,
    confidence: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    reservoir_age: object = None,
    delta_r: object = 0.0,
    delta_r_sigma: object = 0.0,
    curve_sigma: float | None = None,
    seed: int | None = None,
) -> tuple[RadiocarbonCalibration, ...]:
    """一次校准多个定年点（复用同一条曲线，逐点给出结果）。

    每个点用**同一个**曲线与**同一个**种子，因此批量结果与逐个调用
    :func:`calibrate_radiocarbon` 的结果**完全一致**。

    Parameters:
        lab_dets: 各点的 ¹⁴C 年龄（¹⁴C yr BP），非负。
        lab_errs: 各点的 1σ 误差（¹⁴C yr），为正，与 ``lab_dets`` 等长。
        curve: 校准曲线，格式同 :func:`calibrate_radiocarbon`。
        confidence: 置信水平序列。
        reservoir_age: 水库年龄；标量或逐点序列。
        delta_r: 局部储库偏移 ``ΔR``；标量或逐点序列。
        delta_r_sigma: ``ΔR`` 的 1σ；标量或逐点序列。
        curve_sigma: 额外叠加到曲线 σ 上的常数（所有点相同）。
        seed: 随机种子（传给每个点）。

    Returns:
        tuple: 与输入等长的 :class:`RadiocarbonCalibration` 元组。

    Raises:
        ValidationError: 序列为空、长度不一致、长度不足两点，或任何一点的
            参数非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.radiocarbon import calibrate_radiocarbon_batch
        >>> calendar = np.arange(0.0, 1001.0)
        >>> curve = np.column_stack([calendar - 137.0, calendar, np.full(calendar.size, 2.0)])
        >>> results = calibrate_radiocarbon_batch([500.0, 600.0], [20.0, 25.0], curve)
        >>> len(results)
        2
    """
    try:
        ages = list(lab_dets)
        errors = list(lab_errs)
    except TypeError as exc:
        raise ValidationError(
            _("the 14C ages and their uncertainties must be sequences"),
            details={"parameter": "lab_dets"},
            original_exception=exc,
        ) from exc
    if not ages:
        raise ValidationError(
            _("at least one date is required"),
            details={"parameter": "lab_dets"},
        )
    if len(ages) != len(errors):
        raise ValidationError(
            _("the 14C ages and their uncertainties must have the same length"),
            details={"ages": len(ages), "errors": len(errors)},
        )
    if len(ages) < 2:
        raise ValidationError(
            _("at least two dates are required"),
            details={"parameter": "lab_dets", "length": len(ages)},
        )

    reservoirs = _broadcast(0.0 if reservoir_age is None else reservoir_age, len(ages), "reservoir_age")
    offsets = _broadcast(delta_r, len(ages), "delta_r")
    offset_sigmas = _broadcast(delta_r_sigma, len(ages), "delta_r_sigma")

    results: list[RadiocarbonCalibration] = []
    for index, (age, error) in enumerate(zip(ages, errors, strict=True)):
        results.append(
            calibrate_radiocarbon(
                age,
                error,
                curve,
                confidence=confidence,
                reservoir_age=reservoirs[index],
                delta_r=offsets[index],
                delta_r_sigma=offset_sigmas[index],
                curve_sigma=curve_sigma,
                seed=seed,
            )
        )
    return tuple(results)


# =============================================================================
# 供界面调用的门面
# =============================================================================


class RadiocarbonCalibrationAnalyzer:
    """¹⁴C 定年校准，供生成的参数表单驱动。

    校准曲线是**三列的数组**（¹⁴C 年龄、日历年龄、σ），表格的一列装不下
    它；因此它由使用者自己提供曲线**文件**（IntCal 的 ``.14c``），门面按
    路径读取。表格给的是两列：¹⁴C 年龄与其 1σ。

    表格里的**第一行**被校准（两列长度必须相等）；要一次校准多个点，请直接
    调用 :func:`calibrate_radiocarbon_batch`。
    """

    def __init__(self) -> None:
        """门面没有状态：曲线每次调用都从磁盘重新读取。"""

    @staticmethod
    def _column(values: object, name: str) -> npt.NDArray:
        """把表格的一列转成一维浮点数组。"""
        try:
            array = np.atleast_1d(np.asarray(values, dtype=float))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("{0} must be a column of numbers").format(name),
                details={"parameter": name},
                original_exception=exc,
            ) from exc
        return array.ravel()

    def analyze(
        self,
        lab_det: npt.NDArray,
        lab_err: npt.NDArray,
        curve_path: str,
    ) -> RadiocarbonCalibration:
        """校准表格里的一个 ¹⁴C 年龄。

        Parameters:
            lab_det: ¹⁴C 年龄（¹⁴C yr BP），表格的一列；取第一行。
            lab_err: 与 ``lab_det`` 等长的 1σ 误差列（¹⁴C yr）；取第一行。
            curve_path: 校准曲线文件的路径（IntCal 的 ``.14c``）。

        Returns:
            RadiocarbonCalibration: 与直接调用 :func:`calibrate_radiocarbon`
            完全一致的结果。

        Raises:
            ValidationError: 两列长度不一致、列里有非数字、曲线路径为空、
                曲线文件不存在或格式不对、曲线本身不合格。

        Examples:
            >>> import numpy as np
            >>> from stratigraphy.radiocarbon import RadiocarbonCalibrationAnalyzer
            >>> analyzer = RadiocarbonCalibrationAnalyzer()
            >>> analyzer.analyze(np.array([500.0]), np.array([20.0]), "IntCal20.14c")  # doctest: +SKIP
        """
        ages = self._column(lab_det, "lab_det")
        errors = self._column(lab_err, "lab_err")
        if ages.size != errors.size:
            raise ValidationError(
                _("the 14C ages and their uncertainties must have the same length"),
                details={"lab_det": int(ages.size), "lab_err": int(errors.size)},
            )
        if ages.size == 0:
            raise ValidationError(
                _("the 14C age column is empty"),
                details={"parameter": "lab_det"},
            )
        if not isinstance(curve_path, str) or not curve_path.strip():
            raise ValidationError(
                _("the calibration curve path must be a non-empty string"),
                details={"parameter": "curve_path"},
            )

        curve = read_calibration_curve(curve_path.strip())
        return calibrate_radiocarbon(float(ages[0]), float(errors[0]), curve)
