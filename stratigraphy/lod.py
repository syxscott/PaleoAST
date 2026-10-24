# =============================================================================
# FILE: stratigraphy/lod.py
# =============================================================================
"""
一日之长（Length of Day, LOD）的米兰科维奇计算。

LOD 究竟指什么
~~~~~~~~~~~~~~
Acycle 的 "LOD" 工具是 ``code/guicode/LODGUI.m``。把它读完之后得到的结论
是: **LOD = Length of Day, 即"一日之长"（小时）, 配套给出"一年有多少天"**,
既不是分形维数, 也不是对数-对数谱分析。证据都在那个文件里:

* 窗口标题是 ``Milankovitch Calculator``;
* 单点模式调用 ``MilankovitchCal`` (Waltham 2015 模型) 或 ``lodla04``
  (Laskar et al. 2004 模型), 两者返回的都是**日长**与**年长**;
* 纵轴标注是 ``Earth Day`` (小时) 与 ``Days in a Year``;
* 输出文件名是 ``LOD_<age>Ma.txt``。

因此本模块实现的是**一日之长**。函数名 ``lod_dimension`` 沿用注册表里的既有
命名（注册代码依赖这个名字, 不能改), 其中的 "dimension" 指的是**一日的长度**;
其余公开名一律以 ``lod_`` 开头。

依据
----
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for
  the eccentricity and obliquity of the Earth's orbits over 5 Myr.
  *Astronomy and Astrophysics*, 413: 755-767.
* Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A.C.M. &
  Levrard, B., 2004. A long-term numerical solution for the insolation
  quantities of the Earth. *Astronomy & Astrophysics*, 428: 261-285.
  （日长曲线出自这一篇: 其长期解的一阶展开可写成年龄的四次多项式, 今日的
  锚点值 23.934468 h 是该解的定标点。）

单位与符号约定
--------------
* 年龄一律以 **Ma** 给出, **正值向过去**（与 Acycle 的 LODGUI 一致）;
* 拟合式自变量是 **Gyr**, **负值向过去**, 即 ``T = -age_Ma / 1000``;
* 日长单位是**小时**, 年长单位是**天**。

与 Acycle 的关系
----------------
Acycle (GPL) 的 ``code/seriesanalysis/lodla04.m`` 用的是同一篇论文的拟合式。
**本模块是独立实现, 没有翻译那段 MATLAB 代码**; 数值口径（单位、符号约定、
多项式系数）与 Acycle 一致, 可以互相核对。

本模块只实现 Laskar et al. (2004) 这一支。Acycle 的另一支（Waltham 2015
模型）需要 Waltham 的完整解析轨道解才能给出日长, 未在此实现; 因此
:func:`lod_length_of_day` 的 ``model`` 参数目前只有 ``"laskar2004"``。

作者: PaleoAST 开发组
版本: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "LODResult",
    "lod_day_lengthening_rate",
    "lod_days_in_year",
    "lod_dimension",
    "lod_length_of_day",
]

# =============================================================================
# 常量
# =============================================================================

#: 本模块实现的模型名。Laskar et al. (2004) 的日长拟合。
LOD_MODEL = "laskar2004"

#: 今日 (0 Ma) 的日长 (小时)。Laskar et al. (2004) 长期解的锚点值。
LOD_REFERENCE_HOURS = 23.934468

#: 一个儒略年的秒数 (3.15576e7 s = 365.25 平太阳日)。
LOD_YEAR_SECONDS = 3.15576e7

#: 恒星日与平太阳日之比 (1.00273790935)。用于独立核对"年长"。
LOD_SIDEREAL_RATIO = 1.00273790935

#: 深度时间分支 (T <= 0) 的拟合系数: day0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4。
_LOD_PAST_COEFFICIENTS = (7.432167, -0.727046, -0.409572, -0.0589692)

#: 未来分支 (T > 0) 的拟合系数: day0 + b1*T + b2*T^2 + b3*T^3。
_LOD_FUTURE_COEFFICIENTS = (7.444649, -0.715049, 0.458097)

#: 由"每 Gyr 的日长变化"换算成"每世纪的毫秒数": 1e-9 Gyr^-1 -> 1e-9 yr^-1,
#: 再乘 100 (世纪)、3600 (秒/小时)、1000 (毫秒/秒)。
_MS_PER_CENTURY_PER_HOUR_PER_GYR = 0.36


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class LODResult:
    """
    一日之长的估计结果。

    Attributes:
        dimension: 参考年龄处的**一日之长**估计值, 单位与观测序列
            ``values`` 相同（观测以小时给出时即为小时）。
        uncertainty: 该估计值的标准误, 单位同上。
        scales: 模型曲线的年龄栅格 (Ma), 即模型的"尺度轴"。
        values: 栅格上的**拟合日长**曲线, 单位同 ``dimension``。
        reference_age_ma: 参考年龄 (Ma), 即记录中最年轻的一个年龄;
            ``dimension`` 就是这个年龄上的日长。
        model: 使用的模型名（目前只有 ``"laskar2004"``）。
    """

    dimension: float
    uncertainty: float
    scales: npt.NDArray
    values: npt.NDArray
    reference_age_ma: float = 0.0
    model: str = LOD_MODEL

    def summary(self) -> str:
        """
        返回一行人类可读的摘要。

        Returns:
            str: 摘要文本。
        """
        return (
            f"LOD ({self.model}): 日长 {self.dimension:.4f} +/- {self.uncertainty:.4f} h "
            f"@ {self.reference_age_ma:.3f} Ma; 模型曲线 {self.scales.size} 点 "
            f"[{float(np.min(self.scales)):.3f}, {float(np.max(self.scales)):.3f}] Ma"
        )


# =============================================================================
# 模型
# =============================================================================


def _as_finite_float_array(values: npt.NDArray | float, name: str) -> npt.NDArray:
    """
    把输入转成有限浮点数组, 形状原样保留。

    保留形状是有意的: 标量输入返回一个 0 维数组, ``float(...)`` 取它不会
    触发 numpy 的 "ndim > 0 转标量" 弃用警告。
    """
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must contain numbers", details={"name": name}) from exc
    if array.size == 0:
        raise ValidationError(f"{name} must not be empty", details={"name": name})
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite numbers only", details={"name": name})
    return array


def _gigayears(age_ma: npt.NDArray | float) -> npt.NDArray:
    """年龄 (Ma, 正值向过去) 换成拟合式的自变量 (Gyr, 负值向过去)。"""
    age = _as_finite_float_array(age_ma, "age_ma")
    return -age / 1000.0


def lod_length_of_day(age_ma: npt.NDArray | float) -> npt.NDArray:
    """
    给定年龄求一日之长 (小时)。

    用的是 Laskar et al. (2004) 长期解给出的日长拟合式: 深度时间分支是
    年龄的四次多项式, 未来分支是三次多项式, 今日的锚点值为 23.934468 h。

    Parameters:
        age_ma: 年龄 (Ma), **正值向过去**; 标量或一维数组。

    Returns:
        与输入同形的一日之长数组 (小时)。

    Raises:
        ValidationError: 输入含非有限值或形状不对。

    Example:
        >>> float(lod_length_of_day(0.0))
        23.934468
        >>> round(float(lod_length_of_day(500.0)), 3)
        20.084
    """
    t = _gigayears(age_ma)
    a1, a2, a3, a4 = _LOD_PAST_COEFFICIENTS
    b1, b2, b3 = _LOD_FUTURE_COEFFICIENTS

    past = t <= 0.0
    # Horner 形式: 与直接展开等价, 但在深度时间上数值更稳。
    lod_past = LOD_REFERENCE_HOURS + t * (a1 + t * (a2 + t * (a3 + t * a4)))
    lod_future = LOD_REFERENCE_HOURS + t * (b1 + t * (b2 + t * b3))
    return np.where(past, lod_past, lod_future)


def lod_days_in_year(age_ma: npt.NDArray | float) -> npt.NDArray:
    """
    给定年龄求一年的天数（年长）。

    年长由日长直接换算: ``一年 = 3.15576e7 s``, 一年里的天数就是这
    段秒数除以一个日长的秒数。

    Parameters:
        age_ma: 年龄 (Ma), 正值向过去; 标量或一维数组。

    Returns:
        与输入同形的年长数组 (天)。

    Raises:
        ValidationError: 输入含非有限值或形状不对。

    Example:
        >>> round(float(lod_days_in_year(0.0)), 4)
        366.2504
    """
    lod = lod_length_of_day(age_ma)
    return LOD_YEAR_SECONDS / (lod * 3600.0)


def lod_day_lengthening_rate(age_ma: npt.NDArray | float) -> npt.NDArray:
    """
    给定年龄求日长**向未来变长**的速率 (毫秒/世纪)。

    这是拟合式的导数: 正值表示日长正在变长（今天约 2.68 ms/世纪, 与潮汐
    耗散导致的日长增长的公认量级一致）。

    Parameters:
        age_ma: 年龄 (Ma), 正值向过去; 标量或一维数组。

    Returns:
        与输入同形的速率数组 (ms/世纪)。

    Raises:
        ValidationError: 输入含非有限值或形状不对。

    Example:
        >>> round(float(lod_day_lengthening_rate(0.0)), 3)
        2.676
    """
    t = _gigayears(age_ma)
    a1, a2, a3, a4 = _LOD_PAST_COEFFICIENTS
    b1, b2, b3 = _LOD_FUTURE_COEFFICIENTS

    past = t <= 0.0
    derivative_past = a1 + t * (2.0 * a2 + t * (3.0 * a3 + t * (4.0 * a4)))
    derivative_future = b1 + t * (2.0 * b2 + t * (3.0 * b3))
    derivative = np.where(past, derivative_past, derivative_future)
    return _MS_PER_CENTURY_PER_HOUR_PER_GYR * derivative


# =============================================================================
# 估计
# =============================================================================


def lod_dimension(
    time: npt.NDArray,
    values: npt.NDArray | None = None,
    n_scales: int = 20,
    min_scale: int = 2,
    max_scale: int | None = None,
) -> LODResult:
    """
    估计一条记录所在时段的一日之长。

    模型是**已发表的前向模型**: Laskar et al. (2004) 的日长曲线。观测值
    ``values`` 被当作与日长成正比的量（例如潮汐韵律给出的日长, 或任何按
    日长线性变化的记录）, 于是只剩一个自由参数——比例系数 ``c``:

    .. code-block:: text

        values ~ c * LOD_model(time)

    ``c`` 用过原点的最小二乘拟合求出, 参考年龄上的日长就是
    ``c * LOD_model(reference_age)``, 其标准误由拟合残差给出。观测本身就是
    日长（小时）时, ``c`` 应当接近 1。

    Parameters:
        time: 年龄轴 (Ma), **正值向过去**; 至少两个点, 全部有限。
        values: 与 ``time`` 等长的观测序列; 为 ``None`` 时只给模型值
            （此时不确定度为 0）。
        n_scales: 模型曲线在年龄栅格上的取样点数（至少 2）。
        min_scale: 模型栅格的最年轻年龄界 (Ma), 默认 2。若记录的最年轻
            年龄比它更年轻, 栅格改从记录的最年轻年龄开始, 以保证曲线覆盖
            整条记录。
        max_scale: 模型栅格的最老年龄界 (Ma); ``None`` 时取记录的最老年龄。

    Returns:
        :class:`LODResult`: 参考年龄处的日长估计、不确定度, 以及模型曲线。

    Raises:
        ValidationError: 输入长度不匹配、点数太少、含非有限值、栅格参数
            非法, 或拟合退化（模型在观测点上的取值全为零）。

    Example:
        >>> ages = np.linspace(0.0, 500.0, 51)
        >>> observed = lod_length_of_day(ages)
        >>> result = lod_dimension(ages, observed)
        >>> round(result.dimension, 4)
        23.9345
    """
    ages = _as_finite_float_array(time, "time")
    if ages.ndim != 1:
        raise ValidationError(
            "time must be one-dimensional",
            details={"shape": tuple(int(size) for size in ages.shape)},
        )
    if ages.size < 2:
        raise ValidationError(
            "time must contain at least two ages",
            details={"n_points": int(ages.size)},
        )

    try:
        n_points = int(n_scales)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"n_scales must be an integer, got {n_scales!r}") from exc
    if n_points < 2:
        raise ValidationError(f"n_scales must be at least 2, got {n_points}")

    try:
        lower = float(min_scale)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"min_scale must be a number, got {min_scale!r}") from exc
    if not np.isfinite(lower):
        raise ValidationError(f"min_scale must be finite, got {min_scale!r}")

    if max_scale is None:
        upper = float(np.max(ages))
    else:
        try:
            upper = float(max_scale)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"max_scale must be a number or None, got {max_scale!r}") from exc
        if not np.isfinite(upper):
            raise ValidationError(f"max_scale must be finite, got {max_scale!r}")

    reference_age = float(np.min(ages))
    start = min(lower, reference_age)
    if upper <= start:
        raise ValidationError(
            "the age grid is empty: max_scale must be larger than the grid's lower bound",
            details={"min_scale": lower, "max_scale": upper, "reference_age_ma": reference_age},
        )
    grid = np.linspace(start, upper, n_points)

    model_at_ages = lod_length_of_day(ages)
    model_at_reference = float(lod_length_of_day(reference_age))

    if values is None:
        return LODResult(
            dimension=model_at_reference,
            uncertainty=0.0,
            scales=grid,
            values=lod_length_of_day(grid),
            reference_age_ma=reference_age,
            model=LOD_MODEL,
        )

    observed = _as_finite_float_array(values, "values")
    if observed.ndim != 1:
        raise ValidationError(
            "values must be one-dimensional",
            details={"shape": tuple(int(size) for size in observed.shape)},
        )
    if observed.size != ages.size:
        raise ValidationError(
            "values and time must have the same length",
            details={"n_values": int(observed.size), "n_times": int(ages.size)},
        )

    denominator = float(np.sum(model_at_ages**2))
    if denominator <= 0.0:
        raise ValidationError(
            "the day-length model vanishes at every observation; the fit is degenerate",
            details={"n_points": int(ages.size)},
        )

    scale = float(np.sum(model_at_ages * observed) / denominator)
    residuals = observed - scale * model_at_ages
    degrees_of_freedom = int(observed.size) - 1
    if degrees_of_freedom <= 0:
        raise ValidationError(
            "at least two observations are needed to estimate the uncertainty",
            details={"n_points": int(observed.size)},
        )
    residual_variance = float(np.sum(residuals**2) / degrees_of_freedom)
    scale_standard_error = float(np.sqrt(residual_variance / denominator))

    return LODResult(
        dimension=scale * model_at_reference,
        uncertainty=abs(scale_standard_error * model_at_reference),
        scales=grid,
        values=scale * lod_length_of_day(grid),
        reference_age_ma=reference_age,
        model=LOD_MODEL,
    )
