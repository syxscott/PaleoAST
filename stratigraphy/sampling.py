# =============================================================================
# FILE: stratigraphy/sampling.py
# =============================================================================
"""
采样均匀性检测（是否等间距采样）。

一条地层记录被切成若干样本点之后，这些点是否**在时间上均匀分布**，决定了
后面所有频谱/周期分析的结论是否可信（例如 eCOCO 需要年龄轴上采样足够均匀，
COCO 的 Monte Carlo 也假定了规则的采样网格）。

算法
----
本模块给出两个互相印证的判据：

1. **Acycle 口径的"间距规整性"**。``code/seriesanalysis/acycleSamplingIsUneven.m``
   的做法是：取相邻采样点的间距 ``Δ``，用**中位间距**作基准，最大相对偏离
   ``max|Δᵢ − median(Δ)| / median(Δ)`` 超过容差（默认 ``1e-5``，即 10 ppm，
   用来容忍文本序列化带来的抖动）即判为"不均匀"。非有限或非递增的坐标轴
   直接判为不均匀。本模块沿用该判据与该容差。

2. **卡方拟合优度检验**（统计证据）。把时间跨度切成 ``k = max(2, n // 5)`` 个
   等宽箱（期望箱计数 ≈ 5，满足 Pearson 卡方近似的适用条件），统计每箱的
   样本数，与"均匀分布"零假设下的期望箱计数比较：

       χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ,   Eᵢ = n / k,   自由度 k − 1

   p 值取自卡方分布上尾。它回答的是"样本点在跨度上是否与**均匀随机采样**
   相容（有没有整体堆在一处）"。注意它与判据 1 问的不是同一个问题：一个
   只在局部留下巨大空洞的规则栅格，对均匀随机采样而言并不算出格（空洞只
   占跨度的一小段），所以卡方 p 值可以很大 —— 这时是判据 1 把它判为不均匀。
   反过来，一个随机（泊松）采样的序列判据 1 一定判为不均匀，而它的卡方
   p 值可能完全正常。

``is_uniform`` 取两个判据的**合取**：间距相对偏离在容差内 **且** 卡方检验不
显著。之所以要两个判据：间距规整性对**规则栅格**敏感（10 ppm 之内才算
均匀），而卡方检验只看"散不散得开"，二者合起来既不会把规则的栅格误判为
不均匀，也不会把"整体均匀但局部有巨大空洞"的网格放过去。

Acycle 是 GPL 软件，本模块按上述规格**独立实现**，未翻译其 MATLAB 源码。

参考资料
--------
1. Acycle ``code/seriesanalysis/acycleSamplingIsUneven.m`` 的算法说明
   （相对间距容差判据；该文件为 GPL，此处仅参照算法规格）。
2. Pearson, K. 1900. On the criterion that a given system of deviations from
   the probable ... *The London, Edinburgh, and Dublin Philosophical Magazine
   and Journal of Science* 50(302): 157-175.（卡方拟合优度。）
3. Freedman, L.S., 1981. *Randomization*, 2nd ed. W.H. Freeman.（由间距
   判读采样网格规则性的统计背景。）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from config.i18n import _
from utils.exceptions import ValidationError

__all__ = ["DEFAULT_TOLERANCE", "SamplingUniformityResult", "test_uniform_sampling"]

#: 间距相对偏离的默认容差（Acycle 口径：10 ppm）。
DEFAULT_TOLERANCE: float = 1.0e-5

#: 卡方检验显著性水平的默认值（``is_uniform`` 判定用）。
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class SamplingUniformityResult:
    """采样均匀性检验的结果。

    Attributes:
        is_uniform: 是否判定为均匀采样（间距规整且卡方检验不显著）。
        statistic: 卡方统计量（与 ``p_value`` 配对）。
        p_value: 均匀零假设下的 p 值。
        relative_gaps: 各相邻间距除以中位间距之比（1.0 表示恰好等于中位间距）。
    """

    is_uniform: bool
    statistic: float
    p_value: float
    relative_gaps: npt.NDArray

    def summary(self) -> str:
        """返回一段中文摘要。"""
        relative_gaps = np.asarray(self.relative_gaps, dtype=float)
        spacing_error = float(np.max(np.abs(relative_gaps - 1.0))) if relative_gaps.size else 0.0
        lines = [
            _("采样均匀性检验"),
            _("间距数: {0}").format(int(relative_gaps.size)),
            _("最大相对间距偏离: {0:.6g}").format(spacing_error),
            _("卡方统计量: {0:.6g}").format(float(self.statistic)),
            _("p 值: {0:.6g}").format(float(self.p_value)),
            _("判定: {0}").format(_("均匀") if self.is_uniform else _("不均匀")),
        ]
        return "\n".join(lines)


def test_uniform_sampling(
    time: npt.NDArray,
    tolerance: float = DEFAULT_TOLERANCE,
    alpha: float = DEFAULT_ALPHA,
) -> SamplingUniformityResult:
    """检验一条时间轴上的采样是否均匀。

    Parameters:
        time: 采样时刻（一维，至少两个点）。
        tolerance: 间距相对偏离的容差（Acycle 默认 ``1e-5``）。
        alpha: 卡方检验的显著性水平；``p_value`` 低于它即判为不均匀。

    Returns:
        SamplingUniformityResult: 判定、卡方统计量、p 值与相对间距。

    Raises:
        ValidationError: 输入维度、元素个数、有限性或参数不合法时。
    """
    array = np.asarray(time, dtype=float)
    if array.ndim != 1:
        raise ValidationError(
            _("'time' must be a one-dimensional array; got {0} dimensions").format(array.ndim),
            details={"ndim": int(array.ndim)},
        )
    if array.size < 2:
        raise ValidationError(
            _("'time' needs at least 2 samples; got {0}").format(int(array.size)),
            details={"size": int(array.size)},
        )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValidationError(
            _("tolerance must be a finite non-negative number; got {0}").format(tolerance),
            details={"tolerance": float(tolerance)},
        )
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValidationError(
            _("alpha must lie strictly between 0 and 1; got {0}").format(alpha),
            details={"alpha": float(alpha)},
        )

    gaps = np.diff(array)
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0.0):
        # 非有限或非递增的坐标轴：Acycle 口径直接判为"不均匀"。
        return SamplingUniformityResult(
            is_uniform=False,
            statistic=float("inf"),
            p_value=0.0,
            relative_gaps=np.zeros(gaps.size, dtype=float),
        )

    median_gap = float(np.median(gaps))
    relative_gaps = gaps / median_gap
    spacing_error = float(np.max(np.abs(relative_gaps - 1.0)))

    # 箱数取 ``n // 5``：期望箱计数 ≈ 5，满足 Pearson 卡方近似的适用条件
    # （期望频数过小时该近似不成立）。箱数太少又会让局部空洞被摊平。
    n_bins = max(2, int(array.size) // 5)
    counts = np.histogram(array, bins=n_bins, range=(float(array[0]), float(array[-1])))[0]
    expected = float(array.size) / float(n_bins)
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    degrees_of_freedom = n_bins - 1
    p_value = 1.0 if degrees_of_freedom <= 0 else float(chi2.sf(statistic, degrees_of_freedom))

    is_uniform = bool(spacing_error <= tolerance and p_value >= alpha)
    return SamplingUniformityResult(
        is_uniform=is_uniform,
        statistic=statistic,
        p_value=p_value,
        relative_gaps=relative_gaps,
    )


#: pytest 会把名字以 ``test_`` 开头的**函数**当成测试来收集，于是本分析函数
#: 的 ``time`` 形参会被误当成 fixture（报 ``fixture 'time' not found``）。
#: 这里显式关掉收集：它是分析函数，不是测试。
test_uniform_sampling.__test__ = False
