# =============================================================================
# FILE: stratigraphy/lead_lag.py
# =============================================================================
"""
Lead-lag（超前/滞后）判定：让两条序列**错位后 RMSE 最小**的滞后量。

要回答的问题
------------
两条地层记录（一条目标曲线、一条代用指标）之间，究竟哪一条超前、滞后多少？
本模块用**最小二乘**而不是**相关系数**来回答这个问题：在每一个候选滞后量上
把两条序列错位对齐，算出对齐后的均方根误差 RMSE，取 RMSE 最小的那个滞后量。

为什么不是相关系数
------------------
仓库已有的 :func:`stratigraphy.cycles.cross_correlation` 给的是"相关系数最大"
的滞后量。两者回答的不是同一个问题：

* 相关系数最大化只关心**波形形状**是否相似，与两条序列的幅度、基线无关；
  一条被强烈压缩过的曲线和一条没压缩的曲线，相关系数可以接近 1。
* RMSE 最小化是**最小二乘配准**：它同时惩罚形状差异和幅度差异。

因此本模块**不改动** :func:`stratigraphy.cycles.cross_correlation`，而是新增
一个独立函数；两者的结论不一致时（例如两条序列形状相似但幅度差很多），
RMSE 给出的滞后量才是"把一条曲线挪过去就能重合"的答案。

标准化与 RMSE 的定义
---------------------
两条序列各自**在整条记录上标准化一次**（样本均值、样本标准差，N−1 归一），
记为 z₁、z₂。滞后量 L 定义为把 z₂ 的坐标平移 **+L**（沿记录顺序向下为正），
再在 z₁ 的坐标上线性插值取 z₂ 的值，于是

    RMSE(L) = sqrt( mean( ( z₂(t − L) − z₁(t) )² ) )        t 取重叠区间

标准化在**循环之外**做一次（不是每个滞后量各做一次），所以 RMSE 曲线上的
各点可以互相比较；标准化也让"幅度差"进入判据（这正是与相关系数的区别）。
两条互不相关的标准化序列之间的 RMSE 期望值是 √2，因此 RMSE 曲线上的
√2 可以读作"完全对不上"的基准线。

符号约定（写死，不得随意改动）
--------------------------------
**正的滞后量 = 第二条序列滞后第一条序列。**

形式化地，滞后量 L 的配对规则是 ``z₁[i]`` 与 ``z₂[i + L]``：

* ``L > 0``：第二条序列要**往后读 L 个样本**才能与第一条对齐，即第二条序列
  的特征比第一条**晚** L 个样本出现 —— **第二条滞后第一条 L 个样本**；
* ``L < 0``：第二条序列的特征比第一条**早** |L| 个样本出现 —— **第二条超前
  第一条 |L| 个样本**；
* ``L = 0``：两条序列同相。

这个约定只依赖**样本顺序**，与坐标轴是年龄轴（越深越老）还是深度轴无关，
也不随"坐标方向"改变。返回的 :attr:`LeadLagResult.best_lag` 用的就是这套
约定，测试里用"人为延迟 Δ 个样本"的构造真值来钉死它。

重叠不足的处理
--------------
候选滞后量越大，重叠的样本越少。重叠样本少于 :data:`MIN_OVERLAP`（3 个）
的候选**直接跳过**，不出现在结果里——两个点算出来的 RMSE 没有意义。零滞后
总是被评估（它至少与整条记录等长），所以只要记录本身够长，总有解。

本模块是**确定性**的：没有随机过程，因此不需要 ``random_seed``。

参考文献
--------
1. Thomson, D.J. 1982. Spectrum estimation and harmonic analysis. Proceedings
   of the IEEE 70: 1055-1096.（多锥与谐波分析的总体框架。）
2. Percival, D.B. & Walden, A.T. 1993. Spectral Analysis and its
   Applications. Cambridge University Press.（多锥谱估计。）
3. Li, M., Kump, L.R., Hinnov, L.A. & Mann, M.E. 2018. Cyclostratigraphic
   correlation of astronomical forcing and Earth system processes. Earth and
   Planetary Science Letters 501: 165-179.（地层记录之间超前/滞后关系的
   定量表述。）
4. Acycle（Li, M.）的 ``leadlag`` / ``rmse4leadlag`` 系列函数提供了同一问题的
   另一种工程口径；本模块按上面的算法规格与文献**独立实现**，未翻译其
   MATLAB 源码（Acycle 为 GPL 软件）。

作者: PaleoAST 开发组
版本: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 一条序列至少要有多少个样本才值得做超前/滞后分析。
MIN_SAMPLES = 8

#: 一个候选滞后量至少要有多少个重叠样本才参与 RMSE 比较。
#:
#: 两个点算出来的"RMSE"是两点连线的半长，不是均方根误差；重叠更少时
#: 估计量本身就没有定义。因此重叠不足的候选被**跳过**而不是硬算。
MIN_OVERLAP = 3


# =============================================================================
# 1. 结果容器
# =============================================================================


@dataclass(frozen=True)
class LeadLagResult:
    """最小 RMSE 超前/滞后分析的结果。

    Attributes:
        best_lag: 最优滞后量，单位是样本。**正 = 第二条序列滞后第一条**
            （见模块 docstring 的符号约定）。
        best_rmse: 最优滞后量上的标准化 RMSE。两条互不相关的标准化序列之间
            的 RMSE 期望是 √2，可以当作"完全对不上"的基准。
        rmse_curve: 每个候选滞后量上的标准化 RMSE，与 ``lags`` 一一对应。
        lags: 被评估的候选滞后量（整数样本），升序。重叠样本不足
            :data:`MIN_OVERLAP` 的候选不在其中。
    """

    best_lag: int
    best_rmse: float
    rmse_curve: npt.NDArray
    lags: npt.NDArray

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [
            _("Lead-Lag RMSE Analysis"),
            "=" * 50,
            _("Lags evaluated: {0}").format(int(self.lags.size)),
            _("Best lag: {0} samples").format(int(self.best_lag)),
            _("Best standardised RMSE: {0:.6g}").format(float(self.best_rmse)),
            _("Interpretation: {0}").format(self._interpretation()),
        ]
        return "\n".join(lines)

    def _interpretation(self) -> str:
        """把 best_lag 翻成一句话（符号约定见模块 docstring）。"""
        if self.best_lag > 0:
            return _("the second series lags the first by {0} samples").format(int(self.best_lag))
        if self.best_lag < 0:
            return _("the second series leads the first by {0} samples").format(int(-self.best_lag))
        return _("the two series are in phase")

    def to_dict(self) -> dict[str, object]:
        """JSON 友好的视图。"""
        return {
            "best_lag": int(self.best_lag),
            "best_rmse": float(self.best_rmse),
            "lags": [int(v) for v in self.lags],
            "rmse_curve": [float(v) for v in self.rmse_curve],
            "sign_convention": "positive = second series lags first",
        }


# =============================================================================
# 2. 输入清洗
# =============================================================================


def _clean_series(values: npt.NDArray, name: str) -> npt.NDArray:
    """把输入整理成一维有限 float 数组，并检查样本数与方差。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        array = array.flatten()
    if array.size < MIN_SAMPLES:
        raise ValidationError(
            _("'{0}' needs at least {1} samples; got {2}").format(name, MIN_SAMPLES, int(array.size)),
            details={"n": int(array.size), "minimum_required": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("'{0}' must contain only finite values").format(name),
            details={"n_invalid": int(np.sum(~np.isfinite(array)))},
        )
    if float(np.ptp(array)) == 0.0:
        raise ValidationError(_("'{0}' is constant, so there is no shape to align").format(name))
    return array


def _standardize(values: npt.NDArray) -> npt.NDArray:
    """样本标准化（均值 0、样本标准差 1，N−1 归一）。"""
    centred = values - float(np.mean(values))
    deviation = float(np.std(centred, ddof=1))
    if deviation <= 0.0:
        raise ValidationError(_("a series with zero variance cannot be standardised"))
    return centred / deviation


# =============================================================================
# 3. 主分析
# =============================================================================


def lead_lag_rmse(
    first: npt.NDArray,
    second: npt.NDArray,
    max_lag: int | None = None,
) -> LeadLagResult:
    """求使两条序列错位后 RMSE 最小的滞后量。

    算法（确定性，无随机过程，因此不接受 ``random_seed``）：

    1. 两条序列各自在整条记录上标准化一次（样本均值、样本标准差，N−1）；
    2. 对每个候选滞后量 ``L``（``−max_lag … +max_lag`` 的整数），
       把标准化后的第二条序列按 ``L`` 平移，与第一条在重叠区间上逐点配对
       ``z₁[i]`` 与 ``z₂[i + L]``；
    3. 重叠样本少于 :data:`MIN_OVERLAP` 的候选被跳过；
    4. 取 RMSE 最小的候选为最优滞后量；RMSE 完全相同时，先取 ``|L|`` 小的，
       再取带符号值小的那个。

    **符号约定：正的滞后量表示第二条序列滞后第一条序列。** 详见模块
    docstring。

    Parameters
    ----------
    first:
        第一条序列，样本顺序。其"超前/滞后"方向是判定的基准。
    second:
        第二条序列，样本顺序。
    max_lag:
        评估的最大滞后量，单位是样本。默认取两条序列公共长度的 ``n // 2``。
        必须落在 ``[1, n−1]`` 内。

    Returns
    -------
    LeadLagResult

    Raises
    ------
    ValidationError
        样本太少、含非有限值、是常数序列、``max_lag`` 越界，或没有任何候选
        滞后量满足最小重叠要求。
    """
    a = _clean_series(first, "lead-lag first series")
    b = _clean_series(second, "lead-lag second series")

    length = int(min(a.size, b.size))
    if a.size != b.size:
        logger.info(
            "lead_lag_rmse: series lengths differ (%d vs %d); truncating both to %d samples",
            int(a.size),
            int(b.size),
            length,
        )
    a, b = a[:length], b[:length]

    limit = int(length // 2) if max_lag is None else int(max_lag)
    if limit < 1 or limit >= length:
        raise ValidationError(
            _("max_lag must lie in [1, n-1]; got {0} for n = {1}").format(limit, length),
            details={"max_lag": limit, "n": length},
        )

    z_first = _standardize(a)
    z_second = _standardize(b)

    candidates = np.arange(-limit, limit + 1, dtype=int)
    rmse = np.full(candidates.size, np.nan, dtype=float)
    evaluated = np.zeros(candidates.size, dtype=bool)
    for index, lag in enumerate(candidates):
        if lag >= 0:
            left = z_first[: length - lag]
            right = z_second[lag:]
        else:
            left = z_first[-lag:]
            right = z_second[: length + lag]
        if left.size < MIN_OVERLAP:
            # 重叠不足：这个候选没有意义，跳过（而不是拿两三个点算 RMSE）。
            continue
        rmse[index] = float(np.sqrt(np.mean((left - right) ** 2)))
        evaluated[index] = True

    if not np.any(evaluated):
        raise ValidationError(
            _("no lag candidate has at least {0} overlapping samples; the series are too short").format(MIN_OVERLAP),
            details={"n": length, "minimum_overlap": MIN_OVERLAP},
        )

    lags = candidates[evaluated]
    curve = rmse[evaluated]
    best_rmse = float(np.min(curve))
    tied = np.flatnonzero(curve == best_rmse)
    # 平局：先取 |lag| 小的，再取带符号值小的。
    order = np.lexsort((lags[tied], np.abs(lags[tied])))
    best_index = int(tied[order[0]])

    return LeadLagResult(
        best_lag=int(lags[best_index]),
        best_rmse=best_rmse,
        rmse_curve=curve,
        lags=lags,
    )
