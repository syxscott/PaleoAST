# =============================================================================
# FILE: stratigraphy/boxcox.py
# =============================================================================
"""
Box–Cox 幂变换（含 λ 的极大似然估计与逆变换）。

Box–Cox 变换把偏态的正值数据族拉到"接近正态"的形状上：

    y(λ) = ((x + c)^λ − 1) / λ      (λ ≠ 0)
    y(λ) = ln(x + c)                (λ = 0)

其中 ``c`` 是**偏移量**（保证 ``x + c > 0``），``λ`` 是形状参数：``λ = 0``
是对数变换，``λ = 1`` 是（平移后的）线性变换，``λ < 0`` 反转不等性。

λ 的估计
--------
按 Acycle ``code/misc/boxcox.m``（其算法源自 MathWorks 的同名函数）与经典
文献，用**极大似然**求 λ：把轮廓似然

    L(λ) = −(n/2)·ln(σ̂²(y(λ))) + (λ − 1)·Σ ln(xᵢ + c)

在 λ 上求最大，其中 ``σ̂²`` 是变换后数据的方差（总体方差，与 MATLAB
``std(x, 1)`` 一致）。Acycle 用 ``fminsearch`` 做这件事；本模块改用
"粗网格扫描 + 一维有界精细搜索"，结果等价但更稳健、且完全确定（无随机性）。
变换本身与逆变换都在本模块内**自己实现**，不调用 ``scipy.stats.boxcox``：
只有自己算，才能同时给出 λ 与可逆的逆变换。

Acycle 是 GPL 软件，本模块按上述规格与下列文献**独立实现**，未翻译其
MATLAB 源码。

参考资料
--------
1. Box, G.E.P. & Cox, D.R., 1964. An analysis of transformations. *Journal of
   the Royal Statistical Society, Series B* 26(2): 211-243.（变换族本身。）
2. Aitchison, J. & Brown, J.A.C. 1957. The Lognormal Distribution.
   Cambridge University Press.（λ 的极大似然估计与轮廓似然。）
3. MathWorks, 1995-2018. ``boxcox``（Financial Toolbox）。Acycle
   ``code/misc/boxcox.m`` 与 ``code/misc/boxcoxML.m`` 沿用该算法。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from config.i18n import _
from utils.exceptions import ValidationError

__all__ = ["BoxCoxResult", "box_cox_inverse", "box_cox_transform"]

#: λ 粗网格搜索的范围与步长（Ma；无物理单位，纯数值）。
_LAMBDA_GRID_MIN: float = -5.0
_LAMBDA_GRID_MAX: float = 5.0
_LAMBDA_GRID_STEP: float = 0.25


@dataclass(frozen=True)
class BoxCoxResult:
    """Box–Cox 变换的结果。

    Attributes:
        transformed: 变换后的数据。
        lambda_: 形状参数 λ（0 表示对数变换）。
        offset: 偏移量 c，变换作用于 ``x + c``。
    """

    transformed: npt.NDArray
    lambda_: float
    offset: float

    def summary(self) -> str:
        """返回一段中文摘要。"""
        transformed = np.asarray(self.transformed, dtype=float)
        lines = [
            _("Box–Cox 变换"),
            _("λ: {0:.6g}").format(float(self.lambda_)),
            _("偏移量: {0:.6g}").format(float(self.offset)),
            _("变换后均值: {0:.6g}").format(float(np.mean(transformed))),
            _("变换后标准差: {0:.6g}").format(float(np.std(transformed, ddof=1))),
        ]
        return "\n".join(lines)


def _as_values(values: npt.NDArray) -> npt.NDArray:
    """把输入转成一维、有限的浮点数组。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValidationError(
            _("Box-Cox input must be a one-dimensional array; got {0} dimensions").format(array.ndim),
            details={"ndim": int(array.ndim)},
        )
    if array.size == 0:
        raise ValidationError(_("Box-Cox input must not be empty"), details={"size": 0})
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("Box-Cox input contains non-finite values"),
            details={"n_non_finite": int(np.sum(~np.isfinite(array)))},
        )
    return array


def _transform_core(shifted: npt.NDArray, lambda_: float) -> npt.NDArray:
    """对**已平移**的正值数据施加 Box–Cox 变换。

    用 ``expm1(λ·ln x)/λ`` 而不是 ``(x^λ − 1)/λ``：两者数学等价，但前者在
    λ 接近 0 时不会因 ``x^λ − 1`` 的灾难性抵消而丢失精度。
    """
    if lambda_ == 0.0:
        return np.log(shifted)
    return np.expm1(lambda_ * np.log(shifted)) / lambda_


def _profile_log_likelihood(lambda_: float, shifted: npt.NDArray) -> float:
    """Box–Cox 轮廓对数似然（待最大化）。"""
    transformed = _transform_core(shifted, lambda_)
    variance = float(np.var(transformed))
    if not np.isfinite(variance) or variance <= 0.0:
        return -np.inf
    n = float(shifted.size)
    return -0.5 * n * np.log(variance) + (lambda_ - 1.0) * float(np.sum(np.log(shifted)))


def _estimate_lambda(shifted: npt.NDArray) -> float:
    """极大似然估计 λ：粗网格扫描 + 局部一维精细搜索。"""
    grid = np.arange(_LAMBDA_GRID_MIN, _LAMBDA_GRID_MAX + 0.5 * _LAMBDA_GRID_STEP, _LAMBDA_GRID_STEP)
    log_likelihoods = np.array([_profile_log_likelihood(float(value), shifted) for value in grid])
    best_index = int(np.argmax(log_likelihoods))
    if best_index == 0 or best_index == grid.size - 1:
        # 最优点落在搜索边界上：网格范围内没有更好的点，如实返回边界值。
        return float(grid[best_index])

    lower = float(grid[best_index - 1])
    upper = float(grid[best_index + 1])
    refined = minimize_scalar(
        lambda value: -_profile_log_likelihood(float(value), shifted),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not refined.success or not np.isfinite(refined.fun):
        return float(grid[best_index])
    return float(refined.x)


def box_cox_transform(values: npt.NDArray, lambda_: float | None = None) -> BoxCoxResult:
    """对正值数据做 Box–Cox 变换。

    Parameters:
        values: 待变换的数据（一维、有限）。显式给出 ``lambda_`` 时要求全部
            为正（与 Acycle ``boxcox.m`` 的 ``DATA must be positive`` 一致）；
            ``lambda_`` 为 ``None``（自动估计）时若数据含非正值，则自动加一个
            偏移量使其为正，偏移量记录在结果里。
        lambda_: 形状参数。``None``（默认）表示用极大似然自动估计。

    Returns:
        BoxCoxResult: 变换后的数据、λ 与偏移量。

    Raises:
        ValidationError: 输入维度、有限性、正值性或 λ 不合法时。
    """
    array = _as_values(values)

    if lambda_ is None:
        if array.size < 2:
            raise ValidationError(
                _("estimating lambda needs at least 2 values; got {0}").format(int(array.size)),
                details={"size": int(array.size)},
            )
        minimum = float(np.min(array))
        if minimum > 0.0:
            offset = 0.0
        else:
            margin = 1.0e-9 * max(1.0, abs(minimum))
            offset = -minimum + margin
        shifted = array + offset
        estimated = _estimate_lambda(shifted)
        transformed = _transform_core(shifted, estimated)
        return BoxCoxResult(transformed=transformed, lambda_=float(estimated), offset=float(offset))

    if not np.isfinite(lambda_):
        raise ValidationError(
            _("lambda must be a finite number; got {0}").format(lambda_),
            details={"lambda": float(lambda_)},
        )
    if np.any(array <= 0.0):
        raise ValidationError(
            _("Box-Cox data must be positive when lambda is given explicitly"),
            details={"n_non_positive": int(np.sum(array <= 0.0))},
        )
    transformed = _transform_core(array, float(lambda_))
    return BoxCoxResult(transformed=transformed, lambda_=float(lambda_), offset=0.0)


def box_cox_inverse(transformed: npt.NDArray, lambda_: float, offset: float = 0.0) -> npt.NDArray:
    """Box–Cox 逆变换：把变换后的数据还原回原尺度。

    Parameters:
        transformed: 变换后的数据（一维、有限）。
        lambda_: 变换时使用的形状参数。
        offset: 变换时使用的偏移量（默认 0）。

    Returns:
        npt.NDArray: 还原后的数据。

    Raises:
        ValidationError: 输入维度、有限性、λ 或偏移量不合法时。
    """
    array = _as_values(transformed)
    if not np.isfinite(lambda_):
        raise ValidationError(
            _("lambda must be a finite number; got {0}").format(lambda_),
            details={"lambda": float(lambda_)},
        )
    if not np.isfinite(offset):
        raise ValidationError(
            _("offset must be a finite number; got {0}").format(offset),
            details={"offset": float(offset)},
        )
    if lambda_ == 0.0:
        return np.exp(array) - offset
    base = float(lambda_) * array + 1.0
    if np.any(base <= 0.0):
        raise ValidationError(
            _("lambda * y + 1 must be positive to invert the Box-Cox transform"),
            details={"n_non_positive": int(np.sum(base <= 0.0))},
        )
    return np.power(base, 1.0 / float(lambda_)) - offset
