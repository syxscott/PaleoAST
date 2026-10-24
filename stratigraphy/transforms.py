# =============================================================================
# FILE: stratigraphy/transforms.py
# =============================================================================
"""
序列的简单变换 (Acycle 的 Math 菜单: "Standardize" / "Log-transform" /
"Simple Function")。

本模块提供**可直接调用的纯函数**: 输入一个序列, 输出变换后的序列, 不碰任何
界面状态。仓库里 :meth:`controllers.data_controller.DataController.transform_standardize`
与 :meth:`~controllers.data_controller.DataController.transform_log` 是**控制器
方法** (带缓存、进度回调与界面刷新), 通用分析运行器用不到它们; 这里的纯函数
才是分析流水线里该用的那一层。

三个函数
--------
* :func:`standardize` —— 标准化。``method="zscore"`` 给 z 分数
  (减均值除标准差), ``method="minmax"`` 把序列线性映射到 ``[0, 1]``。
* :func:`log_transform` —— 对数变换。**用的是不带偏移的 log**, 即
  ``log_base(x)``; 因此输入必须**全为正数**。非正数会抛
  ``ValidationError``, 而不是被悄悄变成 ``NaN`` 或被截断。
* :func:`apply_function` —— 简单函数: ``"square"``、``"abs"``、``"exp"``
  (外加 ``"sqrt"``)。``exp`` 溢出时抛 ``ValidationError``, 而不是把 ``inf``
  交给下游。

与仓库其它变换的关系
--------------------
* z 分数**复用** :func:`utils.transformations.zscore_standardize` (样本标准差,
  ``ddof=1``), 不另造一套。
* :func:`utils.transformations.log_transform` 是给生态学丰度数据用的**带偏移**
  版本 (``log(x + 1)``, 默认 ``offset=1.0``), 与本模块的口径不同: 本模块的
  :func:`log_transform` **不加偏移**, 因此对 0 与负数直接报错。想做
  ``log1p`` 请显式调用 ``utils.transformations.log_transform(data, offset=1.0)``。
* :func:`utils.transformations.sqrt_transform` 处理丰度数据; 本模块的
  ``"sqrt"`` 只是 ``apply_function`` 的一个选项, 语义相同 (负数报错)。

异常口径
--------
所有非法输入都抛 :class:`utils.exceptions.ValidationError` (仓库的统一口径),
不抛裸 ``ValueError``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from utils.exceptions import ValidationError
from utils.transformations import zscore_standardize

logger = logging.getLogger(__name__)

__all__ = [
    "apply_function",
    "log_transform",
    "standardize",
]

# =============================================================================
# 常量
# =============================================================================

#: :func:`standardize` 支持的方法。
SUPPORTED_STANDARDIZATIONS = ("zscore", "minmax")

#: :func:`apply_function` 支持的函数名。
SUPPORTED_FUNCTIONS = ("square", "abs", "exp", "sqrt")

#: ``exp`` 的溢出阈值 (自然指数在该值之上溢出为 ``inf``)。
_EXP_OVERFLOW_LIMIT = 709.0


# =============================================================================
# 内部工具
# =============================================================================


def _as_series(values: npt.NDArray, name: str = "values") -> npt.NDArray:
    """
    把输入整理成 float 数组并做基本校验。

    Parameters:
        values: 输入数组。
        name: 参数名 (用于报错信息)。

    Returns:
        float 数组 (形状与输入一致)。

    Raises:
        ValidationError: 若为空或含非有限值。
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values only")
    return array


def _is_constant(array: npt.NDArray, axis: int = 0) -> bool:
    """
    判断序列 (或矩阵的每一列) 是否为常数。

    Parameters:
        array: 输入数组。
        axis: 统计方向。

    Returns:
        是否没有可用的变异 (``True`` 表示常数)。
    """
    if array.ndim == 1:
        return bool(np.ptp(array) == 0.0)
    spread = np.ptp(array, axis=axis)
    return bool(np.all(spread == 0.0))


# =============================================================================
# 公开接口
# =============================================================================


def standardize(values: npt.NDArray, method: str = "zscore") -> npt.NDArray:
    """
    标准化序列。

    Parameters:
        values: 输入序列 (一维; 多维输入沿 ``axis=0`` 逐列标准化)。
        method: 标准化方法。

            * ``"zscore"`` (默认) —— z 分数 ``(x - mean) / std``。标准差用
              **样本标准差** (``ddof=1``), 所以结果的标准差恰为 1、均值为 0。
            * ``"minmax"`` —— 线性映射到 ``[0, 1]``: ``(x - min) / (max - min)``,
              最小值映到 0, 最大值映到 1。

    Returns:
        变换后的数组 (形状与输入一致)。

    Raises:
        ValidationError: 若方法未知、输入为空/非有限/元素太少, 或序列是常数
            (常数序列没有可用的尺度, 标准差或极差为零)。
    """
    key = str(method).strip().lower()
    if key not in SUPPORTED_STANDARDIZATIONS:
        raise ValidationError(
            f"Unknown standardization method: {method!r}",
            details={"known": list(SUPPORTED_STANDARDIZATIONS)},
        )

    array = _as_series(values)
    if array.size < 2:
        raise ValidationError(
            "values must contain at least two elements to be standardized",
            details={"size": int(array.size)},
        )
    if _is_constant(array):
        raise ValidationError(
            "Cannot standardize a constant series (zero spread)",
            details={"method": key},
        )

    if key == "zscore":
        # 复用仓库已有的实现 (样本标准差, ddof=1), 不另造一套。
        return np.asarray(zscore_standardize(array), dtype=float)

    axis = 0
    minimum = np.min(array, axis=axis, keepdims=True)
    maximum = np.max(array, axis=axis, keepdims=True)
    spread = maximum - minimum
    if np.any(spread == 0.0):
        raise ValidationError(
            "Cannot min-max standardize a constant series (zero range)",
            details={"method": key},
        )
    return (array - minimum) / spread


def log_transform(values: npt.NDArray, base: float = 10.0) -> npt.NDArray:
    """
    对数变换: ``log_base(x)``。

    **不带偏移** —— 即用的是 ``log``, 不是 ``log1p``。因此输入必须全为正数;
    0 与负数会抛 ``ValidationError``, 而不是被悄悄变成 ``-inf`` / ``NaN``。
    需要 ``log1p`` 语义 (允许 0) 时请显式使用
    :func:`utils.transformations.log_transform` (``offset=1.0``)。

    Parameters:
        values: 输入序列, 全部元素必须 **> 0**。
        base: 对数底, 必须为正且不等于 1 (可以是 10、2 或 ``numpy.e``)。

    Returns:
        变换后的数组 (形状与输入一致)。

    Raises:
        ValidationError: 若底数非法、输入含非正数或非有限值, 或输入为空。
    """
    exponent = float(base)
    if not np.isfinite(exponent) or exponent <= 0.0 or exponent == 1.0:
        raise ValidationError(
            f"base must be a finite positive number different from 1, got {base!r}",
            details={"base": base},
        )

    array = _as_series(values)
    non_positive = array <= 0.0
    if np.any(non_positive):
        count = int(np.count_nonzero(non_positive))
        raise ValidationError(
            "log_transform requires strictly positive values "
            "(the transform is log(x) without an offset, so 0 and negative "
            "values are undefined)",
            details={"non_positive_values": count, "base": exponent},
        )

    return np.log(array) / np.log(exponent)


def apply_function(values: npt.NDArray, function: str) -> npt.NDArray:
    """
    对序列施加一个简单函数。

    Parameters:
        values: 输入序列。
        function: 函数名。

            * ``"square"`` —— ``x ** 2``;
            * ``"abs"`` —— ``|x|``;
            * ``"exp"`` —— ``e ** x``;
            * ``"sqrt"`` —— ``sqrt(x)`` (负数报错)。

    Returns:
        变换后的数组 (形状与输入一致)。

    Raises:
        ValidationError: 若函数名未知、输入含非有限值, 或该函数在给定输入上
            无定义 (``sqrt`` 的负数) / 溢出 (``exp``)。
    """
    key = str(function).strip().lower()
    if key not in SUPPORTED_FUNCTIONS:
        raise ValidationError(
            f"Unknown function: {function!r}",
            details={"known": list(SUPPORTED_FUNCTIONS)},
        )

    array = _as_series(values)

    if key == "square":
        return array**2
    if key == "abs":
        return np.abs(array)
    if key == "sqrt":
        if np.any(array < 0.0):
            count = int(np.count_nonzero(array < 0.0))
            raise ValidationError(
                "sqrt is undefined for negative values",
                details={"negative_values": count},
            )
        return np.sqrt(array)

    # exp: 溢出时报错, 不把 inf 交给下游。
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(array)
    if not np.all(np.isfinite(result)):
        overflow = int(np.count_nonzero(~np.isfinite(result)))
        raise ValidationError(
            "exp overflowed to infinity for part of the input",
            details={"overflow_values": overflow, "limit": _EXP_OVERFLOW_LIMIT},
        )
    return result
