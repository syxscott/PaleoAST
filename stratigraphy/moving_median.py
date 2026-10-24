# =============================================================================
# FILE: stratigraphy/moving_median.py
# =============================================================================
"""
移动中值滤波（moving median）。

要回答的问题
------------
地层记录里偶尔混进一个孤立尖峰（一个测错的样品、一个未剔除的 outliers、
一次采样事故），或者偶尔有一段缺失。用**移动平均**去压这种尖峰是危险的：
平均把尖峰分摊给邻域，尖峰所在位置被邻域拉高，尖峰两侧也被拖偏，尖峰
"变宽"而不是"变没"。移动中值只取窗口里的**选中值**，一个孤立的尖峰在
窗口里占少数，直接被多数票顶掉，而窗口里真正的趋势（中位数）原样保留。

这就是"用中值而不是用平均"的理由，也是本模块存在的理由。

窗口与端点约定
--------------
* **奇数窗口** ``w = 2h + 1``：窗口关于当前样本**中心对称**，覆盖
  ``[i−h, i+h]``。
* **偶数窗口** ``w = 2h``：按 Acycle ``moveMedian`` 的约定，窗口向前多取一个
  点，覆盖 ``[i−(h−1), i+h]``，即"h−1 个向后、h 个向前"。偶数窗口因此是
  **不对称**的，滤波结果相对输入有半个样本的前向偏移；需要中心对称时用
  奇数窗口。
* **端点**：窗口在记录两端被**截断**（不是复制端点值、不是反射），中值只
  在实际存在的样本上取。靠近端点时窗口因此比 ``w`` 短。

本模块是**确定性**的：没有随机过程，因此不需要 ``random_seed``。

参考文献
--------
1. Tukey, J.W. 1977. Exploratory Data Analysis. Addison-Wesley.（中值滤波的
   经典论述：抗孤立离群值，保边缘。）
2. Hampel, F.R. 1974. The influence curve and its robustness properties.
   Journal of the American Statistical Association 69: 383-397.（中值对孤立
   离群值不"敏感"的严格表述。）
3. Acycle（Li, M.）的 ``moveMedian`` 给出同一操作的工程口径；本模块按上面的
   算法规格**独立实现**，未翻译其 MATLAB 源码（Acycle 为 GPL 软件）。

作者: PaleoAST 开发组
版本: 1.0.0
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 允许的最小窗口长度。
#:
#: 窗口为 1 时输出等于输入（没有滤波）；窗口为 2 时"中值"退化成两点的
#: 平均，且每个输出只依赖两个样本，既没有统计意义也没有平滑意义。因此
#: 窗口小于 3 一律拒绝。
MIN_WINDOW = 3


def moving_median(values: npt.NDArray, window: int) -> npt.NDArray:
    """滑动窗口中值滤波。

    对每个位置 ``i``，取窗口覆盖到的样本的中值作为输出。窗口长度见模块
    docstring：奇数窗口中心对称，偶数窗口向前多取一个点；两端截断。

    Parameters
    ----------
    values:
        输入序列，一维。允许非整数长度。
    window:
        窗口长度（样本数），必须是不小于 :data:`MIN_WINDOW` 的整数。

    Returns
    -------
    ndarray
        与输入等长的中值滤波结果。

    Raises
    ------
    ValidationError
        输入为空、含非有限值，或窗口长度小于 :data:`MIN_WINDOW`、不是整数。

    Examples
    --------
    >>> moving_median(np.array([0.0, 0.0, 10.0, 0.0, 0.0]), 3)
    array([0., 0., 0., 0., 0.])
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        array = array.flatten()
    if array.size == 0:
        raise ValidationError(_("moving_median needs a non-empty series"))
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("moving_median needs only finite values"),
            details={"n_invalid": int(np.sum(~np.isfinite(array)))},
        )

    size = int(window)
    if size != window:
        raise ValidationError(
            _("window must be a whole number of samples; got {0}").format(window),
            details={"window": window},
        )
    if size < MIN_WINDOW:
        raise ValidationError(
            _("window must be at least {0} samples; got {1}").format(MIN_WINDOW, size),
            details={"window": size, "minimum_required": MIN_WINDOW},
        )

    half = size // 2
    if size % 2 == 1:
        # 奇数窗口：中心对称。
        back, forward = half, half
    else:
        # 偶数窗口：向后 half−1 个、向前 half 个（Acycle 的约定）。
        back, forward = max(0, half - 1), half

    count = array.size
    output = np.empty(count, dtype=float)
    interior_stop = count - forward

    if interior_stop <= back:
        # 记录比窗口还短：每个位置的窗口都被截断，逐点计算。
        for index in range(count):
            start = max(0, index - back)
            stop = min(count, index + forward + 1)
            output[index] = float(np.median(array[start:stop]))
        return output

    # 内部位置：窗口长度恰好是 size，用滑窗视图一次算完。
    windows = sliding_window_view(array, size)
    output[back:interior_stop] = np.median(windows[0 : interior_stop - back], axis=-1)

    # 两端：窗口被截断。
    for index in range(back):
        output[index] = float(np.median(array[0 : index + forward + 1]))
    for index in range(interior_stop, count):
        output[index] = float(np.median(array[index - back : count]))
    return output
