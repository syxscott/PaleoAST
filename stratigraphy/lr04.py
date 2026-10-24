# =============================================================================
# FILE: stratigraphy/lr04.py
# =============================================================================
"""
LR04 底栖 δ¹⁸O 复合栈（Lisiecki & Raymo 2005）。

本模块提供古气候学最常用的**底栖有孔虫 δ¹⁸O 基准栈** —— LR04 (LR04 stack):
由全球 57 条底栖 δ¹⁸O 记录经自动图形对比对齐而成的 5.32 Myr 复合曲线。
δ¹⁸O 的绝对值直接反映全球冰量与深海温度, 因此该栈既是**古气候代理指标的
基准**, 也是轨道调谐 (astrochronology) 常用的对比目标, 与本仓库已有的
La04 天文解 (:mod:`stratigraphy.astronomy`) 配套使用。

提供的三个能力
--------------
* :class:`LR04Stack` —— 不可变的栈数据容器, 含线性插值 :meth:`~LR04Stack.at`
  与文字摘要 :meth:`~LR04Stack.summary`。
* :func:`load_lr04_stack` —— 读取内置数据文件
  (``data/stratigraphy/lr04_5320ka.txt``)。
* :func:`lr04_target_curve` —— 把栈插值到任意年龄轴上, 作为对比/调谐的
  目标曲线。

数据来源
--------
内置数据取自 Acycle (https://github.com/mingsongli/acycle) 分发的
``data/Examples/LR04stack5320ka.txt`` (取回日期 2026-10-24)。该文件是**已发表
数据**的副本, 不是 Acycle 的代码产物: 它与 NOAA/NCEI 古气候数据库中同一数据集
(study 5847) 的数值逐行一致 (例如 0.0 ka → 3.23 ‰、20.0 ka → 4.99 ‰、
40.0 ka → 4.55 ‰), 行数同为 2115。Acycle 本身是 GPL 许可的软件, 但本文件里的
**数据**是 Lisiecki & Raymo (2005) 发表的结果。

原始文献:
    Lisiecki, L.E. & Raymo, M.E., 2005. A Pliocene-Pleistocene stack of 57
    globally distributed benthic δ¹⁸O records. *Paleoceanography*, 20(2):
    269-305. doi:10.1029/2004PA001071

数据档案:
    https://www.ncei.noaa.gov/access/paleo-search/study/5847

时间覆盖与格式
--------------
* 年龄覆盖 **0 – 5.32 Ma** (5320 ka)。
* 采样间隔: 0–600 ka 为 1 ka; 600–2100 ka 为 2.5 ka; 2100–4420 ka 为 5 ka;
  4420–5320 ka 为 2 ka。共 **2115** 个采样点。
* 数值范围: **2.65 – 5.08 ‰ (VPDB)** (底栖 δ¹⁸O 的绝对值; 冰期高、间冰期低)。

:attr:`LR04Stack.ages_ma` 的顺序与本仓库其它时间序列一致 —— **由老到新**
(最老点在最前, 故年龄数值递减)。原始文件按"由新到老"排列, 载入时反转。

打包说明
--------
数据文件随包分发, 通过 ``importlib.resources`` 定位, 因此在源码运行、wheel
安装与打包环境下均可找到。**该文件需要进 wheel**: ``pyproject.toml`` 的
``[tool.setuptools.package-data]`` 中 ``"data"`` 一项必须包含
``"stratigraphy/*.txt"`` (与既有的 ``"astronomy/*.npz"`` 同一层级), 否则
非 editable 安装时 :func:`load_lr04_stack` 会抛 ``ValidationError``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import numpy.typing as npt

from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "LR04Stack",
    "load_lr04_stack",
    "lr04_target_curve",
]

# =============================================================================
# 常量
# =============================================================================

#: 内置 LR04 栈数据文件名 (位于 ``data/stratigraphy/`` 下)。
LR04_DATA_FILE = "lr04_5320ka.txt"

#: 栈的最老年龄 (Ma)。
LR04_MAX_AGE_MA = 5.32

#: 栈的最新年龄 (Ma)。
LR04_MIN_AGE_MA = 0.0

#: 数据文件里年龄列的单位: 千 (ka)。
_AGE_UNIT_KA = 1000.0

#: 已发表的 LR04 栈 δ¹⁸O 值域 (‰ VPDB); 用于载入时的合理性检查。
_D18O_LIMITS = (-5.0, 6.0)


# =============================================================================
# 数据容器
# =============================================================================


@dataclass(frozen=True)
class LR04Stack:
    """
    LR04 底栖 δ¹⁸O 复合栈。

    Attributes:
        ages_ma: 年龄轴 (Ma), **由老到新** (最老在前, 故数值递减)。
        d18o: 底栖 δ¹⁸O (‰, VPDB), 与 ``ages_ma`` 一一对应。
    """

    ages_ma: npt.NDArray
    d18o: npt.NDArray

    def at(self, ages_ma: npt.NDArray) -> npt.NDArray:
        """
        把栈插值到给定的年龄轴上。

        采用**线性插值**; 插值前把年龄轴排成升序, 因此本方法对
        ``ages_ma`` 本身的顺序不作要求。

        Parameters:
            ages_ma: 目标年龄 (Ma), 须落在栈的年龄范围内
                (0 – 5.32 Ma)。

        Returns:
            与 ``ages_ma`` 等长的 δ¹⁸O 数组 (‰, VPDB)。

        Raises:
            ValidationError: 若 ``ages_ma`` 为空、含非有限值, 或落在栈的
                年龄范围之外 (栈不外推)。
        """
        target = np.atleast_1d(np.asarray(ages_ma, dtype=float))
        if target.size == 0:
            raise ValidationError("ages_ma must contain at least one age")
        if not np.all(np.isfinite(target)):
            raise ValidationError("ages_ma must contain finite ages only")

        lo = float(np.min(self.ages_ma))
        hi = float(np.max(self.ages_ma))
        if float(target.min()) < lo or float(target.max()) > hi:
            raise ValidationError(
                "Requested ages fall outside the LR04 stack's age range",
                details={
                    "range_ma": [lo, hi],
                    "requested_ma": [float(target.min()), float(target.max())],
                },
            )

        # 本栈的年龄轴由老到新 (数值递减), 插值前先排成升序。
        order = np.argsort(self.ages_ma)
        return np.interp(target, self.ages_ma[order], self.d18o[order])

    def summary(self) -> str:
        """
        栈的文字摘要。

        Returns:
            多行字符串, 含采样点数、年龄覆盖、δ¹⁸O 的均值与极值。
        """
        ages = np.asarray(self.ages_ma, dtype=float)
        values = np.asarray(self.d18o, dtype=float)
        return "\n".join(
            [
                "LR04 benthic d18O stack (Lisiecki & Raymo 2005)",
                f"  samples      : {ages.size}",
                f"  age range    : {float(ages.min()):.3f} - {float(ages.max()):.3f} Ma",
                f"  d18O mean    : {float(values.mean()):.3f} permil VPDB",
                f"  d18O range   : {float(values.min()):.3f} - {float(values.max()):.3f} permil VPDB",
                "  order        : oldest first (ages descending)",
            ]
        )


# =============================================================================
# 数据加载
# =============================================================================


def load_lr04_stack() -> LR04Stack:
    """
    载入内置的 LR04 底栖 δ¹⁸O 栈。

    数据文件随包分发 (``data/stratigraphy/lr04_5320ka.txt``), 通过
    ``importlib.resources`` 定位, 因此在源码运行、wheel 安装与打包环境下
    均可找到。

    Returns:
        :class:`LR04Stack`, 其 ``ages_ma`` **由老到新** (5.32 Ma → 0 Ma)。

    Raises:
        ValidationError: 若数据文件缺失、格式不对, 或数值不合理。
    """
    resource = files("data") / "stratigraphy" / LR04_DATA_FILE
    try:
        with as_file(resource) as path:
            text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(
            f"LR04 stack data file not found: {LR04_DATA_FILE}",
            details={"file": LR04_DATA_FILE},
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"LR04 stack data file could not be read: {LR04_DATA_FILE}",
            details={"file": LR04_DATA_FILE, "reason": str(exc)},
        ) from exc

    ages_ka: list[float] = []
    values: list[float] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValidationError(
                "LR04 stack data file has a malformed row",
                details={"line": lineno, "expected_columns": 2, "found": len(parts)},
            )
        try:
            ages_ka.append(float(parts[0]))
            values.append(float(parts[1]))
        except ValueError as exc:
            raise ValidationError(
                "LR04 stack data file has a non-numeric value",
                details={"line": lineno, "reason": str(exc)},
            ) from exc

    if len(ages_ka) < 2:
        raise ValidationError(
            "LR04 stack data file contains fewer than two samples",
            details={"samples": len(ages_ka)},
        )

    age_array = np.asarray(ages_ka, dtype=float) / _AGE_UNIT_KA
    value_array = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(age_array)) or not np.all(np.isfinite(value_array)):
        raise ValidationError("LR04 stack data file contains non-finite values")
    if np.any(np.diff(age_array) <= 0.0):
        raise ValidationError("LR04 stack ages are not strictly increasing")

    low, high = _D18O_LIMITS
    if float(value_array.min()) < low or float(value_array.max()) > high:
        raise ValidationError(
            "LR04 stack d18O values fall outside the plausible benthic range",
            details={"min": float(value_array.min()), "max": float(value_array.max())},
        )

    # 原始文件按"由新到老"排列; 本仓库统一用"由老到新"(年龄数值递减)。
    return LR04Stack(
        ages_ma=age_array[::-1].copy(),
        d18o=value_array[::-1].copy(),
    )


# =============================================================================
# 目标曲线
# =============================================================================


def lr04_target_curve(ages_ma: npt.NDArray, method: str = "interpolate") -> npt.NDArray:
    """
    生成给定年龄轴上的 LR04 目标曲线。

    这是把"底栖 δ¹⁸O 基准栈"变成对比/调谐目标曲线的入口: 例如在
    TimeOpt 一类方法里, 用它与待测记录做相位与包络相关。

    Parameters:
        ages_ma: 目标年龄轴 (Ma), 须落在 0 – 5.32 Ma 内。
        method: 采样方式。

            * ``"interpolate"`` —— 线性插值 (默认);
            * ``"nearest"`` —— 取最近的采样点 (阶梯状, 保留原始采样分辨率)。

    Returns:
        与 ``ages_ma`` 等长的 δ¹⁸O 数组 (‰, VPDB)。

    Raises:
        ValidationError: 若 ``method`` 未知, 或年龄越界/含非有限值。
    """
    key = str(method).strip().lower()
    if key not in ("interpolate", "nearest"):
        raise ValidationError(
            f"Unknown LR04 sampling method: {method!r}",
            details={"known": ["interpolate", "nearest"]},
        )

    stack = load_lr04_stack()
    target = np.atleast_1d(np.asarray(ages_ma, dtype=float))

    if key == "nearest":
        # 最近邻: 栈的年龄轴由老到新, 故先排成升序再取最近点。
        order = np.argsort(stack.ages_ma)
        ages_sorted = stack.ages_ma[order]
        values_sorted = stack.d18o[order]
        if target.size == 0:
            raise ValidationError("ages_ma must contain at least one age")
        if not np.all(np.isfinite(target)):
            raise ValidationError("ages_ma must contain finite ages only")
        low = float(ages_sorted.min())
        high = float(ages_sorted.max())
        if float(target.min()) < low or float(target.max()) > high:
            raise ValidationError(
                "Requested ages fall outside the LR04 stack's age range",
                details={"range_ma": [low, high], "requested_ma": [float(target.min()), float(target.max())]},
            )
        positions = np.clip(np.searchsorted(ages_sorted, target), 1, ages_sorted.size - 1)
        left = ages_sorted[positions - 1]
        right = ages_sorted[positions]
        pick_left = np.abs(target - left) <= np.abs(right - target)
        return np.where(pick_left, values_sorted[positions - 1], values_sorted[positions])

    return stack.at(target)
