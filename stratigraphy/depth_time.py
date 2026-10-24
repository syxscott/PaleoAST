# =============================================================================
# FILE: stratigraphy/depth_time.py
# =============================================================================
"""
深度 → 时间转换（depth-to-time conversion）。

把一条按**深度**采样的地层记录（δ¹³C、伽马测井、孔隙度……）搬到**年龄轴**
上：给定若干"深度–年龄"锚点（生物地层约束、放射性年龄约束），为每一个采样
深度插值出年龄（Ma）。

算法
----
算法规格参照 Acycle ``code/seriesanalysis/depthtotime.m``：

* 锚点之间按**分段线性**插值；
* 锚点范围之外按**端段速率线性外推**（左端用第一段、右端用最后一段的
  深度/时间速率），因此外推段与锚点段连续。

Acycle 是 GPL 软件，本模块按上述规格与下列文献**独立实现**，未翻译其
MATLAB 源码。插值本身复用仓库里已有的年龄模型设施
``stratigraphy.correlation.AgeModelAnalyzer``（``model_type='linear'`` 即分段
线性 + 线性外推，``model_type='spline'`` 即保形单调三次插值 PCHIP），不复制
一份插值代码。

深度与高程的符号约定
--------------------
``AgeModelAnalyzer`` 的坐标轴是**高程**（向上为正、年龄向上变小）。深度是
高程的相反数，因此本模块把深度取负、并在年龄上做同样的反向，再调用既有设施；
这样既复用了仓库的年龄模型代码，也让它算出的沉积速率为正值（与"越深越老"
的地层直觉一致）。

支持的模型
----------
``"linear"``
    分段线性插值 + 端段速率线性外推（Acycle ``depthtotime.m`` 的行为）。
``"pchip"``
    保形单调三次插值（Fritsch & Carlson, 1980 的 PCHIP），锚点处精确通过、
    区间内单调且不振荡；锚点范围外按端点斜率线性外推。

参考资料
--------
1. Acycle ``code/seriesanalysis/depthtotime.m`` 的算法说明（Linda Hinnov,
   2006-05-24；该文件为 GPL，本处仅参照算法规格）。
2. Fritsch, F.N. & Carlson, R.E., 1980. Monotone piecewise cubic
   interpolation. *SIAM Journal on Numerical Analysis* 17(2): 238-246.
   （PCHIP 的原始文献。）
3. Gradstein, F.M., Ogg, J.G. & Wiegert, W. 2012. The geologic time scale.
   *Journal of Stratigraphy* 35: 1110-1130.（Ma/kyr 的单位约定。）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from stratigraphy.correlation import AgeModelAnalyzer, StratigraphicSection
from utils.exceptions import ValidationError

__all__ = ["SUPPORTED_MODELS", "DepthTimeResult", "depth_to_time"]

#: 支持的年龄模型。``"pchip"`` 映射到 ``AgeModelAnalyzer`` 的 ``"spline"``。
SUPPORTED_MODELS: tuple[str, ...] = ("linear", "pchip")

#: 本模块模型名 → ``AgeModelAnalyzer.build_model`` 的 ``model_type``。
_MODEL_TO_AGE_MODEL: dict[str, str] = {"linear": "linear", "pchip": "spline"}


@dataclass(frozen=True)
class DepthTimeResult:
    """深度 → 时间转换的结果。

    Attributes:
        depths: 采样深度（与输入同序，m）。
        ages_ma: 与 ``depths`` 一一对应的年龄（Ma，越深越老）。
    """

    depths: npt.NDArray
    ages_ma: npt.NDArray

    def summary(self) -> str:
        """返回一段中文摘要。"""
        depths = np.asarray(self.depths, dtype=float)
        ages = np.asarray(self.ages_ma, dtype=float)
        lines = [
            _("深度→时间转换"),
            _("深度范围: {0:.3f} – {1:.3f} m").format(float(depths[0]), float(depths[-1])),
            _("年龄范围: {0:.4f} – {1:.4f} Ma").format(float(ages[0]), float(ages[-1])),
        ]
        depth_span = float(depths[-1] - depths[0])
        age_span = float(ages[-1] - ages[0])
        if age_span > 0.0:
            lines.append(_("平均沉积速率: {0:.3f} m/Ma").format(depth_span / age_span))
        return "\n".join(lines)


def _as_strictly_increasing(values: npt.NDArray, name: str) -> npt.NDArray:
    """把输入转成一维、有限、严格递增的浮点数组。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValidationError(
            _("'{0}' must be a one-dimensional array; got {1} dimensions").format(name, array.ndim),
            details={"name": name, "ndim": int(array.ndim)},
        )
    if array.size == 0:
        raise ValidationError(_("'{0}' must not be empty").format(name), details={"name": name})
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("'{0}' contains non-finite values").format(name),
            details={"name": name, "n_non_finite": int(np.sum(~np.isfinite(array)))},
        )
    if np.any(np.diff(array) <= 0.0):
        raise ValidationError(
            _("'{0}' must be strictly increasing").format(name),
            details={"name": name},
        )
    return array


def depth_to_time(
    depths: npt.NDArray,
    age_anchors: npt.NDArray,
    anchor_ages_ma: npt.NDArray,
    model: str = "linear",
) -> DepthTimeResult:
    """把采样深度转换成年龄。

    Parameters:
        depths: 采样深度（m），**严格递增**。
        age_anchors: 锚点的深度（m），**严格递增**，至少两个。
        anchor_ages_ma: 与 ``age_anchors`` 一一对应的年龄（Ma），**严格递增**
            （即"越深越老"）。
        model: ``"linear"``（分段线性 + 线性外推，默认）或 ``"pchip"``
            （保形单调三次插值）。

    Returns:
        DepthTimeResult: 与 ``depths`` 同序的年龄数组。

    Raises:
        ValidationError: 输入维度、有限性、单调性或模型名不合法时。
    """
    depth_array = _as_strictly_increasing(depths, "depths")
    anchor_depths = _as_strictly_increasing(age_anchors, "age_anchors")
    anchor_ages = _as_strictly_increasing(anchor_ages_ma, "anchor_ages_ma")

    if anchor_depths.size != anchor_ages.size:
        raise ValidationError(
            _("'age_anchors' and 'anchor_ages_ma' must have the same length; got {0} and {1}").format(
                int(anchor_depths.size), int(anchor_ages.size)
            ),
            details={"n_anchors": int(anchor_depths.size), "n_ages": int(anchor_ages.size)},
        )
    if anchor_depths.size < 2:
        raise ValidationError(
            _("an age model needs at least 2 anchors; got {0}").format(int(anchor_depths.size)),
            details={"n_anchors": int(anchor_depths.size)},
        )

    key = str(model).strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValidationError(
            _("depth-time model must be one of {0}; got '{1}'").format(", ".join(SUPPORTED_MODELS), model),
            details={"model": str(model), "supported": list(SUPPORTED_MODELS)},
        )

    # 深度取负即高程，年龄同步反向：于是约束点的坐标递增、年龄递减，
    # 正好符合 AgeModelAnalyzer 的"高程向上、年龄向上变小"约定。
    section = StratigraphicSection(
        name="depth_time",
        heights=-depth_array,
        thicknesses=np.diff(depth_array),
        lithologies=[],
    )
    constraints = [
        (float(-anchor_depth), float(anchor_age), 0.0)
        for anchor_depth, anchor_age in zip(anchor_depths[::-1], anchor_ages[::-1], strict=True)
    ]

    age_model = AgeModelAnalyzer().build_model(
        section,
        constraints,
        model_type=_MODEL_TO_AGE_MODEL[key],
    )
    ages = np.asarray(age_model.modeled_ages, dtype=float)
    if ages.shape != depth_array.shape or not np.all(np.isfinite(ages)):
        raise ValidationError(
            _("the age model did not return a finite age for every depth"),
            details={"n_depths": int(depth_array.size)},
        )
    return DepthTimeResult(depths=depth_array.copy(), ages_ma=ages)
