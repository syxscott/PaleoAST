# =============================================================================
# FILE: stratigraphy/coco_advanced.py
# =============================================================================
"""
COCO / eCOCO 的高级模式族：自适应、交叉拟合、交错、轨道追踪与轨道几何。

用途
====
:mod:`stratigraphy.coco` 实现了基础的 COCO（功率谱与目标谱求相关系数，反演最
可能沉积速率）与 eCOCO（逐窗做 COCO，得到随位置变化的速率）。本模块补上
Acycle 在基础模式之上的那一批"高级模式"，它们都建立在同一个统计量上——记录
功率谱与目标谱在共同频带上的 Pearson 相关系数——只是**目标谱从哪来**、**显著
性怎么定**这两件事变了：

* :func:`adaptive_coco` —— **自适应**目标谱：目标谱不再由外部给定，而是在每条
  候选速率上从数据本身重新估计（先估红噪声背景，再估每条轨道周期带里的峰高），
  背景与目标谱互相迭代到不动点为止。
* :func:`crossfit_coco` —— **交叉拟合**：把记录切成互补的两半，一半估计目标谱，
  另一半只做检验（"估计"与"检验"用的不是同一批数据），消除"在同一份数据上
  既估计又检验"的自洽偏倚。
* :func:`interleaved_coco` —— **交错**：按观测序号的奇偶把记录分成两折，奇折
  训练、偶折检验，偶折训练、奇折检验；对不均匀、时有间断的记录尤其有用，因为
  奇偶交错天然覆盖整条时间轴。
* :func:`coco_orbit_tracking` —— **轨道追踪**：在逐窗谱上把显著周期峰串成
  "轨道链"，一条链就是一条被连续追踪到的轨道。
* :func:`num_orbits` —— **轨道计数**：在给定沉积速率下，候选轨道周期里有几条
  真的能被这条记录分辨（记录够长、采样够密）。
* :func:`orbit_groups` / :func:`fixed_target_weights` —— **轨道分组**与**固定目标
  权重**：把九条（或任意条）轨道周期按物理族分组，并给出固定的目标谱权重。

方法
====
频率与单位
~~~~~~~~~~
所有高级模式与 :mod:`stratigraphy.coco` 用**同一套频率口径**：坐标轴除以候选
沉积速率 ``r``（坐标单位/Ma）得到天文时间轴（Ma），频率轴是"每 Ma"的循环
频率，轨道周期 ``p``（kyr）的频率是 ``1000 / p``。栅格由记录长度决定（Rayleigh
分辨率 ``1/T``），而不是由采样密度决定。

目标谱的三种来源
~~~~~~~~~~~~~~~~
1. **固定目标谱**：:func:`stratigraphy.coco.milankovitch_target_spectrum` 的
   Berger (1989) 现代目标谱，或调用方自带的目标谱。这就是基础 COCO 的目标谱。
2. **自适应目标谱**：每条轨道周期带（以该周期为中心、半宽为一个 Rayleigh 分辨
   率的频带）里的**背景以上峰高**作为该轨道的权重，目标谱是这些峰的叠加。
   权重是从数据估出来的，所以统计量的零假设必须**对每条替代序列重新估一遍**
   目标谱——这正是 Acycle 自适应模式的做法。
3. **训练得到的目标谱**：交叉拟合与交错模式在一折上估出目标谱，拿到另一折上
   检验；零假设里目标谱**固定不动**，替代序列只换数据不换目标。

红噪声背景
~~~~~~~~~~
背景曲线取 AR(1) 理论谱（Mann & Lees 1996 的式 3）按观测谱的比例缩放，
缩放系数用**非轨道频带**上的均值估计——轨道峰本身不该把背景抬上去。

背景与目标谱互相迭代：每一轮都把轨道频带里的周期信号（各带峰位上的余弦与
正弦最小二乘拟合）从序列里扣掉，用剩下的残差重新估计背景的 AR(1) 系数（周
期性信号会抬高滞后 1 阶自相关，把红噪声背景高估），再在新背景上重新估计目
标谱；相邻两轮目标谱的相对变化小于容差即收敛。

显著性
~~~~~~
每个模式都给出一条蒙特卡洛显著性链：替代序列是与观测等长、同 AR(1) 系数的
高斯红噪声（``red`` 参数，或由数据估计），每条替代序列按**该模式自己的规则**
构造目标谱（自适应：重新估；交叉拟合/交错：固定用训练得到的那一条），然后
统计有多少条替代序列的相关系数不小于实测值：

    p = (1 + #{零假设相关系数 >= 实测相关系数}) / (1 + 蒙特卡洛条数)

逐轨道周期的显著性另有一条链：自适应模式下统计量是"该周期带里背景以上的峰
高"，零假设里替代序列在自己的背景上取同样的量；交叉拟合与交错模式下统计量
是"留出折在**训练折**给出的频率上的功率"，零假设里替代序列在同一个频率上取
同样的量。

与基础 COCO 的关系
==================
本模块与 :mod:`stratigraphy.coco` 共用同一套口径，并且**直接调用**它的公开
接口：``coco()``（基础 COCO，用于固定目标谱的对照与轨道追踪的定速率）、
``milankovitch_target_spectrum()``（固定目标谱）、``berger1989_periods()``（默
认候选轨道周期表）。频率栅格、重采样与相关系数沿用 ``stratigraphy.coco`` 的
内部工具（``_spectral_grid``、``_resample_to_uniform_time``、``_pearson_rows``），
与包内其它模块的写法一致（例如 ``stratigraphy/coco.py`` 自己也
``from stratigraphy.spectral_analysis import _linear_detrend``），这样基础模式
与高级模式给出的频率轴**逐点相同**，两者可以直接对比。

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定（自
适应峰值提取、双向交叉拟合、奇偶交错、轨道链的连接规则、轨道分组与固定权重
的取值），未翻译或复制其 MATLAB 源码。Li et al. (2018) 给出了 COCO/eCOCO 的
统计框架，但**没有**给出自适应、交叉拟合、交错这三种模式的逐公式定义；下面
的算法是"论文方法框架 + Acycle 实现语义"的工程变体，docstring 里逐条标注了。

参考文献
========
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（COCO / eCOCO
  的统计框架：功率谱与目标谱的相关系数、蒙特卡洛显著性）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的功能划分："COCO"、"eCOCO" 及其高级模式的定位）
* Mann, M. E. & Lees, J. M. (1996). Red noise and the search for periodicity in
  Quaternary climate data. *Paleoceanography*, 11, 499-515.（AR(1) 理论谱作为
  红噪声背景；本文的背景曲线用它）
* Berger, A. (1989). Pleistocene climate from the top of the ice core. In
  *Geochemistry and Geophysics of Climate Change* (eds. Ruddiman, W. & Prell,
  W.). AGU, 13-24.（现代轨道周期表，固定目标谱的周期来源）
* Horne, J. H. & Baliunas, S. (1986). The search for periods in the light curve
  of the suspected solar variability. *The Astrophysical Journal*, 302, 717-735.
  （红噪声零假设下的假警概率）
* Schwarz, J. (2001). Invited contributions: Chronology of the paleoclimate
  record. *Eos Transactions*, 82, 682-685.（轨道计数：记录长度与 Nyquist 对
  可用轨道周期的限制）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import periodogram

from config.i18n import _
from stratigraphy.coco import (
    _pearson_rows,
    _resample_to_uniform_time,
    _spectral_grid,
    _validate_rate_range,
    _validate_record,
    berger1989_periods,
    coco,
    milankovitch_target_spectrum,
)
from stratigraphy.cycles import _ar1_phi
from stratigraphy.signal_generator import generate_red_noise
from stratigraphy.spectral_analysis import ar1_theoretical_spectrum
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

# =============================================================================
# 参数常量
# =============================================================================

#: 参与分析的最少样本数由 ``stratigraphy.coco`` 的记录校验给出（32 个样本），
#: 本模块直接复用那一份校验，不再另立一套口径。

#: 单个追踪窗口至少要有的样本数。
MIN_WINDOW_SAMPLES = 16

#: 候选沉积速率的默认个数。
DEFAULT_RATES = 21

#: 蒙特卡洛替代序列的默认条数。
DEFAULT_MONTE_CARLO = 50

#: 轨道频带的半宽，以 Rayleigh 分辨率为单位。
#:
#: Acycle 的自适应峰值提取在每条轨道周期的 Rayleigh 带内取功率极大值（矩形窗
#: 的等效噪声带宽正好是一个 Rayleigh 分辨率），本模块沿用这个口径。
BAND_HALF_WIDTH = 1.0

#: 轨道频带至少要覆盖的频率栅格点数（半宽的下限，保证带里不是孤零零一个点）。
MIN_BAND_BINS = 3

#: 比较频带至少要有的频率点数。
MIN_COMPARISON_BINS = 8

#: 轨道分组的四个物理族（与 Acycle ``cocoOrbitGroups`` 的命名一致）。
ORBIT_GROUP_NAMES: tuple[str, ...] = (
    "long eccentricity",
    "short eccentricity",
    "obliquity",
    "precession",
)

# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class AdaptiveCocoResult:
    """自适应 COCO 的结果。

    Attributes:
        rates: 实际扫描的候选速率（坐标单位/Ma）。
        correlations: 每个候选速率的相关系数；不可用的速率为 ``nan``。
        p_values: 每个候选速率的蒙特卡洛 p 值；不可用的速率为 ``nan``。
        best_rate: 相关系数最大的候选速率。
        best_correlation: 最优速率处的相关系数。
        best_p_value: 最优速率处的 p 值。
        periods_kyr: 参与检验的候选轨道周期（kyr）。
        period_p_values: 每条候选轨道周期的 p 值（与 ``periods_kyr`` 等长）。
        significant_periods_kyr: p 值不超过显著性水平的轨道周期（kyr）。
        peak_period_kyr: 最优速率处功率最强的轨道周期（kyr），无法判定时为 ``None``。
        target_frequencies: 最优速率处目标谱所在的频率点（cycles/Ma）。
        target_spectrum: 上述频率点上的目标谱功率。
        background_spectrum: 上述频率点上的红噪声背景功率。
        iterations: 最优速率处背景/目标谱不动点迭代实际用掉的轮数；固定目标谱
            模式下没有迭代，这一项是 0。
        converged: 不动点迭代是否在容差内收敛（固定目标谱模式下恒为真）。
        target_change: 最后两轮目标谱之间的相对变化（``L2`` 范数之比）。
        sig_level: 名义显著性水平。
        critical_value: 零假设相关系数分布的 ``1 - sig_level`` 分位数（临界值）。
        significant: 最优速率处的相关系数是否显著。
    """

    rates: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    best_rate: float
    best_correlation: float
    best_p_value: float
    periods_kyr: npt.NDArray
    period_p_values: npt.NDArray
    significant_periods_kyr: npt.NDArray
    peak_period_kyr: float | None
    target_frequencies: npt.NDArray
    target_spectrum: npt.NDArray
    background_spectrum: npt.NDArray
    iterations: int
    converged: bool
    target_change: float
    sig_level: float
    critical_value: float
    significant: bool

    def summary(self) -> str:
        """Generate summary text."""
        periods = ", ".join(f"{value:.4g}" for value in self.significant_periods_kyr)
        lines = [
            _("Adaptive COCO Results"),
            "=" * 50,
            _("Candidate rates tried: {0}").format(int(self.rates.size)),
            _("Best rate: {0:.6g}").format(self.best_rate),
            _("Best correlation: {0:.4f}").format(self.best_correlation),
            _("Best p-value: {0:.4g}").format(self.best_p_value),
            _("Significant periods (kyr): {0}").format(periods or _("none")),
            _("Target iteration: {0} round(s), converged: {1}").format(self.iterations, self.converged),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CrossfitCocoResult:
    """交叉拟合 COCO 的结果。

    Attributes:
        rates: 实际扫描的候选速率（坐标单位/Ma）。
        correlations: 每个候选速率的**双向共识**相关系数（两折相关系数的较小者）。
        p_values: 每个候选速率的蒙特卡洛 p 值。
        best_rate: 共识相关系数最大的候选速率。
        best_correlation: 最优速率处的共识相关系数。
        best_p_value: 最优速率处的 p 值。
        fold_correlations: 每个候选速率在两个方向上的相关系数，形状 ``(2, n_rates)``。
        fold_p_values: 每个候选速率在两个方向上的 p 值，形状 ``(2, n_rates)``。
        periods_kyr: 参与检验的候选轨道周期（kyr）。
        period_p_values: 每折、每条轨道周期的留出检验 p 值，形状 ``(2, n_periods)``。
        significant_periods_kyr: p 值不超过显著性水平的轨道周期（kyr）。
        significant_fraction: 留出检验中显著的比例。
        n_tests: 留出检验的总次数。
        sig_level: 名义显著性水平。
        critical_value: 零假设共识相关系数分布的 ``1 - sig_level`` 分位数。
        significant: 最优速率处的共识相关系数是否显著。
    """

    rates: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    best_rate: float
    best_correlation: float
    best_p_value: float
    fold_correlations: npt.NDArray
    fold_p_values: npt.NDArray
    periods_kyr: npt.NDArray
    period_p_values: npt.NDArray
    significant_periods_kyr: npt.NDArray
    significant_fraction: float
    n_tests: int
    sig_level: float
    critical_value: float
    significant: bool

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Crossfit COCO Results"),
            "=" * 50,
            _("Candidate rates tried: {0}").format(int(self.rates.size)),
            _("Best rate: {0:.6g}").format(self.best_rate),
            _("Best consensus correlation: {0:.4f}").format(self.best_correlation),
            _("Best p-value: {0:.4g}").format(self.best_p_value),
            _("Held-out tests: {0}, significant fraction: {1:.3f}").format(self.n_tests, self.significant_fraction),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class InterleavedCocoResult:
    """交错 COCO 的结果。

    Attributes:
        rates: 实际扫描的候选速率（坐标单位/Ma）。
        correlations: 每个候选速率的双向共识相关系数（奇折/偶折的较小者）。
        p_values: 每个候选速率的蒙特卡洛 p 值。
        best_rate: 共识相关系数最大的候选速率。
        best_correlation: 最优速率处的共识相关系数。
        best_p_value: 最优速率处的 p 值。
        fold_correlations: 每个候选速率在两个方向上的相关系数，形状 ``(2, n_rates)``。
        fold_p_values: 每个候选速率在两个方向上的 p 值，形状 ``(2, n_rates)``。
        periods_kyr: 参与检验的候选轨道周期（kyr）。
        period_p_values: 每折、每条轨道周期的留出检验 p 值，形状 ``(2, n_periods)``。
        significant_periods_kyr: p 值不超过显著性水平的轨道周期（kyr）。
        significant_fraction: 留出检验中显著的比例。
        n_tests: 留出检验的总次数。
        sig_level: 名义显著性水平。
        critical_value: 零假设共识相关系数分布的 ``1 - sig_level`` 分位数。
        significant: 最优速率处的共识相关系数是否显著。
    """

    rates: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    best_rate: float
    best_correlation: float
    best_p_value: float
    fold_correlations: npt.NDArray
    fold_p_values: npt.NDArray
    periods_kyr: npt.NDArray
    period_p_values: npt.NDArray
    significant_periods_kyr: npt.NDArray
    significant_fraction: float
    n_tests: int
    sig_level: float
    critical_value: float
    significant: bool

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Interleaved COCO Results"),
            "=" * 50,
            _("Candidate rates tried: {0}").format(int(self.rates.size)),
            _("Best rate: {0:.6g}").format(self.best_rate),
            _("Best consensus correlation: {0:.4f}").format(self.best_correlation),
            _("Best p-value: {0:.4g}").format(self.best_p_value),
            _("Held-out tests: {0}, significant fraction: {1:.3f}").format(self.n_tests, self.significant_fraction),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class OrbitTrackingResult:
    """轨道追踪的结果。

    Attributes:
        frequencies: 最后一个追踪窗口的频率轴（cycles/Ma）。
        track_frequencies: 每条轨道链上追踪到的频率，元素为数组。
        track_periods_kyr: 每条轨道链的中心周期（kyr）。
        n_orbits: 轨道链的条数。
        window_centres: 各追踪窗口的中心时间（Ma）。
        window_thresholds: 各窗口的显著性阈值（功率）。
        window_p_values: 各窗口最强峰的假警概率（红噪声零假设）。
        sig_level: 名义显著性水平。
        critical_value: 各窗口显著性阈值里的最大值（最强的一条判据）。
        significant: 是否至少追踪到一条轨道链。
    """

    frequencies: npt.NDArray
    track_frequencies: tuple[npt.NDArray, ...]
    track_periods_kyr: npt.NDArray
    n_orbits: int
    window_centres: npt.NDArray
    window_thresholds: npt.NDArray
    window_p_values: npt.NDArray
    sig_level: float
    critical_value: float
    significant: bool

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Orbit Tracking Results"),
            "=" * 50,
            _("Windows: {0}").format(int(self.window_centres.size)),
            _("Orbits tracked: {0}").format(self.n_orbits),
        ]
        for index, period in enumerate(self.track_periods_kyr):
            lines.append(_("Orbit {0}: {1:.4g} kyr").format(index + 1, float(period)))
        return "\n".join(lines)


@dataclass(frozen=True)
class NumOrbitsResult:
    """轨道计数的结果。

    Attributes:
        rates: 参与统计的候选速率（坐标单位/Ma）。
        n_orbits: 每个速率下能被分辨的轨道周期条数。
        resolved: 形状 ``(n_rates, n_periods)`` 的布尔矩阵，标出每条轨道周期
            在该速率下是否可用。
        periods_kyr: 候选轨道周期（kyr）。
        time_span_ma: 每个速率下记录的时间跨度（Ma）。
        sig_level: 名义显著性水平（保留字段：本函数只做几何判定，不做假设检验）。
        critical_value: 临界值；本函数不做假设检验，恒为 ``None``。
        significant: 恒为 ``False``；本函数不做假设检验。
    """

    rates: npt.NDArray
    n_orbits: npt.NDArray
    resolved: npt.NDArray
    periods_kyr: npt.NDArray
    time_span_ma: npt.NDArray
    sig_level: float
    critical_value: float | None = None
    significant: bool = False

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Orbit Count"),
            "=" * 50,
            _("Candidate rates: {0}").format(int(self.rates.size)),
        ]
        for rate, count in zip(self.rates, self.n_orbits, strict=True):
            lines.append(_("Rate {0:.6g}: {1} orbit(s)").format(float(rate), int(count)))
        return "\n".join(lines)


@dataclass(frozen=True)
class OrbitGroupsResult:
    """轨道分组的结果。

    Attributes:
        periods_kyr: 输入的轨道周期（kyr）。
        groups: 每条轨道周期所属的物理族名称。
        index: 每条轨道周期的组号（1..4），按周期从长到短排名给出。
        counts: 四个物理族的成员数。
        names: 四个物理族的名称。
        rule: 分组规则的人读描述。
        sig_level: 名义显著性水平（保留字段：本函数只做分组，不做假设检验）。
        critical_value: 临界值；本函数不做假设检验，恒为 ``None``。
        significant: 恒为 ``False``；本函数不做假设检验。
    """

    periods_kyr: npt.NDArray
    groups: tuple[str, ...]
    index: npt.NDArray
    counts: npt.NDArray
    names: tuple[str, ...]
    rule: str
    sig_level: float
    critical_value: float | None = None
    significant: bool = False

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Orbit Groups"),
            "=" * 50,
            _("Rule: {0}").format(self.rule),
        ]
        for name, count in zip(self.names, self.counts, strict=True):
            lines.append(_("{0}: {1} member(s)").format(name, int(count)))
        return "\n".join(lines)


@dataclass(frozen=True)
class FixedTargetWeightsResult:
    """固定目标权重的结果。

    Attributes:
        periods_kyr: 输入的轨道周期（kyr）。
        weights: 每条轨道周期在目标谱里的权重。
        rule: 权重规则的人读描述。
        sig_level: 名义显著性水平（保留字段：本函数只给权重，不做假设检验）。
        critical_value: 临界值；本函数不做假设检验，恒为 ``None``。
        significant: 恒为 ``False``；本函数不做假设检验。
    """

    periods_kyr: npt.NDArray
    weights: npt.NDArray
    rule: str
    sig_level: float
    critical_value: float | None = None
    significant: bool = False

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Fixed Target Weights"),
            "=" * 50,
            _("Rule: {0}").format(self.rule),
        ]
        for period, weight in zip(self.periods_kyr, self.weights, strict=True):
            lines.append(_("{0:.6g} kyr: {1:.3f}").format(float(period), float(weight)))
        return "\n".join(lines)


# =============================================================================
# 内部工具
# =============================================================================


def _validate_level(sig_level: float) -> float:
    """校验名义显著性水平。"""
    try:
        level = float(sig_level)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("sig_level must be a number between 0 and 1"),
            details={"sig_level": repr(sig_level)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(level) or not (0.0 < level < 1.0):
        raise ValidationError(
            _("sig_level must lie strictly between 0 and 1; got {0}").format(level),
            details={"sig_level": level},
        )
    return level


def _validate_red(red: float | None, values: npt.NDArray) -> float:
    """给出零假设用的 AR(1) 系数：调用方指定，或由数据估计。"""
    if red is None:
        phi = _ar1_phi(values)
    else:
        try:
            phi = float(red)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("red must be a number strictly between -1 and 1"),
                details={"red": repr(red)},
                original_exception=exc,
            ) from exc
        if not np.isfinite(phi) or not (-1.0 < phi < 1.0):
            raise ValidationError(
                _("red must lie strictly between -1 and 1; got {0}").format(phi),
                details={"red": phi},
            )
    return float(np.clip(phi, -0.99, 0.99))


def _validate_periods(periods: npt.NDArray) -> npt.NDArray:
    """校验轨道周期：有限、为正。"""
    values = np.asarray(periods, dtype=float).flatten()
    if values.size == 0:
        raise ValidationError(_("at least one orbital period is required"))
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValidationError(
            _("orbital periods must be finite and positive"),
            details={"n_invalid": int(np.sum(~np.isfinite(values) | (values <= 0.0)))},
        )
    return values


def _resolve_periods(
    periods_kyr: npt.NDArray | None,
    n_period: int | None,
    n_target: int | None,
    target_period: float | None,
) -> tuple[npt.NDArray, int]:
    """定下候选轨道周期表与目标谱里的谐波个数。

    Returns
    -------
    tuple
        ``(periods, harmonics)``：按周期**从长到短**排好的候选周期表（kyr），
        以及目标谱里实际用到的谐波个数。
    """
    if target_period is not None:
        try:
            value = float(target_period)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("target_period must be a positive number of kyr"),
                details={"target_period": repr(target_period)},
                original_exception=exc,
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValidationError(
                _("target_period must be finite and positive; got {0}").format(value),
                details={"target_period": value},
            )
        periods = np.array([value], dtype=float)
    elif periods_kyr is None:
        periods = np.array(sorted(berger1989_periods().values(), reverse=True), dtype=float)
    else:
        periods = _validate_periods(periods_kyr)
        periods = np.sort(periods)[::-1]

    if n_period is not None:
        count = int(n_period)
        if count < 1:
            raise ValidationError(
                _("n_period must be at least 1; got {0}").format(count),
                details={"n_period": count},
            )
        periods = periods[:count]

    harmonics = periods.size if n_target is None else int(n_target)
    if harmonics < 1:
        raise ValidationError(
            _("n_target must be at least 1; got {0}").format(harmonics),
            details={"n_target": harmonics},
        )
    harmonics = int(min(harmonics, periods.size))
    return periods, harmonics


def _rate_grid(rate_range: tuple[float, float] | None, n_rates: int) -> npt.NDArray:
    """候选速率栅格；``rate_range`` 为 ``None`` 时只用速率 1.0。"""
    if rate_range is None:
        return np.array([1.0], dtype=float)
    low, high = _validate_rate_range(rate_range, n_rates)
    return np.linspace(low, high, int(n_rates))


def _validate_iteration_controls(max_iterations: int, tolerance: float) -> tuple[int, float]:
    """校验不动点迭代的轮数上限与收敛判据。"""
    rounds = int(max_iterations)
    if rounds < 1:
        raise ValidationError(
            _("max_iterations must be at least 1; got {0}").format(rounds),
            details={"max_iterations": rounds},
        )
    try:
        limit = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("tolerance must be a positive number"),
            details={"tolerance": repr(tolerance)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValidationError(
            _("tolerance must be finite and positive; got {0}").format(limit),
            details={"tolerance": limit},
        )
    return rounds, limit


def _validate_monte_carlo(n_monte_carlo: int) -> int:
    """校验蒙特卡洛条数。"""
    count = int(n_monte_carlo)
    if count < 0:
        raise ValidationError(
            _("n_monte_carlo must be >= 0; got {0}").format(count),
            details={"n_monte_carlo": count},
        )
    return count


def _ar1_surrogates(values: npt.NDArray, phi: float, count: int, rng: np.random.Generator) -> npt.NDArray:
    """AR(1) 红噪声替代序列，边际标准差与观测一致。"""
    sigma = float(np.std(values))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    out = np.empty((int(count), values.size), dtype=float)
    for index in range(int(count)):
        seed = int(rng.integers(0, 2**31 - 1))
        out[index] = generate_red_noise(values.size, rho=phi, sigma=sigma, random_seed=seed)
    return out


def _temporal_spectrum(
    time: npt.NDArray, values: npt.NDArray, rate: float, nfft: int
) -> tuple[npt.NDArray, npt.NDArray, float, npt.NDArray, npt.NDArray]:
    """把记录调到速率 ``rate`` 的天文时间轴上并求周期图。

    Returns
    -------
    tuple
        ``(grid, resampled, dt, freqs, power)``：``grid`` 是均匀的天文时间轴
        （Ma），``resampled`` 是重采样后的数值，``dt`` 是采样间距（Ma），
        ``freqs``/``power`` 是周期图。
    """
    grid, resampled, dt = _resample_to_uniform_time(time, values, rate)
    freqs, power = periodogram(
        resampled,
        fs=1.0 / dt,
        detrend="linear",
        nfft=int(nfft),
        scaling="spectrum",
    )
    return grid, resampled, float(dt), freqs, power


def _band_half_width(rayleigh: float, spacing: float) -> float:
    """轨道频带的半宽：一个 Rayleigh 分辨率，且至少盖住几个频率点。"""
    return float(max(BAND_HALF_WIDTH * rayleigh, 0.5 * MIN_BAND_BINS * spacing))


def _band_masks(freqs: npt.NDArray, centres: npt.NDArray, half_width: float) -> list[npt.NDArray]:
    """每条轨道周期带内的频率点下标。"""
    masks: list[npt.NDArray] = []
    for centre in centres:
        masks.append(np.flatnonzero((freqs >= centre - half_width) & (freqs <= centre + half_width)))
    return masks


def _ar1_background(
    freqs: npt.NDArray,
    power: npt.NDArray,
    phi: float,
    dt: float,
    excluded: npt.NDArray,
) -> npt.NDArray:
    """红噪声背景曲线。

    取 AR(1) 理论谱（Mann & Lees 1996 的式 3）按观测谱的比例缩放；缩放系数
    用**非轨道频带**上的均值估计，免得轨道峰自己把背景抬上去。
    """
    curve = np.asarray(ar1_theoretical_spectrum(freqs, phi, dt), dtype=float)
    curve = np.where(np.isfinite(curve), curve, 0.0)
    usable = (curve > 0.0) & (~excluded)
    if not np.any(usable):
        usable = curve > 0.0
    if not np.any(usable):
        return np.zeros_like(power)
    scale = float(np.mean(power[usable] / curve[usable]))
    if not np.isfinite(scale) or scale < 0.0:
        scale = 0.0
    return scale * curve


def _target_from_excess(
    freqs: npt.NDArray,
    power: npt.NDArray,
    background: npt.NDArray,
    bands: list[npt.NDArray],
    centres: npt.NDArray,
    half_width: float,
) -> npt.NDArray:
    """由"背景以上的峰高"搭出目标谱。

    每条轨道周期带里取背景以上的最大功率作为该轨道的权重，目标谱是这些权重
    在以轨道频率为中心的三角窗上的叠加——这与 Acycle 自适应模式"在每条轨道的
    Rayleigh 带内取功率极大值作为幅度"的语义一致。
    """
    target = np.zeros_like(power)
    for band, centre in zip(bands, centres, strict=True):
        if band.size == 0:
            continue
        excess = power[band] - background[band]
        peak = float(np.max(excess))
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        shape = np.clip(1.0 - np.abs(freqs[band] - centre) / half_width, 0.0, 1.0)
        target[band] += peak * shape
    return target


def _relative_change(new: npt.NDArray, old: npt.NDArray) -> float:
    """两个向量之间的相对变化（``L2`` 范数之比）。"""
    denominator = float(np.linalg.norm(old))
    numerator = float(np.linalg.norm(new - old))
    if denominator <= 0.0:
        return numerator
    return numerator / denominator


def _remove_band_sinusoids(
    grid: npt.NDArray,
    series: npt.NDArray,
    freqs: npt.NDArray,
    power: npt.NDArray,
    background: npt.NDArray,
    bands: list[npt.NDArray],
) -> npt.NDArray:
    """把轨道频带里的周期信号从序列里扣掉。

    每条频带在它的峰位上用最小二乘拟合 ``a cos(w t) + b sin(w t)``（两个系数
    用 2x2 正规方程解），扣掉之后得到残差序列。残差用来重新估计红噪声背景
    的 AR(1) 系数——周期性信号会把滞后 1 阶自相关抬高，背景因此被高估。
    """
    residual = np.array(series, dtype=float, copy=True)
    for band in bands:
        if band.size == 0:
            continue
        excess = power[band] - background[band]
        peak = float(np.max(excess))
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        local = int(band[int(np.argmax(excess))])
        omega = 2.0 * np.pi * float(freqs[local]) * grid
        cosine = np.cos(omega)
        sine = np.sin(omega)
        cc = float(np.sum(cosine * cosine))
        ss = float(np.sum(sine * sine))
        cs = float(np.sum(cosine * sine))
        cv = float(np.sum(cosine * residual))
        sv = float(np.sum(sine * residual))
        determinant = cc * ss - cs * cs
        if not np.isfinite(determinant) or abs(determinant) < 1e-12:
            continue
        a = (ss * cv - cs * sv) / determinant
        b = (cc * sv - cs * cv) / determinant
        residual = residual - (a * cosine + b * sine)
    return residual


def _fit_target(
    grid: npt.NDArray,
    series: npt.NDArray,
    freqs: npt.NDArray,
    power: npt.NDArray,
    phi: float,
    dt: float,
    bands: list[npt.NDArray],
    centres: npt.NDArray,
    half_width: float,
    excluded: npt.NDArray,
    max_iterations: int,
    tolerance: float,
) -> tuple[npt.NDArray, npt.NDArray, int, float, bool]:
    """背景与目标谱的不动点迭代。

    第一轮用整条谱估背景、用背景以上的峰高搭出目标谱。之后每一轮都做两件事：

    1. 把轨道频带里的周期信号从序列里扣掉，用残差重新估计背景的 AR(1)
       系数（周期性信号会抬高滞后 1 阶自相关，把红噪声背景高估）；
    2. 在新的背景上重新估计目标谱。

    相邻两轮目标谱的相对变化小于 ``tolerance`` 即认为收敛。

    依据：Acycle 的实现语义（自适应的背景与目标谱互相修正）+ Li et al. (2018)
    的方法框架；论文没有给出这一迭代的逐公式定义，属于工程变体。
    """
    background = _ar1_background(freqs, power, phi, dt, excluded)
    target = _target_from_excess(freqs, power, background, bands, centres, half_width)
    change = float("inf")
    iterations = 0
    converged = False
    for iteration in range(1, int(max_iterations) + 1):
        residual = _remove_band_sinusoids(grid, series, freqs, power, background, bands)
        phi = _ar1_phi(residual)
        new_background = _ar1_background(freqs, power, phi, dt, excluded)
        new_target = _target_from_excess(freqs, power, new_background, bands, centres, half_width)
        change = _relative_change(new_target, target)
        target = new_target
        background = new_background
        iterations = iteration
        if change <= float(tolerance):
            converged = True
            break
    return target, background, iterations, change, converged


def _fixed_target(freqs: npt.NDArray) -> npt.NDArray:
    """固定目标谱（Berger 1989 现代目标谱），与基础 COCO 同一个来源。"""
    return np.asarray(milankovitch_target_spectrum(freqs), dtype=float)


def _external_target(
    target_spectrum: tuple[npt.NDArray, npt.NDArray], freqs: npt.NDArray
) -> tuple[npt.NDArray, tuple[float, float]]:
    """把调用方自带的目标谱插值到频率栅格上，并给出它的频率范围。"""
    try:
        target_freq = np.asarray(target_spectrum[0], dtype=float).flatten()
        target_power = np.asarray(target_spectrum[1], dtype=float).flatten()
    except (TypeError, ValueError, IndexError) as exc:
        raise ValidationError(
            _("target_spectrum must be a (frequencies, power) pair of arrays"),
            details={"target_spectrum": repr(target_spectrum)},
            original_exception=exc,
        ) from exc
    if target_freq.size != target_power.size or target_freq.size == 0:
        raise ValidationError(
            _("target_spectrum frequencies and power must have the same non-zero length"),
            details={
                "n_frequencies": int(target_freq.size),
                "n_power": int(target_power.size),
            },
        )
    if not np.all(np.isfinite(target_freq)) or np.any(target_freq <= 0.0):
        raise ValidationError(
            _("target_spectrum frequencies must be finite and positive (cycles per unit time)"),
            details={"n_invalid": int(np.sum(~np.isfinite(target_freq) | (target_freq <= 0.0)))},
        )
    order = np.argsort(target_freq)
    target_freq = target_freq[order]
    target_power = target_power[order]
    interpolated = np.interp(freqs, target_freq, target_power, left=0.0, right=0.0)
    return interpolated, (float(target_freq[0]), float(target_freq[-1]))


def _comparison_mask(
    freqs: npt.NDArray,
    rayleigh: float,
    nyquist: float,
    centres: npt.NDArray,
    half_width: float,
) -> npt.NDArray:
    """相关系数的比较频带：所有目标轨道频带的外包络。"""
    low = max(float(rayleigh), float(np.min(centres)) - half_width)
    high = min(float(nyquist), float(np.max(centres)) + half_width)
    return (freqs >= low) & (freqs <= high)


def _pearson(values: npt.NDArray, target: npt.NDArray) -> float:
    """两条向量的 Pearson 相关系数。"""
    if values.size < 2:
        return float("nan")
    return float(_pearson_rows(values[None, :], target)[0])


def _band_peaks(
    freqs: npt.NDArray,
    power: npt.NDArray,
    background: npt.NDArray,
    bands: list[npt.NDArray],
) -> npt.NDArray:
    """每条轨道频带里"背景以上的峰高"。"""
    peaks = np.zeros(len(bands), dtype=float)
    for index, band in enumerate(bands):
        if band.size == 0:
            continue
        peaks[index] = float(np.max(power[band] - background[band]))
    return peaks


def _interpolate_power(freqs: npt.NDArray, power: npt.NDArray, frequency: float) -> float:
    """在给定频率上线性插值功率。"""
    if not np.isfinite(frequency) or freqs.size < 2:
        return float("nan")
    if frequency <= freqs[0] or frequency >= freqs[-1]:
        return float("nan")
    return float(np.interp(frequency, freqs, power))


def _split_halves(time: npt.NDArray, values: npt.NDArray) -> tuple[tuple, tuple]:
    """把记录按时间中点切成互补的两半。"""
    midpoint = 0.5 * (float(time[0]) + float(time[-1]))
    first = time < midpoint
    if not np.any(first) or np.all(first):
        half = time.size // 2
        first = np.zeros(time.size, dtype=bool)
        first[:half] = True
    second = ~first
    return (time[first], values[first]), (time[second], values[second])


def _interleaved_folds(time: npt.NDArray, values: npt.NDArray) -> tuple[tuple, tuple]:
    """按观测序号的奇偶把记录分成两折（全局分配，不随窗口起点改变）。"""
    odd = slice(0, None, 2)
    even = slice(1, None, 2)
    return (time[odd], values[odd]), (time[even], values[even])


def _p_value(observed: float, null: npt.NDArray) -> float:
    """ "加一"蒙特卡洛 p 值。"""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1.0 + np.sum(null >= observed)) / (null.size + 1.0))


def _quantile(null: npt.NDArray, level: float) -> float:
    """零假设分布的 ``level`` 分位数（临界值）。"""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return float("nan")
    return float(np.quantile(null, level))


def _significant_periods(periods: npt.NDArray, p_values: npt.NDArray, level: float) -> npt.NDArray:
    """p 值不超过显著性水平的轨道周期。"""
    mask = np.isfinite(p_values) & (np.asarray(p_values) <= float(level))
    return periods[mask]


def _parabolic_peak(freqs: npt.NDArray, power: npt.NDArray, index: int) -> float:
    """峰位抛物线插值，比直接取栅格点准得多。"""
    if index <= 0 or index >= freqs.size - 1:
        return float(freqs[index])
    left, centre, right = power[index - 1], power[index], power[index + 1]
    denominator = left - 2.0 * centre + right
    if not np.isfinite(denominator) or denominator == 0.0:
        return float(freqs[index])
    numerator = 0.5 * (left - right)
    shift = float(np.clip(numerator / denominator, -1.0, 1.0))
    spacing = float(freqs[index + 1] - freqs[index - 1]) / 2.0
    return float(freqs[index] + shift * spacing)


def _local_peaks(power: npt.NDArray) -> npt.NDArray:
    """局部极大值的下标。"""
    if power.size < 3:
        return np.array([], dtype=int)
    interior = np.arange(1, power.size - 1)
    keep = (power[interior] >= power[interior - 1]) & (power[interior] > power[interior + 1])
    return interior[keep]


def _align_power(freqs: npt.NDArray, power: npt.NDArray, target_freqs: npt.NDArray) -> npt.NDArray:
    """把一条谱线性插值到另一条谱的频率栅格上。"""
    if freqs.size == target_freqs.size and np.allclose(freqs, target_freqs):
        return power
    return np.interp(target_freqs, freqs, power, left=0.0, right=0.0)


# =============================================================================
# 自适应 COCO
# =============================================================================


def adaptive_coco(
    time: npt.NDArray,
    values: npt.NDArray,
    *,
    rate_range: tuple[float, float] | None = None,
    n_rates: int = DEFAULT_RATES,
    periods_kyr: npt.NDArray | None = None,
    n_target: int | None = None,
    n_period: int | None = None,
    target_period: float | None = None,
    sig_level: float = 0.05,
    red: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    max_iterations: int = 5,
    tolerance: float = 1e-3,
    seed: int | None = None,
    target: str = "adaptive",
    target_spectrum: tuple[npt.NDArray, npt.NDArray] | None = None,
) -> AdaptiveCocoResult:
    """自适应 COCO：目标谱从数据本身迭代地估出来。

    基础 COCO 的目标谱是外部给定的（Berger 1989 现代目标谱，或调用方自带的
    目标谱）。自适应模式把它换成"从这条记录估出来"的目标谱：先估红噪声背景，
    再取每条轨道周期带里背景以上的峰高作为该轨道的权重；背景与目标谱互相修
    正，直到目标谱不再变化。

    **零假设必须与统计量匹配**：目标谱是从数据估出来的，所以每一条替代序列
    都要**重新估一遍自己的目标谱**，否则显著性会被系统性地高估。这是 Acycle
    自适应模式的核心处理（``cocoAdaptiveEvaluate`` 对每条谱独立做自适应峰值
    提取）。

    Parameters
    ----------
    time
        记录的地层坐标轴（严格递增）。缺省 ``rate_range`` 时，这个坐标轴被
        当作**已经是天文时间轴（Ma）**，只用速率 1.0。
    values
        与 ``time`` 等长的测量序列。
    rate_range
        候选沉积速率区间（坐标单位/Ma）；``None`` 表示坐标轴本身已是 Ma。
    n_rates
        扫描的候选速率个数。
    periods_kyr
        候选轨道周期（kyr）；缺省用 Berger (1989) 的九条周期。
    n_target
        目标谱里用到的谐波个数（缺省是全部候选周期）。
    n_period
        候选轨道周期的条数（缺省是全部）。
    target_period
        给定后候选周期表只剩这一条周期（kyr）。
    sig_level
        名义显著性水平。
    red
        零假设的 AR(1) 系数；``None`` 时由数据估计。
    n_monte_carlo
        蒙特卡洛替代序列条数；0 表示不算 p 值。
    max_iterations
        背景/目标谱不动点迭代的最大轮数。
    tolerance
        收敛判据：相邻两轮目标谱的相对变化。
    seed
        随机种子；``None`` 时结果不可复现（按仓库约定发出 ``RuntimeWarning``）。
    target
        ``"adaptive"``（缺省）：目标谱从数据估出来，零假设里每条替代序列都重
        新估一遍自己的目标谱。``"fixed"``：目标谱固定不变（Berger 1989 现代目
        标谱，或调用方给的目标谱），零假设里替代序列只换数据不换目标——这与
        Acycle 的 ``AmplitudeMode`` 两种取值对应。
    target_spectrum
        调用方自带的目标谱 ``(频率, 功率)``；给了它就等于 ``target="fixed"``，
        比较频带取目标谱的频率范围（与 :func:`stratigraphy.coco.coco` 的约定
        一致）。

    Returns
    -------
    AdaptiveCocoResult
        速率扫描结果、逐轨道周期的显著性、最终目标谱与收敛信息。

    Raises
    ------
    ValidationError
        记录不合法、参数越界、没有任何候选速率可用等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import adaptive_coco
    >>> axis = np.linspace(0.0, 3.0, 600)
    >>> record = np.sin(2.0 * np.pi * axis / 0.021) + np.sin(2.0 * np.pi * axis / 0.019)
    >>> result = adaptive_coco(axis, record, periods_kyr=np.array([19.0, 21.0]),
    ...                         n_monte_carlo=20, seed=1)
    >>> bool(result.significant_periods_kyr.size == 2)
    True
    """
    t, v = _validate_record(time, values)
    level = _validate_level(sig_level)
    periods, harmonics = _resolve_periods(periods_kyr, n_period, n_target, target_period)
    rates = _rate_grid(rate_range, n_rates)
    draws = _validate_monte_carlo(n_monte_carlo)
    rounds, limit = _validate_iteration_controls(max_iterations, tolerance)
    mode = str(target).strip().lower()
    if mode not in ("adaptive", "fixed"):
        raise ValidationError(
            _("target must be 'adaptive' or 'fixed'; got {0}").format(target),
            details={"target": target},
        )
    if target_spectrum is not None:
        mode = "fixed"

    phi = _validate_red(red, v)
    rng = make_rng(seed, context="adaptive COCO")
    nfft = _spectral_grid(t.size)
    centres = 1000.0 / periods
    target_centres = centres[:harmonics]

    correlations = np.full(rates.size, np.nan)
    p_values = np.full(rates.size, np.nan)
    states: dict[int, dict[str, object]] = {}

    for index, rate in enumerate(rates):
        grid, resampled, dt, freqs, power = _temporal_spectrum(t, v, float(rate), nfft)
        span = float(grid[-1] - grid[0]) + dt
        rayleigh = 1.0 / span
        nyquist = 0.5 / dt
        spacing = float(freqs[1] - freqs[0]) if freqs.size > 1 else rayleigh
        half_width = _band_half_width(rayleigh, spacing)
        bands = _band_masks(freqs, centres, half_width)
        if any(band.size == 0 for band in bands):
            logger.warning(
                "adaptive COCO: rate %.6g skipped -- an orbital band is not resolved by the record",
                float(rate),
            )
            continue
        excluded = np.zeros(freqs.size, dtype=bool)
        for band in bands:
            excluded[band] = True
        if mode == "fixed" and target_spectrum is not None:
            fixed_target, (target_lo, target_hi) = _external_target(target_spectrum, freqs)
            low = max(rayleigh, target_lo)
            high = min(nyquist, target_hi)
            mask = (freqs >= low) & (freqs <= high)
        else:
            fixed_target = np.zeros_like(power)
            mask = _comparison_mask(freqs, rayleigh, nyquist, target_centres, half_width)
        if int(np.sum(mask)) < MIN_COMPARISON_BINS:
            logger.warning(
                "adaptive COCO: rate %.6g skipped -- fewer than %d frequencies in the comparison band",
                float(rate),
                MIN_COMPARISON_BINS,
            )
            continue

        background = _ar1_background(freqs, power, phi, dt, excluded)
        iterations = 0
        change = 0.0
        converged = True
        if mode == "adaptive":
            target, background, iterations, change, converged = _fit_target(
                grid,
                resampled,
                freqs,
                power,
                phi,
                dt,
                bands[:harmonics],
                target_centres,
                half_width,
                excluded,
                rounds,
                limit,
            )
        else:
            target = fixed_target if target_spectrum is not None else _fixed_target(freqs)
        correlation = _pearson(power[mask], target[mask])
        if not np.isfinite(correlation):
            continue
        correlations[index] = correlation

        null_values = np.empty(0, dtype=float)
        null_peaks = np.empty((0, periods.size), dtype=float)
        if draws > 0:
            surrogates = _ar1_surrogates(resampled, phi, draws, rng)
            _null_freqs, null_power = periodogram(
                surrogates,
                fs=1.0 / dt,
                detrend="linear",
                nfft=int(nfft),
                scaling="spectrum",
                axis=-1,
            )
            null_values = np.empty(draws, dtype=float)
            null_peaks = np.empty((draws, periods.size), dtype=float)
            for sim in range(draws):
                if mode == "adaptive":
                    sim_target, sim_background, _it, _ch, _cv = _fit_target(
                        grid,
                        surrogates[sim],
                        freqs,
                        null_power[sim],
                        phi,
                        dt,
                        bands[:harmonics],
                        target_centres,
                        half_width,
                        excluded,
                        rounds,
                        limit,
                    )
                else:
                    sim_target = target
                    sim_background = _ar1_background(freqs, null_power[sim], phi, dt, excluded)
                null_values[sim] = _pearson(null_power[sim][mask], sim_target[mask])
                null_peaks[sim] = _band_peaks(freqs, null_power[sim], sim_background, bands)
            p_values[index] = _p_value(correlation, null_values)

        states[index] = {
            "rate": float(rate),
            "freqs": freqs,
            "power": power,
            "target": target,
            "background": background,
            "bands": bands,
            "mask": mask,
            "iterations": iterations,
            "change": change,
            "converged": converged,
            "null_values": null_values,
            "null_peaks": null_peaks,
        }

    finite = np.isfinite(correlations)
    if not np.any(finite):
        raise ValidationError(
            _("no candidate rate is admissible: the record is too short or too coarse to resolve the orbital bands"),
            details={"n_rates": int(rates.size)},
        )
    best_index = int(np.nanargmax(correlations))
    best_rate = float(rates[best_index])
    best_correlation = float(correlations[best_index])
    best_p = float(p_values[best_index])
    state = states[best_index]

    null_all = state["null_values"]
    critical = _quantile(null_all, 1.0 - level) if np.size(null_all) else float("nan")

    observed_peaks = _band_peaks(state["freqs"], state["power"], state["background"], state["bands"])
    null_peaks = state["null_peaks"]
    period_p_values = np.full(periods.size, np.nan)
    if np.size(null_peaks):
        for index in range(periods.size):
            period_p_values[index] = _p_value(observed_peaks[index], null_peaks[:, index])

    significant_periods = _significant_periods(periods, period_p_values, level)
    peak_period: float | None = None
    if observed_peaks.size and float(np.max(observed_peaks)) > 0.0:
        peak_period = float(periods[int(np.argmax(observed_peaks))])

    logger.info(
        "adaptive COCO: best rate %.6g (correlation %.4f, p %.4g), %d significant period(s)",
        best_rate,
        best_correlation,
        best_p,
        int(significant_periods.size),
    )

    return AdaptiveCocoResult(
        rates=rates,
        correlations=correlations,
        p_values=p_values,
        best_rate=best_rate,
        best_correlation=best_correlation,
        best_p_value=best_p,
        periods_kyr=periods,
        period_p_values=period_p_values,
        significant_periods_kyr=significant_periods,
        peak_period_kyr=peak_period,
        target_frequencies=state["freqs"][state["mask"]],
        target_spectrum=state["target"][state["mask"]],
        background_spectrum=state["background"][state["mask"]],
        iterations=int(state["iterations"]),
        converged=bool(state["converged"]),
        target_change=float(state["change"]),
        sig_level=level,
        critical_value=critical,
        significant=bool(np.isfinite(best_p) and best_p <= level),
    )


# =============================================================================
# 交叉拟合与交错
# =============================================================================


def _fold_spectra(
    folds: tuple[tuple, tuple],
    rates: npt.NDArray,
    nfft: int,
    centres: npt.NDArray,
    harmonics: int,
    phi: float,
    max_iterations: int,
    tolerance: float,
) -> list[dict[str, object]]:
    """两折在每个速率上的谱、背景与目标谱。

    ``centres`` 是**全部**候选轨道周期的频率（逐周期检验要用），目标谱只用
    前 ``harmonics`` 个。
    """
    states: list[dict[str, object]] = []
    target_centres = centres[:harmonics]
    for rate in rates:
        fold_states: list[dict[str, object]] = []
        for fold_time, fold_values in folds:
            grid, resampled, dt, freqs, power = _temporal_spectrum(fold_time, fold_values, float(rate), nfft)
            span = float(grid[-1] - grid[0]) + dt
            rayleigh = 1.0 / span
            nyquist = 0.5 / dt
            spacing = float(freqs[1] - freqs[0]) if freqs.size > 1 else rayleigh
            half_width = _band_half_width(rayleigh, spacing)
            bands = _band_masks(freqs, centres, half_width)
            excluded = np.zeros(freqs.size, dtype=bool)
            for band in bands:
                excluded[band] = True
            target, background, iterations, change, converged = _fit_target(
                grid,
                resampled,
                freqs,
                power,
                phi,
                dt,
                bands[:harmonics],
                target_centres,
                half_width,
                excluded,
                max_iterations,
                tolerance,
            )
            fold_states.append(
                {
                    "grid": grid,
                    "resampled": resampled,
                    "dt": dt,
                    "nfft": int(nfft),
                    "freqs": freqs,
                    "power": power,
                    "target": target,
                    "background": background,
                    "bands": bands,
                    "half_width": half_width,
                    "rayleigh": rayleigh,
                    "nyquist": nyquist,
                    "iterations": iterations,
                    "change": change,
                    "converged": converged,
                }
            )
        states.append({"rate": float(rate), "folds": fold_states})
    return states


def _crossfit_rate_scan(
    states: list[dict[str, object]],
    centres: npt.NDArray,
    phi: float,
    draws: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """两折的双向相关系数与蒙特卡洛 p 值。

    统计量是**双向共识**：两个方向（用一折训练、另一折检验）的相关系数取较小
    者——这是 Acycle 交叉拟合模式的核心（forward 与 backward 两个方向的较小
    值），避免只在一边成立的巧合。

    零假设里目标谱**固定不动**（它是训练折估出来的），替代序列只换数据。
    """
    correlations = np.full(len(states), np.nan)
    p_values = np.full(len(states), np.nan)
    fold_correlations = np.full((2, len(states)), np.nan)
    fold_p_values = np.full((2, len(states)), np.nan)

    for index, state in enumerate(states):
        first, second = state["folds"]
        freqs = first["freqs"]
        second_power = _align_power(second["freqs"], second["power"], freqs)
        second_target = _align_power(second["freqs"], second["target"], freqs)
        half_width = float(first["half_width"])
        mask = _comparison_mask(
            freqs,
            max(float(first["rayleigh"]), float(second["rayleigh"])),
            min(float(first["nyquist"]), float(second["nyquist"])),
            centres,
            half_width,
        )
        if int(np.sum(mask)) < MIN_COMPARISON_BINS:
            logger.warning(
                "crossfit COCO: rate %.6g skipped -- too few frequencies in the comparison band",
                state["rate"],
            )
            continue

        rho_forward = _pearson(second_power[mask], first["target"][mask])
        rho_backward = _pearson(first["power"][mask], second_target[mask])
        consensus = min(rho_forward, rho_backward)
        fold_correlations[0, index] = rho_forward
        fold_correlations[1, index] = rho_backward
        correlations[index] = consensus

        if draws <= 0:
            continue

        null_forward = np.empty(draws, dtype=float)
        null_backward = np.empty(draws, dtype=float)
        surrogates_first = _ar1_surrogates(first["resampled"], phi, draws, rng)
        surrogates_second = _ar1_surrogates(second["resampled"], phi, draws, rng)
        _first_freqs, null_first = periodogram(
            surrogates_first,
            fs=1.0 / float(first["dt"]),
            detrend="linear",
            nfft=int(first["nfft"]),
            scaling="spectrum",
            axis=-1,
        )
        _second_freqs, null_second = periodogram(
            surrogates_second,
            fs=1.0 / float(second["dt"]),
            detrend="linear",
            nfft=int(second["nfft"]),
            scaling="spectrum",
            axis=-1,
        )
        for sim in range(draws):
            null_forward[sim] = _pearson(
                _align_power(second["freqs"], null_second[sim], freqs)[mask], first["target"][mask]
            )
            null_backward[sim] = _pearson(null_first[sim][mask], second_target[mask])
        null_consensus = np.minimum(null_forward, null_backward)
        p_values[index] = _p_value(consensus, null_consensus)
        fold_p_values[0, index] = _p_value(rho_forward, null_forward)
        fold_p_values[1, index] = _p_value(rho_backward, null_backward)

    return correlations, p_values, fold_correlations, fold_p_values


def _trained_frequencies(trained: dict[str, object], centres: npt.NDArray) -> npt.NDArray:
    """训练折在每条轨道周期带里给出的峰位。"""
    freqs = trained["freqs"]
    power = trained["power"]
    background = trained["background"]
    out = np.full(centres.size, np.nan)
    for index, band in enumerate(trained["bands"]):
        if band.size == 0:
            continue
        excess = power[band] - background[band]
        local = int(band[int(np.argmax(excess))])
        out[index] = float(freqs[local])
    return out


def _held_out_period_p_values(
    state: dict[str, object],
    centres: npt.NDArray,
    phi: float,
    draws: int,
    rng: np.random.Generator,
) -> npt.NDArray:
    """留出检验的逐轨道周期 p 值。

    训练折给出每条轨道周期带里的峰位（频率），检验折在**那个频率**上取功率；
    零假设里替代序列也在同一个频率上取功率。训练折与检验折的数据互不相干，
    所以这个 p 值是诚实的。
    """
    first, second = state["folds"]
    observed = np.zeros((2, centres.size), dtype=float)
    null = np.zeros((2, draws, centres.size), dtype=float)
    trained_first = _trained_frequencies(first, centres)
    trained_second = _trained_frequencies(second, centres)

    directions = ((second, first, trained_first), (first, second, trained_second))
    for direction, (tested, _trainer, trained_frequencies) in enumerate(directions):
        observed[direction] = _power_at(tested["freqs"], tested["power"], trained_frequencies)
        if draws <= 0:
            continue
        surrogates = _ar1_surrogates(tested["resampled"], phi, draws, rng)
        _null_freqs, null_power = periodogram(
            surrogates,
            fs=1.0 / float(tested["dt"]),
            detrend="linear",
            nfft=int(tested["nfft"]),
            scaling="spectrum",
            axis=-1,
        )
        for sim in range(draws):
            null[direction, sim] = _power_at(tested["freqs"], null_power[sim], trained_frequencies)

    p_values = np.full((2, centres.size), np.nan)
    for direction in range(2):
        for index in range(centres.size):
            p_values[direction, index] = _p_value(observed[direction, index], null[direction, :, index])
    return p_values


def _power_at(freqs: npt.NDArray, power: npt.NDArray, frequencies: npt.NDArray) -> npt.NDArray:
    """在给定的一组频率上插值功率。"""
    out = np.full(frequencies.size, np.nan)
    for index, frequency in enumerate(frequencies):
        if not np.isfinite(frequency):
            continue
        value = _interpolate_power(freqs, power, float(frequency))
        if np.isfinite(value):
            out[index] = float(value)
    return out


def _held_out_summary(
    correlations: npt.NDArray,
    period_p_values: npt.NDArray,
    periods: npt.NDArray,
    level: float,
) -> tuple[npt.NDArray, float, int]:
    """留出检验的汇总：显著周期、显著比例、检验次数。"""
    flat = np.asarray(period_p_values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    n_tests = int(flat.size)
    fraction = float(np.mean(flat <= level)) if n_tests else 0.0
    per_period = np.min(np.asarray(period_p_values, dtype=float), axis=0)
    significant = _significant_periods(periods, per_period, level)
    return significant, fraction, n_tests


def crossfit_coco(
    time: npt.NDArray,
    values: npt.NDArray,
    *,
    rate_range: tuple[float, float] | None = None,
    n_rates: int = DEFAULT_RATES,
    periods_kyr: npt.NDArray | None = None,
    n_target: int | None = None,
    n_period: int | None = None,
    target_period: float | None = None,
    sig_level: float = 0.05,
    red: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    max_iterations: int = 5,
    tolerance: float = 1e-3,
    seed: int | None = None,
) -> CrossfitCocoResult:
    """交叉拟合 COCO：一半估目标谱，另一半做检验。

    基础 COCO 的显著性问题在于：目标谱（或速率）是从这份数据估出来的，又拿
    同一份数据去检验它，"既当运动员又当裁判"。交叉拟合把记录切成**互补的两
    半**：前一半估目标谱，后一半只用来算相关系数与 p 值；再反过来做一遍。

    两个方向的相关系数取**较小者**作为共识（Acycle 交叉拟合模式的双向共识），
    因为只在一边成立的巧合不算证据。

    Parameters
    ----------
    time, values
        记录的地层坐标轴与测量序列（语义同 :func:`adaptive_coco`）。
    rate_range, n_rates
        候选沉积速率区间与个数。
    periods_kyr, n_target, n_period, target_period
        候选轨道周期与目标谱谐波个数（语义同 :func:`adaptive_coco`）。
    sig_level
        名义显著性水平。
    red
        零假设的 AR(1) 系数；``None`` 时由数据估计。
    n_monte_carlo
        蒙特卡洛替代序列条数。
    max_iterations, tolerance
        背景/目标谱不动点迭代的轮数上限与收敛判据。
    seed
        随机种子。

    Returns
    -------
    CrossfitCocoResult
        双向共识的速率扫描、留出检验的逐周期显著性与显著比例。

    Raises
    ------
    ValidationError
        记录不合法、参数越界、折的点数太少等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import crossfit_coco
    >>> axis = np.linspace(0.0, 3.0, 600)
    >>> record = np.sin(2.0 * np.pi * axis / 0.021)
    >>> result = crossfit_coco(axis, record, periods_kyr=np.array([21.0]), n_monte_carlo=20, seed=1)
    >>> result.n_tests
    2
    """
    t, v = _validate_record(time, values)
    level = _validate_level(sig_level)
    periods, harmonics = _resolve_periods(periods_kyr, n_period, n_target, target_period)
    rates = _rate_grid(rate_range, n_rates)
    draws = _validate_monte_carlo(n_monte_carlo)
    rounds, limit = _validate_iteration_controls(max_iterations, tolerance)

    phi = _validate_red(red, v)
    rng = make_rng(seed, context="crossfit COCO")
    nfft = _spectral_grid(t.size)
    centres = 1000.0 / periods
    target_centres = centres[:harmonics]

    folds = _split_halves(t, v)
    for fold_time, fold_values in folds:
        _validate_record(fold_time, fold_values)

    states = _fold_spectra(folds, rates, nfft, centres, harmonics, phi, rounds, limit)
    correlations, p_values, fold_correlations, fold_p_values = _crossfit_rate_scan(
        states, target_centres, phi, draws, rng
    )

    finite = np.isfinite(correlations)
    if not np.any(finite):
        raise ValidationError(
            _("no candidate rate is admissible: the record is too short or too coarse to resolve the orbital bands"),
            details={"n_rates": int(rates.size)},
        )
    best_index = int(np.nanargmax(correlations))
    best_rate = float(rates[best_index])
    best_correlation = float(correlations[best_index])
    best_p = float(p_values[best_index])

    period_p_values = _held_out_period_p_values(states[best_index], centres, phi, draws, rng)
    significant_periods, fraction, n_tests = _held_out_summary(correlations, period_p_values, periods, level)

    logger.info(
        "crossfit COCO: best rate %.6g (consensus correlation %.4f, p %.4g), significant fraction %.3f",
        best_rate,
        best_correlation,
        best_p,
        fraction,
    )

    return CrossfitCocoResult(
        rates=rates,
        correlations=correlations,
        p_values=p_values,
        best_rate=best_rate,
        best_correlation=best_correlation,
        best_p_value=best_p,
        fold_correlations=fold_correlations,
        fold_p_values=fold_p_values,
        periods_kyr=periods,
        period_p_values=period_p_values,
        significant_periods_kyr=significant_periods,
        significant_fraction=fraction,
        n_tests=n_tests,
        sig_level=level,
        critical_value=float("nan"),
        significant=bool(np.isfinite(best_p) and best_p <= level),
    )


def interleaved_coco(
    time: npt.NDArray,
    values: npt.NDArray,
    *,
    rate_range: tuple[float, float] | None = None,
    n_rates: int = DEFAULT_RATES,
    periods_kyr: npt.NDArray | None = None,
    n_target: int | None = None,
    n_period: int | None = None,
    target_period: float | None = None,
    sig_level: float = 0.05,
    red: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    max_iterations: int = 5,
    tolerance: float = 1e-3,
    seed: int | None = None,
) -> InterleavedCocoResult:
    """交错 COCO：奇数观测训练、偶数观测检验，再反过来。

    交叉拟合按时间把记录切成两半，**两半的时间覆盖不重叠**；交错模式按观测
    序号取奇偶，奇折与偶折在整个时间轴上**互相穿插**。对有间断、不均匀的记
    录这一点很有用：奇折仍然覆盖整条时间轴。

    折的划分是**全局**的（先对整条记录排序，再按序号取奇偶），所以窗口起点
    落在奇数行还是偶数行不会改变折的归属。

    Parameters
    ----------
    time, values
        记录的地层坐标轴与测量序列（语义同 :func:`adaptive_coco`）。
    rate_range, n_rates
        候选沉积速率区间与个数。
    periods_kyr, n_target, n_period, target_period
        候选轨道周期与目标谱谐波个数（语义同 :func:`adaptive_coco`）。
    sig_level
        名义显著性水平。
    red
        零假设的 AR(1) 系数；``None`` 时由数据估计。
    n_monte_carlo
        蒙特卡洛替代序列条数。
    max_iterations, tolerance
        背景/目标谱不动点迭代的轮数上限与收敛判据。
    seed
        随机种子。

    Returns
    -------
    InterleavedCocoResult
        双向共识的速率扫描、留出检验的逐周期显著性与显著比例。

    Raises
    ------
    ValidationError
        记录不合法、参数越界、折的点数太少等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import interleaved_coco
    >>> axis = np.linspace(0.0, 3.0, 600)
    >>> record = np.sin(2.0 * np.pi * axis / 0.021)
    >>> result = interleaved_coco(axis, record, periods_kyr=np.array([21.0]), n_monte_carlo=20, seed=1)
    >>> result.n_tests
    2
    """
    t, v = _validate_record(time, values)
    level = _validate_level(sig_level)
    periods, harmonics = _resolve_periods(periods_kyr, n_period, n_target, target_period)
    rates = _rate_grid(rate_range, n_rates)
    draws = _validate_monte_carlo(n_monte_carlo)
    rounds, limit = _validate_iteration_controls(max_iterations, tolerance)

    phi = _validate_red(red, v)
    rng = make_rng(seed, context="interleaved COCO")
    nfft = _spectral_grid(t.size)
    centres = 1000.0 / periods
    target_centres = centres[:harmonics]

    folds = _interleaved_folds(t, v)
    for fold_time, fold_values in folds:
        _validate_record(fold_time, fold_values)

    states = _fold_spectra(folds, rates, nfft, centres, harmonics, phi, rounds, limit)
    correlations, p_values, fold_correlations, fold_p_values = _crossfit_rate_scan(
        states, target_centres, phi, draws, rng
    )

    finite = np.isfinite(correlations)
    if not np.any(finite):
        raise ValidationError(
            _("no candidate rate is admissible: the record is too short or too coarse to resolve the orbital bands"),
            details={"n_rates": int(rates.size)},
        )
    best_index = int(np.nanargmax(correlations))
    best_rate = float(rates[best_index])
    best_correlation = float(correlations[best_index])
    best_p = float(p_values[best_index])

    period_p_values = _held_out_period_p_values(states[best_index], centres, phi, draws, rng)
    significant_periods, fraction, n_tests = _held_out_summary(correlations, period_p_values, periods, level)

    logger.info(
        "interleaved COCO: best rate %.6g (consensus correlation %.4f, p %.4g), significant fraction %.3f",
        best_rate,
        best_correlation,
        best_p,
        fraction,
    )

    return InterleavedCocoResult(
        rates=rates,
        correlations=correlations,
        p_values=p_values,
        best_rate=best_rate,
        best_correlation=best_correlation,
        best_p_value=best_p,
        fold_correlations=fold_correlations,
        fold_p_values=fold_p_values,
        periods_kyr=periods,
        period_p_values=period_p_values,
        significant_periods_kyr=significant_periods,
        significant_fraction=fraction,
        n_tests=n_tests,
        sig_level=level,
        critical_value=float("nan"),
        significant=bool(np.isfinite(best_p) and best_p <= level),
    )


# =============================================================================
# 轨道追踪
# =============================================================================


def coco_orbit_tracking(
    time: npt.NDArray,
    values: npt.NDArray,
    *,
    rate_range: tuple[float, float] | None = None,
    n_rates: int = DEFAULT_RATES,
    periods_kyr: npt.NDArray | None = None,
    n_windows: int = 5,
    sig_level: float = 0.05,
    red: float | None = None,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO,
    max_shift: float = 0.05,
    min_track_length: int = 2,
    seed: int | None = None,
) -> OrbitTrackingResult:
    """轨道追踪：把逐窗谱上显著的周期峰串成轨道链。

    Acycle 的 ``ecocotrack`` / ``ecocotrackn`` 在深度方向上追踪"每段的最可能
    沉积速率"，把相邻段里足够接近的解连成一条链；断掉的地方就是"这一段的
    速率跳了"。本模块做的是同一件事在**频率方向**上的版本：追踪显著的周期
    峰——一条链就是一条被连续追到的轨道。

    连接规则沿用 ``ecocotrackn`` 的语义：参考点是这条链最近若干个（至多 5 个）
    成员的**中位数**，下一个窗口里离它最近的显著峰若相对偏差不超过
    ``max_shift`` 就接上，否则这条链断掉。

    Parameters
    ----------
    time, values
        记录的地层坐标轴与测量序列（语义同 :func:`adaptive_coco`）。
    rate_range, n_rates
        候选沉积速率区间与个数；给了区间时先用基础 COCO（``stratigraphy.coco``）
        找出最可能速率，再在那个速率上追踪。
    periods_kyr
        轨道搜索的周期范围（kyr）；缺省用 Berger (1989) 的九条周期。
    n_windows
        追踪窗口数。
    sig_level
        名义显著性水平（红噪声零假设下的假警概率）。
    red
        零假设的 AR(1) 系数；``None`` 时由每个追踪窗口自己估计。
    n_monte_carlo
        每个窗口的蒙特卡洛替代序列条数。
    max_shift
        相邻窗口之间允许的频率相对漂移。
    min_track_length
        一条轨道链至少要连上多少个窗口才算一条轨道。
    seed
        随机种子。

    Returns
    -------
    OrbitTrackingResult
        轨道链、各链的中心周期、各窗口的显著性阈值与假警概率。

    Raises
    ------
    ValidationError
        记录不合法、参数越界、窗口数不足等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import coco_orbit_tracking
    >>> axis = np.linspace(0.0, 3.0, 600)
    >>> record = np.sin(2.0 * np.pi * axis / 0.021) + np.sin(2.0 * np.pi * axis / 0.019)
    >>> result = coco_orbit_tracking(axis, record, n_windows=3, n_monte_carlo=10, seed=1)
    >>> result.n_orbits
    2
    """
    t, v = _validate_record(time, values)
    level = _validate_level(sig_level)
    periods, _harmonics = _resolve_periods(periods_kyr, None, None, None)
    windows = int(n_windows)
    if windows < 2:
        raise ValidationError(
            _("n_windows must be at least 2; got {0}").format(windows),
            details={"n_windows": windows},
        )
    draws = _validate_monte_carlo(n_monte_carlo)
    try:
        shift = float(max_shift)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("max_shift must be a positive number"),
            details={"max_shift": repr(max_shift)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(shift) or shift <= 0.0:
        raise ValidationError(
            _("max_shift must be finite and positive; got {0}").format(shift),
            details={"max_shift": shift},
        )
    length = int(min_track_length)
    if length < 1:
        raise ValidationError(
            _("min_track_length must be at least 1; got {0}").format(length),
            details={"min_track_length": length},
        )

    phi = _validate_red(red, v)
    fixed_red = red is not None
    rng = make_rng(seed, context="orbit tracking")

    if rate_range is None:
        rate = 1.0
    else:
        base = coco(
            t,
            v,
            rate_range=rate_range,
            n_rates=int(n_rates),
            n_monte_carlo=0,
            random_seed=None if seed is None else int(seed),
        )
        rate = float(base.best_rate)

    grid, resampled, dt = _resample_to_uniform_time(t, v, rate)
    span = float(grid[-1] - grid[0]) + dt
    rayleigh = 1.0 / span
    nyquist = 0.5 / dt
    centre_frequencies = 1000.0 / periods
    low = max(rayleigh, float(np.min(centre_frequencies)))
    high = min(nyquist, float(np.max(centre_frequencies)))
    if not high > low:
        raise ValidationError(
            _("the record is too short or too coarse to resolve any orbital period"),
            details={"nyquist": float(nyquist), "rayleigh": float(rayleigh)},
        )

    edges = np.linspace(float(grid[0]), float(grid[-1]), windows + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    thresholds = np.full(windows, np.nan)
    p_values = np.full(windows, np.nan)
    window_peaks: list[npt.NDArray] = []
    all_frequencies = np.empty(0)

    for index in range(windows):
        mask = (grid >= edges[index]) & (grid <= edges[index + 1])
        window_values = resampled[mask]
        if window_values.size < MIN_WINDOW_SAMPLES:
            raise ValidationError(
                _("a tracking window must hold at least {0} samples; got {1}").format(
                    MIN_WINDOW_SAMPLES, int(window_values.size)
                ),
                details={"window": index, "n_samples": int(window_values.size)},
            )
        nfft = _spectral_grid(window_values.size)
        freqs, power = periodogram(
            window_values,
            fs=1.0 / dt,
            detrend="linear",
            nfft=int(nfft),
            scaling="spectrum",
        )
        all_frequencies = freqs
        band = (freqs >= low) & (freqs <= high)
        band_freqs = freqs[band]
        if band_freqs.size < 4:
            raise ValidationError(
                _("a tracking window has too few frequencies in the orbital band"),
                details={"window": index, "n_frequencies": int(band_freqs.size)},
            )
        observed = power[band]
        window_phi = phi if fixed_red else _ar1_phi(window_values)
        spread = float(np.std(window_values))
        if not np.isfinite(spread) or spread <= 0.0:
            spread = 1.0
        null_max = np.empty(draws, dtype=float)
        for sim in range(draws):
            surrogate = generate_red_noise(
                window_values.size,
                rho=window_phi,
                sigma=spread,
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )
            _surrogate_freqs, surrogate_power = periodogram(
                surrogate,
                fs=1.0 / dt,
                detrend="linear",
                nfft=int(nfft),
                scaling="spectrum",
            )
            null_max[sim] = float(np.max(surrogate_power[band]))
        threshold = _quantile(null_max, 1.0 - level) if null_max.size else float("nan")
        thresholds[index] = threshold
        p_values[index] = _p_value(float(np.max(observed)), null_max) if null_max.size else float("nan")
        if not np.isfinite(threshold):
            window_peaks.append(np.empty(0))
            continue
        selected = [position for position in _local_peaks(observed) if observed[position] > threshold]
        window_peaks.append(
            np.array(
                [_parabolic_peak(band_freqs, observed, position) for position in selected],
                dtype=float,
            )
        )

    chains = _chain_peaks(window_peaks, shift)
    kept = [chain for chain in chains if chain.size >= length]
    kept.sort(key=lambda chain: float(np.median(chain)))
    track_periods = np.array([1000.0 / float(np.median(chain)) for chain in kept], dtype=float)
    track_frequencies = tuple(np.asarray(chain, dtype=float) for chain in kept)

    critical = float(np.nanmax(thresholds)) if thresholds.size else float("nan")
    logger.info("orbit tracking: %d orbit chain(s) over %d window(s)", len(kept), windows)

    return OrbitTrackingResult(
        frequencies=all_frequencies,
        track_frequencies=track_frequencies,
        track_periods_kyr=track_periods,
        n_orbits=len(kept),
        window_centres=centres,
        window_thresholds=thresholds,
        window_p_values=p_values,
        sig_level=level,
        critical_value=critical,
        significant=bool(len(kept) > 0),
    )


def _chain_peaks(window_peaks: list[npt.NDArray], max_shift: float) -> list[npt.NDArray]:
    """把逐窗的显著峰串成轨道链。

    连接规则沿用 Acycle ``ecocotrackn`` 的语义：参考点是这条链最近至多 5 个
    成员的中位数；下一个窗口里离它最近的峰若相对偏差不超过 ``max_shift`` 就
    接上，否则断链。没被接走的峰另起一条新链。
    """
    tracks: list[list[float]] = []
    alive: list[bool] = []
    for peaks in window_peaks:
        used = np.zeros(peaks.size, dtype=bool)
        for track_index, track in enumerate(tracks):
            if not alive[track_index]:
                continue
            reference = float(np.median(track[-5:]))
            if peaks.size == 0:
                alive[track_index] = False
                continue
            distances = np.abs(peaks - reference) / max(abs(reference), 1e-12)
            candidate = int(np.argmin(distances))
            if distances[candidate] <= max_shift:
                track.append(float(peaks[candidate]))
                used[candidate] = True
            else:
                alive[track_index] = False
        for peak_index in range(peaks.size):
            if used[peak_index]:
                continue
            tracks.append([float(peaks[peak_index])])
            alive.append(True)
    return [np.asarray(track, dtype=float) for track in tracks]


# =============================================================================
# 轨道几何
# =============================================================================


def num_orbits(
    periods_kyr: npt.NDArray,
    time_span: float,
    *,
    rate: float = 1.0,
    rate_range: tuple[float, float] | None = None,
    n_rates: int = DEFAULT_RATES,
    sample_spacing: float | None = None,
    sig_level: float = 0.05,
) -> NumOrbitsResult:
    """轨道计数：每个候选速率下有几条轨道周期真的能被分辨。

    两条判据（Acycle ``numorbit`` / ``norbits`` 的语义）：

    1. **记录够长**：速率 ``r`` 下记录的时间跨度是 ``time_span / r``（Ma），
       轨道周期 ``p``（kyr）必须不超过这个跨度，否则连一个完整周期都没有。
    2. **采样够密**：轨道频率 ``1000 / p``（cycles/Ma）必须低于 Nyquist
       频率 ``r / (2 * dx)``；给了 ``sample_spacing`` 才会检查这一条。

    Parameters
    ----------
    periods_kyr
        候选轨道周期（kyr）。
    time_span
        记录在坐标轴上的跨度（与 ``time`` 同单位）。
    rate
        沉积速率（坐标单位/Ma）。
    rate_range
        给了就用这一段速率区间代替 ``rate``（扫描若干个速率）。
    n_rates
        速率区间的分割点数。
    sample_spacing
        坐标轴上采样间距的中位数；给了才检查 Nyquist 判据。
    sig_level
        名义显著性水平（保留字段：本函数只做几何判定）。

    Returns
    -------
    NumOrbitsResult
        每个速率下的轨道计数与逐周期的可用性矩阵。

    Raises
    ------
    ValidationError
        周期非法、跨度非法、速率非法等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import num_orbits
    >>> periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
    >>> result = num_orbits(periods, 0.1, rate=1.0)
    >>> int(result.n_orbits[0])
    7
    """
    level = _validate_level(sig_level)
    periods = _validate_periods(periods_kyr)
    try:
        span = float(time_span)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("time_span must be a positive number"),
            details={"time_span": repr(time_span)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(span) or span <= 0.0:
        raise ValidationError(
            _("time_span must be finite and positive; got {0}").format(span),
            details={"time_span": span},
        )

    if rate_range is None:
        try:
            single = float(rate)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("rate must be a positive number"),
                details={"rate": repr(rate)},
                original_exception=exc,
            ) from exc
        if not np.isfinite(single) or single <= 0.0:
            raise ValidationError(
                _("rate must be finite and positive; got {0}").format(single),
                details={"rate": single},
            )
        rates = np.array([single], dtype=float)
    else:
        low, high = _validate_rate_range(rate_range, n_rates)
        rates = np.linspace(low, high, int(n_rates))

    spacing = None
    if sample_spacing is not None:
        spacing = float(sample_spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValidationError(
                _("sample_spacing must be finite and positive; got {0}").format(spacing),
                details={"sample_spacing": spacing},
            )

    period_ma = periods / 1000.0
    counts = np.zeros(rates.size, dtype=int)
    resolved = np.zeros((rates.size, periods.size), dtype=bool)
    spans = np.zeros(rates.size, dtype=float)
    for index, value in enumerate(rates):
        span_ma = span / float(value)
        spans[index] = span_ma
        usable = period_ma <= span_ma
        if spacing is not None:
            nyquist = float(value) / (2.0 * spacing)
            usable = usable & ((1000.0 / periods) < nyquist)
        resolved[index] = usable
        counts[index] = int(np.sum(usable))

    return NumOrbitsResult(
        rates=rates,
        n_orbits=counts,
        resolved=resolved,
        periods_kyr=periods,
        time_span_ma=spans,
        sig_level=level,
    )


def orbit_groups(periods_kyr: npt.NDArray, *, sig_level: float = 0.05) -> OrbitGroupsResult:
    """轨道分组：把轨道周期分进四个物理族。

    九条轨道周期时按**周期从长到短的排名**分组（1 条长偏心率、4 条短偏心率、
    1 条倾角、3 条岁差），这样深时轨道周期整体缩短时某一族的身份不会变——这与
    Acycle ``cocoOrbitGroups`` 的做法一致。周期数不是九条时退回按周期阈值
    分组（≥80 kyr 偏心率、30--80 kyr 倾角、其余岁差），与 ``cocoFixedTargetWeights``
    的历史口径保持一致。

    Parameters
    ----------
    periods_kyr
        轨道周期（kyr）。
    sig_level
        名义显著性水平（保留字段：本函数只做分组）。

    Returns
    -------
    OrbitGroupsResult
        每条周期的族名与族号、各族成员数、分组规则。

    Raises
    ------
    ValidationError
        周期非法等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import orbit_groups
    >>> periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
    >>> result = orbit_groups(periods)
    >>> result.groups[0]
    'long eccentricity'
    """
    level = _validate_level(sig_level)
    periods = _validate_periods(periods_kyr)

    if periods.size == 9 and np.unique(periods).size == 9:
        order = np.argsort(periods)[::-1]
        index = np.zeros(9, dtype=int)
        index[order[0]] = 1
        index[order[1:5]] = 2
        index[order[5]] = 3
        index[order[6:9]] = 4
        counts = np.array([1, 4, 1, 3], dtype=int)
        rule = _("descending period rank: 1 / 2-5 / 6 / 7-9; no fixed period thresholds")
    else:
        index = np.ones(periods.size, dtype=int)
        index[(periods >= 30.0) & (periods < 80.0)] = 3
        index[periods < 30.0] = 4
        counts = np.array([int(np.sum(index == number)) for number in (1, 2, 3, 4)], dtype=int)
        rule = _("period thresholds: >=80 kyr eccentricity, 30-80 kyr obliquity, <30 kyr precession")

    names = ORBIT_GROUP_NAMES
    groups = tuple(names[number - 1] for number in index)

    return OrbitGroupsResult(
        periods_kyr=periods,
        groups=groups,
        index=index,
        counts=counts,
        names=names,
        rule=rule,
        sig_level=level,
    )


def fixed_target_weights(periods_kyr: npt.NDArray, *, sig_level: float = 0.05) -> FixedTargetWeightsResult:
    """固定目标权重：给每条轨道周期一个固定的目标谱权重。

    九条轨道周期时权重按物理族给：偏心率（长 + 短）1.0、倾角 0.8、岁差 0.6。
    其它周期数退回按阈值给：≥80 kyr 为 1.0、30--80 kyr 为 0.8、其余 0.6。

    Parameters
    ----------
    periods_kyr
        轨道周期（kyr）。
    sig_level
        名义显著性水平（保留字段：本函数只给权重）。

    Returns
    -------
    FixedTargetWeightsResult
        每条周期的权重与权重规则。

    Raises
    ------
    ValidationError
        周期非法等。

    Examples
    --------
    >>> import numpy as np
    >>> from stratigraphy.coco_advanced import fixed_target_weights
    >>> result = fixed_target_weights(np.array([405.0, 41.0, 19.0]))
    >>> result.weights.tolist()
    [1.0, 0.8, 0.6]
    """
    level = _validate_level(sig_level)
    periods = _validate_periods(periods_kyr)

    weights = 0.6 * np.ones(periods.size, dtype=float)
    if periods.size == 9 and np.unique(periods).size == 9:
        groups = orbit_groups(periods)
        weights[groups.index == 3] = 0.8
        weights[groups.index <= 2] = 1.0
        rule = _("nine-period ORBIT9: eccentricity 1.0, obliquity 0.8, precession 0.6")
    else:
        weights[periods >= 80.0] = 1.0
        weights[(periods >= 30.0) & (periods < 80.0)] = 0.8
        rule = _("period thresholds: >=80 kyr 1.0, 30-80 kyr 0.8, <30 kyr 0.6")

    return FixedTargetWeightsResult(
        periods_kyr=periods,
        weights=weights,
        rule=rule,
        sig_level=level,
    )
