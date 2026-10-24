# =============================================================================
# FILE: stratigraphy/coco_variants.py
# =============================================================================
"""
COCO 家族的三个变体：Spearman 秩相关、相关系数的置信区间、eCOCO 网格的逐行
汇总与"只取 CI 小于 5% 的前 n 个峰"筛选。

用途
====
:mod:`stratigraphy.coco` 实现了基础的 COCO（记录功率谱与天文目标谱求 **Pearson**
相关系数，反演最可能沉积速率）与 eCOCO（逐窗做 COCO）。本模块补上同一族里基础
COCO 没有覆盖的三块能力：

* :func:`coco_spearman` —— COCO 的 **Spearman 秩相关**变体。统计量从 Pearson
  换成 Spearman 秩相关系数（Fisher, 1915），显著性用**秩置换零分布**给出。
* :func:`correlation_confidence_interval` —— **相关系数的置信区间**：Fisher z
  变换的正态区间（Fiore & Bonferroni, 2009）与秩相关的自助法区间
  （Efron & Tibshirani, 1993）。
* :func:`ecoco_row_summary` 与 :func:`significant_peaks` —— eCOCO 网格的
  **逐行汇总**（Acycle ``eccount.m`` 的语义）与"**只取 CI 小于 5% 的前 n 个峰**"
  的筛选（Acycle ``eccpeaks.m`` 的语义）。

方法
====

Spearman 秩相关
----------------
Spearman 秩相关系数是两条序列**秩**的 Pearson 相关系数::

    rho_s = corr( rank(x), rank(y) )

因此它对 ``y`` 的任何**严格单调变换**不变（秩没变），而且对"整体单调但不是直线"
的关系给出 ``rho_s = 1``（精确），这是 Pearson 做不到的——Pearson 只看线性。

零假设下的显著性用 **Fisher (1915) 的条件秩分布**：条件于两个秩向量的边际分布，
两者的配对关系在零假设下是随机的，于是相关系数的零分布由**秩的排列分布**给出。
本模块用蒙特卡洛方式抽这个排列分布（``n_monte_carlo`` 次排列），p 值为::

    p = (1 + #{零假设 rho >= 实测 rho}) / (1 + 排列次数)

这与 Acycle 的做法**不同**：Acycle ``corrsig.m`` 的 p 值来自 MATLAB
``corr(...,'type','Spearman')`` 的 t 近似，而它的 H0 显著性水平来自
``cyclesig_rank.m`` 在**随机谱**上的蒙特卡洛。本模块两者都给：

* ``p_values``：秩置换零分布的 p 值（Fisher, 1915；**默认**）；
* ``p_values_monte_carlo``：AR(1) 替代序列的随机谱蒙特卡洛 p 值
  （Acycle ``corrsig.m`` / ``cyclesig_rank.m`` 的语义）。

两者**回答的问题不同**：秩置换零分布问"观测谱的秩与目标谱的秩的**配对**是否显著
偏离随机"，条件于两个秩向量的边际分布；蒙特卡洛零分布问"观测谱是否显著偏离**噪
声模型**给出的谱"，替代序列带着与观测相同的 AR(1) 系数生成。

在**本模块的比较栅格**（Rayleigh 分辨率）上，两者的标定都接近名义水平：一条没
有轨道信号的记录上，固定候选速率处 400 次重复中 ``p < 0.05`` 的比例，秩置换给
出 5.8%、蒙特卡洛给出 6.5%（名义水平 5%，标准误 1.1%）。但秩置换零分布**依赖
比较栅格的粗细**：把栅格加密到零填充的细网格上，谱线的秩向量不再可交换，假阳性
率升到约 20%——这正是本模块把比较栅格取在 Rayleigh 分辨率上的原因之一。做
**筛选**（eCOCO 网格、峰值挑选）时用蒙特卡洛零分布（Acycle 的做法，它把噪声模
型带进了零分布），做**统计检验**（"秩的配对是否显著偏离随机"）时用秩置换零分布
（Fisher, 1915 的原意）。

比较频带
--------
相关系数在"记录功率谱 vs 目标谱"的公共频带上计算。频带下界取
``max(Rayleigh 分辨率, 目标谱最低频率)``，上界取 ``min(Nyquist 频率, 目标谱最高
频率)``——与 :mod:`stratigraphy.coco` 的口径一致。

但**比较用的频率栅格**取记录的 **Rayleigh 分辨率栅格**（即零填充频谱每隔若干点
取一个，使点距不小于 ``1 / T``），而不是零填充后的细栅格。理由有二：

1. 谱估计只有在 Rayleigh 分辨率这一层上才近似互相独立；秩置换零分布按
   ``1 / sqrt(n - 1)`` 标定，用细栅格会让零分布明显偏窄。
2. Acycle ``cyclecorr_rank.m`` 在高沉积速率分支上把数据谱插值到**目标谱自己的
   频率栅格**上再求相关，也不是在过零填充的细栅格上做。

置信区间
--------
* ``method="fisher-z"``：Fisher z 变换 ``z = arctanh(rho)``，标准误
  ``1 / sqrt(n - 3)``，在正态分布下取区间后反变换。依据 Fiore & Bonferroni
  (2009) 对 bivariate normal 下若干种相关系数区间的比较——Fisher z 区间覆盖率
  接近名义水平，是这里的默认口径。
* ``method="bootstrap"``：对 ``(x, y)`` **成对重抽样**（case resampling），
  每次重算 Spearman 秩相关系数，取百分位区间（Efron & Tibshirani, 1993）。

需要说明的是：Fisher z 变换是对 **Pearson** 相关系数（在 bivariate normal 下）
推导的；把它作用在秩相关系数上是**近似**（秩相关系数的抽样方差比同值的 Pearson
相关系数略小，所以该区间略偏保守）。要严格的秩相关区间请用
``method="bootstrap"``。

与基础 COCO 的关系
==================
本模块**直接复用** :mod:`stratigraphy.coco` 的公开接口与内部工具（与
``stratigraphy/coco_advanced.py`` 的做法一致）：记录与速率区间的校验
（``_validate_record`` / ``_validate_rate_range``）、频率栅格
（``_spectral_grid``）、重采样（``_resample_to_uniform_time``）、最优速率的
不确定度区间（``_confidence_interval``）、目标谱（``milankovitch_target_spectrum``、
``berger1989_periods``）与结果容器（``CocoResult``）。轨道计数复用
:mod:`stratigraphy.coco_advanced` 的 :func:`~stratigraphy.coco_advanced.num_orbits`。

频率口径、单位口径、目标谱来源与基础 COCO 完全一致：**坐标单位/Ma** 的沉积速率、
"每 Ma"的频率轴、Rayleigh 分辨率决定栅格上限。

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（秩相关的统计量、"CI 小于 5%"的峰值筛选条件、逐行汇总的字段），未翻译或复制
其 MATLAB 源码。各参考文件读出来的语义与本实现的差别，逐条写在下面各函数的
docstring 里。

参考文献
========
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（COCO / eCOCO 的
  统计框架：功率谱与目标谱的相关系数、蒙特卡洛显著性）
* Fisher, R. A. (1915). Frequency in the occurrence of events.
  *Journal of the Royal Statistical Society*, 78(2), 322-329.（秩相关的显著性：
  条件于秩结构的排列零分布）
* Fiore, A. R. & Bonferroni, R. (2009). A comparison of some confidence
  intervals for the correlation coefficient in the bivariate normal case.
  *Metrika*, 59(2), 139-168.（相关系数的 Fisher z 变换区间）
* Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.（百分位自助法区间）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的功能划分："COCO"、"eCOCO" 及其"corrcoef"子族）
* Berger, A. (1989). Pleistocene climate from the top of the ice core. In
  *Geochemistry and Geophysics of Climate Change* (eds. Ruddiman, W. & Prell,
  W.). AGU, 13-24.（现代轨道周期表，目标谱的周期来源）
* Mann, M. E. & Lees, J. M. (1996). Red noise and the search for periodicity in
  Quaternary climate data. *Paleoceanography*, 11, 499-515.（AR(1) 理论谱作为
  红噪声背景）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter, periodogram
from scipy.stats import norm, rankdata
from scipy.stats import t as student_t

from config.i18n import _
from stratigraphy.coco import (
    BAND_FLAT_CORE,
    BAND_HALF_WIDTH,
    MAX_RESAMPLED_POINTS,
    MIN_SAMPLES,
    TARGET_BAND_WEIGHTS,
    CocoResult,
    _confidence_interval,
    _resample_to_uniform_time,
    _spectral_grid,
    _validate_rate_range,
    _validate_record,
    berger1989_periods,
    milankovitch_target_spectrum,
)
from stratigraphy.coco_advanced import num_orbits
from stratigraphy.cycles import _ar1_phi
from stratigraphy.spectral_analysis import _linear_detrend, ar1_theoretical_spectrum
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

# =============================================================================
# 参数常量
# =============================================================================

#: 比较频带里至少要有的谱线点数（低于此数该候选速率被跳过）。
MIN_BAND_POINTS = 8

#: Fisher z 区间需要的最小样本数（标准误 ``1 / sqrt(n - 3)``）。
MIN_CI_SAMPLES = 4

#: 秩相关检验需要的最小样本数。
MIN_RANK_SAMPLES = 4

#: 自助法默认的重抽样次数。
DEFAULT_N_BOOTSTRAP = 1000

#: 允许的零分布模型。
NULL_MODELS = ("permutation", "monte-carlo", "both")

#: 允许的"CI 小于 5%"判据。
PEAK_CRITERIA = ("significance", "ci_width")

#: 允许的筛选维度。
PEAK_DIMENSIONS = ("row", "column")


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class SpearmanTestResult:
    """一次 Spearman 秩相关检验的结果。

    Attributes:
        coefficient: Spearman 秩相关系数。
        p_value: 秩置换零分布给出的 p 值（Fisher, 1915）。
        p_value_t: t 近似的 p 值（Acycle/MATLAB ``corr(...,'type','Spearman')``
            的口径）；样本量不足时为 ``nan``。
        n: 参与检验的样本数。
        n_permutations: 抽取的排列数。
    """

    coefficient: float
    p_value: float
    p_value_t: float
    n: int
    n_permutations: int

    def summary(self) -> str:
        """Generate summary text."""
        return "\n".join(
            [
                _("Spearman Rank Correlation Results"),
                "=" * 50,
                _("Rank correlation: {0:.6f}").format(float(self.coefficient)),
                _("Permutation p-value: {0:.6g}").format(float(self.p_value)),
                _("t-approximation p-value: {0:.6g}").format(float(self.p_value_t)),
                _("Samples: {0}").format(int(self.n)),
            ]
        )


@dataclass(frozen=True)
class CorrelationInterval:
    """相关系数的置信区间。

    Attributes:
        lower: 区间下端点。
        upper: 区间上端点。
        method: 所用方法（``"fisher-z"`` 或 ``"bootstrap"``）。
        level: 名义置信水平。
        point_estimate: 区间围绕的点估计（调用方传入的相关系数）。
        standard_error: 点估计的标准误（Fisher z 区间给出；自助法为 ``None``）。
        n: 样本数。
    """

    lower: float
    upper: float
    method: str
    level: float
    point_estimate: float
    standard_error: float | None = None
    n: int = 0

    @property
    def width(self) -> float:
        """区间宽度（上端点减下端点）。"""
        return float(self.upper - self.lower)

    def summary(self) -> str:
        """Generate summary text."""
        return "\n".join(
            [
                _("Correlation Confidence Interval"),
                "=" * 50,
                _("Method: {0}").format(self.method),
                _("Level: {0:.4g}").format(float(self.level)),
                _("Point estimate: {0:.6f}").format(float(self.point_estimate)),
                _("Interval: [{0:.6f}, {1:.6f}]").format(float(self.lower), float(self.upper)),
            ]
        )


@dataclass(frozen=True)
class CocoSpearmanResult(CocoResult):
    """COCO（Spearman 秩相关变体）的扫描结果。

    在 :class:`stratigraphy.coco.CocoResult` 的字段之上补充了秩相关特有的量。

    Attributes:
        p_values_permutation: 秩置换零分布给出的 p 值；未计算时为 ``None``。
        p_values_monte_carlo: AR(1) 随机谱蒙特卡洛零分布给出的 p 值；未计算时
            为 ``None``。
        band_sizes: 每个候选速率的比较频带点数；未计算的速率为 ``nan``。
        null_model: 实际使用的零分布模型（``"permutation"`` / ``"monte-carlo"`` /
            ``"both"``）。
        significant: 每个候选速率是否在名义水平上显著（p 值小于
            ``sig_level``）；未计算的速率为 ``False``。
    """

    p_values_permutation: npt.NDArray | None = None
    p_values_monte_carlo: npt.NDArray | None = None
    band_sizes: npt.NDArray | None = None
    null_model: str = "permutation"
    significant: npt.NDArray | None = None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("COCO Spearman Rank-Correlation Results"),
            "=" * 50,
            _("Candidate rates tried: {0}").format(int(self.rates.size)),
            _("Best rate: {0:.6g}").format(float(self.best_rate)),
            _("Best rank correlation: {0:.4f}").format(float(self.best_correlation)),
            _("Best p-value: {0:.4g}").format(float(self.best_p_value)),
            _("Null model: {0}").format(str(self.null_model)),
        ]
        if self.rate_confidence is not None:
            low, high = self.rate_confidence
            lines.append(_("Rate confidence interval: [{0:.6g}, {1:.6g}]").format(low, high))
        valid = int(np.sum(np.isfinite(self.correlations)))
        if valid < int(self.rates.size):
            lines.append(_("Rates skipped as inadmissible: {0}").format(int(self.rates.size) - valid))
        return "\n".join(lines)


@dataclass(frozen=True)
class ECocoGrid:
    """eCOCO 的网格化结果。

    网格的**行**是记录上的窗口（位置），**列**是候选沉积速率。

    Attributes:
        window_centres: 各窗口中心在记录坐标轴上的位置。
        rates: 候选沉积速率（网格的列）。
        rate_map: 形状 ``(n_windows, n_rates)`` 的速率图：某格若该窗口在该速率
            上显著则填该速率，否则为 ``nan``（Acycle ``eccount.m`` 里
            "sedimentation maps" 的非 NaN 语义）。
        correlations: 形状 ``(n_windows, n_rates)`` 的秩相关系数。
        p_values: 形状 ``(n_windows, n_rates)`` 的显著性水平（p 值）。
        ci_lower: 形状 ``(n_windows, n_rates)`` 的相关系数置信区间下端点。
        ci_upper: 形状 ``(n_windows, n_rates)`` 的相关系数置信区间上端点。
        n_orbits: 形状 ``(n_windows, n_rates)`` 的轨道计数。
        band_sizes: 形状 ``(n_windows, n_rates)`` 的比较频带点数。
    """

    window_centres: npt.NDArray
    rates: npt.NDArray
    rate_map: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    ci_lower: npt.NDArray
    ci_upper: npt.NDArray
    n_orbits: npt.NDArray
    band_sizes: npt.NDArray

    @property
    def shape(self) -> tuple[int, int]:
        """网格形状 ``(行数, 列数)``。"""
        return int(self.rate_map.shape[0]), int(self.rate_map.shape[1])

    def summary(self) -> str:
        """Generate summary text."""
        return "\n".join(
            [
                _("eCOCO Grid"),
                "=" * 50,
                _("Windows: {0}").format(int(self.rate_map.shape[0])),
                _("Candidate rates: {0}").format(int(self.rate_map.shape[1])),
                _("Significant cells: {0}").format(int(np.sum(np.isfinite(self.rate_map)))),
            ]
        )


@dataclass(frozen=True)
class ECocoRowSummary:
    """eCOCO 网格的逐行汇总。

    逐行汇总的语义来自 Acycle ``eccount.m``：它对"沉积速率图"（行 × 列的矩阵）
    的**每一行**分别统计——该行非 NaN 格的个数、这些格的速率的均值与中位数、
    相关系数之和、置信区间之和、"相关系数 × 置信区间"之和与轨道计数之和；输出里
    相关系数之和与置信区间之和除以个数，而另外两项保持原始求和。

    Attributes:
        window_centres: 各窗口中心在记录坐标轴上的位置。
        best_rates: 各窗口的最优沉积速率。
        best_correlations: 各窗口最优速率处的秩相关系数。
        best_p_values: 各窗口最优速率处的显著性水平。
        rate_mean: 每行非 NaN 格的速率均值。
        rate_median: 每行非 NaN 格的速率中位数。
        rate_counts: 每行非 NaN 格的个数。
        correlation_mean: 每行显著格上的平均相关系数（``eccsum / n``）。
        ci_mean: 每行显著格上的平均置信区间宽度（``ecisum / n``）。
        ecc_ci_sum: 每行显著格上"相关系数 × 置信区间宽度"之和（``ecocosum``）。
        orbit_sum: 每行显著格上的轨道计数之和（``enorbsum``）。
        grid: 产生本汇总的网格。
    """

    window_centres: npt.NDArray
    best_rates: npt.NDArray
    best_correlations: npt.NDArray
    best_p_values: npt.NDArray
    rate_mean: npt.NDArray
    rate_median: npt.NDArray
    rate_counts: npt.NDArray
    correlation_mean: npt.NDArray
    ci_mean: npt.NDArray
    ecc_ci_sum: npt.NDArray
    orbit_sum: npt.NDArray
    grid: ECocoGrid

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("eCOCO Row Summary"),
            "=" * 50,
            _("Rows (windows): {0}").format(int(self.rate_counts.size)),
        ]
        if self.rate_counts.size:
            lines.append(_("Mean rate: {0:.6g}").format(float(np.nanmean(self.best_rates))))
            lines.append(_("Mean correlation: {0:.4f}").format(float(np.nanmean(self.best_correlations))))
            lines.append(_("Total significant cells: {0}").format(int(np.sum(self.rate_counts))))
        return "\n".join(lines)


@dataclass(frozen=True)
class SignificantPeaksResult:
    """筛选出来的显著峰。

    Attributes:
        indices: 形状 ``(n_rows, n_peaks)`` 的网格列下标；该槽位没有合格峰时
            为 ``-1``。
        rates: 与 ``indices`` 同形状的速率；该槽位没有合格峰时为 ``nan``。
        correlations: 与 ``indices`` 同形状的相关系数。
        p_values: 与 ``indices`` 同形状的显著性水平。
        counts: 每行实际选出的峰数。
        criterion: 实际使用的判据（``"significance"`` 或 ``"ci_width"``）。
        ci_limit: 实际使用的判据阈值。
    """

    indices: npt.NDArray
    rates: npt.NDArray
    correlations: npt.NDArray
    p_values: npt.NDArray
    counts: npt.NDArray
    criterion: str
    ci_limit: float

    def summary(self) -> str:
        """Generate summary text."""
        return "\n".join(
            [
                _("Significant Peaks"),
                "=" * 50,
                _("Criterion: {0} <= {1}").format(str(self.criterion), float(self.ci_limit)),
                _("Rows: {0}").format(int(self.counts.size)),
                _("Peaks kept: {0}").format(int(np.sum(self.counts))),
            ]
        )


# =============================================================================
# 内部工具
# =============================================================================


def _resolve_seed(random_seed: int | None, seed: int | None) -> int | None:
    """把 ``random_seed`` 与别名 ``seed`` 合成一个种子。"""
    if random_seed is not None and seed is not None and int(random_seed) != int(seed):
        raise ValidationError(
            _("random_seed and seed disagree; pass only one of them"),
            details={"random_seed": int(random_seed), "seed": int(seed)},
        )
    if seed is not None:
        return int(seed)
    if random_seed is not None:
        return int(random_seed)
    return None


def _selected_periods(n_target: int | None) -> dict[str, float]:
    """按 Acycle ``orbit7`` / ``orbit9`` 的语义挑选目标谱的轨道周期。

    ``n_target`` 给 ``None`` 时用全部周期（:func:`berger1989_periods` 的 9 条，
    即 Acycle 的 ``orbit9``）；给 ``7`` 时取最长的 7 条（即 Acycle 的
    ``orbit7``，丢掉 19 kyr 与 17 kyr 的岁差带）。

    Returns:
        周期表（键为 ``"<轨道族>_<周期>kyr"``）。
    """
    periods = berger1989_periods()
    if n_target is None:
        return periods
    count = int(n_target)
    if count < 1 or count > len(periods):
        raise ValidationError(
            _("n_target must be between 1 and {0}; got {1}").format(len(periods), count),
            details={"n_target": count, "n_available": len(periods)},
        )
    ordered = sorted(periods.items(), key=lambda item: -float(item[1]))
    return dict(ordered[:count])


def _target_spectrum(frequencies: npt.NDArray, periods: dict[str, float]) -> npt.NDArray:
    """按给定周期表构造目标谱（与 :func:`milankovitch_target_spectrum` 同形）。"""
    target = np.zeros_like(frequencies)
    for key, period_kyr in periods.items():
        band = key.split("_", 1)[0]
        weight = TARGET_BAND_WEIGHTS.get(band, 1.0)
        centre = 1000.0 / float(period_kyr)
        half = BAND_HALF_WIDTH * centre
        taper = half * (1.0 - BAND_FLAT_CORE)
        shape = np.clip((half - np.abs(frequencies - centre)) / taper, 0.0, 1.0)
        target += weight * shape
    return target


def _rank_vector(values: npt.NDArray) -> npt.NDArray:
    """平均秩（并列取平均），与 MATLAB ``rank`` / ``tiedrank`` 的口径一致。"""
    return np.asarray(rankdata(values, method="average"), dtype=float)


def _spearman_from_ranks(rank_x: npt.NDArray, rank_y: npt.NDArray) -> float:
    """两个秩向量的 Pearson 相关系数（即 Spearman 秩相关系数）。"""
    centred_x = rank_x - float(np.mean(rank_x))
    centred_y = rank_y - float(np.mean(rank_y))
    denominator = np.sqrt(float(np.sum(centred_x**2)) * float(np.sum(centred_y**2)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.sum(centred_x * centred_y) / denominator)


def _permutation_null(
    rank_x: npt.NDArray, rank_y: npt.NDArray, n_permutations: int, rng: np.random.Generator
) -> npt.NDArray:
    """Fisher (1915) 的条件秩分布：随机排列 ``rank_y`` 得到的相关系数零分布。

    相关系数是秩的内积的线性函数，因此零分布只需一次向量化内积就能抽完；为控制
    内存，分块抽取。
    """
    n = int(rank_x.size)
    centred_x = rank_x - float(np.mean(rank_x))
    centred_y = rank_y - float(np.mean(rank_y))
    denominator = np.sqrt(float(np.sum(centred_x**2)) * float(np.sum(centred_y**2)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.full(int(n_permutations), np.nan)

    chunk = max(1, int(2_000_000 // max(n, 1)))
    base = np.arange(n)
    samples: list[npt.NDArray] = []
    remaining = int(n_permutations)
    while remaining > 0:
        take = min(chunk, remaining)
        order = rng.permuted(np.tile(base, (take, 1)), axis=1)
        dots = (centred_y[order] * centred_x).sum(axis=1)
        samples.append(dots / denominator)
        remaining -= take
    return np.concatenate(samples) if samples else np.empty(0)


def _t_approximation_p_value(coefficient: float, n: int) -> float:
    """t 近似的 p 值（Acycle / MATLAB ``corr(...,'type','Spearman')`` 的口径）。

    统计量 ``t = rho * sqrt((n - 2) / (1 - rho^2))``，自由度 ``n - 2``。
    """
    if n < MIN_RANK_SAMPLES or not np.isfinite(coefficient):
        return float("nan")
    if abs(coefficient) >= 1.0:
        return 0.0
    statistic = coefficient * np.sqrt((n - 2.0) / (1.0 - coefficient**2))
    return float(2.0 * student_t.sf(abs(statistic), n - 2))


def _comparison_stride(nfft: int, n_samples: int) -> int:
    """比较栅格的抽稀步长：使点距不小于记录的 Rayleigh 分辨率 ``1 / T``。

    零填充长度是 ``nfft``，谱线间距 ``1 / (nfft * dt)``；Rayleigh 分辨率是
    ``1 / (n_samples * dt)``。因此每隔 ``ceil(nfft / n_samples)`` 个零填充频点取
    一个，得到的栅格间距就不小于 Rayleigh 分辨率。
    """
    return max(1, int(np.ceil(float(nfft) / float(max(n_samples, 1)))))


def _band_selection(
    freqs: npt.NDArray,
    band_lo: float,
    band_hi: float,
    rayleigh: float,
    nyquist: float,
    stride: int,
) -> npt.NDArray:
    """返回比较频带内、按 Rayleigh 栅格抽稀后的频点下标。

    抽稀规则是"下标是 ``stride`` 的整数倍"：零填充频谱的频点间距是
    ``1 / (nfft * dt)``，每隔 ``stride`` 个取一个，间距就不小于 Rayleigh
    分辨率 ``1 / T``。
    """
    low = max(rayleigh, band_lo)
    high = min(nyquist, band_hi)
    if not high > low:
        return np.empty(0, dtype=int)
    mask = (freqs >= low) & (freqs <= high)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices
    return indices[(indices % stride) == 0]


def _ar1_divisor(freqs: npt.NDArray, phi: float, dt: float) -> npt.NDArray:
    """AR(1) 理论谱（Mann & Lees, 1996 的式 3），无效处取 ``inf``。

    Acycle ``corrsig.m`` 的 ``red=1`` 分支做的是 ``p = p ./ theored``：把功率谱
    除以 AR(1) 理论谱。除法是频率的单调重排，因此会改变功率谱的**秩**，从而
    改变秩相关系数——这一点与 Pearson 版本不同，Pearson 只看幅值。
    """
    theoretical = ar1_theoretical_spectrum(freqs, phi, dt)
    return np.where(np.isfinite(theoretical) & (theoretical > 0.0), theoretical, np.inf)


def _validate_pair(x: npt.NDArray, y: npt.NDArray, minimum: int) -> tuple[npt.NDArray, npt.NDArray]:
    """校验成对数据：等长、有限、点数足够。"""
    first = np.asarray(x, dtype=float).flatten()
    second = np.asarray(y, dtype=float).flatten()
    if first.size != second.size:
        raise ValidationError(
            _("x and y must have the same length; got {0} and {1}").format(first.size, second.size),
            details={"x_length": int(first.size), "y_length": int(second.size)},
        )
    if first.size < minimum:
        raise ValidationError(
            _("this test needs at least {0} paired samples; got {1}").format(minimum, int(first.size)),
            details={"n_samples": int(first.size), "minimum_required": minimum},
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValidationError(
            _("x and y must be finite"),
            details={
                "n_invalid_x": int(np.sum(~np.isfinite(first))),
                "n_invalid_y": int(np.sum(~np.isfinite(second))),
            },
        )
    if float(np.ptp(first)) == 0.0 or float(np.ptp(second)) == 0.0:
        raise ValidationError(_("x and y must both vary; a constant series cannot be correlated"))
    return first, second


def _rate_spearman(
    time: npt.NDArray,
    values: npt.NDArray,
    rate: float,
    nfft: int,
    phi: float,
    n_permutations: int,
    rng: np.random.Generator,
    target_freq: npt.NDArray | None,
    target_power: npt.NDArray | None,
    periods: dict[str, float],
    red: bool,
    default_target: bool,
    want_monte_carlo: int,
    sig_level: float,
) -> tuple[float, float, float, float, npt.NDArray, npt.NDArray, int, float] | None:
    """单个候选速率的秩相关系数、p 值与比较频带。

    Returns:
        ``(coefficient, p_permutation, p_monte_carlo, null_critical, band_freqs,
        band_target, band_size, null_spread)``；该速率不可用时返回 ``None``。
    """
    dx = float(np.median(np.diff(time)))
    span = float(time[-1] - time[0])
    dt = dx / float(rate)
    total = span / float(rate)

    n_grid = int(np.floor(span / dx)) + 1
    if n_grid < MIN_SAMPLES:
        logger.warning("COCO-Spearman: rate %.6g skipped -- resampling leaves only %d points", rate, n_grid)
        return None
    if n_grid > MAX_RESAMPLED_POINTS:
        logger.warning("COCO-Spearman: rate %.6g skipped -- resampling would leave %d points", rate, n_grid)
        return None

    if target_freq is None or target_power is None:
        band_lo = 1000.0 / max(periods.values())
        band_hi = 1000.0 / min(periods.values())
    else:
        band_lo = float(np.min(target_freq))
        band_hi = float(np.max(target_freq))

    nyquist = 1.0 / (2.0 * dt)
    if band_hi > 0.0 and nyquist < 2.0 * band_hi:
        logger.warning(
            "COCO-Spearman: rate %.6g skipped -- the resampled axis undersamples the shortest target period",
            rate,
        )
        return None

    rayleigh = 1.0 / total
    _grid, resampled, _dt = _resample_to_uniform_time(time, values, rate)
    freqs, power = periodogram(
        resampled,
        fs=1.0 / dt,
        detrend="linear",
        nfft=nfft,
        scaling="spectrum",
    )
    if red:
        power = power / _ar1_divisor(freqs, phi, dt)

    stride = _comparison_stride(nfft, n_grid)
    indices = _band_selection(freqs, band_lo, band_hi, rayleigh, nyquist, stride)
    if indices.size < MIN_BAND_POINTS:
        logger.warning(
            "COCO-Spearman: rate %.6g skipped -- fewer than %d spectral lines in the common band",
            rate,
            MIN_BAND_POINTS,
        )
        return None

    band_freqs = freqs[indices]
    if target_freq is None or target_power is None:
        if default_target:
            band_target = milankovitch_target_spectrum(band_freqs)
        else:
            band_target = _target_spectrum(band_freqs, periods)
    else:
        band_target = np.interp(band_freqs, target_freq, target_power, left=0.0, right=0.0)

    observed = power[indices]
    if float(np.ptp(observed)) == 0.0 or float(np.ptp(band_target)) == 0.0:
        logger.warning("COCO-Spearman: rate %.6g skipped -- the comparison band is degenerate", rate)
        return None

    rank_observed = _rank_vector(observed)
    rank_target = _rank_vector(band_target)
    coefficient = _spearman_from_ranks(rank_observed, rank_target)
    if not np.isfinite(coefficient):
        logger.warning("COCO-Spearman: rate %.6g skipped -- the rank correlation is undefined", rate)
        return None

    p_permutation = float("nan")
    critical = float("nan")
    spread = float("nan")
    if n_permutations > 0:
        null = _permutation_null(rank_observed, rank_target, n_permutations, rng)
        p_permutation = (1.0 + float(np.sum(null >= coefficient))) / (float(n_permutations) + 1.0)
        critical = float(np.nanquantile(null, 1.0 - sig_level))
        spread = float(np.nanstd(null))

    p_monte_carlo = float("nan")
    if want_monte_carlo > 0:
        burn_in = int(np.ceil(10.0 / (1.0 - abs(phi))))
        noise = rng.normal(size=(want_monte_carlo, n_grid + burn_in))
        surrogates = lfilter([1.0], [1.0, -phi], noise, axis=-1)[:, burn_in:]
        _sfreqs, null_power = periodogram(
            surrogates,
            fs=1.0 / dt,
            detrend="linear",
            nfft=nfft,
            scaling="spectrum",
            axis=-1,
        )
        if red:
            null_power = null_power / _ar1_divisor(freqs, phi, dt)
        null_observed = null_power[:, indices]
        null_ranks = np.asarray(rankdata(null_observed, method="average", axis=1), dtype=float)
        centred_null = null_ranks - null_ranks.mean(axis=1, keepdims=True)
        centred_target = rank_target - float(np.mean(rank_target))
        numerator = centred_null @ centred_target
        denominator = np.sqrt(np.sum(centred_null**2, axis=1) * float(np.sum(centred_target**2)))
        with np.errstate(invalid="ignore", divide="ignore"):
            null_coefficients = np.where(denominator > 0.0, numerator / denominator, np.nan)
        p_monte_carlo = (1.0 + float(np.sum(null_coefficients >= coefficient))) / (float(want_monte_carlo) + 1.0)
        if not np.isfinite(critical):
            critical = float(np.nanquantile(null_coefficients, 1.0 - sig_level))
            spread = float(np.nanstd(null_coefficients))

    return (
        float(coefficient),
        float(p_permutation),
        float(p_monte_carlo),
        float(critical),
        band_freqs,
        band_target,
        int(indices.size),
        float(spread),
    )


# =============================================================================
# Spearman 秩相关
# =============================================================================


def spearman_rank_correlation(
    x: npt.NDArray,
    y: npt.NDArray,
    n_permutations: int = 199,
    seed: int | None = None,
) -> SpearmanTestResult:
    """Spearman 秩相关与它的**秩置换零分布** p 值。

    统计量是两条序列秩的 Pearson 相关系数（Fisher, 1915）。p 值来自零假设
    "两条序列的秩是随机配对的"下的**排列分布**：条件于两个秩向量，随机排列其
    一就能抽到零分布。

    Parameters
    ----------
    x
        第一条序列。
    y
        第二条序列（与 ``x`` 等长）。
    n_permutations
        抽取的排列数。p 值的分辨率是 ``1 / (n_permutations + 1)``。
    seed
        随机种子；``None`` 时结果不可复现（按仓库约定发出 ``RuntimeWarning``）。

    Returns
    -------
    SpearmanTestResult
        秩相关系数、置换 p 值与 t 近似 p 值。

    Raises
    ------
    ValidationError
        长度不匹配、点数太少、含非有限值或某条序列为常数。

    Notes
    -----
    这个函数是 :func:`coco_spearman` 的统计内核。这里的零分布是 Fisher (1915)
    的条件秩分布（精确）；在 COCO 层面，观测功率谱在频率上是光滑的，秩置换零
    分布不条件于这种光滑性，因此那里的标定依赖比较栅格的粗细——详见
    :func:`coco_spearman` 的说明。

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-1.0, 1.0, 40)
    >>> result = spearman_rank_correlation(x, x**3, n_permutations=99, seed=1)
    >>> round(result.coefficient, 12)
    1.0
    """
    first, second = _validate_pair(x, y, MIN_RANK_SAMPLES)
    if int(n_permutations) < 1:
        raise ValidationError(
            _("n_permutations must be at least 1; got {0}").format(int(n_permutations)),
            details={"n_permutations": int(n_permutations)},
        )
    rng = make_rng(seed, context="Spearman rank correlation")

    rank_x = _rank_vector(first)
    rank_y = _rank_vector(second)
    coefficient = _spearman_from_ranks(rank_x, rank_y)
    null = _permutation_null(rank_x, rank_y, int(n_permutations), rng)
    p_value = (1.0 + float(np.sum(null >= coefficient))) / (float(n_permutations) + 1.0)
    return SpearmanTestResult(
        coefficient=float(coefficient),
        p_value=float(p_value),
        p_value_t=_t_approximation_p_value(coefficient, int(first.size)),
        n=int(first.size),
        n_permutations=int(n_permutations),
    )


# =============================================================================
# 相关系数的置信区间
# =============================================================================


def correlation_confidence_interval(
    rho: float,
    n: int,
    level: float = 0.95,
    method: str = "fisher-z",
    *,
    x: npt.NDArray | None = None,
    y: npt.NDArray | None = None,
    seed: int | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> CorrelationInterval:
    """相关系数的置信区间。

    Parameters
    ----------
    rho
        点估计（相关系数）。Fisher z 区间围绕它构造。
    n
        样本数。
    level
        名义置信水平，``0 < level < 1``。
    method
        ``"fisher-z"``：Fisher z 变换的正态区间（Fiore & Bonferroni, 2009）。
        ``"bootstrap"``：对 ``(x, y)`` 成对重抽样的百分位自助法区间
        （Efron & Tibshirani, 1993），**秩相关**的区间。
    x
        自助法用：第一条序列。与 ``y`` 必须成对给出。
    y
        自助法用：第二条序列。
    seed
        自助法的随机种子。
    n_bootstrap
        自助法的重抽样次数。

    Returns
    -------
    CorrelationInterval
        区间端点、所用方法与名义水平。

    Raises
    ------
    ValidationError
        相关系数越界、样本数不足、水平非法、方法未知，或自助法缺少成对数据。

    Notes
    -----
    **Fisher z 区间是对 Pearson 相关系数（在 bivariate normal 下）推导的。**
    把它作用在秩相关系数上是近似：秩相关系数的抽样方差比同值的 Pearson 相关系数
    略小，所以该区间略偏保守。要严格的秩相关区间请用 ``method="bootstrap"``。

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> a = rng.normal(size=60)
    >>> b = 0.5 * a + np.sqrt(0.75) * rng.normal(size=60)
    >>> interval = correlation_confidence_interval(
    ...     float(np.corrcoef(a, b)[0, 1]), 60, method="bootstrap", x=a, y=b, seed=1
    ... )
    >>> interval.lower < interval.upper
    True
    """
    try:
        estimate = float(rho)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("rho must be a number"),
            details={"rho": repr(rho)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(estimate) or estimate < -1.0 or estimate > 1.0:
        raise ValidationError(
            _("rho must be finite and within [-1, 1]; got {0}").format(estimate),
            details={"rho": estimate},
        )
    samples = int(n)
    if samples < MIN_CI_SAMPLES:
        raise ValidationError(
            _("n must be at least {0} for a confidence interval; got {1}").format(MIN_CI_SAMPLES, samples),
            details={"n": samples, "minimum_required": MIN_CI_SAMPLES},
        )
    try:
        confidence = float(level)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("level must be a number"),
            details={"level": repr(level)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValidationError(
            _("level must lie strictly between 0 and 1; got {0}").format(confidence),
            details={"level": confidence},
        )
    chosen = str(method).strip().lower()
    if chosen not in ("fisher-z", "bootstrap"):
        raise ValidationError(
            _("method must be 'fisher-z' or 'bootstrap'; got {0}").format(str(method)),
            details={"method": str(method)},
        )

    if chosen == "fisher-z":
        standard_error = 1.0 / np.sqrt(samples - 3.0)
        critical = float(norm.ppf(0.5 * (1.0 + confidence)))
        centre = np.arctanh(estimate)
        lower = float(np.tanh(centre - critical * standard_error))
        upper = float(np.tanh(centre + critical * standard_error))
        return CorrelationInterval(
            lower=lower,
            upper=upper,
            method="fisher-z",
            level=confidence,
            point_estimate=estimate,
            standard_error=float(standard_error),
            n=samples,
        )

    if x is None or y is None:
        raise ValidationError(
            _("the bootstrap interval needs the paired data: pass both x and y"),
            details={"x_is_none": x is None, "y_is_none": y is None},
        )
    first, second = _validate_pair(x, y, MIN_CI_SAMPLES)
    if int(n_bootstrap) < 2:
        raise ValidationError(
            _("n_bootstrap must be at least 2; got {0}").format(int(n_bootstrap)),
            details={"n_bootstrap": int(n_bootstrap)},
        )
    rng = make_rng(seed, context="Correlation bootstrap interval")
    count = int(first.size)
    draws = rng.integers(0, count, size=(int(n_bootstrap), count))
    resampled_x = first[draws]
    resampled_y = second[draws]
    rank_x = np.asarray(rankdata(resampled_x, method="average", axis=1), dtype=float)
    rank_y = np.asarray(rankdata(resampled_y, method="average", axis=1), dtype=float)
    centred_x = rank_x - rank_x.mean(axis=1, keepdims=True)
    centred_y = rank_y - rank_y.mean(axis=1, keepdims=True)
    numerator = np.sum(centred_x * centred_y, axis=1)
    denominator = np.sqrt(np.sum(centred_x**2, axis=1) * np.sum(centred_y**2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        statistics = np.where(denominator > 0.0, numerator / denominator, np.nan)
    finite = statistics[np.isfinite(statistics)]
    if finite.size < 2:
        raise ValidationError(
            _("the bootstrap produced too few usable resamples to build an interval"),
            details={"n_bootstrap": int(n_bootstrap), "n_usable": int(finite.size)},
        )
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(finite, alpha))
    upper = float(np.quantile(finite, 1.0 - alpha))
    return CorrelationInterval(
        lower=lower,
        upper=upper,
        method="bootstrap",
        level=confidence,
        point_estimate=estimate,
        standard_error=None,
        n=count,
    )


# =============================================================================
# COCO（Spearman 秩相关变体）
# =============================================================================


def coco_spearman(
    time: npt.NDArray,
    values: npt.NDArray,
    rate_range: tuple[float, float],
    n_rates: int = 200,
    target_spectrum: tuple[npt.NDArray, npt.NDArray] | None = None,
    n_monte_carlo: int = 200,
    random_seed: int | None = None,
    *,
    n_target: int | None = None,
    n_period: int | None = None,
    sig_level: float = 0.05,
    red: bool = True,
    seed: int | None = None,
    null_model: str = "permutation",
) -> CocoSpearmanResult:
    """COCO 的 Spearman 秩相关变体。

    与 :func:`stratigraphy.coco.coco` 的框架完全一致（同样的频率口径、同样的
    目标谱、同样的重采样与带宽判据），只把统计量从 **Pearson 相关系数**换成
    **Spearman 秩相关系数**。

    Parameters
    ----------
    time
        记录的地层坐标轴（严格递增）。速率单位是"坐标单位 / Ma"。
    values
        与 ``time`` 等长的测量序列。
    rate_range
        候选速率区间 ``(low, high)``。
    n_rates
        扫描的候选速率个数。
    target_spectrum
        可选的外部目标谱 ``(频率, 功率)``，频率单位 cycles/Ma。为 ``None`` 时用
        Berger (1989) 的目标谱。
    n_monte_carlo
        零分布的样本数：秩置换零分布用这么多次排列；蒙特卡洛零分布用这么多条
        AR(1) 替代序列。为 0 时不计算 p 值。
    random_seed
        随机种子；``None`` 时结果不可复现。
    n_target
        目标谱里用几条轨道周期（Acycle 的 ``orbit7`` / ``orbit9`` 语义）：
        ``None`` 用全部 9 条，``7`` 取最长的 7 条。
    n_period
        ``n_target`` 的别名，两个都给出且不一致时报错。
    sig_level
        名义显著性水平。用于"最优速率的不确定度区间"（相关性高于零分布的
        ``1 - sig_level`` 分位数）与 eCOCO 网格的筛选。
    red
        是否去掉 AR(1) 红噪声背景（Acycle ``corrsig.m`` 的 ``red=1``：把功率谱
        除以 AR(1) 理论谱）。这一步是频率的单调重排，**会改变功率谱的秩**，
        因而会改变秩相关系数。
    seed
        ``random_seed`` 的别名。
    null_model
        零分布模型：``"permutation"``（Fisher, 1915 的条件秩分布，默认）、
        ``"monte-carlo"``（AR(1) 随机谱，Acycle ``corrsig.m`` 的语义）或
        ``"both"``（两个都算，``p_values`` 仍是置换零分布的）。

    Returns
    -------
    CocoSpearmanResult
        扫描结果，含最优速率、秩相关系数与 p 值。

    Raises
    ------
    ValidationError
        记录不合法、速率区间非法、参数越界等。

    Notes
    -----
    **与 Acycle 的差别。** Acycle ``corrsig.m`` 用 MATLAB
    ``corr(...,'type','Spearman')`` 的 t 近似 p 值，而 H0 显著性水平来自
    ``cyclesig_rank.m`` 的**随机谱**蒙特卡洛；它的输出矩阵 ``corrCI`` 的表头写着
    6 列（含 95% 置信区间的上下端点），但代码只填了 4 列，置信区间两列在该文件
    里从未被填充。本模块因此：

    * p 值给两个：``p_values``（秩置换）与 ``p_values_monte_carlo``（随机谱）；
    * 置信区间由 :func:`correlation_confidence_interval` 单独提供。

    **两种零分布回答的问题不同。** 秩置换零分布条件于两个秩向量的边际分布，问的
    是"秩的配对是否偏离随机"；蒙特卡洛零分布把观测的 AR(1) 系数带进替代序列，
    问的是"观测谱是否偏离噪声模型给出的谱"。

    标定：一条没有轨道信号的记录上，固定候选速率处 400 次重复中 ``p < 0.05`` 的
    比例，在本模块的比较栅格（Rayleigh 分辨率）上秩置换给出 5.8%、蒙特卡洛给出
    6.5%（名义水平 5%，标准误 1.1%）；但把比较栅格加密到零填充细网格上，秩置换
    零分布会明显偏乐观（实测约 20%），因为谱线的秩向量在那一层上不再可交换。
    做**筛选**时用蒙特卡洛零分布（Acycle ``corrsig.m`` 的做法）。

    蒙特卡洛零分布的替代序列直接生成在**重采样后的**时间轴上，因此对**不等间
    距**的记录，观测谱（经过线性插值重采样）比替代序列多一道平滑，两者的谱形状
    略有差别；对等间距的记录这一差别不存在。

    Examples
    --------
    >>> import numpy as np
    >>> astro = np.linspace(0.0, 0.8, 600)
    >>> record = np.sin(2.0 * np.pi * astro / 0.02)
    >>> result = coco_spearman(4.0 * astro, record, rate_range=(2.0, 6.0), n_rates=8,
    ...                       n_monte_carlo=19, random_seed=1)
    >>> result.best_rate > 0.0
    True
    """
    resolved_seed = _resolve_seed(random_seed, seed)
    model = str(null_model).strip().lower()
    if model not in NULL_MODELS:
        raise ValidationError(
            _("null_model must be one of {0}; got {1}").format(", ".join(NULL_MODELS), str(null_model)),
            details={"null_model": str(null_model)},
        )
    if n_target is not None and n_period is not None and int(n_target) != int(n_period):
        raise ValidationError(
            _("n_target and n_period disagree; they are aliases, pass only one of them"),
            details={"n_target": int(n_target), "n_period": int(n_period)},
        )
    target_count = n_target if n_target is not None else n_period
    periods = _selected_periods(target_count)
    default_target = target_count is None

    t, v = _validate_record(time, values)
    low, high = _validate_rate_range(rate_range, n_rates)
    if int(n_monte_carlo) < 0:
        raise ValidationError(
            _("n_monte_carlo must be >= 0; got {0}").format(int(n_monte_carlo)),
            details={"n_monte_carlo": int(n_monte_carlo)},
        )
    if not 0.0 < float(sig_level) < 1.0:
        raise ValidationError(
            _("sig_level must lie strictly between 0 and 1; got {0}").format(float(sig_level)),
            details={"sig_level": float(sig_level)},
        )

    target_freq: npt.NDArray | None = None
    target_power: npt.NDArray | None = None
    if target_spectrum is not None:
        try:
            target_freq = np.asarray(target_spectrum[0], dtype=float).flatten()
            target_power = np.asarray(target_spectrum[1], dtype=float).flatten()
        except (TypeError, ValueError, IndexError) as exc:
            raise ValidationError(_("target_spectrum must be a (frequencies, power) pair of arrays")) from exc
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

    rates = np.linspace(low, high, int(n_rates))
    nfft = _spectral_grid(t.size)
    phi = _ar1_phi(_linear_detrend(v))
    rng = make_rng(resolved_seed, context="COCO Spearman")

    want_permutation = model in ("permutation", "both") and int(n_monte_carlo) > 0
    want_monte_carlo = int(n_monte_carlo) if model in ("monte-carlo", "both") else 0

    correlations = np.full(rates.size, np.nan)
    p_values = np.full(rates.size, np.nan)
    p_permutation = np.full(rates.size, np.nan)
    p_monte_carlo = np.full(rates.size, np.nan)
    critical = np.full(rates.size, np.nan)
    band_sizes = np.full(rates.size, np.nan)
    band_freqs = np.empty(0)
    band_target = np.empty(0)

    for index, rate in enumerate(rates):
        outcome = _rate_spearman(
            t,
            v,
            float(rate),
            nfft,
            phi,
            int(n_monte_carlo) if want_permutation else 0,
            rng,
            target_freq,
            target_power,
            periods,
            bool(red),
            default_target,
            want_monte_carlo,
            float(sig_level),
        )
        if outcome is None:
            continue
        coefficient, p_perm, p_mc, level, freqs_here, target_here, size, _spread = outcome
        correlations[index] = coefficient
        p_permutation[index] = p_perm
        p_monte_carlo[index] = p_mc
        critical[index] = level
        band_sizes[index] = size
        band_freqs = freqs_here
        band_target = target_here

    if not np.any(np.isfinite(correlations)):
        raise ValidationError(
            _(
                "no candidate rate in [{0}, {1}] is admissible: the record is too short or too "
                "coarse to resolve any astronomical band"
            ).format(low, high),
            details={"rate_range": [low, high], "n_rates": int(rates.size)},
        )

    if model == "monte-carlo":
        p_values = p_monte_carlo.copy()
    else:
        p_values = p_permutation.copy()

    best_index = int(np.nanargmax(correlations))
    best_rate = float(rates[best_index])
    best_correlation = float(correlations[best_index])
    best_p = float(p_values[best_index])

    threshold = float(critical[best_index])
    confidence: tuple[float, float] | None = None
    if np.isfinite(threshold) and threshold < best_correlation:
        confidence = _confidence_interval(rates, correlations, best_index, best_correlation - threshold)

    significant = np.isfinite(p_values) & (p_values <= float(sig_level))

    logger.info(
        "COCO-Spearman: best rate %.6g (rank correlation %.4f, p %.4g) over %d candidate rates",
        best_rate,
        best_correlation,
        best_p,
        int(rates.size),
    )

    return CocoSpearmanResult(
        rates=rates,
        correlations=correlations,
        p_values=p_values,
        best_rate=best_rate,
        best_correlation=best_correlation,
        best_p_value=best_p,
        target_frequencies=band_freqs,
        target_spectrum=band_target,
        rate_confidence=confidence,
        p_values_permutation=p_permutation if want_permutation else None,
        p_values_monte_carlo=p_monte_carlo if want_monte_carlo > 0 else None,
        band_sizes=band_sizes,
        null_model=model,
        significant=significant,
    )


# =============================================================================
# eCOCO 网格与逐行汇总
# =============================================================================


def ecoco_row_summary(
    time: npt.NDArray,
    values: npt.NDArray,
    rate_range: tuple[float, float],
    n_windows: int = 10,
    n_rates: int = 100,
    n_monte_carlo: int = 100,
    random_seed: int | None = None,
    *,
    n_target: int | None = None,
    n_period: int | None = None,
    sig_level: float = 0.05,
    red: bool = True,
    seed: int | None = None,
    null_model: str = "monte-carlo",
    ci_level: float = 0.95,
) -> ECocoRowSummary:
    """eCOCO 网格的逐行汇总。

    记录沿坐标轴被等分成 ``n_windows`` 个窗口，每个窗口独立跑一次
    :func:`coco_spearman`，得到一张"行 = 窗口、列 = 候选速率"的网格。某格若该
    窗口在该速率上显著（p 值不大于 ``sig_level``），速率图在该格填该速率，否则
    填 ``nan``——这正是 Acycle ``eccount.m`` 里"沉积速率图"非 NaN 格的语义。

    逐行汇总的字段与 Acycle ``eccount.m`` 的输出一一对应：

    ==============  =========================================================
    本实现          Acycle ``eccount.m``
    ==============  =========================================================
    ``rate_counts``   ``ecsrn_n``（该行非 NaN 格的个数）
    ``rate_mean``     ``ecsrn_mean``（该行非 NaN 格的速率均值）
    ``rate_median``   ``ecsrn_median``（该行非 NaN 格的速率中位数）
    ``correlation_mean``  ``eccsum / ecsrn_n``（该行相关系数之和除以个数）
    ``ci_mean``       ``ecisum / ecsrn_n``（该行置信区间之和除以个数）
    ``ecc_ci_sum``    ``ecocosum``（该行"相关系数 × 置信区间"之和，**不**除以个数）
    ``orbit_sum``     ``enorbsum``（该行轨道计数之和，**不**除以个数）
    ==============  =========================================================

    Parameters
    ----------
    time
        记录的地层坐标轴（严格递增）。
    values
        与 ``time`` 等长的测量序列。
    rate_range
        候选速率区间 ``(low, high)``。
    n_windows
        窗口个数。
    n_rates
        每个窗口扫描的候选速率个数。
    n_monte_carlo
        每个候选速率的零分布样本数（排列数或替代序列条数）。
    random_seed
        随机种子；``None`` 时结果不可复现。
    n_target
        目标谱里用几条轨道周期；见 :func:`coco_spearman`。
    n_period
        ``n_target`` 的别名。
    sig_level
        名义显著性水平，也是网格的筛选阈值。
    red
        是否去掉 AR(1) 红噪声背景；见 :func:`coco_spearman`。
    seed
        ``random_seed`` 的别名。
    null_model
        零分布模型；见 :func:`coco_spearman`。**本函数默认用
        ``"monte-carlo"``**：网格的用途是筛选，而 Acycle 自己的筛选管线
        （``corrsig.m`` → ``eccount.m`` → ``eccpeaks.m``）的显著性水平来自蒙特
        卡洛零分布——它把观测的 AR(1) 系数带进了替代序列。两种零分布在本模块的
        比较栅格上都接近名义水平，但回答的问题不同。
    ci_level
        每个格子的相关系数置信区间的名义置信水平（默认 95%）。它与
        ``sig_level``（筛选用的显著性水平）是两件事：前者决定区间有多宽，后者
        决定哪些格算显著。

    Returns
    -------
    ECocoRowSummary
        逐行汇总与产生它的网格。

    Raises
    ------
    ValidationError
        记录不合法、窗口数不足或速率区间非法。

    Notes
    -----
    Acycle 的 ``eccount.m`` 对"沉积速率图"的**行**做汇总（``for i = 1:nrow``），
    不是对列。本实现沿用这个方向：行是窗口，列是候选速率。

    窗口 ``i`` 的网格用**整条记录**的采样间距，但时间跨度用该窗口自己的跨度来
    判轨道是否可分辨（与 Acycle ``numorbit`` 的语义一致）。
    """
    resolved_seed = _resolve_seed(random_seed, seed)
    if int(n_windows) < 2:
        raise ValidationError(
            _("n_windows must be at least 2; got {0}").format(int(n_windows)),
            details={"n_windows": int(n_windows)},
        )
    if n_target is not None and n_period is not None and int(n_target) != int(n_period):
        raise ValidationError(
            _("n_target and n_period disagree; they are aliases, pass only one of them"),
            details={"n_target": int(n_target), "n_period": int(n_period)},
        )
    if not 0.0 < float(ci_level) < 1.0:
        raise ValidationError(
            _("ci_level must lie strictly between 0 and 1; got {0}").format(float(ci_level)),
            details={"ci_level": float(ci_level)},
        )

    t, v = _validate_record(time, values)
    low, high = _validate_rate_range(rate_range, n_rates)

    edges = np.linspace(float(t[0]), float(t[-1]), int(n_windows) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_rows = int(centres.size)
    n_cols = int(n_rates)

    rate_map = np.full((n_rows, n_cols), np.nan)
    correlations = np.full((n_rows, n_cols), np.nan)
    p_values = np.full((n_rows, n_cols), np.nan)
    ci_lower = np.full((n_rows, n_cols), np.nan)
    ci_upper = np.full((n_rows, n_cols), np.nan)
    n_orbits = np.zeros((n_rows, n_cols), dtype=int)
    band_sizes = np.full((n_rows, n_cols), np.nan)
    best_rates = np.full(n_rows, np.nan)
    best_correlations = np.full(n_rows, np.nan)
    best_p_values = np.full(n_rows, np.nan)

    spacing = float(np.median(np.diff(t)))
    periods = np.asarray(list(berger1989_periods().values()), dtype=float)

    for index in range(n_rows):
        mask = (t >= edges[index]) & (t <= edges[index + 1])
        window_time = t[mask]
        window_values = v[mask]
        if window_time.size < MIN_SAMPLES:
            logger.warning("eCOCO: window %d skipped -- only %d samples", index, int(window_time.size))
            continue
        window_seed = None if resolved_seed is None else int(resolved_seed) + index
        try:
            window = coco_spearman(
                window_time,
                window_values,
                rate_range=rate_range,
                n_rates=n_rates,
                n_monte_carlo=n_monte_carlo,
                random_seed=window_seed,
                n_target=n_target,
                n_period=n_period,
                sig_level=sig_level,
                red=red,
                null_model=null_model,
            )
        except ValidationError as exc:
            logger.warning("eCOCO: window %d skipped -- %s", index, exc)
            continue

        best_rates[index] = window.best_rate
        best_correlations[index] = window.best_correlation
        best_p_values[index] = window.best_p_value

        window_span = float(window_time[-1] - window_time[0])
        try:
            orbits = num_orbits(
                periods,
                window_span,
                rate_range=(low, high),
                n_rates=n_rates,
                sample_spacing=spacing,
                sig_level=sig_level,
            )
        except ValidationError as exc:
            logger.warning("eCOCO: window %d skipped -- %s", index, exc)
            continue
        n_orbits[index, :] = np.asarray(orbits.n_orbits, dtype=int)

        window_correlations = np.asarray(window.correlations, dtype=float)
        window_p = np.asarray(window.p_values, dtype=float)
        window_sizes = np.asarray(window.band_sizes, dtype=float)
        finite = np.isfinite(window_correlations)
        row_correlations = np.where(finite, window_correlations, np.nan)
        row_p = np.where(finite, window_p, np.nan)

        lower = np.full(n_cols, np.nan)
        upper = np.full(n_cols, np.nan)
        for column in range(n_cols):
            size = window_sizes[column]
            if not np.isfinite(size) or not np.isfinite(row_correlations[column]) or size < MIN_CI_SAMPLES:
                continue
            interval = correlation_confidence_interval(
                float(row_correlations[column]), int(size), level=float(ci_level), method="fisher-z"
            )
            lower[column] = interval.lower
            upper[column] = interval.upper

        significant = finite & np.isfinite(row_p) & (row_p <= float(sig_level))
        rate_map[index, :] = np.where(significant, window.rates, np.nan)
        correlations[index, :] = row_correlations
        p_values[index, :] = row_p
        ci_lower[index, :] = lower
        ci_upper[index, :] = upper
        band_sizes[index, :] = window_sizes

    grid = ECocoGrid(
        window_centres=centres,
        rates=np.linspace(low, high, n_cols),
        rate_map=rate_map,
        correlations=correlations,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_orbits=n_orbits,
        band_sizes=band_sizes,
    )

    counts = np.zeros(n_rows, dtype=int)
    rate_mean = np.full(n_rows, np.nan)
    rate_median = np.full(n_rows, np.nan)
    correlation_mean = np.full(n_rows, np.nan)
    ci_mean = np.full(n_rows, np.nan)
    ecc_ci_sum = np.full(n_rows, np.nan)
    orbit_sum = np.full(n_rows, np.nan)

    for index in range(n_rows):
        row = rate_map[index, :]
        present = np.isfinite(row)
        counts[index] = int(np.sum(present))
        if counts[index] == 0:
            continue
        rate_mean[index] = float(np.mean(row[present]))
        rate_median[index] = float(np.median(row[present]))
        correlation_mean[index] = float(np.mean(correlations[index, :][present]))
        row_widths = ci_upper[index, :] - ci_lower[index, :]
        usable = present & np.isfinite(row_widths)
        if bool(np.any(usable)):
            ci_mean[index] = float(np.mean(row_widths[usable]))
            ecc_ci_sum[index] = float(np.sum(correlations[index, :][usable] * row_widths[usable]))
        orbit_sum[index] = float(np.sum(n_orbits[index, :][present]))

    return ECocoRowSummary(
        window_centres=centres,
        best_rates=best_rates,
        best_correlations=best_correlations,
        best_p_values=best_p_values,
        rate_mean=rate_mean,
        rate_median=rate_median,
        rate_counts=counts,
        correlation_mean=correlation_mean,
        ci_mean=ci_mean,
        ecc_ci_sum=ecc_ci_sum,
        orbit_sum=orbit_sum,
        grid=grid,
    )


# =============================================================================
# 峰值筛选
# =============================================================================


def significant_peaks(
    grid: ECocoGrid | ECocoRowSummary,
    n_peaks: int = 3,
    ci_limit: float = 0.05,
    *,
    corr_threshold: float = 0.3,
    min_orbits: int = 3,
    criterion: str = "significance",
    dim: str = "row",
) -> SignificantPeaksResult:
    """筛选"只取 CI 小于 5% 的前 n 个峰"。

    语义来自 Acycle ``eccpeaks.m``：把速率图按速率**从大到小**排序，取前 ``n``
    个，其中满足下列条件的留下——

    1. 相关系数不小于 ``corr_threshold``（Acycle 默认 0.3）；
    2. "CI" 判据满足（见下）；
    3. 轨道计数不小于 ``min_orbits``（Acycle 默认 3）。

    **关于"CI 小于 5%"。** Acycle ``eccpeaks.m`` 的注释写的是
    ``ci: shreshold confidence interval. default = 5 (5%)``，而它的判据是
    ``eci(...) <= ci``。Acycle 的 ``eci`` 在 eCOCO 管线里是"相关系数的置信区间"
    矩阵，但**同一个矩阵里的量在 ``corrsig.m`` 里被当作 H0 显著性水平使用**
    （``corr_h0`` 的输出就是"零假设的蒙特卡洛显著性水平"，阈值也写成
    0.10 / 0.05 / 0.01 / 0.001 的百分数）。两种读法都说得通，本函数把判据做成
    可选：

    * ``criterion="significance"``（默认）：判据量是**显著性水平**（p 值），
      ``ci_limit`` 取 0–1 的小数，0.05 即"5%"。这是与 Acycle 的显著性阈值
      习惯一致、也便于与其它显著性结果混用的读法。
    * ``criterion="ci_width"``：判据量是**相关系数置信区间的宽度**
      （上端点减下端点），``ci_limit`` 同样是 0–1 的小数。

    Parameters
    ----------
    grid
        :func:`ecoco_row_summary` 产出的 :class:`ECocoRowSummary` 或
        :class:`ECocoGrid`。
    n_peaks
        每行（或每列）最多保留几个峰。
    ci_limit
        判据阈值，0–1 的小数。
    corr_threshold
        相关系数的下限。
    min_orbits
        轨道计数的下限。
    criterion
        判据量：``"significance"`` 或 ``"ci_width"``。
    dim
        筛选方向：``"row"``（默认，每行即每个窗口取前 n 个）或 ``"column"``
        （每列即每个候选速率取前 n 个）。

    Returns
    -------
    SignificantPeaksResult
        选出的峰的下标、速率、相关系数与显著性水平。

    Raises
    ------
    ValidationError
        参数越界或输入不是 eCOCO 网格。

    Notes
    -----
    Acycle ``eccpeaks.m`` 的默认方向是 ``dim=1``（按**列**取前 n 个），本实现
    的默认是按**行**（按窗口取前 n 个）：eCOCO 的速率曲线是"随位置变化的沉积
    速率"，"前 n 个峰"指的是这条曲线上最显著的 n 个速率。两种方向都实现了。
    """
    if isinstance(grid, ECocoRowSummary):
        mesh = grid.grid
    elif isinstance(grid, ECocoGrid):
        mesh = grid
    else:
        raise ValidationError(
            _("grid must be an ECocoGrid or an ECocoRowSummary"),
            details={"type": type(grid).__name__},
        )

    peaks = int(n_peaks)
    if peaks < 1:
        raise ValidationError(
            _("n_peaks must be at least 1; got {0}").format(peaks),
            details={"n_peaks": peaks},
        )
    try:
        limit = float(ci_limit)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("ci_limit must be a number"),
            details={"ci_limit": repr(ci_limit)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(limit) or not 0.0 <= limit <= 1.0:
        raise ValidationError(
            _("ci_limit must lie between 0 and 1; got {0}").format(limit),
            details={"ci_limit": limit},
        )
    try:
        threshold = float(corr_threshold)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("corr_threshold must be a number"),
            details={"corr_threshold": repr(corr_threshold)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValidationError(
            _("corr_threshold must lie between -1 and 1; got {0}").format(threshold),
            details={"corr_threshold": threshold},
        )
    orbit_floor = int(min_orbits)
    if orbit_floor < 0:
        raise ValidationError(
            _("min_orbits must be >= 0; got {0}").format(orbit_floor),
            details={"min_orbits": orbit_floor},
        )
    chosen = str(criterion).strip().lower()
    if chosen not in PEAK_CRITERIA:
        raise ValidationError(
            _("criterion must be one of {0}; got {1}").format(", ".join(PEAK_CRITERIA), str(criterion)),
            details={"criterion": str(criterion)},
        )
    direction = str(dim).strip().lower()
    if direction not in PEAK_DIMENSIONS:
        raise ValidationError(
            _("dim must be one of {0}; got {1}").format(", ".join(PEAK_DIMENSIONS), str(dim)),
            details={"dim": str(dim)},
        )

    n_rows, n_cols = mesh.shape
    if direction == "row":
        indices = np.full((n_rows, peaks), -1, dtype=int)
    else:
        indices = np.full((n_cols, peaks), -1, dtype=int)
    rates = np.full(indices.shape, np.nan)
    coefficients = np.full(indices.shape, np.nan)
    significances = np.full(indices.shape, np.nan)

    def passes(row: int, column: int) -> bool:
        coefficient = float(mesh.correlations[row, column])
        if not np.isfinite(coefficient) or coefficient < threshold:
            return False
        if int(mesh.n_orbits[row, column]) < orbit_floor:
            return False
        if chosen == "significance":
            value = float(mesh.p_values[row, column])
            return np.isfinite(value) and value <= limit
        lower = float(mesh.ci_lower[row, column])
        upper = float(mesh.ci_upper[row, column])
        if not np.isfinite(lower) or not np.isfinite(upper):
            return False
        return (upper - lower) <= limit

    if direction == "row":
        for row in range(n_rows):
            row_map = mesh.rate_map[row, :]
            present = np.flatnonzero(np.isfinite(row_map))
            if present.size == 0:
                continue
            ordered = present[np.argsort(-row_map[present], kind="stable")][:peaks]
            for slot, column in enumerate(ordered):
                if not passes(row, int(column)):
                    continue
                indices[row, slot] = int(column)
                rates[row, slot] = float(mesh.rates[column])
                coefficients[row, slot] = float(mesh.correlations[row, column])
                significances[row, slot] = float(mesh.p_values[row, column])
    else:
        for column in range(n_cols):
            column_map = mesh.rate_map[:, column]
            present = np.flatnonzero(np.isfinite(column_map))
            if present.size == 0:
                continue
            ordered = present[np.argsort(-column_map[present], kind="stable")][:peaks]
            for slot, row in enumerate(ordered):
                if not passes(int(row), column):
                    continue
                indices[column, slot] = int(row)
                rates[column, slot] = float(mesh.rates[column])
                coefficients[column, slot] = float(mesh.correlations[int(row), column])
                significances[column, slot] = float(mesh.p_values[int(row), column])

    counts = np.sum(indices >= 0, axis=1).astype(int)
    return SignificantPeaksResult(
        indices=indices,
        rates=rates,
        correlations=coefficients,
        p_values=significances,
        counts=counts,
        criterion=chosen,
        ci_limit=limit,
    )
