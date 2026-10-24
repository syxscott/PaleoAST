# =============================================================================
# FILE: stratigraphy/tuning.py
# =============================================================================
"""
轨道调谐（astrochronological tuning）: TimeOpt / eTimeOpt、年龄标尺与沉积速率年龄模型。

本模块把一条地层记录校准到天文时间轴上, 是 PaleoAST 目前完全空白的一环:
``stratigraphy.spectral_analysis`` 只能回答"这条记录里有没有周期", 而调谐要回答
"这条记录对应哪条天文时间轴" —— 这是把地层记录挂到年代框架上的最后一步。

提供的四个能力
--------------
* :func:`time_opt` —— TimeOpt (Meyers 2015): 搜索一个压缩/膨胀系数（沉积速率的
  倒数）与一个常数相位偏移, 使记录重采样到均匀时间轴后, 功率谱在偏心率相关谐波
  （约 100 kyr 的长周期族）上的能量最大, 且岁差调制（约 20 kyr）与该谐波相位一致
  （"bundle"）。
* :func:`etime_opt` —— eTimeOpt (Meyers 2019): 把 TimeOpt 推广到**非恒定**沉积速率,
  速率本身随时间分段变化, 优化对象是整条时间轴。
* :func:`build_age_scale` —— 年龄标尺: 由锚点 (深度/年龄) 插值出深度→年龄, 并给出
  逐深度的年龄不确定度（口径与 ``stratigraphy.correlation.AgeModelAnalyzer`` 一致）。
* :func:`sedimentation_rate_to_age_model` —— 恒定沉积速率 → 年龄轴。

单位与口径
----------
* 深度: 任意深度单位（地质上通常为 m）。
* 时间: 本模块内部一律用 **kyr**；输出年龄用 **Ma**（``datum_age_ma`` / ``ages_ma``）。
* 频率: 一律是"每单位时间"（cycles/kyr），**不是**"每样本"。频率栅格由记录长度
  （Rayleigh 分辨率 1/T）决定, 而不是由采样密度决定 —— 采样更密不会让周期变差
  （与 ``stratigraphy.spectral_analysis`` 修过的同一个 bug 保持一致）。

评分函数（TimeOpt）
-------------------
对每个候选压缩系数 C（绝对沉积速率 S = C · S_ref）, 记录被重采样到均匀时间轴
t (kyr) 后计算三项:

1. ``ecc_frac`` —— 原始功率谱落在**偏心率带**（主目标周期 ±15%）内的能量占比。
   记录的长周期能量越集中在 ~100 kyr, 该值越大。
2. ``pre_frac`` —— 原始功率谱落在**岁差带**内的能量占比: 该记录必须真的含有
   ~20 kyr 的节律, 否则谈不上 bundle。
3. ``env_ecc_frac`` —— **岁差带的振幅包络**（Hilbert 变换）在偏心率带内的能量占比。
   这是 TimeOpt 的核心判据（"偏心率调制最强"）。
4. ``bundle`` —— 岁差包络与偏心率参考余弦之间的归一化相关 ∈ [-1, 1]。
   它同时给出**相位偏移**: 使两者对齐的相位即为结果里的 ``best_phase``。

得分 = 0.30·ecc_frac + 0.15·pre_frac + 0.25·env_ecc_frac + 0.30·bundle。
相位偏移以弧度给出, 对应把时间轴整体平移 Δt = φ·T_ecc/(2π)。

参考速率 S_ref 的确定
---------------------
压缩系数必须相对于某个参考速率才有意义, 而 API 不接受外部速率, 因此 S_ref 由
记录自身估计: 取记录在**深度域**的主峰周期 P_dom, 分别假设它属于每一个候选轨道
周期族（偏心率 ~100 kyr / 倾角 ~41 kyr / 岁差 ~21 kyr）, 得到候选速率
S = P_dom / T, 用**同一个评分函数**选出得分最高的那个作为 S_ref。于是"岁差主导"
的记录（环地层学里最常见的情形）不会因为主峰是岁差而被锚歪。

Acycle (https://github.com/mingsongli/acycle) 为 GPL 许可, 本模块是按 Meyers
(2015, 2019) 论文与上述算法规格**独立实现**的, 未翻译或复制其 MATLAB 源码。

参考文献
--------
* Meyers, S.R., 2015. The evaluation of eccentricity-related amplitude modulation
  and bundling in paleoclimate data: An inverse approach for astrochronologic
  testing and time scale optimization. *Paleoceanography*, 30(10): 2241-2272.
  doi:10.1002/2015PA002850
* Meyers, S.R., 2019. Cyclostratigraphy and the problem of astrochronologic
  testing. *Earth-Science Reviews*, 190: 190-223.
* Berger, A., 1978. Long-term climatic variation: astronomical theory.
  In Understanding the Climate. Academic Press.
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for the
  eccentricity and obliquity of the Earth's orbits over 5 Myr. *Astronomy and
  Astrophysics*, 413: 755-767.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

from config.i18n import _
from stratigraphy.correlation import _depth_varying_half_width
from stratigraphy.spectral_analysis import SpectralAnalyzer
from utils.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "AgeScaleResult",
    "ETimeOptResult",
    "TimeOptResult",
    "build_age_scale",
    "etime_opt",
    "milankovitch_periods",
    "sedimentation_rate_to_age_model",
    "time_opt",
]

# =============================================================================
# 常量
# =============================================================================

#: 调谐所需的最少点数（谱分析本身至少要 4 点, 但稳健的周期搜索需要更多）。
MIN_POINTS = 8

#: 偏心率族（长周期族）的下限 (kyr): 不小于该值的周期视为偏心率及其谐波。
ECCENTRICITY_MIN_KYR = 50.0

#: 岁差族的下限 (kyr)。
PRECESSION_MIN_KYR = 12.0

#: 岁差族的上限 (kyr)。
PRECESSION_MAX_KYR = 30.0

#: 偏心率滤波带的半宽（相对主目标周期）。
ECCENTRICITY_BAND_WIDTH = 0.15

#: 岁差滤波带相对岁差目标周期集合的下限系数。
PRECESSION_BAND_LOW = 0.80

#: 岁差滤波带相对岁差目标周期集合的上限系数。
PRECESSION_BAND_HIGH = 1.25

#: 深度域主峰搜索时允许的最长周期（以记录深度跨度的倍数计）。
MAX_ANCHOR_PERIOD_FRACTION = 3.0

#: 评分函数中偏心率能量占比的权重。
WEIGHT_ECCENTRICITY = 0.30

#: 评分函数中岁差能量占比的权重。
WEIGHT_PRECESSION = 0.15

#: 评分函数中岁差包络偏心率能量占比的权重。
WEIGHT_ENVELOPE = 0.25

#: 评分函数中岁差-偏心率相位一致性（bundle）的权重。
WEIGHT_BUNDLE = 0.30

#: 均匀重采样网格的点数上限。
MAX_RESAMPLE_POINTS = 16384

#: 均匀重采样网格的点数下限。
MIN_RESAMPLE_POINTS = 64

#: 每个岁差周期至少采样点数（决定重采样网格的步长上限）。
SAMPLES_PER_CYCLE = 6

#: 局部随机细化（围绕网格最优解 ±1 格）的采样点数。
N_LOCAL_REFINEMENTS = 8

#: eTimeOpt 的多起点重启次数（不含确定性的起点）。
N_ETIME_RESTARTS = 2

#: eTimeOpt 坐标下降的扫描轮数。
N_ETIME_PASSES = 1

#: eTimeOpt 内部使用的相位网格点数。
ETIME_N_PHASE = 36

#: 米兰科维奇周期表 (kyr)。键名即周期族, 数值取经典的平均轨道周期。
_MILANKOVITCH_PERIODS: dict[str, float] = {
    "eccentricity_100": 100.0,
    "obliquity_41": 41.0,
    "precession_21": 21.0,
    "precession_19": 19.0,
    "precession_23": 23.0,
}


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class TimeOptResult:
    """
    TimeOpt 调谐结果。

    Attributes:
        compression_factors: 每个候选的压缩系数 (无量纲, = S/S_ref)。
        phase_offsets: 每个候选的最优相位偏移 (弧度, ∈ [0, 2π))。
        scores: 每个候选的最优得分。
        best_compression: 最优压缩系数。
        best_phase: 最优压缩系数处的相位偏移 (弧度)。
        best_score: 最优得分。
        tuned_times: 调谐后的时间轴 (kyr), 与输入点一一对应。
    """

    compression_factors: npt.NDArray
    phase_offsets: npt.NDArray
    scores: npt.NDArray
    best_compression: float
    best_phase: float
    best_score: float
    tuned_times: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("TimeOpt 轨道调谐结果"),
            "=" * 50,
            _("候选压缩系数个数: {0}").format(len(self.compression_factors)),
            _("最佳压缩系数: {0:.4f}").format(self.best_compression),
            _("最佳相位偏移: {0:.4f} rad").format(self.best_phase),
            _("最佳得分: {0:.4f}").format(self.best_score),
            _("调谐时间轴点数: {0}").format(len(self.tuned_times)),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ETimeOptResult:
    """
    eTimeOpt（非恒定沉积速率）调谐结果。

    Attributes:
        window_centres: 每个窗口中心在调谐时间轴上的位置 (kyr)。
        compression_factors: 每个窗口的压缩系数 (无量纲)。
        scores: 每个窗口的局部分数。
        tuned_times: 调谐后的时间轴 (kyr), 与输入点一一对应。
    """

    window_centres: npt.NDArray
    compression_factors: npt.NDArray
    scores: npt.NDArray
    tuned_times: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("eTimeOpt 非恒定速率调谐结果"),
            "=" * 50,
            _("窗口数: {0}").format(len(self.compression_factors)),
            _("压缩系数范围: {0:.4f} - {1:.4f}").format(
                float(np.min(self.compression_factors)), float(np.max(self.compression_factors))
            ),
            _("平均得分: {0:.4f}").format(float(np.mean(self.scores))),
            _("调谐时间轴点数: {0}").format(len(self.tuned_times)),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class AgeScaleResult:
    """
    年龄标尺结果。

    Attributes:
        depths: 求值深度。
        ages_ma: 对应年龄 (Ma)。
        age_uncertainty: 对应年龄的 1σ 不确定度 (Ma)。
    """

    depths: npt.NDArray
    ages_ma: npt.NDArray
    age_uncertainty: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("年龄标尺结果"),
            "=" * 50,
            _("深度点数: {0}").format(len(self.depths)),
            _("年龄范围: {0:.4f} - {1:.4f} Ma").format(float(np.min(self.ages_ma)), float(np.max(self.ages_ma))),
            _("平均年龄不确定度 (1σ): {0:.4f} Ma").format(float(np.mean(self.age_uncertainty))),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class _TargetBands:
    """由候选轨道周期导出的滤波带（内部使用）。"""

    primary_kyr: float
    ecc_lo: float
    ecc_hi: float
    pre_lo: float
    pre_hi: float
    precession_kyr: tuple[float, ...]

    @property
    def has_precession(self) -> bool:
        """是否存在可用的岁差周期族。"""
        return len(self.precession_kyr) > 0

    @property
    def min_precession_kyr(self) -> float:
        """岁差族中最短的周期 (kyr)。"""
        return min(self.precession_kyr) if self.precession_kyr else 0.0


# =============================================================================
# 输入校验与工具
# =============================================================================


def _prepare_series(
    depths: npt.NDArray,
    values: npt.NDArray,
    min_points: int = MIN_POINTS,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    校验并整理 (深度, 数值) 序列。

    Returns:
        (升序深度, 升序数值, 升序索引)。升序索引用于把结果映射回输入顺序。
    """
    d = np.asarray(depths, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()

    if d.size != v.size:
        raise ValidationError(
            "Depths and values must have the same length",
            details={"n_depths": int(d.size), "n_values": int(v.size)},
        )
    if d.size < min_points:
        raise ValidationError(
            f"At least {min_points} data points are required for astrochronological tuning",
            details={"n_points": int(d.size), "min_points": int(min_points)},
        )
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(v)):
        raise ValidationError("Depths and values must be finite (no NaN/Inf)")

    diffs = np.diff(d)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValidationError(
            "Depths must be strictly monotonic (either increasing or decreasing)",
            details={"n_points": int(d.size)},
        )

    order = np.argsort(d, kind="stable")
    return d[order], v[order], order


def _resolve_periods(periods_kyr: tuple[float, ...] | None) -> npt.NDArray:
    """解析并校验候选轨道周期, 返回升序去重的数组 (kyr)。"""
    if periods_kyr is None:
        periods = np.array(sorted(_MILANKOVITCH_PERIODS.values()), dtype=float)
    else:
        periods = np.asarray(periods_kyr, dtype=float).ravel()
    if periods.size == 0:
        raise ValidationError("periods_kyr must contain at least one orbital period")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValidationError("periods_kyr must contain finite, strictly positive periods (kyr)")
    return np.unique(periods)


def _target_bands(periods: npt.NDArray) -> _TargetBands:
    """由候选周期导出偏心率带与岁差带（频率单位 cycles/kyr）。"""
    ecc = periods[periods >= ECCENTRICITY_MIN_KYR]
    if ecc.size == 0:
        ecc = np.array([float(np.max(periods))])
        logger.warning(
            "No eccentricity-family period in %s; falling back to the longest period (%.3f kyr) as the "
            "eccentricity target",
            periods.tolist(),
            float(ecc[0]),
        )
    primary = float(np.min(ecc))

    pre = periods[(periods >= PRECESSION_MIN_KYR) & (periods <= PRECESSION_MAX_KYR)]
    if pre.size == 0:
        logger.warning("No precession-family period in %s; the bundle test is disabled", periods.tolist())
        return _TargetBands(
            primary_kyr=primary,
            ecc_lo=1.0 / (primary * (1.0 + ECCENTRICITY_BAND_WIDTH)),
            ecc_hi=1.0 / (primary * (1.0 - ECCENTRICITY_BAND_WIDTH)),
            pre_lo=0.0,
            pre_hi=0.0,
            precession_kyr=(),
        )

    return _TargetBands(
        primary_kyr=primary,
        ecc_lo=1.0 / (primary * (1.0 + ECCENTRICITY_BAND_WIDTH)),
        ecc_hi=1.0 / (primary * (1.0 - ECCENTRICITY_BAND_WIDTH)),
        pre_lo=1.0 / (float(np.max(pre)) * PRECESSION_BAND_HIGH),
        pre_hi=1.0 / (float(np.min(pre)) * PRECESSION_BAND_LOW),
        precession_kyr=tuple(float(p) for p in pre),
    )


def _validate_compression_range(compression_range: tuple[float, float]) -> tuple[float, float]:
    """校验压缩系数区间。"""
    try:
        lo, hi = (float(compression_range[0]), float(compression_range[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValidationError("compression_range must be a (min, max) pair of numbers") from exc
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValidationError("compression_range bounds must be finite")
    if lo <= 0:
        raise ValidationError(
            "compression_range bounds must be strictly positive (a compression factor scales a rate)",
            details={"min": lo, "max": hi},
        )
    if lo >= hi:
        raise ValidationError(
            "compression_range must satisfy min < max",
            details={"min": lo, "max": hi},
        )
    return lo, hi


def _phase_grid(n_phase: int) -> npt.NDArray:
    """构造 [0, 2π) 上的相位扫描网格 (弧度)。"""
    if n_phase < 1:
        raise ValidationError("n_phase must be >= 1", details={"n_phase": int(n_phase)})
    return np.linspace(0.0, 2.0 * np.pi, int(n_phase), endpoint=False)


# =============================================================================
# 谱计算: 均匀时间轴上的 TimeOpt 评分
# =============================================================================


def _score_time_axis(
    times: npt.NDArray,
    values: npt.NDArray,
    bands: _TargetBands,
    phases: npt.NDArray,
) -> tuple[float, float]:
    """
    在给定（升序）时间轴上计算 TimeOpt 得分。

    频率栅格由 FFT 的 ``df = 1/T`` 决定（T 为记录的时间跨度）, 因此它只依赖记录
    长度, 与采样密度无关 —— 加密采样不会让可分辨的周期变差。

    Returns:
        (最优得分, 最优相位偏移/弧度)。输入不可用时返回 ``(-inf, 0.0)``。
    """
    span = float(times[-1] - times[0])
    if span <= 0:
        return -np.inf, 0.0
    dt_data = float(np.median(np.diff(times)))
    if dt_data <= 0:
        return -np.inf, 0.0

    dt_target = dt_data
    if bands.has_precession:
        dt_target = min(dt_target, bands.min_precession_kyr / SAMPLES_PER_CYCLE)
    n_uniform = round(span / dt_target) + 1
    n_uniform = int(np.clip(n_uniform, MIN_RESAMPLE_POINTS, MAX_RESAMPLE_POINTS))
    dt_uniform = span / (n_uniform - 1)
    t_uniform = np.linspace(times[0], times[-1], n_uniform)
    signal = np.interp(t_uniform, times, values)
    signal = signal - float(np.mean(signal))
    if not np.any(signal):
        return -np.inf, 0.0

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_uniform, dt_uniform)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power <= 0:
        return -np.inf, 0.0

    ecc_mask = (freqs >= bands.ecc_lo) & (freqs <= bands.ecc_hi)
    ecc_frac = float(np.sum(power[ecc_mask]) / total_power) if np.any(ecc_mask) else 0.0

    pre_frac = 0.0
    env_ecc_frac = 0.0
    align = np.zeros(len(phases), dtype=float)

    if bands.has_precession:
        pre_mask = (freqs >= bands.pre_lo) & (freqs <= bands.pre_hi)
        if np.any(pre_mask):
            pre_frac = float(np.sum(power[pre_mask]) / total_power)
            precession_signal = np.fft.irfft(spectrum * pre_mask, n_uniform)
            envelope = np.abs(hilbert(precession_signal))
            envelope = envelope - float(np.mean(envelope))

            env_spectrum = np.fft.rfft(envelope)
            env_power = np.abs(env_spectrum) ** 2
            env_total = float(np.sum(env_power))
            if env_total > 0 and np.any(ecc_mask):
                env_ecc_frac = float(np.sum(env_power[ecc_mask]) / env_total)

            env_norm = float(np.linalg.norm(envelope))
            if env_norm > 0:
                omega = 2.0 * np.pi / bands.primary_kyr
                cos_w = np.cos(omega * t_uniform)
                sin_w = np.sin(omega * t_uniform)
                cos_w = cos_w - float(np.mean(cos_w))
                sin_w = sin_w - float(np.mean(sin_w))
                proj_cos = float(np.dot(envelope, cos_w))
                proj_sin = float(np.dot(envelope, sin_w))
                norm_cos = float(np.dot(cos_w, cos_w))
                norm_sin = float(np.dot(sin_w, sin_w))
                norm_cross = float(np.dot(cos_w, sin_w))

                cos_p = np.cos(phases)
                sin_p = np.sin(phases)
                numerator = proj_cos * cos_p - proj_sin * sin_p
                denominator = norm_cos * cos_p**2 - 2.0 * norm_cross * cos_p * sin_p + norm_sin * sin_p**2
                usable = denominator > 0
                align = np.zeros(len(phases), dtype=float)
                align[usable] = numerator[usable] / (env_norm * np.sqrt(denominator[usable]))

    base = WEIGHT_ECCENTRICITY * ecc_frac + WEIGHT_PRECESSION * pre_frac + WEIGHT_ENVELOPE * env_ecc_frac
    total_scores = base + WEIGHT_BUNDLE * align
    best = int(np.argmax(total_scores))
    return float(total_scores[best]), float(phases[best])


# =============================================================================
# 参考速率（锚点）与压缩系数搜索
# =============================================================================


def _dominant_depth_period(depths_asc: npt.NDArray, values_asc: npt.NDArray) -> float | None:
    """
    深度域主峰周期（深度单位）。

    频率栅格沿用 ``stratigraphy.spectral_analysis.SpectralAnalyzer`` 的规则
    （由记录长度决定的 Rayleigh 栅格）。为避免长周期趋势主导锚点, 只在
    "记录深度跨度的 1/3 以内" 的周期区间里找主峰。
    """
    span = float(depths_asc[-1] - depths_asc[0])
    if span <= 0:
        return None
    try:
        spectral = SpectralAnalyzer().analyze(
            depths_asc,
            values_asc,
            n_frequencies=1000,
            detrend=True,
            ar1_significance=False,
        )
    except ComputationError as exc:
        logger.warning("Depth-domain spectrum failed (%s); using the span-based fallback", exc)
        return None

    usable = spectral.periods <= span / MAX_ANCHOR_PERIOD_FRACTION
    if not np.any(usable):
        return None
    masked = np.where(usable, spectral.power, -np.inf)
    best = int(np.argmax(masked))
    period = float(spectral.periods[best])
    if not np.isfinite(period) or period <= 0:
        return None
    return period


def _reference_rate(
    depths_asc: npt.NDArray,
    values_asc: npt.NDArray,
    periods: npt.NDArray,
    bands: _TargetBands,
    phases: npt.NDArray,
) -> float:
    """
    估计参考沉积速率 S_ref（深度单位/kyr）。

    做法: 取深度域主峰周期 P_dom, 分别假设它属于每个候选轨道周期族
    （S = P_dom / T）, 用与搜索完全相同的评分函数选出得分最高者。这样"岁差主导"
    的记录不会被错误地锚到 100 kyr 上。
    """
    span = float(depths_asc[-1] - depths_asc[0])
    period = _dominant_depth_period(depths_asc, values_asc)
    if period is None:
        fallback = span / (10.0 * bands.primary_kyr)
        logger.warning("No usable depth-domain peak; falling back to S_ref = %.6g / kyr", fallback)
        return fallback

    best_rate = period / bands.primary_kyr
    best_score = -np.inf
    for target in periods:
        rate = period / float(target)
        if rate <= 0:
            continue
        times = (depths_asc - depths_asc[0]) / rate
        score, _ = _score_time_axis(times, values_asc, bands, phases)
        if score > best_score:
            best_score = score
            best_rate = rate
    logger.info("Reference rate S_ref = %.6g /kyr (depth-domain peak %.6g, score %.4f)", best_rate, period, best_score)
    return float(best_rate)


def _search_compression(
    depths_asc: npt.NDArray,
    values_asc: npt.NDArray,
    reference_rate: float,
    grid: npt.NDArray,
    bands: _TargetBands,
    phases: npt.NDArray,
    rng: np.random.Generator,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    在压缩系数网格上搜索最佳 (压缩系数, 相位偏移, 得分)。

    Returns:
        (压缩系数数组, 相位偏移数组, 得分数组)。三者一一对应。
    """
    factors: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []

    for factor in grid:
        rate = reference_rate * float(factor)
        times = (depths_asc - depths_asc[0]) / rate
        score, phase = _score_time_axis(times, values_asc, bands, phases)
        factors.append(float(factor))
        offsets.append(phase)
        scores.append(score)

    factor_arr = np.asarray(factors, dtype=float)
    phase_arr = np.asarray(offsets, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    best = int(np.argmax(score_arr))

    # 局部随机细化: 在网格最优解 ±1 格内再取若干候选。细化结果并入同一组数组,
    # 因此 compression_factors / phase_offsets / scores 始终严格一一对应。
    if N_LOCAL_REFINEMENTS > 0 and len(grid) > 1:
        step = float(grid[1] - grid[0])
        low = float(grid[0])
        high = float(grid[-1])
        for candidate in rng.uniform(factor_arr[best] - step, factor_arr[best] + step, N_LOCAL_REFINEMENTS):
            clipped = float(np.clip(candidate, low, high))
            rate = reference_rate * clipped
            times = (depths_asc - depths_asc[0]) / rate
            score, phase = _score_time_axis(times, values_asc, bands, phases)
            factor_arr = np.append(factor_arr, clipped)
            phase_arr = np.append(phase_arr, phase)
            score_arr = np.append(score_arr, score)
            if score > score_arr[best]:
                best = int(len(score_arr) - 1)

    return factor_arr, phase_arr, score_arr


# =============================================================================
# 公开 API: TimeOpt
# =============================================================================


def time_opt(
    depths: npt.NDArray,
    values: npt.NDArray,
    periods_kyr: tuple[float, ...] | None = None,
    compression_range: tuple[float, float] = (0.5, 2.0),
    n_compression: int = 60,
    n_phase: int = 36,
    random_seed: int | None = None,
) -> TimeOptResult:
    """
    TimeOpt 轨道调谐（Meyers 2015）。

    在"压缩系数 × 相位偏移"的二维空间里搜索, 使记录重采样到均匀时间轴后
    偏心率族的能量最大、且岁差调制与偏心率参考周期相位一致（bundle）。

    Parameters:
        depths: 深度轴, 严格单调（升序或降序均可）, 长度与 ``values`` 相同。
        values: 记录数值。
        periods_kyr: 候选轨道周期 (kyr)。``None`` 时使用
            :func:`milankovitch_periods` 的默认集合。
        compression_range: 压缩系数扫描区间 (min, max)。压缩系数是沉积速率的
            倒数比例, 即 S = C · S_ref。
        n_compression: 压缩系数的扫描点数。
        n_phase: 相位偏移的扫描点数（0-2π 上的等分网格）。
        random_seed: 局部随机细化的随机种子。``None`` 时用固定种子 0, 结果可复现。

    Returns:
        TimeOptResult: ``tuned_times`` 是调谐后的时间轴 (kyr), 与输入点一一对应
        （按输入顺序）。
    """
    d_asc, v_asc, order = _prepare_series(depths, values)
    lo, hi = _validate_compression_range(compression_range)
    if n_compression < 2:
        raise ValidationError("n_compression must be >= 2", details={"n_compression": int(n_compression)})
    phases = _phase_grid(n_phase)
    periods = _resolve_periods(periods_kyr)
    bands = _target_bands(periods)

    rng = np.random.default_rng(0 if random_seed is None else int(random_seed))
    reference_rate = _reference_rate(d_asc, v_asc, periods, bands, phases)

    grid = np.linspace(lo, hi, int(n_compression))
    factors, offsets, scores = _search_compression(d_asc, v_asc, reference_rate, grid, bands, phases, rng)
    best = int(np.argmax(scores))

    best_factor = float(factors[best])
    best_phase = float(offsets[best])
    best_score = float(scores[best])

    # 调谐时间轴: 最优速率给出的时间, 再按最优相位整体平移。
    rate = reference_rate * best_factor
    times = (d_asc - d_asc[0]) / rate
    times = times + best_phase * bands.primary_kyr / (2.0 * np.pi)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    tuned_times = times[inverse]

    logger.info(
        "TimeOpt: best compression %.4f, phase %.4f rad, score %.4f (S_ref = %.6g /kyr)",
        best_factor,
        best_phase,
        best_score,
        reference_rate,
    )
    return TimeOptResult(
        compression_factors=factors,
        phase_offsets=offsets,
        scores=scores,
        best_compression=best_factor,
        best_phase=best_phase,
        best_score=best_score,
        tuned_times=tuned_times,
    )


# =============================================================================
# 公开 API: eTimeOpt
# =============================================================================


def _window_bounds(n_points: int, n_windows: int) -> list[tuple[int, int]]:
    """按深度跨度把记录切成 ``n_windows`` 个连续窗口, 返回索引区间 [start, stop)。"""
    edges = np.linspace(0, n_points, n_windows + 1).astype(int)
    bounds: list[tuple[int, int]] = []
    for i in range(n_windows):
        start = int(edges[i])
        stop = int(edges[i + 1])
        if stop - start < 2:  # 至少要两个点才能构成一个区间
            continue
        bounds.append((start, stop))
    return bounds


def _windowed_times(
    depths_asc: npt.NDArray,
    bounds: list[tuple[int, int]],
    rates: npt.NDArray,
) -> npt.NDArray:
    """
    按窗口内各自的速率拼接出一条连续、严格递增的时间轴 (kyr)。

    分段常数沉积速率的物理含义是"年龄随深度连续累积":
    ``dt_i = Δdepth_i / rate(window(i))``。因此时间轴必须**逐段累加**,
    而不能把每个窗口的局部时间各自平移到前一个窗口的末点上 —— 后者会让
    窗口边界上的两个相邻深度共享同一个时间（时间差为 0）, 时间轴不再严格
    递增, 这既违反物理, 也破坏了下游对频率栅格与窗口中心的所有假设。
    """
    rates_per_point = np.empty(depths_asc.size, dtype=float)
    for index, (start, stop) in enumerate(bounds):
        rates_per_point[start:stop] = float(rates[index])

    times = np.empty(depths_asc.size, dtype=float)
    times[0] = 0.0
    times[1:] = np.cumsum(np.diff(depths_asc) / rates_per_point[1:])
    return times


def etime_opt(
    depths: npt.NDArray,
    values: npt.NDArray,
    periods_kyr: tuple[float, ...] | None = None,
    compression_range: tuple[float, float] = (0.5, 2.0),
    n_windows: int = 8,
    n_compression: int = 40,
    random_seed: int | None = None,
) -> ETimeOptResult:
    """
    eTimeOpt 轨道调谐（Meyers 2019）: 允许沉积速率随时间变化。

    与 :func:`time_opt` 的区别在于优化对象: TimeOpt 只搜索一个全局的线性压缩,
    eTimeOpt 把记录切成若干窗口, 每个窗口有自己的压缩系数（等价于分段常数的
    沉积速率）, 并对整条时间轴做坐标下降优化。

    Parameters:
        depths: 深度轴, 严格单调。
        values: 记录数值。
        periods_kyr: 候选轨道周期 (kyr), ``None`` 时用默认集合。
        compression_range: 每个窗口压缩系数的搜索区间 (min, max)。
        n_windows: 窗口数。
        n_compression: 每个窗口压缩系数的扫描点数。
        random_seed: 多起点重启的随机种子。``None`` 时用固定种子 0, 结果可复现。

    Returns:
        ETimeOptResult: ``tuned_times`` 为调谐后的时间轴 (kyr), 与输入点一一对应。
    """
    d_asc, v_asc, order = _prepare_series(depths, values)
    lo, hi = _validate_compression_range(compression_range)
    if n_windows < 1:
        raise ValidationError("n_windows must be >= 1", details={"n_windows": int(n_windows)})
    if n_compression < 2:
        raise ValidationError("n_compression must be >= 2", details={"n_compression": int(n_compression)})

    phases = _phase_grid(ETIME_N_PHASE)
    periods = _resolve_periods(periods_kyr)
    bands = _target_bands(periods)

    rng = np.random.default_rng(0 if random_seed is None else int(random_seed))
    reference_rate = _reference_rate(d_asc, v_asc, periods, bands, phases)
    grid = np.linspace(lo, hi, int(n_compression))

    bounds = _window_bounds(d_asc.size, int(n_windows))
    if not bounds:
        raise ValidationError(
            "n_windows is too large for the number of data points",
            details={"n_points": int(d_asc.size), "n_windows": int(n_windows)},
        )
    n_eff_windows = len(bounds)

    # 先用全局 TimeOpt 的解作为起点: 恒定速率是分段常速率的特例。
    factors, _, scores = _search_compression(d_asc, v_asc, reference_rate, grid, bands, phases, rng)
    global_best = float(factors[int(np.argmax(scores))])

    def evaluate(assignment: npt.NDArray) -> float:
        rates = reference_rate * assignment
        times = _windowed_times(d_asc, bounds, rates)
        score, _ = _score_time_axis(times, v_asc, bands, phases)
        return score

    best_assignment = np.full(n_eff_windows, global_best, dtype=float)
    best_score = evaluate(best_assignment)

    for _ in range(N_ETIME_RESTARTS):
        # 随机起点: 每个窗口从压缩系数网格里随机取一个初值。
        assignment = grid[rng.integers(0, len(grid), size=n_eff_windows)].astype(float)
        for _ in range(N_ETIME_PASSES):
            for window in rng.permutation(n_eff_windows):
                local_best = float(assignment[window])
                local_score = evaluate(assignment)
                for candidate in grid:
                    trial = assignment.copy()
                    trial[window] = float(candidate)
                    score = evaluate(trial)
                    if score > local_score:
                        local_score = score
                        local_best = float(candidate)
                assignment[window] = local_best
        score = evaluate(assignment)
        if score > best_score:
            best_score = score
            best_assignment = assignment.copy()

    rates = reference_rate * best_assignment
    times = _windowed_times(d_asc, bounds, rates)

    window_centres = np.empty(n_eff_windows, dtype=float)
    local_scores = np.empty(n_eff_windows, dtype=float)
    for index, (start, stop) in enumerate(bounds):
        centre_index = (start + stop - 1) // 2
        window_centres[index] = float(times[centre_index])
        window_times = times[start:stop]
        window_values = v_asc[start:stop]
        score, _ = _score_time_axis(window_times, window_values, bands, phases)
        local_scores[index] = score

    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    tuned_times = times[inverse]

    logger.info(
        "eTimeOpt: %d windows, best score %.4f, compression %.4f - %.4f",
        n_eff_windows,
        best_score,
        float(np.min(best_assignment)),
        float(np.max(best_assignment)),
    )
    return ETimeOptResult(
        window_centres=window_centres,
        compression_factors=best_assignment,
        scores=local_scores,
        tuned_times=tuned_times,
    )


# =============================================================================
# 公开 API: 年龄标尺
# =============================================================================


def build_age_scale(
    depths: npt.NDArray,
    anchors: npt.NDArray,
    anchor_ages_ma: npt.NDArray,
    model: str = "linear",
    anchor_uncertainties_ma: npt.NDArray | None = None,
) -> AgeScaleResult:
    """
    由锚点构建年龄标尺（深度 → 年龄）。

    插值口径与 ``stratigraphy.correlation.AgeModelAnalyzer`` 保持一致:
    ``"linear"`` 为分段线性（区间外按端点斜率线性外推）, ``"pchip"`` 为单调
    三次 Hermite 插值（PCHIP, 稀疏节点间不振荡, 不会出现负沉积速率）。

    不确定度沿用同一套"深度可变 CI"口径: 反距离加权混合各锚点的不确定度,
    并在锚点区间之外按距离惩罚放大（上限 3 倍最大锚点误差）。

    Parameters:
        depths: 求值深度。
        anchors: 锚点深度, 至少 2 个且严格递增（内部会排序）。
        anchor_ages_ma: 锚点年龄 (Ma), 与 ``anchors`` 一一对应。
        model: ``"linear"`` / ``"piecewise"``（分段线性）或 ``"pchip"`` / ``"spline"``。
        anchor_uncertainties_ma: 各锚点的 1σ 年龄不确定度。``None`` 时由锚点相对
            全局线性拟合的残差（MAD）估计 —— 锚点完全共线时该估计为 0。

    Returns:
        AgeScaleResult: 逐深度的年龄 (Ma) 与 1σ 不确定度 (Ma)。
    """
    d = np.asarray(depths, dtype=float).ravel()
    a = np.asarray(anchors, dtype=float).ravel()
    ages = np.asarray(anchor_ages_ma, dtype=float).ravel()

    if d.size == 0:
        raise ValidationError("depths must not be empty")
    if a.size != ages.size:
        raise ValidationError(
            "anchors and anchor_ages_ma must have the same length",
            details={"n_anchors": int(a.size), "n_ages": int(ages.size)},
        )
    if a.size < 2:
        raise ValidationError(
            "An age scale needs at least 2 anchors",
            details={"n_anchors": int(a.size)},
        )
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(ages)):
        raise ValidationError("depths, anchors and anchor_ages_ma must be finite (no NaN/Inf)")
    if np.any(np.diff(np.sort(a)) <= 0):
        raise ValidationError("anchors must be distinct")

    model_key = str(model).strip().lower()
    if model_key in ("linear", "piecewise"):
        use_pchip = False
    elif model_key in ("pchip", "spline"):
        use_pchip = True
    else:
        raise ValidationError(
            "model must be one of linear, piecewise, pchip, spline",
            details={"model": model},
        )

    order = np.argsort(a, kind="stable")
    anchor_depths = a[order]
    anchor_ages = ages[order]

    if use_pchip:
        from scipy.interpolate import PchipInterpolator

        interpolator = PchipInterpolator(anchor_depths, anchor_ages, extrapolate=True)
        modelled = np.asarray(interpolator(d), dtype=float)
    else:
        modelled = np.interp(d, anchor_depths, anchor_ages)
        slope_low = (anchor_ages[1] - anchor_ages[0]) / (anchor_depths[1] - anchor_depths[0])
        slope_high = (anchor_ages[-1] - anchor_ages[-2]) / (anchor_depths[-1] - anchor_depths[-2])
        below = d < anchor_depths[0]
        above = d > anchor_depths[-1]
        if np.any(below):
            modelled[below] = anchor_ages[0] + slope_low * (d[below] - anchor_depths[0])
        if np.any(above):
            modelled[above] = anchor_ages[-1] + slope_high * (d[above] - anchor_depths[-1])

    if anchor_uncertainties_ma is None:
        slope, intercept = np.polyfit(anchor_depths, anchor_ages, 1)
        residuals = anchor_ages - (slope * anchor_depths + intercept)
        sigma = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
        anchor_sigma = np.full(anchor_depths.size, sigma, dtype=float)
    else:
        sigma_arr = np.asarray(anchor_uncertainties_ma, dtype=float).ravel()
        if sigma_arr.size != a.size:
            raise ValidationError(
                "anchor_uncertainties_ma must have the same length as anchors",
                details={"n_uncertainties": int(sigma_arr.size), "n_anchors": int(a.size)},
            )
        if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr < 0):
            raise ValidationError("anchor_uncertainties_ma must be finite and non-negative")
        anchor_sigma = sigma_arr[order]

    uncertainty = _depth_varying_half_width(d, anchor_depths, anchor_sigma, z_score=1.0)

    logger.info(
        "Age scale built: %d depths, %d anchors, model=%s, age range %.4f-%.4f Ma",
        d.size,
        anchor_depths.size,
        model_key,
        float(np.min(modelled)),
        float(np.max(modelled)),
    )
    return AgeScaleResult(depths=d, ages_ma=modelled, age_uncertainty=uncertainty)


# =============================================================================
# 公开 API: 沉积速率 → 年龄模型
# =============================================================================


def sedimentation_rate_to_age_model(
    depths: npt.NDArray,
    sedimentation_rate: float,
    datum_age_ma: float = 0.0,
) -> npt.NDArray:
    """
    由恒定沉积速率构建年龄轴。

    积分规则与 Acycle 的 ``acycleSedimentationRateAgeModel`` 一致（分段常数速率、
    左端点取值）, 但按 PaleoAST 的 Ma 口径重写, 单位是 **深度单位/Myr**。

    Parameters:
        depths: 深度轴, 严格单调, 至少 2 个点。
        sedimentation_rate: 沉积速率, 单位为深度单位/Myr, 必须为正。
            （Acycle 用 cm/kyr: 1 cm/kyr = 100 深度单位/Myr。）
        datum_age_ma: 第一个深度点的年龄 (Ma)。

    Returns:
        npt.NDArray: 与 ``depths`` 等长的年龄数组 (Ma)。年龄沿深度增大的方向增大
        （越深越老）。
    """
    d = np.asarray(depths, dtype=float).ravel()
    if d.size < 2:
        raise ValidationError(
            "At least 2 depth points are required to build an age model",
            details={"n_points": int(d.size)},
        )
    if not np.all(np.isfinite(d)):
        raise ValidationError("depths must be finite (no NaN/Inf)")
    diffs = np.diff(d)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValidationError(
            "depths must be strictly monotonic (either increasing or decreasing)",
            details={"n_points": int(d.size)},
        )
    try:
        rate = float(sedimentation_rate)
    except (TypeError, ValueError) as exc:
        raise ValidationError("sedimentation_rate must be a number") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValidationError(
            "sedimentation_rate must be finite and strictly positive",
            details={"sedimentation_rate": rate},
        )
    try:
        datum = float(datum_age_ma)
    except (TypeError, ValueError) as exc:
        raise ValidationError("datum_age_ma must be a number") from exc
    if not np.isfinite(datum):
        raise ValidationError("datum_age_ma must be finite", details={"datum_age_ma": datum})

    ages = datum + (d - d[0]) / rate
    logger.info(
        "Sedimentation-rate age model: %d points, rate %.6g /Myr, age %.4f-%.4f Ma",
        d.size,
        rate,
        float(np.min(ages)),
        float(np.max(ages)),
    )
    return ages


# =============================================================================
# 公开 API: 米兰科维奇周期表
# =============================================================================


def milankovitch_periods() -> dict[str, float]:
    """
    返回米兰科维奇轨道周期表 (kyr)。

    包含偏心率 (~100 kyr)、倾角 (~41 kyr) 与岁差 (~21 kyr, 以及 19/23 kyr 的
    短周期项)。这些是构造合成目标曲线用的经典平均轨道周期, 不是 Laskar et al.
    (2004) 的数值解。

    Returns:
        dict[str, float]: 周期族名称 → 周期 (kyr)。
    """
    return dict(_MILANKOVITCH_PERIODS)
