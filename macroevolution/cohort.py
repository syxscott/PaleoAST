"""
================================================================================
PaleoAST Macroevolution - Cohort Survivorship Analysis
================================================================================

本模块实现Foote (1997, 2000)的边界交叉法存活分析。

数学理论:
================================================================================

1. 问题定义
--------------------------------------------------------------------------------
给定地层剖面中化石记录的时间区间 [t₁, t₂]，计算：
- 起步率 (Origination rate, λ)
- 灭绝率 (Extinction rate, μ)
- 存活概率 (Survival probability)

2. 边界交叉者 (Boundary Crosser) 方法
--------------------------------------------------------------------------------
Foote (2000) 定义:

a) 起始边界交叉者 (FB → LA):
    在区间底部之前起源，顶部仍然存活
    N_FB = 采样剖面中满足 o_i < t₁ 且 L_i > t₂ 的物种数

b) 终止边界交叉者 (LB → FA):
    在区间底部仍然存活，顶部之后灭绝
    N_LB = 采样剖面中满足 L_i < t₂ 且 o_i > t₁ 的物种数

c) 全部存活者 (Total Survivors):
    N_surv = 区间内始终存活的物种数

3. 存活分析
--------------------------------------------------------------------------------
使用二项分布模型:

    令 p = 存活概率

    观测: N_surv 存活，N_FB + N_LB 死亡

    极大似然估计:
        p̂ = N_surv / (N_surv + N_FB + N_LB)

    方差:
        Var(p̂) = p̂(1-p̂) / (N_surv + N_FB + N_LB)

4. Foote公式
--------------------------------------------------------------------------------
对于恒定出生-死亡过程:

    p = (1 - q) / (1 - q^(n+1))

其中:
    q = μ/λ (灭绝/出生比率)
    n = 区间内的年龄区间数

5. 边缘存活分析
--------------------------------------------------------------------------------
每个分类单元的边缘存活:

    φ_i = I(L_i > t₂) / I(o_i < t₁)

其中 I 是指示函数。

6. 置信区间
--------------------------------------------------------------------------------
使用Wilson score interval:

    CI = (p̂ + z²/2N ± z√(p̂(1-p̂)/N + z²/4N²)) / (1 + z²/N)

其中 z 是标准正态分位数。

7. 拉扎勒斯效应 (Lazarus Taxa)
--------------------------------------------------------------------------------
突然消失又出现的分类单元:

    N_Lazarus = 预期数 - 观测数

8. 多区间分析
--------------------------------------------------------------------------------
对连续时间区间序列:

    λ_i = -ln(S_i) / Δt_i
    μ_i = -ln(O_i) / Δt_i

其中 S_i 是存活比例，O_i 是灭绝比例。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class IntervalData(NamedTuple):
    """时间区间数据"""

    t_start: float
    t_end: float
    n_fb: int  # 起始边界交叉者 (Foote 2000): 区间内起源, 存活到区间后
    n_lb: int  # 终止边界交叉者 (Foote 2000): 区间前已存在, 区间内灭绝
    n_surv: int  # 存活者: 区间前已存在, 存活到区间后
    n_total: int  # 总数
    # Foote 1997 cohort variables
    n_bt: int  # 向后存续: 在C中存在且在C之前已知的分类单元数
    n_bl: int  # 向后灭绝: 在C中首次出现且在C之前未知的分类单元数
    n_ft: int  # 向前存续: 在C中存活到其后的分类单元数
    n_fl: int  # 向前灭绝: 在C中首次出现且在C之后灭绝的分类单元数


@dataclass
class SurvivorshipResult:
    """
    存活分析结果

    属性:
        intervals: 时间区间列表
        survival_rates: 存活率
        origination_rates: 起步率
        extinction_rates: 灭绝率
        confidence_intervals: 置信区间
        extinction_probs: 灭绝概率
        # Foote 1997 cohort rates
        foote97_origination: np.ndarray  # Foote 1997 p = -ln(N_bt/N_t)/Δt
        foote97_extinction: np.ndarray   # Foote 1997 q = -ln(N_bL/N_t)/Δt
        foote00_origination: np.ndarray  # Foote 2000 p_F = N_Ft/N_t
        foote00_extinction: np.ndarray   # Foote 2000 q_F = N_FL/N_t
    """

    intervals: list[IntervalData]
    survival_rates: np.ndarray
    origination_rates: np.ndarray
    extinction_rates: np.ndarray
    confidence_intervals: list[tuple[float, float]]
    extinction_probs: np.ndarray
    foote97_origination: np.ndarray = None
    foote97_extinction: np.ndarray = None
    foote00_origination: np.ndarray = None
    foote00_extinction: np.ndarray = None

    def __post_init__(self):
        if self.foote97_origination is None:
            self.foote97_origination = np.zeros(len(self.intervals))
        if self.foote97_extinction is None:
            self.foote97_extinction = np.zeros(len(self.intervals))
        if self.foote00_origination is None:
            self.foote00_origination = np.zeros(len(self.intervals))
        if self.foote00_extinction is None:
            self.foote00_extinction = np.zeros(len(self.intervals))

    def get_rate_ratio(self) -> np.ndarray:
        """获取λ/μ比率"""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.origination_rates / self.extinction_rates
            ratio[np.isinf(ratio)] = np.nan
        return ratio


class CohortSurvivorshipAnalysis:
    """
    边界交叉法存活分析

    实现Foote的存活分析算法。

    使用示例:
        >>> analysis = CohortSurvivorshipAnalysis()
        >>>
        >>> # 输入化石记录: (起源时间, 灭绝时间)
        >>> records = [
        ...     (10.0, 5.0),  # 存活于5-10Ma
        ...     (8.0, 3.0),
        ...     (12.0, 6.0),
        ... ]
        >>>
        >>> intervals = [(0, 5), (5, 10)]
        >>> result = analysis.analyze(records, intervals)
        >>> print(result.survival_rates)
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        初始化存活分析

        参数:
            confidence_level: 置信水平
        """
        self._conf_level = confidence_level
        self._z = stats.norm.ppf((1 + confidence_level) / 2)
        self._logger = logging.getLogger(f"{__name__}.CohortSurvivorship")

    def analyze(
        self, fossil_records: list[tuple[float, float]], intervals: list[tuple[float, float]]
    ) -> SurvivorshipResult:
        """
        执行存活分析

        参数:
            fossil_records: 化石记录列表，每个 (起源时间, 灭绝时间)
            intervals: 时间区间列表，每个 (起始, 终止)

        返回:
            SurvivorshipResult对象

        注: 时间从新到老递减，如 5.0 Ma 表示5百万年前
        """
        records = [(float(o), float(L)) for o, L in fossil_records]
        intervals = [(float(t1), float(t2)) for t1, t2 in intervals]

        self._logger.info(f"Analyzing {len(records)} records across {len(intervals)} intervals")

        interval_data_list = []
        survival_rates = np.zeros(len(intervals))
        origination_rates = np.zeros(len(intervals))
        extinction_rates = np.zeros(len(intervals))
        confidence_intervals = []
        extinction_probs = np.zeros(len(intervals))
        # Foote 1997 cohort rates
        foote97_origination = np.zeros(len(intervals))
        foote97_extinction = np.zeros(len(intervals))
        foote00_origination = np.zeros(len(intervals))
        foote00_extinction = np.zeros(len(intervals))

        for i, (t_start, t_end) in enumerate(intervals):
            # 统计边界交叉者 (Foote 2000)
            n_fb = 0  # 起始边界交叉者: o < t_start, L > t_end
            n_lb = 0  # 终止边界交叉者: L < t_end, o > t_start
            n_surv = 0  # 存活者: o < t_start, L > t_end

            # Foote 1997 cohort counts
            # N_bt = 向后存续: 在C中存在且在C之前已知
            # N_bL = 向后灭绝: 在C中首次出现且在C之前未知
            # N_Ft = 向前存续: 在C中存活到其后
            # N_FL = 向前灭绝: 在C中首次出现且在C之后灭绝
            n_bt = 0  # backward persistence
            n_bl = 0  # backward extinction (originated in interval)
            n_ft = 0  # forward persistence (survived past interval)
            n_fl = 0  # forward extinction

            for o, L in records:
                # Check temporal relationships
                started_before = o < t_start
                started_in = t_start <= o < t_end
                started_after = o >= t_end
                ended_before = L < t_start
                ended_in = t_start <= L < t_end
                ended_after = L >= t_end

                if started_before and ended_after:
                    # Through-timer: existed before interval, survived past interval
                    n_surv += 1
                    n_bt += 1  # Backward persistence
                    n_ft += 1  # Forward persistence
                elif started_before and ended_in:
                    # Existed before, went extinct during interval
                    n_lb += 1
                    n_bt += 1  # Backward persistence
                    n_fl += 1  # Forward extinction
                elif started_in and ended_after:
                    # Originated in interval, survived past
                    n_fb += 1
                    n_bl += 1  # Backward extinction
                    n_ft += 1  # Forward persistence
                elif started_in and ended_in:
                    # Originated and went extinct in same interval
                    n_bl += 1  # Backward extinction
                    n_fl += 1  # Forward extinction
                elif started_before and ended_before:
                    # Entirely before interval - not counted
                    pass
                elif started_after and ended_after:
                    # Entirely after interval - not counted
                    pass

            n_total = n_fb + n_lb + n_surv

            # N_t = total taxa in cohort (appearing in interval)
            # = n_bt + n_bl = n_ft + n_fl
            n_t = n_bt + n_bl

            interval_data_list.append(
                IntervalData(
                    t_start=t_start, t_end=t_end,
                    n_fb=n_fb, n_lb=n_lb, n_surv=n_surv, n_total=n_total,
                    n_bt=n_bt, n_bl=n_bl, n_ft=n_ft, n_fl=n_fl
                )
            )

            if n_total > 0:
                # 存活率
                p = n_surv / n_total
                survival_rates[i] = p
                extinction_probs[i] = 1 - p

                # Wilson置信区间
                z = self._z
                n = n_total
                center = p + z**2 / (2 * n)
                width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

                ci_lower = (center - width) / (1 + z**2 / n)
                ci_upper = (center + width) / (1 + z**2 / n)
                confidence_intervals.append((ci_lower, ci_upper))

                # 起源率 λ 与灭绝率 μ (Foote 1999 per-capita rates)
                dt = t_start - t_end
                if dt > 0:
                    # 修正: 公式正确,origination = -ln(1-p)/dt, extinction = -ln(p)/dt
                    if p < 1:
                        origination_rates[i] = -np.log(1 - p) / dt
                    else:
                        # p = 1 ⇒ 无人起源 ⇒ λ = 0
                        origination_rates[i] = 0.0
                    if p > 0:
                        extinction_rates[i] = -np.log(p) / dt
                    else:
                        # p = 0 ⇒ 无人存活 ⇒ μ → ∞
                        extinction_rates[i] = float("inf")

                    # Foote 1997 cohort rates (Marshall 1990 style)
                    # p = -ln(N_bt / N_t) / Δt
                    # q = -ln(N_bL / N_t) / Δt
                    if n_t > 0 and n_bt > 0:
                        foote97_origination[i] = -np.log(n_bt / n_t) / dt
                    else:
                        foote97_origination[i] = np.nan
                    if n_t > 0 and n_bl > 0:
                        foote97_extinction[i] = -np.log(n_bl / n_t) / dt
                    else:
                        foote97_extinction[i] = np.nan

                    # Foote 2000 simplified rates (without sampling correction)
                    # p_F = N_Ft / N_t
                    # q_F = N_FL / N_t
                    if n_t > 0:
                        foote00_origination[i] = n_ft / n_t
                        foote00_extinction[i] = n_fl / n_t
            else:
                survival_rates[i] = np.nan
                extinction_probs[i] = np.nan
                origination_rates[i] = np.nan
                extinction_rates[i] = np.nan
                foote97_origination[i] = np.nan
                foote97_extinction[i] = np.nan
                foote00_origination[i] = np.nan
                foote00_extinction[i] = np.nan
                confidence_intervals.append((np.nan, np.nan))

        return SurvivorshipResult(
            intervals=interval_data_list,
            survival_rates=survival_rates,
            origination_rates=origination_rates,
            extinction_rates=extinction_rates,
            confidence_intervals=confidence_intervals,
            extinction_probs=extinction_probs,
            foote97_origination=foote97_origination,
            foote97_extinction=foote97_extinction,
            foote00_origination=foote00_origination,
            foote00_extinction=foote00_extinction,
        )

    def foote_analysis(self, n_surv: int, n_total: int, dt: float) -> dict[str, float]:
        """
        Foote (1997) 的边缘存活分析

        参数:
            n_surv: 存活者数量
            n_total: 总数
            dt: 时间间隔

        返回:
            分析结果字典
        """
        if n_total == 0:
            return {
                "survival_prob": np.nan,
                "extinction_prob": np.nan,
                "origination_rate": np.nan,
                "extinction_rate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }

        p = n_surv / n_total

        # 置信区间
        z = self._z
        n = n_total
        center = p + z**2 / (2 * n)
        width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

        ci_lower = (center - width) / (1 + z**2 / n)
        ci_upper = (center + width) / (1 + z**2 / n)

        # 速率：与 analyze() 保持一致 (Foote 1999)
        #   λ = -ln(1 - p) / Δt   (起源率)
        #   μ = -ln(p)     / Δt   (灭绝率)
        # 旧实现把两式互换，导致返回字典里的 "origination_rate"
        # 实为 μ、"extinction_rate" 实为 λ。
        if dt > 0:
            if p < 1:
                lambda_rate = -np.log(1 - p) / dt
            else:
                lambda_rate = 0.0  # p=1 ⇒ 无人起源
            if p > 0:
                extinction_rate = -np.log(p) / dt
            else:
                extinction_rate = float("inf")  # p=0 ⇒ 全部灭绝
        else:
            lambda_rate = np.nan
            extinction_rate = np.nan

        return {
            "survival_prob": p,
            "extinction_prob": 1 - p,
            "origination_rate": lambda_rate,
            "extinction_rate": extinction_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def per_capita_rates(self, survival_rate: float, dt: float) -> tuple[float, float]:
        """
        计算人均出生/灭绝率

        参数:
            survival_rate: 存活率 p
            dt: 时间间隔

        返回:
            (λ, μ)
        """
        if dt <= 0 or survival_rate <= 0 or survival_rate >= 1:
            return np.nan, np.nan

        # Foote (1999) per-capita rates, swapped to correct labels.
        #   λ (origination) = -ln(1 - p) / Δt
        #   μ (extinction)  = -ln(p)     / Δt
        # 旧实现的两式相互颠倒，调用方拿到 (λ, μ) 时实际收到的是 (μ, λ)。
        lambda_rate = -np.log(1 - survival_rate) / dt
        extinction_rate = -np.log(survival_rate) / dt

        return lambda_rate, extinction_rate

    def test_equilibrium(self, origination_rate: float, extinction_rate: float) -> tuple[float, float]:
        """
        检验是否处于平衡态

        参数:
            origination_rate: 起步率 λ
            extinction_rate: 灭绝率 μ

        返回:
            (比率 λ/μ, p值)
        """
        if extinction_rate <= 0:
            return np.inf, 0.0

        ratio = origination_rate / extinction_rate

        # 平衡态检验: H0: λ = μ
        # 使用z检验
        se = np.sqrt(origination_rate**2 + extinction_rate**2)

        if se > 0:
            z_stat = (origination_rate - extinction_rate) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0

        return ratio, p_value


def analyze_cohort_survivorship(
    fossil_records: list[tuple[float, float]], intervals: list[tuple[float, float]], confidence_level: float = 0.95
) -> SurvivorshipResult:
    """
    存活分析的便捷函数

    参数:
        fossil_records: 化石记录
        intervals: 时间区间
        confidence_level: 置信水平

    返回:
        SurvivorshipResult
    """
    analysis = CohortSurvivorshipAnalysis(confidence_level)
    return analysis.analyze(fossil_records, intervals)
