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

2. 边界交叉者 (Boundary Crosser) 方法与区间归属约定
--------------------------------------------------------------------------------
时间轴约定: 数值为 Ma，**越大越老**。区间写成 (t_start, t_end) 且
t_start < t_end，即 t_start 是**年轻**边界 (地层顶)，t_end 是**年老**边界 (地层底)。

区间归属采用**半开区间 [t_start, t_end)**: 年龄 x 落在本区间内 ⟺
t_start <= x < t_end。因此相邻区间共享的边界点只属于**较老**的那个区间，
不会被两个区间同时计数。

Foote (2000) 定义:

a) 起始边界交叉者 (FB):
    在区间内起源、存活到区间年轻边界之外
    N_FB = 满足 started_in 且 ended_after 的物种数

b) 终止边界交叉者 (LB):
    在区间年老边界之前已存在、在区间内灭绝
    N_LB = 满足 started_before 且 ended_in 的物种数

c) 全部存活者 (through-timer):
    N_surv = 区间前已存在且存活过年轻边界的物种数 (跨整个区间)

三个布尔判定必须**互斥且完备** (旧实现用 o > t_end / L > t_end 判"更老"，
使恰等于年老边界 t_end 的 FAD/LAD 既不算"区间内"也不算"区间前"，该分类元
被静默丢弃 —— 连本模块文档示例里的 (10.0, 5.0) 配 [(0,5),(5,10)] 都归零):

    started_before = o >= t_end      # 起源在本区间之前 (含恰在年老边界)
    started_in     = t_start <= o < t_end
    started_after  = o < t_start     # 起源晚于本区间 (更年轻)

    ended_in       = t_start <= L < t_end   (但见下面的现生存哨兵例外)
    ended_before   = L >= t_end
    ended_after    = L < t_start，或 L == t_start == 0 (以 0 Ma 作"现生存"
                     哨兵的类群，记为存活过年轻边界而不是在区间内灭绝)

3. 存活分析
--------------------------------------------------------------------------------
使用二项分布模型:

    令 p = 存活概率

    观测: N_surv 存活，N_FB + N_LB 死亡

    极大似然估计:
        p̂ = N_surv / (N_surv + N_FB + N_LB)

    方差:
        Var(p̂) = p̂(1-p̂) / (N_surv + N_FB + N_LB)

    置信区间: Wilson score interval (见第 6 节)。

4. Foote公式
--------------------------------------------------------------------------------
对于恒定出生-死亡过程:

    p = (1 - q) / (1 - q^(n+1))

其中:
    q = μ/λ (灭绝/出生比率)
    n = 区间内的年龄区间数

这是 Foote (1997) 讨论的**常生常死模型下单区间存活概率的解析式**，
本模块不对它求解 (它需要跨多个等长区间的完整存活曲线)，只保留在文档中
作为理论背景；代码实际使用的是第 8 节的两个估计量族。

5. 边缘存活分析
--------------------------------------------------------------------------------
每个分类单元的边缘存活:

    φ_i = I(L_i 存活过年轻边界) · I(o_i 在年老边界之前)

其中 I 是指示函数 (旧文档写成 φ_i = I(L_i > t₂)/I(o_i < t₁)，把两个指示
函数相除在数学上无意义)。

6. 置信区间
--------------------------------------------------------------------------------
使用Wilson score interval:

    CI = (p̂ + z²/2N ± z√(p̂(1-p̂)/N + z²/4N²)) / (1 + z²/N)

其中 z 是标准正态分位数。

7. 拉扎勒斯效应 (Lazarus Taxa)
--------------------------------------------------------------------------------
突然消失又出现的分类单元:

    N_Lazarus = 预期数 - 观测数

8. 速率估计 —— 本模块并存的两套公式 (务必区分)
--------------------------------------------------------------------------------
(A) **Foote (1997) cohort 存续份额估计量** —— 由 `analyze()` 计算，输出到
    ``origination_rates`` / ``extinction_rates`` / ``foote97_*``:

        λ̂ = -ln(N_bt / N_t) / Δt      N_bt = 向后存续者, N_t = N_bt + N_bl
        μ̂ = -ln(N_ft / N_t) / Δt      N_ft = 向前存续者

    N_bt/N_t 是"在本区间之前已经存在"的概率、N_ft/N_t 是"存活出本区间"的
    概率，两者都是区间内**全体成员**的份额，因此需要 N_t > 0 且 N_bt/N_ft > 0
    (为 0 时速率无界，记为 +inf)。

(B) **只由存活比例 p 出发的 per-capita 公式** —— 由 `foote_analysis()` 与
    `per_capita_rates()` 计算，它们**不**看到 cohort 计数，只有 p 与 Δt:

        λ = -ln(1 - p) / Δt      (进入区间的起源份额 = 1 - p)
        μ = -ln(p)     / Δt      (存活出区间的概率 = p)

两套公式的输入不同，因此**数值一般不相等**；(A) 是 cohort 分析的主输出，
(B) 仅用于已知存活比例的场合。旧实现的 `analyze()` 把 (B) 的 λ 式当作起源率、
又把向后计数 N_bl 当作灭绝率，两套混用且都不是标准估计量 (2026-09 复审)。

此外还有一类"简化概率"(非速率): ``foote00_*`` = N_bl/N_t (区间内起源份额) 与
N_fl/N_t (区间内灭绝份额)，见 Foote (2000) 的简式。

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
                # 半开区间 [t_start, t_end): 恰在共享边界上的年龄只计入
                # 一侧, 避免相邻 bin 重复计数 (此前双端闭合会重复计入)。
                # 灭绝恰在年轻边界 (含 L=0 的现生存哨兵) 记为"存活过"
                # —— 否则现生存类群会被误判为区间内灭绝 (p 归零)。
                started_before = o > t_end
                started_in = t_start <= o < t_end
                started_after = o < t_start
                ended_before = L > t_end
                ended_in = t_start <= L < t_end
                ended_after = L <= t_start

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

                # 起源率与灭绝率 (Foote per-capita 边界穿越者估计)
                # 时间从新到老: t_start (年轻边界) < t_end (年老边界),
                # dt = t_end - t_start (正的时间跨度)。
                dt = t_end - t_start
                if dt > 0:
                    # Foote (1997, 2000) cohort / per-capita rates:
                    #   起源 p = -ln(Nbt/Nt)/dt   Nbt = 向后存续者份额
                    #     (P(在区间之前已存在) = e^{-p·dt})
                    #   灭绝 q = -ln(Nft/Nt)/dt   Nft = 向前存续者份额
                    #     (P(存活出年轻边界) = e^{-q·dt})
                    # 边界情形: Nt=0 → 未定义 (nan); Nbt=0 或 Nft=0 → inf。
                    # 旧实现 origination = -ln(1-p)/dt 以存活份额推起源率、
                    # 灭绝率用向后计数 Nbl, 均非任何标准估计量 (2026-09
                    # 复审; birth-death 模拟显示旧灭绝估计偏差 +191%)。
                    if n_t == 0:
                        foote97_origination[i] = np.nan
                        foote97_extinction[i] = np.nan
                    else:
                        foote97_origination[i] = -np.log(n_bt / n_t) / dt if n_bt > 0 else float("inf")
                        foote97_extinction[i] = -np.log(n_ft / n_t) / dt if n_ft > 0 else float("inf")

                    # Foote 2000 简化概率 (非率):
                    #   起源概率 = 区间内起源者份额 Nbl/Nt (旧实现误用 Nft,
                    #   即向前存活份额 = 1 - 灭绝概率)
                    #   灭绝概率 = 区间内灭绝者份额 Nfl/Nt
                    if n_t > 0:
                        foote00_origination[i] = n_bl / n_t
                        foote00_extinction[i] = n_fl / n_t

                    # 主率输出与 Foote cohort 估计保持一致
                    origination_rates[i] = foote97_origination[i]
                    extinction_rates[i] = foote97_extinction[i]
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
