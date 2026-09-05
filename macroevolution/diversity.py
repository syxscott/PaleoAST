"""
================================================================================
PaleoAST Macroevolution - Diversity Dynamics
================================================================================

本模块实现多样性动态建模和可视化。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiversityCurve:
    """多样性曲线数据"""

    times: np.ndarray
    richness: np.ndarray
    origination_rates: np.ndarray
    extinction_rates: np.ndarray
    turnover_rate: np.ndarray

    @property
    def n_intervals(self) -> int:
        """区间数量"""
        return len(self.times)


class DiversityDynamics:
    """
    多样性动态建模

    分析和模拟生物多样性随时间的变化。

    使用示例:
        >>> dyn = DiversityDynamics()
        >>>
        >>> # 从化石记录估计多样性
        >>> records = [(origination, extinction), ...]
        >>> curve = dyn.estimate_diversity(records, intervals)
        >>>
        >>> # 拟合随机过程
        >>> params = dyn.fit_geometric_brownian(curve)
    """

    def __init__(self):
        """初始化多样性动态分析"""
        self._logger = logging.getLogger(f"{__name__}.DiversityDynamics")

    def estimate_diversity(
        self, fossil_records: list[tuple[float, float]], intervals: list[tuple[float, float]]
    ) -> DiversityCurve:
        """
        从化石记录估计多样性曲线

        时间采用 Ma 约定: 数值越大越老。区间 (t_start, t_end) 中
        t_start 为年轻边界, t_end 为年老边界 (t_start <= t_end,
        输入顺序相反时自动纠正)。分类单元 (起源年龄 o, 灭绝年龄 L):
        o >= L, 输入顺序相反时自动纠正。

        参数:
            fossil_records: 化石记录 [(起源时间, 灭绝时间), ...]
            intervals: 时间区间

        返回:
            DiversityCurve (origination/extinction_rates 为 Foote 式
            边界穿越者 per-capita 估计, 相邻区间无重叠数据时为 nan)
        """
        self._logger.info(
            f"Estimating diversity from {len(fossil_records)} fossil records across {len(intervals)} intervals"
        )

        # 统一约定: o = 起源年龄 (更老), L = 灭绝年龄 (更年轻)
        records = []
        for o, L in fossil_records:
            o, L = float(o), float(L)
            if o < L:
                o, L = L, o
            records.append((o, L))

        # 统一区间方向: t_start 年轻边界 <= t_end 年老边界
        norm_intervals = []
        for t_start, t_end in intervals:
            t_start, t_end = float(t_start), float(t_end)
            if t_start > t_end:
                t_start, t_end = t_end, t_start
            norm_intervals.append((t_start, t_end))

        # 存在性 = 寿命区间 [L, o] 与时间区间 [t_start, t_end] 有重叠:
        #   o >= t_start (起源不晚于年老侧) 且 L <= t_end (灭绝不早于年轻侧)。
        # 此前条件 o <= t_start 且 L >= t_end 在 Ma 约定下几何上几乎
        # 不可满足, 导致 richness 恒为 0 (2026-09 复审)。
        def _present(idx: int) -> set[int]:
            t_s, t_e = norm_intervals[idx]
            return {k for k, (o, L) in enumerate(records) if o >= t_s and L <= t_e}

        present = [_present(i) for i in range(len(norm_intervals))]

        times = []
        richness = []
        origination_rates = []
        extinction_rates = []

        # 区间按从年轻到老的输入顺序处理: i-1 为更年轻邻区, i+1 为更老邻区
        for i, (t_start, t_end) in enumerate(norm_intervals):
            times.append((t_start + t_end) / 2)
            richness.append(len(present[i]))

            dt = t_end - t_start
            n_t = len(present[i])

            # Foote (1999, 2000) per-capita 率 (边界穿越者估计):
            #   起源 p = -ln(Nbt/Nt)/dt, Nbt = 本区与更老邻区均出现
            #   灭绝 q = -ln(Nft/Nt)/dt, Nft = 本区与更年轻邻区均出现
            # 此前的实现把净变化 dR 整体记为起源或灭绝, 在起源与灭绝
            # 并发时两者都被掩盖, 不是任何 per-capita 估计量。
            if i + 1 < len(norm_intervals) and dt > 0 and n_t > 0:
                n_bt = len(present[i] & present[i + 1])
                origination_rates.append(-np.log(n_bt / n_t) / dt if n_bt > 0 else float("inf"))
            else:
                origination_rates.append(float("nan"))

            if i > 0 and dt > 0 and n_t > 0:
                n_ft = len(present[i] & present[i - 1])
                extinction_rates.append(-np.log(n_ft / n_t) / dt if n_ft > 0 else float("inf"))
            else:
                extinction_rates.append(float("nan"))

        orig_arr = np.array(origination_rates, dtype=float)
        ext_arr = np.array(extinction_rates, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            turnover = ext_arr / (orig_arr + ext_arr)

        self._logger.info(
            f"Diversity estimation complete: {len(times)} intervals, max richness = {max(richness) if richness else 0}"
        )
        return DiversityCurve(
            times=np.array(times),
            richness=np.array(richness),
            origination_rates=orig_arr,
            extinction_rates=ext_arr,
            turnover_rate=turnover,
        )

    def fit_exponential_model(self, times: np.ndarray, richness: np.ndarray) -> tuple[float, float]:
        """
        拟合指数增长模型

        dN/dt = rN

        返回:
            (r, N0) 增长率和初始多样性
        """
        self._logger.info(f"Fitting exponential model to {len(times)} data points")
        log_N = np.log(richness + 1e-10)

        coeffs = np.polyfit(times, log_N, deg=1)

        r = coeffs[0]  # 增长率
        N0 = np.exp(coeffs[1])  # 初始值

        self._logger.info(f"Exponential model fit: r={r:.4f}, N0={N0:.4f}")
        return r, N0

    def fit_logistic_model(self, times: np.ndarray, richness: np.ndarray) -> tuple[float, float, float]:
        """
        拟合逻辑斯蒂模型

        dN/dt = rN(1 - N/K)

        返回:
            (r, K, N0) 增长率、承载力和初始多样性
        """
        from scipy.optimize import curve_fit

        self._logger.info(f"Fitting logistic model to {len(times)} data points")

        def logistic(t, r, K, N0):
            return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

        try:
            params, _ = curve_fit(
                logistic, times, richness, p0=[0.1, max(richness) * 2, richness[0]], bounds=([0, 0, 0], [10, 1e6, 1e6])
            )
            self._logger.info(f"Logistic model fit: r={params[0]:.4f}, K={params[1]:.4f}, N0={params[2]:.4f}")
            return tuple(params)
        except (RuntimeError, ValueError) as e:
            self._logger.error(f"Logistic model fitting failed: {e}")
            raise ValueError(f"Logistic model fitting failed: {e}") from e

    def simulate_neutral(
        self, n_taxa: int, duration: float, speciation_rate: float = 0.1, extinction_rate: float = 0.05, dt: float = 0.1
    ) -> DiversityCurve:
        """
        模拟中性随机过程

        返回:
            模拟的多样性曲线
        """
        self._logger.info(
            f"Simulating neutral process: {n_taxa} taxa, duration={duration}, speciation={speciation_rate}, extinction={extinction_rate}"
        )
        n_steps = int(duration / dt)

        times = np.zeros(n_steps)
        richness = np.zeros(n_steps)
        orig_rates = np.zeros(n_steps)
        ext_rates = np.zeros(n_steps)

        N = n_taxa
        times[0] = 0
        richness[0] = N

        for i in range(1, n_steps):
            times[i] = i * dt

            # 随机出生-死亡
            births = np.random.poisson(speciation_rate * N * dt)
            deaths = min(N, np.random.poisson(extinction_rate * N * dt))

            N = max(0, N + births - deaths)
            richness[i] = N

            orig_rates[i] = births / (N * dt + 1e-10)
            ext_rates[i] = deaths / (N * dt + 1e-10)

        turnover = ext_rates / (orig_rates + ext_rates + 1e-10)

        self._logger.info(f"Neutral simulation complete: {n_steps} steps, final richness = {int(richness[-1])}")
        return DiversityCurve(
            times=times,
            richness=richness,
            origination_rates=orig_rates,
            extinction_rates=ext_rates,
            turnover_rate=turnover,
        )
