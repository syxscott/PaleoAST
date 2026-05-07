"""
================================================================================
PaleoAST Macroevolution - Diversity Dynamics
================================================================================

本模块实现多样性动态建模和可视化。

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import logging

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
        self,
        fossil_records: List[Tuple[float, float]],
        intervals: List[Tuple[float, float]]
    ) -> DiversityCurve:
        """
        从化石记录估计多样性曲线
        
        参数:
            fossil_records: 化石记录 [(起源时间, 灭绝时间), ...]
            intervals: 时间区间
        
        返回:
            DiversityCurve
        """
        times = []
        richness = []
        origination_rates = []
        extinction_rates = []
        
        records = fossil_records
        
        for i, (t_start, t_end) in enumerate(intervals):
            times.append((t_start + t_end) / 2)
            
            # 计算该区间内存在的物种数
            count = sum(1 for o, L in records if o >= t_end and L <= t_start)
            richness.append(count)
            
            # 计算速率
            if i > 0:
                dt = intervals[i][0] - intervals[i-1][0]
                if dt > 0:
                    dR = count - richness[i-1]
                    if dR > 0:
                        orig_rate = dR / dt
                        ext_rate = 0.0
                    else:
                        orig_rate = 0.0
                        ext_rate = -dR / dt
                    origination_rates.append(orig_rate)
                    extinction_rates.append(ext_rate)
            else:
                origination_rates.append(0.0)
                extinction_rates.append(0.0)
        
        turnover = np.array(extinction_rates) / (
            np.array(origination_rates) + np.array(extinction_rates) + 1e-10
        )
        
        return DiversityCurve(
            times=np.array(times),
            richness=np.array(richness),
            origination_rates=np.array(origination_rates),
            extinction_rates=np.array(extinction_rates),
            turnover_rate=turnover
        )
    
    def fit_exponential_model(
        self,
        times: np.ndarray,
        richness: np.ndarray
    ) -> Tuple[float, float]:
        """
        拟合指数增长模型
        
        dN/dt = rN
        
        返回:
            (r, N0) 增长率和初始多样性
        """
        log_N = np.log(richness + 1e-10)
        
        coeffs = np.polyfit(times, log_N, deg=1)
        
        r = coeffs[0]  # 增长率
        N0 = np.exp(coeffs[1])  # 初始值
        
        return r, N0
    
    def fit_logistic_model(
        self,
        times: np.ndarray,
        richness: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        拟合逻辑斯蒂模型
        
        dN/dt = rN(1 - N/K)
        
        返回:
            (r, K, N0) 增长率、承载力和初始多样性
        """
        from scipy.optimize import curve_fit
        
        def logistic(t, r, K, N0):
            return K / (1 + ((K - N0) / N0) * np.exp(-r * t))
        
        try:
            params, _ = curve_fit(
                logistic,
                times,
                richness,
                p0=[0.1, max(richness) * 2, richness[0]],
                bounds=([0, 0, 0], [10, 1e6, 1e6])
            )
            return tuple(params)
        except:
            return 0.0, max(richness), richness[0]
    
    def simulate_neutral(
        self,
        n_taxa: int,
        duration: float,
        speciation_rate: float = 0.1,
        extinction_rate: float = 0.05,
        dt: float = 0.1
    ) -> DiversityCurve:
        """
        模拟中性随机过程
        
        返回:
            模拟的多样性曲线
        """
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
        
        return DiversityCurve(
            times=times,
            richness=richness,
            origination_rates=orig_rates,
            extinction_rates=ext_rates,
            turnover_rate=turnover
        )
