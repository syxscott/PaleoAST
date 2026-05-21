# stratigraphy/correlation.py
"""
Stratigraphic Correlation and Age Modeling for PaleoAST

Provides tools for:
- Correlating multiple stratigraphic sections using DTW
- Building geological age models from biostratigraphic constraints
- Computing sedimentation rates

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class StratigraphicSection:
    """单条地层剖面数据"""
    name: str
    heights: np.ndarray  # 层位高度 (m)
    thicknesses: np.ndarray  # 层厚 (m)
    lithologies: List[str]  # 岩性描述
    ages: Optional[np.ndarray] = None  # 绝对年龄 (如果有)
    age_errors: Optional[np.ndarray] = None  # 年龄误差
    notes: Optional[List[str]] = None  # 备注

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'heights': self.heights.tolist(),
            'thicknesses': self.thicknesses.tolist(),
            'lithologies': self.lithologies,
            'ages': self.ages.tolist() if self.ages is not None else None,
            'age_errors': self.age_errors.tolist() if self.age_errors is not None else None
        }


@dataclass
class StratigraphicCorrelationResult:
    """相关性分析结果"""
    sections: List[StratigraphicSection]
    correlation_matrix: np.ndarray
    best_matches: List[Tuple[int, int, float]]

    def summary(self) -> str:
        n = len(self.sections)
        triu_indices = np.triu_indices(n, k=1)
        if len(triu_indices[0]) > 0:
            avg_corr = np.mean(self.correlation_matrix[triu_indices])
        else:
            avg_corr = 1.0
        lines = [
            f"地层相关性分析结果 ({n} 个剖面)",
            f"平均相关性: {avg_corr:.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'sections': [s.to_dict() for s in self.sections],
            'correlation_matrix': self.correlation_matrix.tolist()
        }


@dataclass
class AgeModelResult:
    """年龄模型结果"""
    section: StratigraphicSection
    modeled_ages: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    sedimentation_rates: np.ndarray

    def summary(self) -> str:
        lines = [
            f"年龄模型: {self.section.name}",
            f"年龄范围: {self.modeled_ages.min():.2f} - {self.modeled_ages.max():.2f} Ma",
            f"平均沉积速率: {np.mean(self.sedimentation_rates):.2f} m/Myr",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'section': self.section.to_dict(),
            'modeled_ages': self.modeled_ages.tolist(),
            'ci_lower': self.confidence_intervals[0].tolist(),
            'ci_upper': self.confidence_intervals[1].tolist(),
            'sedimentation_rates': self.sedimentation_rates.tolist()
        }


class StratigraphicCorrelationAnalyzer:
    """地层相关性分析器 (使用动态时间规整 DTW)"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.StratigraphicCorrelationAnalyzer")

    def analyze(
        self,
        sections: List[StratigraphicSection],
        method: str = 'dtw',
        max_matches: int = 5
    ) -> StratigraphicCorrelationResult:
        """
        分析多个地层剖面的相关性

        Parameters:
            sections: 地层剖面列表
            method: 'dtw' (动态时间规整) 或 'euclidean' (欧氏距离)
            max_matches: 最大返回的匹配对数量

        Returns:
            StratigraphicCorrelationResult
        """
        n = len(sections)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = self._compute_correlation(sections[i], sections[j], method)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        best_matches = self._find_best_matches(corr_matrix, sections, max_matches)

        self._logger.info(f"Correlation analysis complete: {n} sections, avg corr={np.mean(corr_matrix):.3f}")

        return StratigraphicCorrelationResult(
            sections=sections,
            correlation_matrix=corr_matrix,
            best_matches=best_matches
        )

    def _compute_correlation(
        self,
        sec_a: StratigraphicSection,
        sec_b: StratigraphicSection,
        method: str
    ) -> float:
        if method == 'dtw':
            return self._dtw_correlation(sec_a, sec_b)
        elif method == 'euclidean':
            return self._euclidean_correlation(sec_a, sec_b)
        else:
            return self._dtw_correlation(sec_a, sec_b)

    def _dtw_correlation(self, sec_a: StratigraphicSection, sec_b: StratigraphicSection) -> float:
        """动态时间规整相关性"""
        h_a = sec_a.heights
        h_b = sec_b.heights

        n, m = len(h_a), len(h_b)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(h_a[i - 1] - h_b[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        if n + m == 0:
            return 1.0
        similarity = 1 / (1 + dtw[n, m] / (n + m))
        return similarity

    def _euclidean_correlation(self, sec_a: StratigraphicSection, sec_b: StratigraphicSection) -> float:
        """欧氏距离相关性"""
        h_a = sec_a.heights
        h_b = sec_b.heights

        min_len = min(len(h_a), len(h_b))
        if min_len == 0:
            return 0.0
        diff = np.sum((h_a[:min_len] - h_b[:min_len]) ** 2)
        similarity = 1 / (1 + np.sqrt(diff) / min_len)
        return similarity

    def _find_best_matches(
        self,
        corr_matrix: np.ndarray,
        sections: List[StratigraphicSection],
        max_matches: int = 5
    ) -> List[Tuple[int, int, float]]:
        """找相关性最高的剖面对"""
        matches = []
        n = len(sections)
        for i in range(n):
            for j in range(i + 1, n):
                matches.append((i, j, corr_matrix[i, j]))
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:max_matches]


class AgeModelAnalyzer:
    """地质年龄模型构建器"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.AgeModelAnalyzer")

    def build_model(
        self,
        section: StratigraphicSection,
        age_constraints: List[Tuple[float, float, float]],
        model_type: str = 'linear'
    ) -> AgeModelResult:
        """
        构建地层年龄模型

        Parameters:
            section: 地层剖面
            age_constraints: 年龄约束点 [(height, age, error), ...]
            model_type: 'linear' (线性插值) 或 'spline' (样条插值)

        Returns:
            AgeModelResult
        """
        from scipy.interpolate import interp1d

        constraint_heights = np.array([c[0] for c in age_constraints])
        constraint_ages = np.array([c[1] for c in age_constraints])
        constraint_errors = np.array([c[2] for c in age_constraints])

        kind = 'linear' if model_type == 'linear' else 'cubic'
        age_func = interp1d(
            constraint_heights,
            constraint_ages,
            kind=kind,
            fill_value='extrapolate'
        )

        modeled_ages = age_func(section.heights)

        rates = np.zeros(len(section.heights))
        if len(section.heights) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Note: In geological convention, age decreases with height
                # so dHeight/dAge < 0, hence the negation to get positive rates
                rates = np.diff(section.heights) / np.diff(modeled_ages)
                rates = -rates  # Make positive (sedimentation rate)
                rates = np.concatenate([[rates[0]], rates])

        ci_lower = modeled_ages - 1.96 * constraint_errors.mean()
        ci_upper = modeled_ages + 1.96 * constraint_errors.mean()

        self._logger.info(
            f"Age model built: {len(section.heights)} points, "
            f"range {modeled_ages.min():.2f}-{modeled_ages.max():.2f} Ma"
        )

        return AgeModelResult(
            section=section,
            modeled_ages=modeled_ages,
            confidence_intervals=(ci_lower, ci_upper),
            sedimentation_rates=rates
        )


class SedimentationRateAnalyzer:
    """沉积速率分析器"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.SedimentationRateAnalyzer")

    def calculate(
        self,
        section: StratigraphicSection,
        smooth: bool = True,
        frac: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算沉积速率

        Parameters:
            section: 地层剖面 (需要包含 ages)
            smooth: 是否使用 LOWESS 平滑
            frac: 平滑参数

        Returns:
            (heights, rates, smoothed_rates)
        """
        heights = section.heights
        ages = section.ages

        if ages is None:
            raise ValueError("需要年龄数据来计算沉积速率")

        rates = np.zeros(len(heights))
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.gradient(heights, ages)
            rates = -rates  # 沉积为正

        smoothed_rates = rates.copy()
        if smooth and len(heights) > 3:
            try:
                from scipy.interpolate import Lowess
                sorted_idx = np.argsort(heights)
                result = Lowess(rates[sorted_idx], heights[sorted_idx], frac=frac, return_sorted=False)
                smoothed_rates[sorted_idx] = result[:, 1]
            except Exception as e:
                self._logger.warning(f"LOWESS smoothing failed: {e}")

        self._logger.info(
            f"Sedimentation rates computed: {len(heights)} points, "
            f"mean={np.mean(rates):.2f} m/Myr"
        )

        return heights, rates, smoothed_rates
