# =============================================================================
# FILE: statistics/simper.py
# =============================================================================
"""
SIMPER (Similarity Percentages) Analysis Module for PaleoAST

SIMPER decomposes Bray-Curtis dissimilarity to identify which variables
(species) contribute most to between-group differences.

Mathematical Foundation:

For two groups A and B, the average Bray-Curtis dissimilarity between
all pairs (i in A, j in B) is:

    δ̄ = (1 / (n_A × n_B)) Σ_i Σ_j δ_ij

where δ_ij = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)

The contribution of variable k to the overall dissimilarity is:

    δ_k = mean over pairs (i, j) of [ |x_ik - x_jk| / Σ_l (x_il + x_jl) ]

Because |a - b| = (a + b) - 2*min(a, b), the per-pair contributions
satisfy Σ_k |x_ik - x_jk| / Σ_l (x_il + x_jl) = δ_ij exactly, and
therefore Σ_k δ_k = δ̄: species contributions sum to the overall
dissimilarity and cumulative percentages reach 100%.

SIMPER reports:
    - Average contribution of each variable across all pairs
    - Standard deviation of contributions
    - Cumulative contribution percentage
    - Ratio: average / SD (higher = more consistent contributor)

Reference: Clarke (1993) Non-parametric multivariate analyses of
changes in community structure. Australian Journal of Ecology, 18, 117-143.

Author: PaleoAST Development Team
version: 1.1.0 (fixed species contribution formula)
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class VariableContribution:
    """Contribution of a single variable (species)."""

    name: str
    index: int
    average: float
    std: float
    cumulative: float
    ratio: float
    group_mean_a: float
    group_mean_b: float


@dataclass
class SimperResult:
    """
    Container for SIMPER analysis results.

    Attributes:
        overall_dissimilarity: Overall average between-group dissimilarity
        contributions: Per-variable contribution details
        group_pairs: List of (group_a, group_b) pairs analyzed
        n_groups: Number of groups
        n_variables: Number of variables
        metric: Distance metric used
    """

    overall_dissimilarity: float
    contributions: list[VariableContribution]
    group_pairs: list[tuple[int, int]]
    n_groups: int
    n_variables: int
    metric: str = "bray_curtis"

    def top_contributors(self, n: int = 10) -> list[VariableContribution]:
        """Get top N contributors sorted by average contribution."""
        return sorted(self.contributions, key=lambda c: c.average, reverse=True)[:n]

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("SIMPER Analysis (Similarity Percentages)"),
            "=" * 50,
            f"{_('Overall average dissimilarity')}: {self.overall_dissimilarity:.4f}",
            f"{_('Groups')}: {self.n_groups}, {_('Variables')}: {self.n_variables}",
            "",
            f"{'Variable':<20} {'Avg':>8} {'SD':>8} {'Cum%':>8} {'Ratio':>8}",
            "-" * 55,
        ]
        for c in sorted(self.contributions, key=lambda x: x.average, reverse=True):
            lines.append(f"{c.name:<20} {c.average:>8.4f} {c.std:>8.4f} {c.cumulative * 100:>7.1f}% {c.ratio:>8.2f}")
        return "\n".join(lines)


class SimperAnalyzer:
    """
    SIMPER analysis engine.

    Computes the contribution of each variable to the average
    between-group Bray-Curtis dissimilarity.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.SimperAnalyzer")
        self._lock = threading.RLock()
        self._last_result: SimperResult | None = None

    def analyze(
        self,
        data: npt.NDArray,
        groups: list[int],
        variable_names: list[str] | None = None,
        metric: str = "bray_curtis",
    ) -> SimperResult:
        """
        Perform SIMPER analysis.

        Parameters:
            data: Data matrix (n_samples x n_variables)
            groups: Group assignment for each sample
            variable_names: Names for each variable column
            metric: Distance metric (default: Bray-Curtis)

        Returns:
            SimperResult
        """
        with self._lock:
            data = validate_data_array(data, allow_nan=False, name="data")
            n_samples, n_vars = data.shape

            if len(groups) != n_samples:
                raise ComputationError(f"Group length ({len(groups)}) must match number of samples ({n_samples})")

            if variable_names is None:
                variable_names = [f"Var_{i + 1}" for i in range(n_vars)]

            unique_groups = sorted(set(groups))
            n_groups = len(unique_groups)

            if n_groups < 2:
                raise ComputationError("SIMPER requires at least 2 groups")

            self._logger.info(f"SIMPER: {n_samples} samples, {n_vars} variables, {n_groups} groups")

            # Build group pair list
            group_pairs = []
            for gi in range(n_groups):
                for gj in range(gi + 1, n_groups):
                    group_pairs.append((unique_groups[gi], unique_groups[gj]))

            # Single pass over all cross-group pairs:
            # per-pair Bray-Curtis (for overall δ̄) and per-variable
            # contributions δ_k(ij) = |x_ik - x_jk| / Σ_l (x_il + x_jl).
            # Per Clarke 1993 the δ_k(ij) over variables sum to δ_ij.
            delta_rows: list[npt.NDArray] = []
            pairwise_dissimilarities: list[float] = []
            for gi in range(n_groups):
                for gj in range(gi + 1, n_groups):
                    ga, gb = unique_groups[gi], unique_groups[gj]
                    idx_a = [i for i, g in enumerate(groups) if g == ga]
                    idx_b = [i for i, g in enumerate(groups) if g == gb]
                    data_a = data[idx_a]
                    data_b = data[idx_b]

                    for i in range(len(idx_a)):
                        for j in range(len(idx_b)):
                            den = np.sum(data_a[i] + data_b[j])
                            if den <= 0:
                                continue
                            pairwise_dissimilarities.append(
                                np.sum(np.abs(data_a[i] - data_b[j])) / den
                            )
                            delta_rows.append(np.abs(data_a[i] - data_b[j]) / den)

            overall_dissimilarity = float(np.mean(pairwise_dissimilarities)) if pairwise_dissimilarities else 0.0

            # Per-variable stats. SD is taken across individual (i, j) pairs
            # (Clarke's Av/SD consistency measure), not across group pairs:
            # with the standard 2-group design there is only one group pair
            # and a across-pair SD would collapse to 0 (ratio = inf).
            if delta_rows:
                delta_matrix = np.vstack(delta_rows)
                avg_vec = delta_matrix.mean(axis=0)
                std_vec = delta_matrix.std(axis=0, ddof=1) if delta_matrix.shape[0] > 1 else np.zeros(n_vars)
            else:
                avg_vec = np.zeros(n_vars)
                std_vec = np.zeros(n_vars)

            contrib_list = []
            means_a = []
            means_b = []
            for ga, gb in group_pairs:
                idx_a = [i for i, g in enumerate(groups) if g == ga]
                idx_b = [i for i, g in enumerate(groups) if g == gb]
                means_a.append(np.mean(data[idx_a], axis=0))
                means_b.append(np.mean(data[idx_b], axis=0))
            mean_a_vec = np.mean(means_a, axis=0) if means_a else np.zeros(n_vars)
            mean_b_vec = np.mean(means_b, axis=0) if means_b else np.zeros(n_vars)

            for k in range(n_vars):
                contrib_list.append(
                    {
                        "index": k,
                        "average": float(avg_vec[k]),
                        "std": float(std_vec[k]),
                        "ratio": float(avg_vec[k] / std_vec[k]) if std_vec[k] > 0 else float("inf"),
                        "mean_a": float(mean_a_vec[k]),
                        "mean_b": float(mean_b_vec[k]),
                    }
                )

            # Sort by average contribution descending
            contrib_list.sort(key=lambda x: x["average"], reverse=True)

            # Cumulative %: contributions sum to overall_dissimilarity
            # (Σ_k δ_k = δ̄), so the last species reaches exactly 100%.
            total = float(np.sum(avg_vec))
            cum = 0.0
            result_contribs = []
            for c in contrib_list:
                cum += c["average"]
                result_contribs.append(
                    VariableContribution(
                        name=variable_names[c["index"]],
                        index=c["index"],
                        average=c["average"],
                        std=c["std"],
                        cumulative=cum / total if total > 0 else 0.0,
                        ratio=c["ratio"],
                        group_mean_a=c["mean_a"],
                        group_mean_b=c["mean_b"],
                    )
                )

            result = SimperResult(
                overall_dissimilarity=overall_dissimilarity,
                contributions=result_contribs,
                group_pairs=group_pairs,
                n_groups=n_groups,
                n_variables=n_vars,
                metric=metric,
            )

            self._last_result = result
            self._logger.info(f"SIMPER complete: overall dissimilarity={overall_dissimilarity:.4f}")
            return result

    def _pairwise_contributions(self, data_a: npt.NDArray, data_b: npt.NDArray) -> npt.NDArray:
        """Compute average contribution of each variable across all between-group pairs.

        Per Clarke 1993, species k contribution is:
        δ_k = mean over pairs of [ |x_ik - x_jk| / Σ_l (x_il + x_jl) ]

        Since |a - b| = (a + b) - 2*min(a, b), the δ_k over variables sum
        to the Bray-Curtis dissimilarity of each pair.
        """
        n_a, n_vars = data_a.shape
        n_b = data_b.shape[0]
        contributions = np.zeros(n_vars)

        for i in range(n_a):
            for j in range(n_b):
                den = np.sum(data_a[i] + data_b[j])
                if den > 0:
                    contributions += np.abs(data_a[i] - data_b[j]) / den

        contributions /= n_a * n_b
        return contributions

    @property
    def last_result(self) -> SimperResult | None:
        with self._lock:
            return self._last_result
