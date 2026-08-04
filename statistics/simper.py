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

    c_k = 2 * Σ_i Σ_j min(x_ik, x_jk) / Σ_i Σ_j Σ_k (x_ik + x_jk)

Equivalently: c_k = mean over pairs of [ 2 * min(x_ik, x_jk) / Σ_k (x_ik + x_jk) ]

The factor of 2 ensures that Σ_k c_k = δ̄ (species contributions sum to overall dissimilarity).

This represents the contribution of species k to the overall
dissimilarity between samples i and j.

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

            # Compute overall dissimilarity as mean of pairwise Bray-Curtis values
            # Per Clarke 1993: δ̄ = (1/(n_A*n_B)) Σ_i Σ_j δ_ij
            # where δ_ij = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)
            all_pairwise_dissimilarities = []
            for gi in range(n_groups):
                for gj in range(gi + 1, n_groups):
                    ga, gb = unique_groups[gi], unique_groups[gj]
                    idx_a = [i for i, g in enumerate(groups) if g == ga]
                    idx_b = [i for i, g in enumerate(groups) if g == gb]
                    data_a = data[idx_a]
                    data_b = data[idx_b]

                    for i in range(len(idx_a)):
                        for j in range(len(idx_b)):
                            num = np.sum(np.abs(data_a[i] - data_b[j]))
                            den = np.sum(data_a[i] + data_b[j])
                            if den > 0:
                                all_pairwise_dissimilarities.append(num / den)

            overall_dissimilarity = np.mean(all_pairwise_dissimilarities) if all_pairwise_dissimilarities else 0.0

            # Compute per-variable stats across all group pairs
            # Species contribution per Clarke 1993: min(x_ik, x_jk) / Σ(x_ik + x_jk)
            contrib_list = []
            for k in range(n_vars):
                # Collect contributions from each pair for std estimation
                pair_vals = []
                means_a = []
                means_b = []
                for gi in range(n_groups):
                    for gj in range(gi + 1, n_groups):
                        ga, gb = unique_groups[gi], unique_groups[gj]
                        idx_a = [i for i, g in enumerate(groups) if g == ga]
                        idx_b = [i for i, g in enumerate(groups) if g == gb]
                        data_a = data[idx_a]
                        data_b = data[idx_b]

                        contrib_k = self._single_variable_contribution(data_a, data_b, k)
                        pair_vals.append(contrib_k)
                        means_a.append(np.mean(data_a[:, k]))
                        means_b.append(np.mean(data_b[:, k]))

                avg_k = np.mean(pair_vals)
                std_k = np.std(pair_vals, ddof=1) if len(pair_vals) > 1 else 0.0

                contrib_list.append(
                    {
                        "index": k,
                        "average": avg_k,
                        "std": std_k,
                        "ratio": avg_k / std_k if std_k > 0 else float("inf"),
                        "mean_a": np.mean(means_a),
                        "mean_b": np.mean(means_b),
                    }
                )

            # Sort by average contribution descending
            contrib_list.sort(key=lambda x: x["average"], reverse=True)

            # Compute cumulative - contributions sum to overall_dissimilarity
            total = overall_dissimilarity
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
        c_k = 2 * Σ min(x_ik, x_jk) / Σ (x_ik + x_jk)

        The factor of 2 ensures that species contributions sum to the
        Bray-Curtis dissimilarity (since |x_ik - x_jk| = max - min and
        max + min = x_ik + x_jk, so max - min = 2*min - (x_ik + x_jk)
        which gives Bray-Curtis = 2*min/(x_ik + x_jk) for each species).

        This represents the contribution of species k to the overall
        dissimilarity between samples i and j.
        """
        n_a, n_vars = data_a.shape
        n_b = data_b.shape[0]
        contributions = np.zeros(n_vars)

        for i in range(n_a):
            for j in range(n_b):
                # Species contribution uses 2*min to match Bray-Curtis
                num = 2.0 * np.minimum(data_a[i], data_b[j])
                den = data_a[i] + data_b[j]
                total = np.sum(den)
                if total > 0:
                    contributions += num / total

        contributions /= n_a * n_b
        return contributions

    def _single_variable_contribution(self, data_a: npt.NDArray, data_b: npt.NDArray, var_idx: int) -> float:
        """Compute average contribution of a single variable for one group pair.

        Per Clarke 1993, species contribution is:
        c_k = 2 * Σ min(x_ik, x_jk) / Σ (x_ik + x_jk)

        The factor of 2 ensures contributions sum to Bray-Curtis.
        """
        n_a = data_a.shape[0]
        n_b = data_b.shape[0]
        contrib = 0.0

        for i in range(n_a):
            for j in range(n_b):
                # Species contribution uses 2*min to match Bray-Curtis
                num = 2.0 * min(data_a[i, var_idx], data_b[j, var_idx])
                den = np.sum(data_a[i] + data_b[j])
                if den > 0:
                    contrib += num / den

        return contrib / (n_a * n_b)

    @property
    def last_result(self) -> SimperResult | None:
        with self._lock:
            return self._last_result
