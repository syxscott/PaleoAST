# =============================================================================
# FILE: ecology/advanced.py
# =============================================================================
"""
Advanced Ecology Models for PaleoAST

Implements abundance distribution models (log-normal, geometric series,
broken stick) and SHE analysis for community structure assessment.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

from config.i18n import _
from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


# =========================================================================
# Abundance Distribution Models
# =========================================================================


@dataclass
class AbundanceModelFit:
    """Result of fitting an abundance model."""

    model_name: str
    parameters: dict
    observed: npt.NDArray
    predicted: npt.NDArray
    r_squared: float
    aic: float

    def summary(self) -> str:
        lines = [
            f"{_('Abundance Model')}: {self.model_name}",
            f"R² = {self.r_squared:.4f}",
            f"AIC = {self.aic:.2f}",
        ]
        for k, v in self.parameters.items():
            lines.append(f"  {k} = {v:.4f}")
        return "\n".join(lines)


class AbundanceModelFitter:
    """
    Fits theoretical species-abundance distributions to observed data.

    Common models in paleoecology:
        - Log-normal (Preston 1948)
        - Geometric series (Motomura 1932)
        - Broken stick (MacArthur 1957)
        - Log-series (Fisher 1943)
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.AbundanceModelFitter")

    def fit_all(self, abundances: npt.NDArray) -> dict[str, AbundanceModelFit]:
        """Fit all available models and return results sorted by AIC."""
        results = {}
        for name, method in [
            ("log_normal", self.fit_log_normal),
            ("geometric", self.fit_geometric),
            ("broken_stick", self.fit_broken_stick),
            ("log_series", self.fit_log_series),
        ]:
            try:
                results[name] = method(abundances)
            except Exception as e:
                self._logger.warning(f"Model {name} fitting failed: {e}")
        return dict(sorted(results.items(), key=lambda x: x[1].aic))

    def fit_log_normal(self, abundances: npt.NDArray) -> AbundanceModelFit:
        """
        Fit Preston's log-normal distribution.

        S(R) = S_0 * exp(-a * R²)

        where R is the octave (log2 abundance class).
        """
        abundances = np.sort(abundances[abundances > 0])[::-1]
        n_species = len(abundances)

        # Bin into octaves (log2 classes)
        log_abund = np.log2(abundances + 1)
        max_octave = int(np.ceil(max(log_abund)))
        octaves = np.arange(0, max_octave + 1)
        hist = np.zeros(len(octaves))
        for v in log_abund:
            idx = min(int(v), len(octaves) - 1)
            hist[idx] += 1

        # Fit parabola in log space: ln(S) = ln(S0) - a*R^2
        valid = hist > 0
        if valid.sum() < 3:
            raise ComputationError("Not enough data for log-normal fit")

        R = octaves[valid]
        ln_S = np.log(hist[valid])

        coeffs = np.polyfit(R**2, ln_S, deg=1)
        a = -coeffs[0]
        ln_S0 = coeffs[1]
        S0 = np.exp(ln_S0)

        predicted_oct = S0 * np.exp(-a * octaves**2)
        # Map back to species abundances
        predicted = self._octave_to_rank(predicted_oct, n_species)

        r_sq = self._r_squared(abundances, predicted)
        aic_val = self._aic(abundances, predicted, 2)

        return AbundanceModelFit(
            model_name="Log-normal (Preston)",
            parameters={"S0": S0, "a": a, "sigma": 1 / np.sqrt(2 * a) if a > 0 else np.inf},
            observed=abundances,
            predicted=predicted,
            r_squared=r_sq,
            aic=aic_val,
        )

    def fit_geometric(self, abundances: npt.NDArray) -> AbundanceModelFit:
        """
        Fit geometric series (Motomura 1932).

        The k-th species has abundance: a_k = N * c^(k-1) * (1-c) / (1-c^S)
        where c is the common ratio.
        """
        abundances = np.sort(abundances[abundances > 0])[::-1]
        n_species = len(abundances)
        N = np.sum(abundances)

        # Find c by minimizing sum of squared residuals
        def residual(c):
            if c <= 0 or c >= 1:
                return 1e10
            expected = np.array([N * c ** (k) * (1 - c) / (1 - c**n_species) for k in range(n_species)])
            return np.sum((abundances - expected) ** 2)

        result = optimize.minimize_scalar(residual, bounds=(0.01, 0.99), method="bounded")
        c = result.x

        predicted = np.array([N * c ** (k) * (1 - c) / (1 - c**n_species) for k in range(n_species)])

        r_sq = self._r_squared(abundances, predicted)
        aic_val = self._aic(abundances, predicted, 1)

        return AbundanceModelFit(
            model_name="Geometric Series (Motomura)",
            parameters={"c": c},
            observed=abundances,
            predicted=predicted,
            r_squared=r_sq,
            aic=aic_val,
        )

    def fit_broken_stick(self, abundances: npt.NDArray) -> AbundanceModelFit:
        """
        Fit broken stick model (MacArthur 1957).

        Expected abundance of the k-th species:
            E(n_k) = (N / S) * Σ_{i=k}^{S} 1/i
        """
        abundances = np.sort(abundances[abundances > 0])[::-1]
        n_species = len(abundances)
        N = np.sum(abundances)

        predicted = np.array(
            [(N / n_species) * np.sum(1.0 / np.arange(k + 1, n_species + 1)) for k in range(n_species)]
        )

        r_sq = self._r_squared(abundances, predicted)
        # Broken stick has no free parameters, but n_params=1 for AIC (variance only)
        aic_val = self._aic(abundances, predicted, 1)

        return AbundanceModelFit(
            model_name="Broken Stick (MacArthur)",
            parameters={},
            observed=abundances,
            predicted=predicted,
            r_squared=r_sq,
            aic=aic_val,
        )

    def fit_log_series(self, abundances: npt.NDArray) -> AbundanceModelFit:
        """
        Fit Fisher's log-series distribution.

        S(n) = α * x^n / n

        where α is Fisher's alpha and x is estimated from N/S.
        """
        abundances = np.asarray(abundances, dtype=float)
        abundances = abundances[~np.isnan(abundances)]
        abundances = abundances[abundances > 0]
        S = len(abundances)
        N = np.sum(abundances)

        # Estimate x from N/S = x / (1-x) * (-ln(1-x))^{-1}
        # Use numerical solution
        def equation(x):
            if x <= 0 or x >= 1:
                return 1e10
            return N / S - x / ((1 - x) * (-np.log(1 - x)))

        try:
            x = optimize.brentq(equation, 0.001, 0.999)
        except ValueError:
            self._logger.warning(
                f"Log-series brentq solver failed for S={S}, N={N}, using fallback x=0.5"
            )
            x = 0.5

        alpha = S * (1 - x) / (-np.log(1 - x)) if abs(1 - x) > 1e-10 else S

        # Predicted abundances
        max_n = int(np.max(abundances))
        predicted_freq = np.array([alpha * x**n / n for n in range(1, max_n + 1)])
        # Sort descending and take top S
        predicted = np.sort(predicted_freq)[::-1][:S]

        if len(predicted) < S:
            predicted = np.pad(predicted, (0, S - len(predicted)))

        r_sq = self._r_squared(abundances, predicted)
        aic_val = self._aic(abundances, predicted, 1)

        return AbundanceModelFit(
            model_name="Log-series (Fisher)",
            parameters={"alpha": alpha, "x": x},
            observed=abundances,
            predicted=predicted,
            r_squared=r_sq,
            aic=aic_val,
        )

    def _octave_to_rank(self, octave_counts: npt.NDArray, n_species: int) -> npt.NDArray:
        """Convert octave class counts to rank-abundance array."""
        result = []
        for octave_idx, count in enumerate(octave_counts):
            abundance = 2**octave_idx
            for _ in range(round(count)):
                result.append(abundance)
        result = sorted(result, reverse=True)[:n_species]
        while len(result) < n_species:
            result.append(0)
        return np.array(result, dtype=float)

    def _r_squared(self, observed: npt.NDArray, predicted: npt.NDArray) -> float:
        n = min(len(observed), len(predicted))
        obs = observed[:n]
        pred = predicted[:n]
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _aic(self, observed: npt.NDArray, predicted: npt.NDArray, n_params: int) -> float:
        n = min(len(observed), len(predicted))
        obs = observed[:n]
        pred = predicted[:n]
        pred = np.maximum(pred, 1e-10)
        ss_res = np.sum((obs - pred) ** 2)
        if ss_res <= 0:
            ss_res = 1e-10
        log_lik = -n / 2 * np.log(ss_res / n)
        return 2 * n_params - 2 * log_lik


# =========================================================================
# SHE Analysis
# =========================================================================


@dataclass
class SHEResult:
    """Result of SHE analysis."""

    sample_sizes: npt.NDArray
    s_values: npt.NDArray  # Species richness
    h_values: npt.NDArray  # Shannon H'
    e_values: npt.NDArray  # Evenness (Pielou's J)

    def summary(self) -> str:
        return (
            f"SHE Analysis: {len(self.sample_sizes)} data points\n"
            f"S range: [{self.s_values.min():.0f}, {self.s_values.max():.0f}]\n"
            f"H range: [{self.h_values.min():.4f}, {self.h_values.max():.4f}]\n"
            f"E range: [{self.e_values.min():.4f}, {self.e_values.max():.4f}]"
        )


class SHEAnalyzer:
    """
    SHE Analysis: Separates S, H, and E trends with increasing sample size.

    Used to detect whether diversity patterns are driven by
    richness changes or evenness changes.
    """

    def analyze(self, abundance_matrix: npt.NDArray) -> SHEResult:
        """
        Perform SHE analysis on cumulative sample subsets.

        Parameters:
            abundance_matrix: (n_samples x n_species) abundance data.
                Row order is meaningful: samples are accumulated in the
                order supplied (typically stratigraphic/temporal order),
                as required by SHE analysis (Hayek & Buzas 1997). Rows
                are NOT sorted by abundance.

        Returns:
            SHEResult
        """
        n_samples, _n_species = abundance_matrix.shape

        # Accumulate samples in input (stratigraphic) order. SHE analysis
        # (Hayek & Buzas 1997) tracks how S, H and E change along the
        # accumulation sequence, so the row order must be preserved.
        data = np.asarray(abundance_matrix)

        sample_sizes = []
        s_vals = []
        h_vals = []
        e_vals = []

        for k in range(2, n_samples + 1):
            subset = data[:k]
            pooled = np.nansum(subset, axis=0)
            pooled = pooled[pooled > 0]

            S = len(pooled)
            if S == 0:
                continue

            # Shannon H'
            proportions = pooled / np.sum(pooled)
            proportions = proportions[proportions > 0]
            H = -np.sum(proportions * np.log(proportions))

            # Pielou's evenness
            E = H / np.log(S) if S > 1 else 1.0

            sample_sizes.append(k)
            s_vals.append(S)
            h_vals.append(H)
            e_vals.append(E)

        return SHEResult(
            sample_sizes=np.array(sample_sizes),
            s_values=np.array(s_vals),
            h_values=np.array(h_vals),
            e_values=np.array(e_vals),
        )
