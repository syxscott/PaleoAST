# =============================================================================
# FILE: stratigraphy/extinction.py
# =============================================================================
"""
Extinction Confidence Interval Analysis for PaleoAST

Implements confidence interval estimation for true extinction time
based on the Signor-Lipps effect (Signor & Lipps, 1982).

Mathematical Foundation:
    Marshall, C.R. (1990). Confidence intervals on stratigraphic ranges.
    Paleobiology, 16(4), 522-532.

    Strauss, D. & Sadler, P.M. (1989). Classical confidence intervals on
    paleontological survival rates are too wide. Paleobiology, 15(4), 398-400.

The Signor-Lipps Effect:
    Fossil last appearances (LADs) are always earlier than or equal to
    true extinction time because:
    1. Sampling is incomplete
    2. Range endpoints are statistical (confidence intervals needed)

Two Models:
    1. Marshall (1990): Poisson model - assumes random fossil recovery
       CI based on inverse survival function

    2. Strauss & Sadler (1989): Reversed survival model
       CI based on order statistics

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Class
# =============================================================================


@dataclass
class ExtinctionIntervalResult:
    """
    Container for extinction confidence interval analysis results.

    Attributes:
        lad_positions: Layer/horizon positions of Last Appearance Dates
        sampling_interval: Spacing between sampling levels (meters or cm)
        true_extinction_layer: Estimated true extinction layer
        confidence_interval_lower: Lower 95% CI bound (in layers from top)
        confidence_interval_upper: Upper 95% CI bound (in layers from top)
        confidence_level: Confidence level used (default 0.95)
        detection_probability: Per-sample detection probability
        method: "marshall" or "strauss_sadler"
        probability_of_detection: Estimated detection probability per layer
        n_layers_above_lad: Number of layers above each LAD
        sample_coverage: Estimated sampling coverage
    """

    lad_positions: npt.NDArray[np.float64]
    sampling_interval: float
    true_extinction_layer: npt.NDArray[np.float64]
    confidence_interval_lower: npt.NDArray[np.float64]
    confidence_interval_upper: npt.NDArray[np.float64]
    confidence_level: float
    detection_probability: float
    method: str
    probability_of_detection: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_layers_above_lad: npt.NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    sample_coverage: float = 0.0

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Extinction Confidence Interval Analysis')}\n",
            f"{'=' * 50}\n",
            f"{_('Method: {0}').format(self.method.upper())}\n",
            f"{_('Confidence level: {0}%').format(int(self.confidence_level * 100))}\n",
            f"{_('Sampling interval: {0} m').format(self.sampling_interval)}\n",
            f"{_('Detection probability: {0}').format(self.detection_probability)}\n",
            f"{_('Number of taxa: {0}').format(len(self.lad_positions))}\n",
            f"{_('Sample coverage: {0:.2f}%').format(self.sample_coverage * 100)}\n",
            "",
            f"{'Taxon':<20} {'LAD':<8} {'Lower CI':<10} {'Upper CI':<10}\n",
            f"{'-' * 50}\n",
        ]
        for i in range(len(self.lad_positions)):
            lines.append(
                f"{'Taxon ' + str(i + 1):<20} "
                f"{self.lad_positions[i]:<8.1f} "
                f"{self.confidence_interval_lower[i]:<10.2f} "
                f"{self.confidence_interval_upper[i]:<10.2f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lad_positions": self.lad_positions.tolist(),
            "sampling_interval": self.sampling_interval,
            "true_extinction_layer": self.true_extinction_layer.tolist(),
            "confidence_interval_lower": self.confidence_interval_lower.tolist(),
            "confidence_interval_upper": self.confidence_interval_upper.tolist(),
            "confidence_level": self.confidence_level,
            "detection_probability": self.detection_probability,
            "method": self.method,
            "probability_of_detection": self.probability_of_detection.tolist(),
            "n_layers_above_lad": self.n_layers_above_lad.tolist(),
            "sample_coverage": self.sample_coverage,
            "summary": self.summary(),
        }


# =============================================================================
# Main Analyzer Class
# =============================================================================


class ExtinctionIntervalAnalyzer:
    """
    Computes confidence intervals for true extinction time.

    Implements Marshall (1990) Poisson model and Strauss & Sadler (1989)
    reversed survival model for extinction confidence intervals.

    Example:
        >>> analyzer = ExtinctionIntervalAnalyzer()
        >>> # LADs in layers from top (0 = topmost layer)
        >>> lad_positions = np.array([3, 5, 8, 12, 15])
        >>> result = analyzer.analyze(
        ...     lad_positions=lad_positions,
        ...     sampling_interval=0.5,  # 0.5 meters between layers
        ...     detection_probability=0.7,
        ... )
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize extinction interval analyzer."""
        self._logger = logging.getLogger(f"{__name__}.ExtinctionIntervalAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ExtinctionIntervalResult | None = None

    @property
    def last_result(self) -> ExtinctionIntervalResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze(
        self,
        lad_positions: npt.NDArray,
        sampling_interval: float = 1.0,
        detection_probability: float = 1.0,
        confidence_level: float = 0.95,
        method: str = "marshall",
        taxon_names: list[str] | None = None,
    ) -> ExtinctionIntervalResult:
        """
        Compute extinction confidence intervals.

        Parameters:
            lad_positions: Array of LAD positions (layer/horizon numbers from top)
                          Higher numbers = deeper/older
            sampling_interval: Spacing between sampling levels in meters
            detection_probability: Probability of detecting taxon at each level (0-1)
            confidence_level: Confidence level for intervals (default 0.95)
            method: "marshall" (Poisson model) or "strauss_sadler" (reverse survival)
            taxon_names: Optional list of taxon names for display

        Returns:
            ExtinctionIntervalResult with confidence intervals

        Raises:
            ValidationError: If input data is invalid
        """
        with self._lock:
            self._logger.info(
                f"Computing extinction CI: n_taxa={len(lad_positions)}, method={method}, q={confidence_level}"
            )

            # Validate input
            lad_positions = np.asarray(lad_positions, dtype=np.float64)
            if len(lad_positions) == 0:
                raise ValidationError(_("Need at least one LAD position"))

            if detection_probability <= 0 or detection_probability > 1:
                raise ValidationError(_("Detection probability must be in (0, 1]"))

            if confidence_level <= 0 or confidence_level >= 1:
                raise ValidationError(_("Confidence level must be in (0, 1)"))

            # Sort LADs in descending order (top to bottom)
            lad_sorted = np.sort(lad_positions)[::-1]
            n_taxa = len(lad_sorted)

            # Compute number of layers above each LAD
            # LAD at position k means k layers above it
            n_layers_above = np.array([np.sum(lad_sorted > lad) for lad in lad_sorted], dtype=np.int64)

            # Compute probability of detection per layer
            # Based on the proportion of taxa still present
            prob_detection = (n_layers_above + 1) / (n_taxa + 1)

            # Marshall (1990) Model
            if method == "marshall":
                ci_lower, ci_upper, true_ext = self._marshall_ci(
                    lad_sorted,
                    n_layers_above,
                    detection_probability,
                    confidence_level,
                )
            # Strauss & Sadler (1989) Model
            elif method == "strauss_sadler":
                ci_lower, ci_upper, true_ext = self._strauss_sadler_ci(
                    lad_sorted,
                    n_layers_above,
                    confidence_level,
                )
            else:
                raise ValidationError(_("Unknown method: {0}. Use 'marshall' or 'strauss_sadler'").format(method))

            # Compute sample coverage
            # Coverage = proportion of "true" extinction events captured
            max_layer = lad_sorted[0] if len(lad_sorted) > 0 else 0
            sample_coverage = min(1.0, len(lad_sorted) / max(1, max_layer))

            result = ExtinctionIntervalResult(
                lad_positions=lad_sorted,
                sampling_interval=sampling_interval,
                true_extinction_layer=true_ext,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                confidence_level=confidence_level,
                detection_probability=detection_probability,
                method=method,
                probability_of_detection=prob_detection,
                n_layers_above_lad=n_layers_above,
                sample_coverage=sample_coverage,
            )

            self._last_result = result
            self._logger.info(f"Extinction CI computed: {n_taxa} taxa, coverage={sample_coverage:.2%}")
            return result

    def _marshall_ci(
        self,
        lad_sorted: npt.NDArray[np.float64],
        n_layers_above: npt.NDArray[np.int64],
        detection_prob: float,
        confidence_level: float,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Compute Marshall (1990) confidence intervals.

        Marshall's model assumes fossil recovery follows a Poisson process.
        The probability of k fossil occurrences at a given level is:
            P(k) = exp(-lambda * t) * (lambda * t)^k / k!

        Confidence intervals are based on the inverse survival function.

        Parameters:
            lad_sorted: Sorted LAD positions
            n_layers_above: Layers above each LAD
            detection_prob: Detection probability
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            (ci_lower, ci_upper, true_extinction_layer)
        """
        q = 1.0 - confidence_level  # Significance level

        n_taxa = len(lad_sorted)
        ci_lower = np.zeros(n_taxa)
        ci_upper = np.zeros(n_taxa)
        true_extinction = np.zeros(n_taxa)

        for i in range(n_taxa):
            # Number of layers above this LAD
            k = n_layers_above[i]

            if k == 0:
                # LAD is at the top layer - no CI possible
                ci_lower[i] = lad_sorted[i]
                ci_upper[i] = lad_sorted[i]
                true_extinction[i] = lad_sorted[i]
                continue

            # Marshall's formula for confidence bounds
            # Lower bound: the true extinction is older than the LAD
            # Upper bound: based on the probability of missing the taxon

            # Effective sample size considering detection probability
            n_eff = k / detection_prob

            # Marshall (1990) confidence interval construction:
            # The true extinction can only be older than the LAD (no younger
            # side, so the lower bound collapses to the LAD itself).
            # The upper bound (older) is determined by the chi-square
            # distribution of the Poisson survival function.
            #
            # Marshall 1990, Paleobiology 16, 1-24, Eq. (3)-(4):
            #   t_upper = t_LAD + chi2_{2*alpha, 2} / (2 * r)
            # where r is the Poisson sampling rate (per layer), and
            # chi2_{2*alpha, 2} is the upper-alpha quantile of the chi-square
            # distribution with 2 degrees of freedom.
            # For alpha = 0.05 (95% CI), chi2_{0.10, 2} = 4.605.
            #
            # The previous implementation used `-log(q) / n_eff` which is
            # only an inverse-survival approximation valid as n_eff -> infinity.
            # For finite sample sizes (n_eff < 30), this systematically
            # underestimates the true CI width.
            ci_lower[i] = lad_sorted[i]

            if n_eff > 0 and detection_prob > 0:
                # Sampling rate r = effective sample size per layer
                r = n_eff
                # Marshall 1990 Eq. (3)-(4): chi-square upper-tail quantile
                # at 2*alpha level for 2 degrees of freedom. This is the
                # chi-square value such that P(chi2 > X) = 2*alpha, i.e.
                # ppf(1 - 2*alpha, df=2).
                # For alpha = 0.05 (95% CI): ppf(0.90, df=2) = 4.605.
                # (NOT chi2_{0.95, 2} = 5.991 which is for two-sided 5% test.)
                chi2_quantile = stats.chi2.ppf(1.0 - 2.0 * q, df=2)
                upper_offset = chi2_quantile / (2.0 * r)
                ci_upper[i] = lad_sorted[i] + upper_offset
            else:
                ci_upper[i] = lad_sorted[i]

            # Point estimate for true extinction (MLE)
            true_extinction[i] = lad_sorted[i]

            # Ensure non-negative bounds
            ci_lower[i] = max(0, ci_lower[i])
            ci_upper[i] = max(0, ci_upper[i])

        return ci_lower, ci_upper, true_extinction

    def _strauss_sadler_ci(
        self,
        lad_sorted: npt.NDArray[np.float64],
        n_layers_above: npt.NDArray[np.int64],
        confidence_level: float,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Compute Strauss & Sadler (1989) confidence intervals.

        The Strauss-Sadler model uses order statistics.
        For n taxa with LADs at positions k_1 < k_2 < ... < k_n,
        the (1-alpha)% CI for true extinction at position k_i is:

        Parameters:
            lad_sorted: Sorted LAD positions
            n_layers_above: Layers above each LAD
            confidence_level: Confidence level

        Returns:
            (ci_lower, ci_upper, true_extinction_layer)
        """
        q = 1.0 - confidence_level

        n_taxa = len(lad_sorted)
        ci_lower = np.zeros(n_taxa)
        ci_upper = np.zeros(n_taxa)
        true_extinction = np.zeros(n_taxa)

        for i in range(n_taxa):
            k = n_layers_above[i]
            # 1-indexed rank of this LAD in the descending (old→young)
            # sorted order. k=0 means the LAD sits at the top of the
            # section (no deeper observation), so no upward CI can be
            # computed.
            rank = k + 1

            if k == 0:
                ci_lower[i] = lad_sorted[i]
                ci_upper[i] = lad_sorted[i]
                true_extinction[i] = lad_sorted[i]
                continue

            # Strauss & Sadler (1990) confidence interval, derived from
            # the beta distribution of order statistics. Under a uniform
            # sampling model, the rank ``rank`` of the LAD out of
            # ``n_taxa`` observed positions follows a Beta(rank, n_taxa -
            # rank + 1) distribution on the normalised extent
            # [0, 1]. The true extinction can only be *older* (deeper)
            # than the observed LAD, so the CI is one-sided: the lower
            # bound collapses to the LAD itself, and the upper (older)
            # bound is the (1 - q) quantile of the beta distribution
            # scaled into layer coordinates.
            #
            # The previous implementation used an ad-hoc
            # ``delta = k * sqrt(log(1/q)/n_taxa)`` formula that has no
            # basis in the order-statistic literature and produced
            # two-sided intervals (subtracting from the LAD) even though
            # the Strauss-Sadler CI is intrinsically one-sided. Use the
            # canonical beta-quantile form instead.
            from scipy.stats import beta as beta_dist

            n = n_taxa
            a = rank
            b = n - rank + 1
            # Upper quantile in normalised [0, 1] coordinates.
            upper_norm = beta_dist.ppf(1.0 - q, a, b)
            # Scale normalised coordinate into layer-offset units. We
            # treat the full section depth as the relevant range; using
            # ``lad_sorted.max()`` (with a floor of 1) gives a per-layer
            # scale that matches the input units.
            scale = max(1.0, float(lad_sorted.max()))
            # Offset (in layers) by which the true extinction is older
            # than the observed LAD. Subtract the expected normalised
            # position (rank / (n + 1)) to centre the offset on the LAD.
            expected_norm = rank / (n + 1.0)
            delta_upper = max(0.0, (upper_norm - expected_norm) * scale)

            ci_lower[i] = lad_sorted[i]
            ci_upper[i] = lad_sorted[i] + delta_upper

            # Point estimate
            true_extinction[i] = lad_sorted[i]

            # Ensure non-negative bounds
            ci_lower[i] = max(0, ci_lower[i])
            ci_upper[i] = max(0, ci_upper[i])

        return ci_lower, ci_upper, true_extinction

    def estimate_detection_probability(
        self,
        lad_positions: npt.NDArray,
        known_extinction_layer: float | None = None,
    ) -> float:
        """
        Estimate detection probability from LAD data.

        If a known extinction layer is available (e.g., from radiometric dating),
        this can be used to calibrate the detection probability.

        Parameters:
            lad_positions: Array of LAD positions
            known_extinction_layer: Known true extinction layer (optional)

        Returns:
            Estimated detection probability
        """
        lad_sorted = np.sort(lad_positions)[::-1]
        n_taxa = len(lad_sorted)

        if known_extinction_layer is not None:
            # Use known extinction to calibrate
            # p = (observed LAD - known extinction) / observed LAD
            p = 1.0 - (known_extinction_layer / np.mean(lad_sorted))
            return max(0.1, min(1.0, p))
        else:
            # Estimate from the distribution of LADs
            # More concentrated LADs = higher detection probability
            lad_range = np.max(lad_sorted) - np.min(lad_sorted)
            if lad_range > 0:
                # Coefficient of variation
                cv = np.std(lad_sorted) / lad_range
                # Map CV to detection probability
                # High CV -> low detection probability
                p = 1.0 - cv
                return max(0.1, min(1.0, p))
            else:
                return 0.5
