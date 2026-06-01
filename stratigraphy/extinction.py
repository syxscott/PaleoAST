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
            # The upper bound (older) is determined by the inverse of the
            # Poisson survival function, scaled by the effective sample size.
            #
            #   upper_offset = -ln(q) / n_eff
            #
            # The previous implementation used math.log(q) which is negative
            # (q in (0,1)), producing ci_lower > lad_sorted and making the
            # interval degenerate (ci_lower == ci_upper).
            ci_lower[i] = lad_sorted[i]

            if n_eff > 0:
                ci_upper[i] = lad_sorted[i] - math.log(q) / n_eff
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
            i + 1  # 1-indexed rank

            if k == 0:
                ci_lower[i] = lad_sorted[i]
                ci_upper[i] = lad_sorted[i]
                true_extinction[i] = lad_sorted[i]
                continue

            # Strauss-Sadler confidence interval
            # Based on the beta distribution of order statistics
            # Lower bound: k - (k / (1 - q/2))^(1/(n+1))
            # Upper bound: k + ((k+1) / (q/2))^(1/(n+1)) - 1

            # For simplicity, use an approximation
            # Lower: LAD is older than observed by approximately
            #   delta_L = (k / n_taxa) * log(1 / q)
            # Upper: LAD is younger by approximately
            #   delta_U = log(1 / q) / (k / n_taxa + 1)

            delta_lower = k * math.pow(math.log(1.0 / q) / n_taxa, 0.5)
            delta_upper = math.log(1.0 / q) / (k / n_taxa + 1)

            ci_lower[i] = lad_sorted[i] - delta_lower
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
        len(lad_sorted)

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
