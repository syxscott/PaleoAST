# =============================================================================
# FILE: stratigraphy/directional.py
# =============================================================================
"""
Directional Statistics & Rose Diagrams for PaleoAST

Provides circular/directional statistical analysis for paleocurrent
data, fossil orientation, and other directional measurements.

Mathematical Foundation:

For n directional observations θ_1, ..., θ_n:

    Resultant length: R = sqrt((Σcosθ)² + (Σsinθ)²)
    Mean direction: θ̄ = atan2(Σsinθ, Σcosθ)
    Mean resultant length: R̄ = R / n
    Circular variance: V = 1 - R̄
    Circular standard deviation: S = sqrt(-2 ln R̄)

Rayleigh test (uniformity):
    Z = n × R̄²
    p ≈ exp(-Z) for large n

Reference: Mardia & Jupp (2000) "Directional Statistics."
Wiley, Chichester.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _

logger = logging.getLogger(__name__)


@dataclass
class DirectionalResult:
    """Result of directional statistics analysis."""

    mean_direction: float  # radians
    mean_direction_deg: float
    resultant_length: float
    mean_resultant: float
    circular_variance: float
    circular_std: float
    rayleigh_z: float
    rayleigh_p: float
    is_significant: bool
    n_observations: int
    raw_data: npt.NDArray

    def summary(self) -> str:
        sig = "**" if self.rayleigh_p < 0.01 else ("*" if self.rayleigh_p < 0.05 else "ns")
        return (
            f"{_('Directional Statistics')}\n"
            f"{'=' * 40}\n"
            f"{_('Mean direction')}: {self.mean_direction_deg:.1f}°\n"
            f"{_('Mean resultant (R̄)')}: {self.mean_resultant:.4f}\n"
            f"{_('Circular variance')}: {self.circular_variance:.4f}\n"
            f"{_('Circular std')}: {np.degrees(self.circular_std):.1f}°\n"
            f"Rayleigh Z = {self.rayleigh_z:.4f}, p = {self.rayleigh_p:.4f} {sig}\n"
            f"n = {self.n_observations}"
        )


class DirectionalAnalyzer:
    """Directional statistics engine."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.DirectionalAnalyzer")

    def analyze(self, angles_deg: npt.NDArray) -> DirectionalResult:
        """
        Compute directional statistics.

        Parameters:
            angles_deg: Array of angles in degrees (0-360)

        Returns:
            DirectionalResult
        """
        angles_rad = np.deg2rad(angles_deg)
        n = len(angles_rad)

        if n < 2:
            raise ValueError("Need at least 2 observations")

        # Resultant components
        C = np.sum(np.cos(angles_rad))
        S = np.sum(np.sin(angles_rad))

        # Resultant length
        R = np.sqrt(C ** 2 + S ** 2)
        R_bar = R / n

        # Mean direction
        mean_dir = np.arctan2(S, C)
        if mean_dir < 0:
            mean_dir += 2 * np.pi

        # Circular variance and std
        V = 1 - R_bar
        circ_std = np.sqrt(-2 * np.log(R_bar)) if R_bar > 0 else np.inf

        # Rayleigh test
        Z = n * R_bar ** 2
        # Approximate p-value (valid for moderate to large n)
        p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))
        p = min(max(p, 0), 1)

        return DirectionalResult(
            mean_direction=float(mean_dir),
            mean_direction_deg=float(np.rad2deg(mean_dir)),
            resultant_length=float(R),
            mean_resultant=float(R_bar),
            circular_variance=float(V),
            circular_std=float(circ_std),
            rayleigh_z=float(Z),
            rayleigh_p=float(p),
            is_significant=p < 0.05,
            n_observations=n,
            raw_data=angles_deg,
        )

    def bin_for_rose(
        self, angles_deg: npt.NDArray, n_bins: int = 12
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Bin angles into a rose diagram.

        Parameters:
            angles_deg: Angles in degrees
            n_bins: Number of bins (default: 12 = 30° each)

        Returns:
            (bin_edges_deg, counts) for plotting
        """
        bin_edges = np.linspace(0, 360, n_bins + 1)
        counts, _ = np.histogram(angles_deg % 360, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts
