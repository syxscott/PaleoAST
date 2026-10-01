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

Rayleigh test (uniformity) — exact p-value (Greenwood & Durand 1955,
Mardia & Jupp 2000):
    Z = n × R̄²
    p = exp(-Z) × Σ_{k=0..∞} (2Z)^k / (k! × (k+1)) × ...
    which scipy.stats.chi2.sf(2Z, df=2) evaluates in closed form.

AXIAL vs POLAR DATA
-------------------
Polar (vector) data: each measurement has a *direction*; e.g. paleocurrent
vector azimuth, glacier-flow direction. θ and θ+180° are distinct.

Axial (undirected-line) data: each measurement is an *axis* — the
measurement at θ is the same as at θ+180°; e.g. tectonic lineation
bearing, paleomagnetic foliation plane, slickenside striation, glacial
striation. Treating axial data as polar produces the classic
"quadrant bias": a unimodal axial set near θ̄=90° appears bimodal because
the bimodal "image" at 270° pushes the vector mean off the true axis.

The standard cure is the double-angle transform (Mardia & Jupp 2000,
§3.5.2; Fisher 1993):
    φ_i = 2 θ_i (mod 2π)
and the resulting φ̄ halved back to [0, π). Rayleigh p is computed on
the doubled sample of size n at 2θ.

Reference: Mardia, K. V. & Jupp, P. E. (2000). "Directional Statistics."
Wiley, Chichester.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as _scipy_stats

from config.i18n import _

logger = logging.getLogger(__name__)


@dataclass
class DirectionalResult:
    """Result of directional statistics analysis."""

    mean_direction: float | None  # radians, or None when undefined (R̄≈0)
    mean_direction_deg: float | None  # degrees, or None when undefined
    resultant_length: float
    mean_resultant: float
    circular_variance: float
    circular_std: float
    rayleigh_z: float
    rayleigh_p: float
    is_significant: bool
    n_observations: int
    raw_data: npt.NDArray
    data_type: str = "polar"  # 'polar' or 'axial'

    def summary(self) -> str:
        sig = "**" if self.rayleigh_p < 0.01 else ("*" if self.rayleigh_p < 0.05 else "ns")
        if self.mean_direction_deg is None:
            mean_str = f"{_('undefined')} (R̄ ≈ 0)"
        else:
            mean_str = f"{self.mean_direction_deg:.1f}°"
        return (
            f"{_('Directional Statistics')}\n"
            f"{'=' * 40}\n"
            f"{_('Data type')}: {self.data_type}\n"
            f"{_('Mean direction')}: {mean_str}\n"
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

    def analyze(
        self,
        angles_deg: npt.NDArray,
        axial: bool = False,
    ) -> DirectionalResult:
        """
        Compute directional statistics.

        Parameters
        ----------
        angles_deg : array-like
            Angles in degrees. Polar data are expected on [0, 360);
            axial data on [0, 180). Values outside their natural range
            are wrapped into it.
        axial : bool, default False
            False → polar (vector) data: classical 1θ circular stats.
            True → axial (undirected-line) data: double-angle transform
            is applied internally and the result is folded back into
            [0, 180). The mean is reported in [0, 180) and the Rayleigh
            test uses the doubled sample of size n.

        Returns
        -------
        DirectionalResult
            ``mean_direction`` / ``mean_direction_deg`` are ``None``
            (and a warning is emitted) when R̄ is so close to zero that
            the mean direction is meaningless (the data are essentially
            uniform).
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        n = len(angles_deg)

        if n < 2:
            raise ValueError("Need at least 2 observations")

        if axial:
            # Wrap into [0, 180) and apply double-angle transform. The
            # resulting sample is on the 2θ circle (period π), so the
            # mean computed on φ is then halved back to the original axis.
            angles_wrapped = np.mod(angles_deg, 180.0)
            angles_rad = np.deg2rad(angles_wrapped)
            doubled_rad = (2.0 * angles_rad) % (2.0 * np.pi)
            data_type = "axial"
        else:
            angles_rad = np.deg2rad(np.mod(angles_deg, 360.0))
            doubled_rad = angles_rad
            data_type = "polar"

        # Resultant components on the working sample (φ for axial, θ for polar)
        C = np.sum(np.cos(doubled_rad))
        S = np.sum(np.sin(doubled_rad))
        R = np.sqrt(C**2 + S**2)
        R_bar = R / n

        # Mean direction. Treat as undefined when R̄ is so small that
        # the Rayleigh test statistic Z = n R̄² is below 1 — in that
        # regime the mean direction is statistically meaningless even
        # though arctan2(S, C) still returns a finite value. Compute Z
        # first so the threshold is principled.
        Z = n * R_bar**2
        mean_dir_undef = (Z < 1.0)
        if mean_dir_undef:
            mean_dir = None
            mean_dir_deg: float | None = None
            self._logger.warning(
                f"R̄ ≈ {R_bar:.3f} (Z = {Z:.3f}) — the Rayleigh test "
                "does not reject uniformity at the 5 % level; the mean "
                "direction is statistically undefined. Returning None."
            )
        else:
            mean_dir_phi = float(np.arctan2(S, C))
            if mean_dir_phi < 0:
                mean_dir_phi += 2 * np.pi
            if axial:
                # φ̄ halved back to the original [0, π) axis, expressed
                # in [0, 180) degrees.
                mean_dir_phi = mean_dir_phi / 2.0
                # wrap into [0, π) just in case of rounding
                if mean_dir_phi >= np.pi:
                    mean_dir_phi -= np.pi
                if mean_dir_phi < 0:
                    mean_dir_phi += np.pi
            mean_dir = mean_dir_phi
            mean_dir_deg = float(np.rad2deg(mean_dir_phi))

        # Circular variance and std
        V = 1 - R_bar
        R_bar_safe = min(max(R_bar, 0.0), 1.0)
        circ_std = np.sqrt(-2 * np.log(R_bar_safe)) if R_bar_safe > 0 else np.inf

        # Rayleigh test — exact form. For a sample of n directions the
        # Rayleigh Z = n R̄² has p-value given by
        #     p = exp(-Z) × Σ_{k=0}^{∞} (2Z)^k / (k! · 2^(k+1)) · ...
        # which scipy.stats.chi2.sf(2Z, df=2) evaluates in closed form
        # (Greenwood & Durand 1955). The Padé approximation previously
        # used here is accurate but only valid for moderate-to-large n;
        # the chi2.sf form is exact and used by the R circular package.
        p = float(_scipy_stats.chi2.sf(2.0 * Z, df=2))
        p = min(max(p, 0.0), 1.0)

        return DirectionalResult(
            mean_direction=mean_dir,
            mean_direction_deg=mean_dir_deg,
            resultant_length=float(R),
            mean_resultant=float(R_bar),
            circular_variance=float(V),
            circular_std=float(circ_std),
            rayleigh_z=float(Z),
            rayleigh_p=float(p),
            is_significant=p < 0.05,
            n_observations=n,
            raw_data=angles_deg,
            data_type=data_type,
        )

    def bin_for_rose(self, angles_deg: npt.NDArray, n_bins: int = 12) -> tuple[npt.NDArray, npt.NDArray]:
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
