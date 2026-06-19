# =============================================================================
# FILE: morphometrics/efa.py
# =============================================================================
"""
Elliptic Fourier Analysis (EFA) for PaleoAST

EFA decomposes closed contours into a sum of harmonically related
elliptic Fourier functions. Each harmonic adds 4 coefficients
(a, b, c, d) for x and y components.

Mathematical Foundation:

A closed contour (x(t), y(t)) parameterized by cumulative chord
length is approximated by N harmonics:

    x(t) = A_0 + Σ_{n=1}^{N} [a_n cos(2nπt/T) + b_n sin(2nπt/T)]
    y(t) = C_0 + Σ_{n=1}^{N} [c_n cos(2nπt/T) + d_n sin(2nπt/T)]

where T is the total perimeter length and t is the cumulative
chord length parameter.

The Fourier coefficients capture shape at different spatial frequencies:
    - Low harmonics: overall shape (ellipticity, triangularity)
    - High harmonics: fine detail (lobes, constrictions)

Reference: Kuhl & Giardina (1982) "Elliptic Fourier features of a
closed contour." Computer Graphics and Image Processing, 18, 236-258.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


@dataclass
class EFAHarmonic:
    """Fourier coefficients for a single harmonic."""

    n: int
    a: float
    b: float
    c: float
    d: float

    @property
    def amplitude_x(self) -> float:
        return np.sqrt(self.a**2 + self.b**2)

    @property
    def amplitude_y(self) -> float:
        return np.sqrt(self.c**2 + self.d**2)


@dataclass
class EFAResult:
    """Result of Elliptic Fourier Analysis."""

    harmonics: list[EFAHarmonic]
    coefficients: npt.NDArray  # (n_harmonics, 4) flat array
    n_harmonics: int
    n_points: int
    a0: float
    c0: float
    reconstructed: npt.NDArray  # (n_points, 2) reconstructed contour
    original: npt.NDArray  # (n_points, 2) original contour

    def summary(self) -> str:
        lines = [
            _("Elliptic Fourier Analysis"),
            "=" * 50,
            f"{_('Harmonics')}: {self.n_harmonics}",
            f"{_('Contour points')}: {self.n_points}",
            "",
            f"{'Harmonic':<10} {'a':>10} {'b':>10} {'c':>10} {'d':>10}",
            "-" * 55,
        ]
        for h in self.harmonics:
            lines.append(f"{h.n:<10} {h.a:>10.4f} {h.b:>10.4f} {h.c:>10.4f} {h.d:>10.4f}")
        return "\n".join(lines)


class EFAAnalyzer:
    """Elliptic Fourier Analysis engine."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.EFAAnalyzer")

    def analyze(
        self,
        contour: npt.NDArray,
        n_harmonics: int = 10,
        n_points: int = 200,
    ) -> EFAResult:
        """
        Perform EFA on a closed contour.

        Parameters:
            contour: (N, 2) array of (x, y) coordinates (closed or open)
            n_harmonics: Number of harmonics to compute
            n_points: Number of points for resampling

        Returns:
            EFAResult
        """
        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ComputationError("Contour must be (N, 2) array of (x, y) coordinates")

        # Close contour if not already
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # Resample contour to uniform spacing along perimeter
        resampled = self._resample_contour(contour, n_points)

        x = resampled[:, 0]
        y = resampled[:, 1]

        # Compute cumulative chord length parameter
        dx = np.diff(x)
        dy = np.diff(y)
        chord_lengths = np.sqrt(dx**2 + dy**2)
        t = np.zeros(n_points)
        t[1:] = np.cumsum(chord_lengths)
        T = t[-1]

        # Compute Fourier coefficients
        a0 = np.mean(x)
        c0 = np.mean(y)

        harmonics = []
        coefficients = []

        if T == 0:
            raise ValueError("EFA requires non-zero contour length (T=0). All contour points may be identical.")

        for n in range(1, n_harmonics + 1):
            omega = 2 * np.pi * n / T

            # Kuhl & Giardina (1982) closed-form coefficients for a
            # piecewise-linear contour. The contour is parameterised by
            # cumulative chord length t, so each segment k from t_{k-1}
            # to t_k has slope Δx_k / Δt_k and Δy_k / Δt_k. K&G integrate
            # the piecewise-constant derivative analytically, yielding:
            #
            #   a_n = (T / (2 π² n²)) Σ (Δx_k/Δt_k) · (cos ωt_k − cos ωt_{k-1})
            #   b_n = (T / (2 π² n²)) Σ (Δx_k/Δt_k) · (sin ωt_k − sin ωt_{k-1})
            #   c_n, d_n analogous with Δy_k/Δt_k
            #
            # The previous implementation integrated ``x(t) cos(ωt)`` by
            # the rectangle rule, which (a) is only a first-order
            # approximation of the integral and (b) is *not* the K&G
            # formulation — it loses the exactness property that makes
            # EFA invariant to the choice of starting point on a
            # piecewise-linear contour. Use the canonical K&G form.
            cos_t = np.cos(omega * t)
            sin_t = np.sin(omega * t)
            dcos = cos_t[1:] - cos_t[:-1]   # cos(ω t_k) − cos(ω t_{k-1})
            dsin = sin_t[1:] - sin_t[:-1]
            # Per-segment slopes (guard against zero-length segments).
            seg_len = np.diff(t)
            seg_len_safe = np.where(seg_len > 0, seg_len, 1.0)
            slope_x = dx / seg_len_safe
            slope_y = dy / seg_len_safe

            factor = T / (2.0 * (np.pi**2) * (n**2))
            a_n = factor * np.sum(slope_x * dcos)
            b_n = factor * np.sum(slope_x * dsin)
            c_n = factor * np.sum(slope_y * dcos)
            d_n = factor * np.sum(slope_y * dsin)

            harmonics.append(EFAHarmonic(n=n, a=a_n, b=b_n, c=c_n, d=d_n))
            coefficients.append([a_n, b_n, c_n, d_n])

        coefficients = np.array(coefficients)

        # Reconstruct contour
        reconstructed = self._reconstruct(a0, c0, harmonics, t, T, n_points)

        return EFAResult(
            harmonics=harmonics,
            coefficients=coefficients,
            n_harmonics=n_harmonics,
            n_points=n_points,
            a0=a0,
            c0=c0,
            reconstructed=reconstructed,
            original=resampled,
        )

    def reconstruct_from_coefficients(
        self,
        a0: float,
        c0: float,
        coefficients: npt.NDArray,
        n_points: int = 200,
        period: float | None = None,
    ) -> npt.NDArray:
        """Reconstruct a contour from Fourier coefficients.

        Parameters
        ----------
        a0, c0 : float
            DC (zeroth-harmonic) offsets along x and y.
        coefficients : ndarray of shape (n_harmonics, 4)
            Rows ``(a_n, b_n, c_n, d_n)`` for harmonics n = 1..N.
        n_points : int, default 200
            Number of points in the reconstructed contour.
        period : float, optional
            Contour period ``T`` used when the coefficients were
            originally computed (typically the cumulative chord length
            of the contour). If omitted, ``T = 2π`` is assumed, which is
            the correct inverse only when the coefficients were
            computed on a normalised ``t ∈ [0, 2π)`` parameterisation.
            Mismatching ``T`` between analysis and reconstruction
            stretches the harmonic frequencies and produces a visibly
            wrong contour, so callers that obtained the coefficients
            from :meth:`analyze` should pass the original perimeter.
        """
        n_harmonics = coefficients.shape[0]
        T = 2 * np.pi if period is None else period
        t = np.linspace(0, T, n_points, endpoint=False)

        x = np.full(n_points, a0)
        y = np.full(n_points, c0)

        for n in range(n_harmonics):
            a, b, c, d = coefficients[n]
            omega = (n + 1) * 2 * np.pi / T
            x += a * np.cos(omega * t) + b * np.sin(omega * t)
            y += c * np.cos(omega * t) + d * np.sin(omega * t)

        return np.column_stack([x, y])

    def _resample_contour(self, contour: npt.NDArray, n_points: int) -> npt.NDArray:
        """Resample contour to n_points with uniform chord-length spacing."""
        # Compute cumulative arc length
        dx = np.diff(contour[:, 0])
        dy = np.diff(contour[:, 1])
        arc_lengths = np.sqrt(dx**2 + dy**2)
        cum_arc = np.zeros(len(contour))
        cum_arc[1:] = np.cumsum(arc_lengths)
        total_length = cum_arc[-1]

        # New uniform parameter values
        new_arc = np.linspace(0, total_length, n_points, endpoint=False)

        # Interpolate x and y. Use ``np.interp`` directly — the previous
        # ``from numpy import interp`` form triggers a DeprecationWarning
        # under numpy >= 1.20 because ``numpy.interp`` is meant to be
        # accessed via the package namespace.
        x_new = np.interp(new_arc, cum_arc, contour[:, 0])
        y_new = np.interp(new_arc, cum_arc, contour[:, 1])

        return np.column_stack([x_new, y_new])

    def _reconstruct(
        self, a0: float, c0: float, harmonics: list, t: npt.NDArray, T: float, n_points: int
    ) -> npt.NDArray:
        """Reconstruct contour from coefficients."""
        x = np.full(n_points, a0)
        y = np.full(n_points, c0)

        for h in harmonics:
            omega = 2 * np.pi * h.n / T
            x += h.a * np.cos(omega * t) + h.b * np.sin(omega * t)
            y += h.c * np.cos(omega * t) + h.d * np.sin(omega * t)

        return np.column_stack([x, y])


@dataclass
class EigenshapeResult:
    """Result of Eigenshape analysis."""

    scores: npt.NDArray
    eigenvalues: npt.NDArray
    explained_variance: npt.NDArray
    cumulative_variance: npt.NDArray
    n_specimens: int
    n_components: int

    def summary(self) -> str:
        lines = [
            _("Eigenshape Analysis"),
            "=" * 50,
            f"{_('Specimens')}: {self.n_specimens}",
        ]
        cum = 0.0
        for i in range(min(self.n_components, 10)):
            cum += self.explained_variance[i]
            lines.append(f"ES{i + 1}: {self.explained_variance[i]:.2%} (cum: {cum:.2%})")
        return "\n".join(lines)


class EigenshapeAnalyzer:
    """Eigenshape analysis from EFA coefficients."""

    def analyze(
        self,
        efa_coefficients_list: list[npt.NDArray],
        n_components: int | None = None,
    ) -> EigenshapeResult:
        """
        Perform Eigenshape analysis on a set of EFA coefficient vectors.

        Parameters:
            efa_coefficients_list: List of (n_harmonics, 4) coefficient arrays
            n_components: Number of eigenshape components

        Returns:
            EigenshapeResult
        """
        # Flatten each specimen's coefficients into a single vector
        vectors = np.array([c.flatten() for c in efa_coefficients_list])
        n_specimens, n_vars = vectors.shape

        if n_components is None:
            n_components = min(n_specimens - 1, n_vars)

        # Center the data
        mean_vec = np.mean(vectors, axis=0)
        centered = vectors - mean_vec

        if n_specimens < 2:
            raise ValueError("Eigenshape analysis requires at least 2 specimens")

        # SVD-based PCA on the coefficient matrix
        _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = S**2 / (n_specimens - 1)
        explained = eigenvalues / np.sum(eigenvalues)
        cumulative = np.cumsum(explained)

        scores = centered @ Vt[:n_components].T

        return EigenshapeResult(
            scores=scores,
            eigenvalues=eigenvalues,
            explained_variance=explained,
            cumulative_variance=cumulative,
            n_specimens=n_specimens,
            n_components=n_components,
        )
