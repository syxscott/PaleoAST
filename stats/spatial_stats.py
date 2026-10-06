# =============================================================================
# FILE: stats/spatial_stats.py
# =============================================================================
"""
Spatial statistics that sit alongside Ripley's K: autocorrelation,
interpolation, nearest-neighbour pattern tests and spherical means.

WHY THIS IS SEPARATE FROM ``stats/spatial.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``stats/spatial.py`` holds Ripley's K, which answers one question -- is the
*point pattern* more clustered than random, as a function of distance? The
four analyses here answer the other spatial questions a palaeontologist
actually reaches for, and none of them is a function of radius, so none of
them fits the ``K(r)`` container:

1. **Moran's I** -- are *values* attached to sites spatially autocorrelated?
   Species richness on a grid, a taxon range, a delta value per core.
2. **Gridding** -- turn irregularly spaced samples into a regular surface
   to plot or to feed a further analysis.
3. **Nearest-neighbour index** -- the Clark & Evans style summary of the
   same pattern question K answers across all radii, in one number.
4. **Spherical statistics** -- the mean of *directions* (paleocurrent sets,
   magnetic polarity vectors, aligned elongate fossils) on S^2.

``stats/spatial.py`` is left alone: its behaviour is depended on elsewhere.

WHY A PERMUTATION TEST AND NOT A NORMAL APPROXIMATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Moran's I is conventionally reported with a z-score against a normal
approximation, ``z = (I - E[I]) / SE[I]``. That approximation is only
reasonable for large *n*. A single fossil site sample is emphatically not:
at ``n = 25`` the sampling distribution of I is visibly skewed and its
variance depends on the weight structure in a way the textbook SE does not
capture. So the p-value here comes from permuting the values among the
sites -- the exact conditional distribution the statistic was defined
against -- and the z-score is reported *from that same permuted null*,
so the two numbers cannot disagree. Reporting the normal-approximation
p-value alone would claim a precision these samples do not have.

WHY THE SPHERICAL PART IS NOT A REUSE OF ``stratigraphy/directional.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
That module reduces its input to scalar 2-D angles and works on the unit
circle. Its circular mean and its Rayleigh R are functions of a *single*
angle per observation. Spherical statistics are a different space: the mean
direction is the normalised 3-vector sum, the resultant length is that
sum's magnitude, and the mean resultant is normalised by *n*, not by
``n - 1`` -- the ``n - 1`` convention belongs to unbiased sample variance,
not to a resultant, and mixing the two is the classic error in this
family of statistics. ``stratigraphy/directional.py`` is not a 2-D special
case that can be widened, so this is genuinely new code; the tests pin the
reduction by comparing against an independently written atan2 circular
mean.

SCOPE: KRIGING IS NOT HERE
~~~~~~~~~~~~~~~~~~~~~~~~~~
Ordinary kriging is not implemented. It is not a variant of inverse
distance weighting with a different name: the estimate depends on a fitted
*variogram model* (nugget, sill, range, and possibly a functional form),
and the prediction variance, the kriging weights and the assessment of
whether the fit is admissible are the analysis. Shipping a "kriging" label
over an IDW formula, or over a spherical-model variogram with no way to
check the fit, would produce a surface that looks defensible and is not.
It deserves its own pass with the variogram fitting as a first-class step.

References
----------
1. Moran, P. A. 1950. Notes on continuous stochastic phenomena. Biometrika
   37: 17-23.
2. Cliff, N., Ord, J. K. & Thompson, A. R. 1974. Spatial autocorrelation
   and regional analysis. Transactions of the Institute of British
   Geographers 40: 119-128. (the E[I] = -1/(n-1) expectation)
3. Legendre, P. & Legendre, L. 2012. Numerical Ecology, 3rd ed. Elsevier,
   §14. (the conditional permutation distribution, §14.1)
4. Shepard, D. 1968. A two-dimensional interpolation function for
   irregularly-spaced data. Proceedings of the 23rd ACM National Conference.
5. Cressie, N. 1993. Statistics for Spatial Data, 2nd ed. Wiley. (§2.2,
   the Rayleigh nearest-neighbour distance distribution under CSR)
6. Clark, P. J. & Evans, F. C. 1954. On some aspects of spatial pattern
   measures. Biometrics 10: 287-297.
7. Fisher, N. I. 1953. Dispersion on a sphere. Proceedings of the Royal
   Society of London A 217: 295-305. (R-bar, the small-sample kappa
   correction, and the confidence cone)
8. Mardia, K. V. & Jupp, P. E. 2000. Directional Statistics. Wiley. (the
   von Mises-Fisher density and its angular cdf)

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.spatial import Delaunay, cKDTree

from config.i18n import _
from stats.geometry import GeometryAnalyzer
from utils.exceptions import ComputationError, DataValidationError, MatrixDimensionError
from utils.statistics_core import permutation_pvalue
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

# ``ValidationError`` is an alias of ``DataValidationError`` in
# utils.exceptions, so importing both would import the same class twice.
# Input problems are raised as ``DataValidationError`` here.

__all__ = [
    "GridInterpolationResult",
    "MoransIResult",
    "NearestNeighbourResult",
    "SpatialStatsAnalyzer",
    "SphericalStatsResult",
    "contiguity_weights",
]

# Moran's I needs at least 4 sites: below that the permutation null has so
# few distinct arrangements that a p-value means nothing.
MORANS_MIN_POINTS = 4

# Additive guard in the inverse-distance kernel, as a fraction of the median
# inter-site distance. Only ever applied to *weights*, never to the surface
# estimate: a site sitting exactly on a prediction node is handled by
# returning its own value, which is exact and needs no epsilon at all.
_IDW_EPS_FRACTION = 1e-9

# R-bar at or above this is treated as "no dispersion at all" (all vectors
# identical), where Fisher's kappa is 0/0 and the answer is +inf.
_PERFECT_CONCENTRATION = 1.0 - 1e-12

# R-bar at or below this means the weighted vector sum is zero, so the mean
# direction does not exist. Any non-degenerate sample is well above it; a
# set of exactly cancelling vectors is at the 1e-16 level.
_MIN_MEAN_CONCENTRATION = 1e-8


# =============================================================================
# Shared validation
# =============================================================================


def _as_points(
    coordinates: npt.NDArray,
    *,
    min_points: int,
    min_dims: int,
    name: str = "coordinates",
) -> npt.NDArray:
    """Validate a coordinate block and return it as a float ``(n, d)`` array.

    ``preserve_dimensions=True`` is essential: the default silently turns a
    1-D input into an ``(n, 1)`` array, which would let a single coordinate
    column through every shape check.
    """
    pts = validate_data_array(
        coordinates,
        allow_nan=False,
        allow_inf=False,
        name=name,
        preserve_dimensions=True,
    )
    if pts.ndim != 2:
        raise MatrixDimensionError(
            f"{name} must be a 2-D array of points with shape (n, d), got shape {pts.shape}",
            details={"shape": tuple(int(x) for x in pts.shape)},
        )
    if pts.shape[1] < min_dims:
        raise MatrixDimensionError(
            f"{name} needs at least {min_dims} dimension(s), got {pts.shape[1]}",
            details={"n_dims": int(pts.shape[1]), "required": min_dims},
        )
    if pts.shape[0] < min_points:
        raise DataValidationError(
            f"{name} needs at least {min_points} points, got {pts.shape[0]}",
            details={"n_points": int(pts.shape[0]), "required": min_points},
        )
    return pts


def _reject_coincident(points: npt.NDArray, *, what: str) -> None:
    """Refuse a point set with two identical members.

    Coincident sites are not a near-duplicate to be tolerated: a nearest
    neighbour distance of zero makes a mean NN distance meaningless, and a
    zero off-diagonal in a contiguity matrix makes that row normalise to
    nothing.
    """
    if points.shape[0] < 2:
        return
    uniques = np.unique(np.round(points, 12), axis=0)
    if uniques.shape[0] < points.shape[0]:
        raise DataValidationError(
            f"{what}: {points.shape[0] - uniques.shape[0]} point(s) are exactly "
            "coincident. Distinct sites must have distinct coordinates -- a "
            "zero distance makes the spatial statistic undefined rather than "
            "merely inaccurate.",
            details={
                "n_points": int(points.shape[0]),
                "n_unique": int(uniques.shape[0]),
            },
        )


def contiguity_weights(
    coordinates: npt.NDArray,
    *,
    method: str = "knn",
    k: int = 4,
) -> npt.NDArray:
    """Build a binary contiguity matrix from coordinates.

    Parameters
    ----------
    coordinates:
        Point coordinates, ``(n, d)`` with ``d >= 2``.
    method:
        ``"knn"`` -- each site is joined to its ``k`` nearest neighbours, and
        the relation is symmetrised so the matrix describes adjacency rather
        than a directed "who was asked". The k-nearest rule needs no distance
        scale, which is the point: choosing a bandwidth for a quarry that is
        8 m across in one dimension and 400 m in the other is a judgement
        call, and a different choice would give a different I.
        ``"delaunay"`` -- edges of the Delaunay triangulation, which is the
        natural contiguity for a scattered sample because it is the planar
        graph whose faces contain no other point.
    k:
        Neighbours per site for ``"knn"``; clipped to ``n - 1``.

    Returns
    -------
    npt.NDArray
        Symmetric ``(n, n)`` matrix of ones, zero diagonal. Pass it straight
        to :meth:`SpatialStatsAnalyzer.morans_i`, which row-standardises it.
    """
    pts = _as_points(coordinates, min_points=MORANS_MIN_POINTS, min_dims=2)
    _reject_coincident(pts, what="contiguity_weights")
    n = pts.shape[0]
    w = np.zeros((n, n), dtype=float)

    if method == "knn":
        kk = int(min(max(int(k), 1), n - 1))
        _dist, idx = cKDTree(pts).query(pts, k=kk + 1)
        rows = np.repeat(np.arange(n), kk)
        w[rows, np.asarray(idx)[:, 1:].ravel()] = 1.0
        # Symmetrise: a directed k-nearest relation is not a contiguity, and
        # an asymmetric W is not row-standardisable symmetrically.
        w = np.maximum(w, w.T)
    elif method == "delaunay":
        try:
            triangulation = Delaunay(pts)
        except Exception as exc:  # Qhull raises its own error type
            raise ComputationError(
                f"contiguity_weights: Delaunay triangulation of the {n} points "
                f"failed ({exc}). The points are collinear or coincident, so no "
                "planar contiguity exists.",
                details={"n_points": n, "method": method},
            ) from exc
        # scipy's Delaunay exposes the triangles as ``simplices``, not as an
        # edge list; the contiguity is the union of each triangle's three
        # sides, which is the planar graph with no point inside any face.
        triangles = np.asarray(triangulation.simplices, dtype=int)
        for tri in triangles:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((i, j), (j, k), (i, k)):
                w[u, v] = 1.0
                w[v, u] = 1.0
    else:
        raise DataValidationError(
            f"contiguity_weights: method must be 'knn' or 'delaunay', got {method!r}",
            details={"method": method},
        )

    np.fill_diagonal(w, 0.0)
    return w


# =============================================================================
# Moran's I
# =============================================================================


def _morans_i_statistic(values: npt.NDArray, weights: npt.NDArray) -> float:
    """Moran's I for a ready weight matrix.

        I = (n / S0) * sum_ij w_ij z_i z_j / sum_i z_i^2,   z = values - mean

    The permutation p-value and the z-score both call this with the same
    weights, so only the value vector changes between the observed statistic
    and each of the null replicates. That is exactly the conditional null
    the test is defined against, and it is why the weights are row-
    standardised once, outside the loop, rather than per replicate.
    """
    z = values - float(np.mean(values))
    denominator = float(np.sum(z * z))
    s0 = float(np.sum(weights))
    if denominator <= 0.0 or s0 <= 0.0:
        return float("nan")
    return (values.size / s0) * float(z @ weights @ z) / denominator


@dataclass
class MoransIResult:
    """Outcome of a Moran's I test.

    Attributes:
        statistic: Observed Moran's I.
        expected: ``-1 / (n - 1)``, the expectation under no spatial
            autocorrelation (Cliff, Ord & Thompson 1974).
        z_score: Observed I standardised against the *permuted* null, not
            against the normal approximation.
        p_value: Two-sided permutation p-value.
        p_value_greater: One-sided p for "more clustered than random".
        n_obs: Number of sites.
        n_permutations: Number of permutations run.
        weight_scheme: ``"inverse-distance"``, ``"supplied"`` or the
            ``"knn"/"delaunay"`` contiguity actually used.
        weights: The row-standardised weight matrix, carried for plotting.
    """

    statistic: float
    expected: float
    z_score: float
    p_value: float
    p_value_greater: float
    n_obs: int
    n_permutations: int
    weight_scheme: str
    weights: npt.NDArray

    @property
    def significant(self) -> bool:
        """Whether the test rejects at the 0.05 level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Generate summary text."""
        verdict = _("CLUSTERED") if self.statistic > 0 else _("DISPERSED")
        if not self.significant:
            verdict = _("not significant")
        return (
            f"{_('Morans I (Global spatial autocorrelation)')}\n"
            f"{'=' * 45}\n"
            f"{_('Number of sites: {0}').format(self.n_obs)}\n"
            f"{_('Weights: {0}').format(self.weight_scheme)}\n"
            f"{_('Morans I: {0:.4f}').format(self.statistic)}\n"
            f"{_('Expected I (no autocorrelation): {0:.4f}').format(self.expected)}\n"
            f"{_('Z-score: {0:.4f}').format(self.z_score)}\n"
            f"{_('p-value (two-sided, permutation): {0:.4f}').format(self.p_value)}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"{_('Interpretation:')} {verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, without the weight matrix."""
        return {
            "statistic": float(self.statistic),
            "expected": float(self.expected),
            "z_score": float(self.z_score),
            "p_value": float(self.p_value),
            "p_value_greater": float(self.p_value_greater),
            "n_obs": int(self.n_obs),
            "n_permutations": int(self.n_permutations),
            "weight_scheme": self.weight_scheme,
            "significant": self.significant,
        }


# =============================================================================
# Gridding
# =============================================================================


@dataclass
class GridInterpolationResult:
    """An interpolated surface and the fit at the sample locations.

    Attributes:
        grid_x / grid_y: The regular grid axes, ``n_points`` each.
        surface: Interpolated values, ``surface[row, col]`` is the value at
            ``(grid_x[col], grid_y[row])``.
        residuals: ``values - fitted`` at each sample location. For IDW these
            are exactly zero, because a sample is its own nearest neighbour;
            a non-zero entry here is the signal that a sample was not
            reproduced.
        method / power / k: Interpolation settings as supplied.
        n_points: Number of samples.
        n_grid: Grid size per axis.
        rmse: Root mean squared residual, zero for an exact interpolator.
    """

    grid_x: npt.NDArray
    grid_y: npt.NDArray
    surface: npt.NDArray
    residuals: npt.NDArray
    method: str
    power: float
    k: int
    n_points: int
    n_grid: int
    rmse: float

    @property
    def max_abs_residual(self) -> float:
        """Largest absolute residual at a sample."""
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Spatial Interpolation')}\n"
            f"{'=' * 45}\n"
            f"{_('Method: {0}').format(self.method)}\n"
            f"{_('Samples: {0}').format(self.n_points)}\n"
            f"{_('Grid: {0} x {0}').format(self.n_grid)}\n"
            f"{_('Neighbours used (k): {0}').format(self.k)}\n"
            f"{_('IDW power: {0:.2f}').format(self.power)}\n"
            f"{_('RMSE at samples: {0:.6f}').format(self.rmse)}\n"
            f"{_('Max |residual| at samples: {0:.6f}').format(self.max_abs_residual)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, without the surface or residual arrays."""
        return {
            "method": self.method,
            "power": float(self.power),
            "k": int(self.k),
            "n_points": int(self.n_points),
            "n_grid": int(self.n_grid),
            "rmse": float(self.rmse),
            "max_abs_residual": self.max_abs_residual,
        }


# =============================================================================
# Nearest neighbour
# =============================================================================


@dataclass
class NearestNeighbourResult:
    """Outcome of a nearest-neighbour point pattern analysis.

    Attributes:
        nn_distances: Nearest-neighbour distance for every site *used*, which
            is the reduced sample when the edge correction is on.
        mean_nn: Observed mean nearest-neighbour distance.
        expected_nn: The CSR expectation ``0.5 * sqrt(A / n)``, the mean of a
            Rayleigh variate with rate ``lambda * pi`` where
            ``lambda = n / A``.
        index: ``mean_nn / expected_nn``. Below 1 the sites sit closer
            together than random (clustering), above 1 they are further
            apart than random (regularity / inhibition), 1 is CSR.
        z_score: ``mean_nn`` standardised against the Monte Carlo CSR null.
        p_value: Two-sided Monte Carlo p-value, counting null patterns at
            least as far from the null mean as the observed one.
        n_sites: Number of sites the statistics were computed on.
        n_sites_total: Number of sites supplied, before any edge reduction.
        area: The window the statistics were computed on -- the inscribed
            window when the edge correction is on, so that the CSR reference
            and the Monte Carlo null use the same area.
        edge_shrink: Border width excluded by the edge correction, equal to
            the observed mean NN distance. Zero when the correction is off or
            could not be applied.
        n_simulations: Monte Carlo replicates run.
        edge_correction: Whether the CSR null was generated in the inscribed
            (reduced-sample) window.
        r_values: Radii at which the NN distance cdf was tabulated, or None.
        cumulative_fraction: Observed fraction of sites with ``NN <= r``.
        csr_expected_fraction: ``1 - exp(-pi r^2 n / A)``, the infinite-window
            Poisson cdf. It is an analytic reference curve and carries no edge
            correction, so near the window border the observed fraction sits
            above it even for a perfectly random pattern.
    """

    nn_distances: npt.NDArray
    mean_nn: float
    expected_nn: float
    index: float
    z_score: float
    p_value: float
    n_sites: int
    n_sites_total: int
    area: float
    edge_shrink: float
    n_simulations: int
    edge_correction: bool
    r_values: npt.NDArray | None = None
    cumulative_fraction: npt.NDArray | None = None
    csr_expected_fraction: npt.NDArray | None = None

    @property
    def significant(self) -> bool:
        """Whether the Monte Carlo test rejects at the 0.05 level."""
        return self.p_value < 0.05

    @property
    def pattern(self) -> str:
        """Clustered, regular or random, judged against the CSR index."""
        if self.index < 1.0:
            return _("CLUSTERED")
        if self.index > 1.0:
            return _("REGULAR")
        return _("RANDOM")

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Nearest Neighbour Analysis')}\n"
            f"{'=' * 45}\n"
            f"{_('Number of sites: {0}').format(self.n_sites)}\n"
            f"{_('Sites supplied: {0}').format(self.n_sites_total)}\n"
            f"{_('Study area: {0:.4f}').format(self.area)}\n"
            f"{_('Mean NN distance: {0:.4f}').format(self.mean_nn)}\n"
            f"{_('Expected NN distance (CSR): {0:.4f}').format(self.expected_nn)}\n"
            f"{_('R index (observed/expected): {0:.4f}').format(self.index)}\n"
            f"{_('Z-score (Monte Carlo CSR): {0:.4f}').format(self.z_score)}\n"
            f"{_('p-value: {0:.4f}').format(self.p_value)}\n"
            f"{_('Monte Carlo simulations: {0}').format(self.n_simulations)}\n"
            f"{_('Edge correction: {0}').format(_('on') if self.edge_correction else _('off'))}\n"
            f"{_('Interpretation:')} {self.pattern}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, without the distance vectors."""
        return {
            "mean_nn": float(self.mean_nn),
            "expected_nn": float(self.expected_nn),
            "index": float(self.index),
            "z_score": float(self.z_score),
            "p_value": float(self.p_value),
            "n_sites": int(self.n_sites),
            "n_sites_total": int(self.n_sites_total),
            "area": float(self.area),
            "edge_shrink": float(self.edge_shrink),
            "n_simulations": int(self.n_simulations),
            "edge_correction": bool(self.edge_correction),
            "pattern": str(self.pattern),
        }


# =============================================================================
# Spherical statistics
# =============================================================================


def _von_mises_fisher_cdf(omega: float, kappa: float) -> float:
    """Probability that an angle from the mean direction is at most ``omega``.

    For the vMF density ``p(theta) = kappa sin(theta) exp(kappa cos theta) /
    (2 sinh kappa)`` the angular cdf is

        P(theta <= omega) = (exp(kappa) - exp(kappa cos omega)) / (2 sinh kappa)

    Factoring out ``exp(kappa)`` and writing both halves with ``expm1`` keeps
    it finite as kappa -> 0, where it tends to ``(1 - cos omega) / 2`` -- the
    uniform distribution, which is the correct answer for "no preferred
    direction". A naive ``exp(kappa)`` difference cancels to 0/0 there.
    """
    numerator = -np.expm1(kappa * (np.cos(omega) - 1.0))
    denominator = -np.expm1(-2.0 * kappa)
    return float(numerator / denominator)


def _confidence_cone(kappa: float, probability: float = 0.95) -> float:
    """Half-angle of the smallest cone about the mean holding ``probability``.

    Fisher's large-kappa form is ``cos(omega) = 1 + ln(1 - p) / kappa``; this
    solves the exact vMF cdf instead, so the answer is right for the weakly
    concentrated samples where the approximation is not.

    Validity: as ``kappa -> 0`` the cone grows towards the uniform limit
    (154 degrees at 95 %), so a wide cone means "no preferred direction",
    not a well-measured but imprecise one. At small ``n`` the estimate of
    kappa itself is poor -- Fisher's correction is a first-order fix for
    that bias, not a cure -- so the cone inherits it. Both caveats are why
    this is reported next to kappa and R-bar rather than on its own.
    """
    if not np.isfinite(kappa):
        # kappa == +inf is the degenerate "every vector identical" case,
        # where the cone collapses to the mean direction.
        return 0.0 if kappa > 0 else float("nan")
    if kappa <= 0.0:
        # The vMF model is unidentifiable at kappa <= 0; report the uniform
        # limit rather than a negative kappa's meaningless cone.
        return float(np.arccos(1.0 - 2.0 * probability))
    return float(
        brentq(
            lambda omega: _von_mises_fisher_cdf(omega, kappa) - probability,
            0.0,
            float(np.pi),
            xtol=1e-10,
        )
    )


@dataclass
class SphericalStatsResult:
    """Outcome of a spherical (S^2) mean direction analysis.

    Attributes:
        mean_direction: Unit vector along the mean direction, or None when
            R is too small for the mean to be defined.
        mean_azimuth_deg: Azimuth of the mean direction in [0, 360).
        mean_polar_deg: Polar angle of the mean direction in [0, 180],
            measured from +z.
        resultant_length: ``R = |sum_i w_i u_i|``, at most ``sum(w)``.
        mean_resultant_length: ``R / sum(w)``. The divisor is ``n``, not
            ``n - 1``: identical vectors give R = n and R-bar = 1, and that
            is the reduction the tests pin.
        kappa: Concentration, with Fisher's (1953) small-sample correction.
            ``inf`` when every vector is identical.
        cone_half_angle_deg / _rad: 95 % confidence cone half-angle.
        n_vectors: Number of vectors.
        effective_n: ``sum(weights)``; the ``n`` Fisher's formulas use.
        concentrated: Whether R-bar is high enough for the mean direction to
            be meaningful.
    """

    mean_direction: npt.NDArray | None
    mean_azimuth_deg: float
    mean_polar_deg: float
    resultant_length: float
    mean_resultant_length: float
    kappa: float
    cone_half_angle_deg: float
    cone_half_angle_rad: float
    n_vectors: int
    effective_n: float
    concentrated: bool

    def summary(self) -> str:
        """Generate summary text."""
        if self.concentrated and self.mean_direction is not None:
            where = (
                f"{_('Azimuth')}: {self.mean_azimuth_deg:.1f} deg, {_('polar angle')}: {self.mean_polar_deg:.1f} deg"
            )
        else:
            where = _("undefined (R-bar ~ 0, no preferred direction)")
        kappa_str = "inf" if np.isinf(self.kappa) else f"{self.kappa:.4f}"
        return (
            f"{_('Spherical Statistics (mean direction on the unit sphere)')}\n"
            f"{'=' * 45}\n"
            f"{_('Vectors: {0}').format(self.n_vectors)}\n"
            f"{_('Effective n (sum of weights): {0:.2f}').format(self.effective_n)}\n"
            f"{_('Mean direction:')} {where}\n"
            f"{_('Resultant length R: {0:.4f}').format(self.resultant_length)}\n"
            f"{_('Mean resultant length R-bar: {0:.4f}').format(self.mean_resultant_length)}\n"
            f"{_('Concentration kappa (Fisher, corrected): {0}').format(kappa_str)}\n"
            f"{_('95% confidence cone half-angle: {0:.2f} deg').format(self.cone_half_angle_deg)}\n"
            f"{_('Interpretation:')} {_('concentrated') if self.concentrated else _('dispersed')}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "mean_direction": ([float(x) for x in self.mean_direction] if self.mean_direction is not None else None),
            "mean_azimuth_deg": float(self.mean_azimuth_deg),
            "mean_polar_deg": float(self.mean_polar_deg),
            "resultant_length": float(self.resultant_length),
            "mean_resultant_length": float(self.mean_resultant_length),
            "kappa": float(self.kappa),
            "cone_half_angle_deg": float(self.cone_half_angle_deg),
            "cone_half_angle_rad": float(self.cone_half_angle_rad),
            "n_vectors": int(self.n_vectors),
            "effective_n": float(self.effective_n),
            "concentrated": bool(self.concentrated),
        }


# =============================================================================
# Analyzer
# =============================================================================


class SpatialStatsAnalyzer:
    """Spatial autocorrelation, interpolation and direction analysis.

    A single entry point for the four analyses, each of which sets
    ``last_result`` and keeps its own ``last_<analysis>`` handle. All
    randomness goes through the shared permutation driver, so a seeded call
    is reproducible and an unseeded one never disturbs the global
    ``np.random`` stream.
    """

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.SpatialStatsAnalyzer")
        self._lock = threading.RLock()
        # Reused rather than reimplemented: pairwise_distances is the one
        # genuinely generic (n, n_dims) spatial primitive in the codebase and
        # Moran's I needs exactly that from coordinates.
        self._geometry = GeometryAnalyzer()
        self._last_result: Any = None
        self._last_morans_i: MoransIResult | None = None
        self._last_interpolation: GridInterpolationResult | None = None
        self._last_nearest_neighbour: NearestNeighbourResult | None = None
        self._last_spherical: SphericalStatsResult | None = None
        self._logger.info("SpatialStatsAnalyzer initialized")

    # -- Moran's I ---------------------------------------------------------

    def morans_i(
        self,
        values: npt.NDArray,
        coordinates: npt.NDArray | None = None,
        weights: npt.NDArray | None = None,
        permutations: int = 999,
        random_seed: int | None = None,
    ) -> MoransIResult:
        """Global spatial autocorrelation of one value per site.

        Parameters
        ----------
        values:
            One value per site, ``(n,)``.
        coordinates:
            Site coordinates, ``(n, d)`` with ``d >= 2``. Required unless
            ``weights`` is given.
        weights:
            Contiguity matrix, ``(n, n)``. Required unless ``coordinates`` is
            given. Zero diagonal expected; it is forced to zero if not, since
            a non-zero diagonal is a constant added to every pair and
            therefore no information about neighbours. The matrix is
            row-standardised.
        permutations:
            Number of permutations for the null. The default 999 gives a
            minimum attainable p-value of 0.001.
        random_seed:
            Seed for the permutation. Unseeded calls are not reproducible.

        Returns
        -------
        MoransIResult

        Notes
        -----
        The default contiguity, when only coordinates are given, is
        **inverse distance over all pairs**: ``w_ij = 1 / (d_ij + eps)``
        with ``eps`` a billionth of the median inter-site distance. Chosen
        over a fixed-distance neighbourhood and over k-nearest because both
        of those need a *spatial scale*, and a fossil site sample has no
        defensible one -- the same quarry outcrop may be 8 m tall and 400 m
        along strike, and a bandwidth that suits one axis destroys the
        other's structure. Inverse distance needs no scale choice, degrades
        gracefully, and gives a strictly positive weight to every pair, so
        no site is ever isolated. If a defensible scale *is* known, pass
        ``weights`` (see :func:`contiguity_weights`) instead.
        """
        with self._lock:
            vals = validate_data_array(
                values,
                allow_nan=False,
                allow_inf=False,
                name="values",
                preserve_dimensions=True,
            )
            if vals.ndim != 1:
                raise MatrixDimensionError(
                    f"Moran's I needs one value per site, a 1-D array; got "
                    f"shape {vals.shape}. Reshape to (n,) explicitly rather "
                    "than having it guessed.",
                    details={"shape": tuple(int(x) for x in vals.shape)},
                )
            n = int(vals.shape[0])
            if n < MORANS_MIN_POINTS:
                raise DataValidationError(
                    f"Moran's I needs at least {MORANS_MIN_POINTS} sites, got "
                    f"{n}. Below that the permutation null has too few "
                    "distinct arrangements for a p-value to mean anything.",
                    details={"n_obs": n, "required": MORANS_MIN_POINTS},
                )
            if float(np.ptp(vals)) == 0.0:
                raise ComputationError(
                    "Moran's I is undefined for a constant variable: with zero "
                    "variance the numerator and denominator are both zero, so "
                    "the ratio is 0/0. The statistic would have to invent an "
                    "answer, and every site is equally autocorrelated with "
                    "every other in the only sense available.",
                    details={"n_obs": n},
                )

            if weights is None and coordinates is None:
                raise DataValidationError(
                    "Moran's I needs either coordinates or a weight matrix; both were None",
                    details={"n_obs": n},
                )

            if weights is not None:
                w, scheme = self._prepare_weights(weights, n)
            else:
                pts = _as_points(coordinates, min_points=MORANS_MIN_POINTS, min_dims=2)
                if int(pts.shape[0]) != n:
                    raise MatrixDimensionError(
                        f"Moran's I: got {n} values but {pts.shape[0]} "
                        "coordinate rows; there must be one value per site",
                        details={"n_values": n, "n_coordinates": int(pts.shape[0])},
                    )
                _reject_coincident(pts, what="Moran's I")
                w, scheme = self._inverse_distance_weights(pts)

            observed = _morans_i_statistic(vals, w)
            expected = -1.0 / (n - 1)

            def _permuted(rng: np.random.Generator) -> float:
                return _morans_i_statistic(vals[rng.permutation(n)], w)

            perm = permutation_pvalue(
                observed,
                _permuted,
                n_permutations=permutations,
                random_seed=random_seed,
                context="Moran's I",
                alternative="two_sided",
            )
            null = perm.null_distribution
            null_sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
            z_score = (observed - float(np.mean(null))) / null_sd if null_sd > 0 else float("nan")

            result = MoransIResult(
                statistic=float(observed),
                expected=float(expected),
                z_score=float(z_score),
                # Two-sided is the headline: I is signed, so "more clustered
                # than random" is only half the question, and the one-sided
                # form reports p = 1.0 for a perfectly dispersed pattern
                # however significant that pattern is.
                p_value=float(perm.p_value_two_sided),
                p_value_greater=float(perm.p_value),
                n_obs=n,
                n_permutations=int(perm.n_permutations),
                weight_scheme=scheme,
                weights=w,
            )
            self._last_morans_i = result
            self._last_result = result
            self._logger.info(
                "Moran's I completed: I=%.4f (E=%.4f), z=%.2f, p=%.4f, weights=%s",
                result.statistic,
                result.expected,
                result.z_score,
                result.p_value,
                scheme,
            )
            return result

    def _inverse_distance_weights(self, points: npt.NDArray) -> tuple[npt.NDArray, str]:
        """Inverse-distance contiguity over every pair, row-standardised."""
        distances = self._geometry.pairwise_distances(points, metric="euclidean")
        n = points.shape[0]
        off_diagonal = ~np.eye(n, dtype=bool)
        off_values = distances[off_diagonal]
        eps = _IDW_EPS_FRACTION * float(np.median(off_values))
        weights = np.zeros((n, n), dtype=float)
        weights[off_diagonal] = 1.0 / (off_values + eps)
        return self._row_standardise(weights, "inverse-distance")

    @staticmethod
    def _row_standardise(weights: npt.NDArray, scheme: str) -> tuple[npt.NDArray, str]:
        """Row-standardise W and confirm no site is isolated.

        Row-standardisation is the usual convention: each site then averages
        its own neighbours' values with weights summing to one, and the
        permutation expectation ``-1/(n-1)`` is unchanged because the
        standardisation of a symmetric zero-diagonal matrix is itself
        symmetric.
        """
        row_sums = np.sum(weights, axis=1, keepdims=True)
        isolated = np.flatnonzero(row_sums.ravel() <= 0)
        if isolated.size:
            raise ComputationError(
                f"Moran's I: {isolated.size} site(s) have no neighbours under the "
                f"{scheme} weight scheme (first: index {int(isolated[0])}), so "
                "they cannot be row-standardised. Use a wider neighbourhood or "
                "a different contiguity.",
                details={"n_isolated": int(isolated.size)},
            )
        return weights / row_sums, scheme

    def _prepare_weights(self, weights: npt.NDArray, n: int) -> tuple[npt.NDArray, str]:
        """Validate a supplied weight matrix and row-standardise it."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise MatrixDimensionError(
                f"Moran's I: weights must be a square (n, n) matrix, got shape {w.shape}",
                details={"shape": tuple(int(x) for x in w.shape)},
            )
        if w.shape[0] != n:
            raise MatrixDimensionError(
                f"Moran's I: got {n} values but a {w.shape[0]}x{w.shape[1]} "
                "weight matrix; there must be one value per site",
                details={"n_values": n, "weight_shape": tuple(int(x) for x in w.shape)},
            )
        if not np.all(np.isfinite(w)):
            raise DataValidationError(
                "Moran's I: weights contain non-finite values",
                details={"n_bad": int(np.sum(~np.isfinite(w)))},
            )
        if np.any(w < 0):
            raise DataValidationError(
                "Moran's I: weights must be non-negative; a negative weight makes the contiguity meaningless",
                details={"n_negative": int(np.sum(w < 0))},
            )
        # Symmetrise defensively: an asymmetric W breaks the row
        # standardisation into something that is no longer a spatial
        # average, and the -1/(n-1) expectation with it.
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
        return self._row_standardise(w, "supplied")

    # -- Gridding ----------------------------------------------------------

    def grid_interpolate(
        self,
        values: npt.NDArray,
        coordinates: npt.NDArray,
        method: str = "idw",
        power: float = 2.0,
        n_points: int = 40,
        k: int = 8,
    ) -> GridInterpolationResult:
        """Interpolate scattered samples onto a regular grid.

        Parameters
        ----------
        values:
            One value per sample, ``(n,)``.
        coordinates:
            Sample locations, ``(n, 2)``. Two dimensions only: a gridding
            surface is a map-view construct, and silently griddng the first
            two of three coordinate columns would drop the third without
            saying so.
        method:
            ``"idw"`` -- inverse distance weighting (Shepard 1968) with
            weights ``w = 1 / d^p`` over the ``k`` nearest samples, and
            ``"nearest"`` -- the value of the closest sample, kept as the
            reference that IDW is compared against. ``"kriging"`` is
            rejected by name: it needs a fitted variogram model, not a
            different exponent.
        power:
            Exponent for IDW. 2.0 is the common default; larger values
            sharpen the surface towards ``"nearest"``.
        n_points:
            Grid nodes per axis, spanning the bounding box of the samples.
        k:
            Neighbours per node for IDW, clipped to ``n``. ``k = n``
            reproduces every sample exactly (see the test).

        Returns
        -------
        GridInterpolationResult
            Including the residuals at the sample locations, which are
            exactly zero for both methods here because a sample is its own
            nearest neighbour. A non-zero residual is therefore a bug
            signal, not a fit statistic.
        """
        with self._lock:
            vals = validate_data_array(
                values,
                allow_nan=False,
                allow_inf=False,
                name="values",
                preserve_dimensions=True,
            )
            if vals.ndim != 1:
                raise MatrixDimensionError(
                    f"grid_interpolate: values must be 1-D with one entry per sample, got shape {vals.shape}",
                    details={"shape": tuple(int(x) for x in vals.shape)},
                )
            if method not in ("idw", "nearest"):
                if method == "kriging":
                    raise DataValidationError(
                        "grid_interpolate: kriging is not implemented. It is "
                        "not a variant of inverse distance weighting -- the "
                        "estimate depends on a fitted variogram model (nugget, "
                        "sill, range, and often a functional form), together "
                        "with a prediction variance and a check that the fit "
                        "is admissible. Choose 'idw' or 'nearest', or fit a "
                        "variogram separately first.",
                        details={"method": method},
                    )
                raise DataValidationError(
                    f"grid_interpolate: method must be 'idw' or 'nearest', got {method!r}",
                    details={"method": method},
                )
            if int(n_points) < 2:
                raise DataValidationError(
                    f"grid_interpolate: n_points must be >= 2, got {n_points}",
                    details={"n_points": int(n_points)},
                )
            if not np.isfinite(power) or float(power) <= 0:
                raise DataValidationError(
                    f"grid_interpolate: power must be a positive finite number, got {power!r}",
                    details={"power": power},
                )

            pts = _as_points(coordinates, min_points=2, min_dims=2, name="coordinates")
            if int(pts.shape[1]) != 2:
                raise MatrixDimensionError(
                    f"grid_interpolate needs 2-D coordinates (n, 2) to build a "
                    f"gridding surface, got {pts.shape[1]} columns",
                    details={"n_dims": int(pts.shape[1]), "required": 2},
                )
            if int(pts.shape[0]) != int(vals.shape[0]):
                raise MatrixDimensionError(
                    f"grid_interpolate: got {vals.shape[0]} values but "
                    f"{pts.shape[0]} coordinate rows; there must be one value "
                    "per sample",
                    details={
                        "n_values": int(vals.shape[0]),
                        "n_coordinates": int(pts.shape[0]),
                    },
                )
            _reject_coincident(pts, what="grid_interpolate")

            n = int(pts.shape[0])
            kk = int(min(max(int(k), 1), n)) if method == "idw" else 1
            tree = cKDTree(pts)
            xs = np.linspace(float(pts[:, 0].min()), float(pts[:, 0].max()), int(n_points))
            ys = np.linspace(float(pts[:, 1].min()), float(pts[:, 1].max()), int(n_points))
            mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="xy")

            surface = self._evaluate(tree, vals, mesh_x.ravel(), mesh_y.ravel(), method, float(power), kk).reshape(
                int(n_points), int(n_points)
            )
            fitted = self._evaluate(tree, vals, pts[:, 0], pts[:, 1], method, float(power), kk)
            residuals = vals - fitted
            rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0

            result = GridInterpolationResult(
                grid_x=xs,
                grid_y=ys,
                surface=surface,
                residuals=residuals,
                method=method,
                power=float(power),
                k=kk,
                n_points=n,
                n_grid=int(n_points),
                rmse=rmse,
            )
            self._last_interpolation = result
            self._last_result = result
            self._logger.info(
                "grid_interpolate completed: method=%s, %d samples -> %dx%d grid, RMSE=%.3e",
                method,
                n,
                int(n_points),
                int(n_points),
                rmse,
            )
            return result

    @staticmethod
    def _evaluate(
        tree: cKDTree,
        values: npt.NDArray,
        query_x: npt.NDArray,
        query_y: npt.NDArray,
        method: str,
        power: float,
        k: int,
    ) -> npt.NDArray:
        """Evaluate the interpolant at a set of query points."""
        targets = np.column_stack([query_x, query_y])
        distance, index = tree.query(targets, k=k)
        # ``k=1`` is the one case scipy returns 1-D (shape (m,)), and
        # atleast_2d would turn that into (1, m) -- transposing the pairing
        # between targets and neighbours, so the neighbour of every target
        # silently became a different target's.
        distance = np.asarray(distance, dtype=float)
        index = np.asarray(index, dtype=int)
        if distance.ndim == 1:
            distance = distance.reshape(-1, 1)
            index = index.reshape(-1, 1)
        if distance.shape[0] != targets.shape[0] or distance.shape[1] != k:
            raise ComputationError(
                "grid_interpolate: the KD-tree returned an unexpected neighbour "
                f"array of shape {distance.shape} for {targets.shape[0]} targets "
                f"and k={k}",
                details={"neighbour_shape": tuple(int(x) for x in distance.shape)},
            )

        if method == "nearest":
            return values[index[:, 0]]

        out = np.empty(targets.shape[0], dtype=float)
        # A query point sitting exactly on a sample is that sample, not a
        # blend dominated by an infinite weight. Handling it here rather than
        # hoping an epsilon is small enough is what makes the residual at a
        # sample exactly 0.0 instead of "small but not zero".
        coincident = distance[:, 0] == 0.0
        if np.any(coincident):
            out[coincident] = values[index[coincident, 0]]
        blend = ~coincident
        if np.any(blend):
            d = distance[blend]
            # d > 0 on these rows, by construction of ``coincident``.
            weights = 1.0 / np.power(d, power)
            out[blend] = np.sum(weights * values[index[blend]], axis=1) / np.sum(weights, axis=1)
        return out

    # -- Nearest neighbour -------------------------------------------------

    def nearest_neighbour_stats(
        self,
        coordinates: npt.NDArray,
        r_values: npt.NDArray | None = None,
        edge_correction: bool = True,
        n_simulations: int = 199,
        random_seed: int | None = None,
    ) -> NearestNeighbourResult:
        """Nearest-neighbour point pattern statistics with a CSR reference.

        Parameters
        ----------
        coordinates:
            Site coordinates, ``(n, 2)``.
        r_values:
            Optional radii at which to tabulate the cdf of nearest-neighbour
            distances -- the observed fraction of sites with ``NN <= r``
            against the Poisson expectation ``1 - exp(-pi r^2 n / A)``.
        edge_correction:
            Apply the reduced-sample (inscribed window) edge correction to
            the Monte Carlo null. A site within one interaction distance of
            the window border has part of its neighbourhood outside the
            window, which inflates its NN distance; a null generated in the
            same window is therefore biased upward, and a genuinely random
            pattern reads as strongly clustered. With the correction on, CSR
            replicates are drawn in a window shrunk by the observed mean NN
            distance on every side, so no replicate point's neighbourhood is
            truncated. The observed statistic still uses every site, so for
            a pattern reaching the border the observed mean can sit slightly
            below the corrected null.
        n_simulations:
            Monte Carlo replicates. 199 gives a minimum p of 0.005.
        random_seed:
            Seed for the Monte Carlo null.

        Returns
        -------
        NearestNeighbourResult

        Notes
        -----
        ``index = mean_NN / E[mean_NN under CSR]``, and the expectation is
        ``0.5 * sqrt(A / n)``. For a Poisson process of intensity
        ``lambda = n / A`` the nearest-neighbour distance is Rayleigh,
        ``P(D > d) = exp(-lambda pi d^2)``, so

            E[D] = integral of exp(-lambda pi d^2) over d = 1 / (2 sqrt(lambda))

        The index is read as: below 1 the sites are closer together than
        random (clustering), above 1 further apart (regularity, as in an
        evenly spaced species packing), 1 is CSR. A single cutoff is not the
        test -- ``z_score`` and ``p_value`` are, because at small n an index
        of 0.9 is well inside sampling noise. The nearest-neighbour distance
        comes from one ``cKDTree`` query rather than the per-radius
        ``query_ball_point`` scan Ripley's K performs: every site's NN is a
        single ``k=2`` query, so the cost is one tree build plus O(n log n)
        instead of one full neighbour scan per radius.

        The area is the bounding box of the sites, which is a *biased*
        estimate of the study area and the bias lands directly on the index.
        For a perfectly regular ``k x k`` lattice of spacing ``a`` the true
        area is ``k^2 a^2`` but the bounding box is ``(k-1)^2 a^2``, so the
        index is ``2k / (k - 1)`` = 2.18 at k = 12 rather than the ideal 2.0.
        The 9 % inflation is the area estimator, not the pattern, and a
        pattern that reaches the window border will look correspondingly more
        regular than it is.
        """
        with self._lock:
            pts = _as_points(coordinates, min_points=3, min_dims=2)
            if int(pts.shape[1]) != 2:
                raise MatrixDimensionError(
                    f"nearest_neighbour_stats needs 2-D coordinates (n, 2), got {pts.shape[1]} columns",
                    details={"n_dims": int(pts.shape[1]), "required": 2},
                )
            _reject_coincident(pts, what="nearest_neighbour_stats")
            if int(n_simulations) < 1:
                raise DataValidationError(
                    f"nearest_neighbour_stats: n_simulations must be >= 1, got {n_simulations}",
                    details={"n_simulations": int(n_simulations)},
                )

            n = int(pts.shape[0])
            x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
            area = (x_max - x_min) * (y_max - y_min)
            if area <= 0:
                raise ComputationError(
                    "nearest_neighbour_stats: the study area is zero. Every site "
                    "shares a x or y coordinate with every other, so there is no "
                    "window to measure a pattern in.",
                    details={"area": area, "n_sites": n},
                )

            nn_all = self._nearest_neighbour_distances(pts)
            if np.any(nn_all <= 0):
                raise DataValidationError(
                    "nearest_neighbour_stats: at least one site has a "
                    "nearest-neighbour distance of zero, which makes the mean "
                    "undefined. Check for duplicated coordinates.",
                    details={"n_sites": n},
                )

            # Edge correction. A site within one interaction distance of the
            # window border has part of its neighbourhood outside the window,
            # which inflates its NN distance. A null generated in the same
            # window is therefore biased upward, and a genuinely random
            # pattern reads as strongly clustered -- by z = -5 on a 250-site
            # sample, which is precisely the failure this correction exists
            # to prevent.
            #
            # The scheme is the reduced-sample (minus sampling) one: BOTH
            # sides are restricted to sites at least one mean NN distance
            # from every border, and the CSR replicates are drawn in the
            # matching inscribed window. Correcting only the null is not
            # enough and is actively worse than doing nothing: the observed
            # mean would then sit above the null for *every* pattern, and
            # random samples were rejected as significantly regular in four
            # cases out of five. The observed statistic is therefore reported
            # on the reduced sample, and ``n_sites_total`` carries the input
            # count so the reduction is never invisible.
            lower_window = np.array([x_min, y_min])
            upper_window = np.array([x_max, y_max])
            shrink = 0.0
            used = np.ones(n, dtype=bool)
            if edge_correction:
                shrink = float(np.mean(nn_all))
                if np.all(upper_window - lower_window > 2.0 * shrink):
                    inside = np.all(
                        (pts >= lower_window + shrink) & (pts <= upper_window - shrink),
                        axis=1,
                    )
                    if int(np.sum(inside)) >= 3:
                        used = inside
                    else:
                        self._logger.warning(
                            "nearest_neighbour_stats: only %d of %d sites lie at "
                            "least %.4f from the window border, too few to form a "
                            "reduced sample; the CSR null is generated uncorrected "
                            "and carries edge bias",
                            int(np.sum(inside)),
                            n,
                            shrink,
                        )
                        shrink = 0.0
                else:
                    self._logger.warning(
                        "nearest_neighbour_stats: the window (%.2f x %.2f) is too "
                        "small to shrink by the mean NN distance (%.4f); the CSR "
                        "null is generated uncorrected and carries edge bias",
                        x_max - x_min,
                        y_max - y_min,
                        shrink,
                    )
                    shrink = 0.0

            nn_distances = nn_all[used]
            n_used = int(nn_distances.size)
            lower = lower_window + shrink
            upper = upper_window - shrink
            observed = float(np.mean(nn_distances))
            # E[D] for a Poisson process of intensity lambda = n / A. The NN
            # distance is Rayleigh, P(D > d) = exp(-lambda pi d^2), so
            #
            #     E[D] = integral_0^inf exp(-lambda pi d^2) dd
            #          = sqrt(pi) / (2 sqrt(lambda pi))
            #          = 1 / (2 sqrt(lambda)) = 0.5 sqrt(A / n)
            #
            # The pi cancels against the pi in the exponent, which is
            # exactly why the familiar-looking 0.5 sqrt(pi A / n) is wrong:
            # it carries a spurious sqrt(pi) = 1.77 and would report a random
            # pattern as strongly regular.
            used_area = float((upper[0] - lower[0]) * (upper[1] - lower[1]))
            expected = 0.5 * float(np.sqrt(used_area / n_used))
            index = observed / expected

            def _simulated(rng: np.random.Generator) -> float:
                candidate = rng.uniform(lower, upper, size=(n_used, 2))
                return float(np.mean(self._nearest_neighbour_distances(candidate)))

            perm = permutation_pvalue(
                observed,
                _simulated,
                n_permutations=int(n_simulations),
                random_seed=random_seed,
                context="Nearest-neighbour index",
                # Two-sided: the index is signed, and a one-sided test reports
                # p = 1.0 for a strongly *regular* pattern however far above
                # the CSR mean it sits.
                alternative="two_sided",
            )
            null = perm.null_distribution
            null_sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
            z_score = (observed - float(np.mean(null))) / null_sd if null_sd > 0 else float("nan")

            observed_radii: npt.NDArray | None = None
            cumulative: npt.NDArray | None = None
            csr_expected: npt.NDArray | None = None
            if r_values is not None:
                radii = np.asarray(r_values, dtype=float)
                if radii.ndim != 1 or radii.size == 0:
                    raise DataValidationError(
                        f"nearest_neighbour_stats: r_values must be a non-empty 1-D array, got shape {radii.shape}",
                        details={"shape": tuple(int(x) for x in radii.shape)},
                    )
                if not np.all(np.isfinite(radii)) or np.any(radii < 0):
                    raise DataValidationError(
                        "nearest_neighbour_stats: r_values must be finite and non-negative",
                        details={"n_bad": int(np.sum(~np.isfinite(radii) | (radii < 0)))},
                    )
                observed_radii = np.sort(radii)
                cumulative = np.array([float(np.mean(nn_distances <= r)) for r in observed_radii])
                csr_expected = 1.0 - np.exp(-np.pi * observed_radii**2 * n_used / used_area)

            result = NearestNeighbourResult(
                nn_distances=nn_distances,
                mean_nn=observed,
                expected_nn=expected,
                index=index,
                z_score=float(z_score),
                p_value=float(perm.p_value_two_sided),
                n_sites=n_used,
                n_sites_total=n,
                area=used_area,
                edge_shrink=shrink,
                n_simulations=int(perm.n_permutations),
                edge_correction=bool(edge_correction),
                r_values=observed_radii,
                cumulative_fraction=cumulative,
                csr_expected_fraction=csr_expected,
            )
            self._last_nearest_neighbour = result
            self._last_result = result
            self._logger.info(
                "nearest_neighbour_stats completed: mean_NN=%.4f, expected=%.4f, R=%.4f, z=%.2f",
                result.mean_nn,
                result.expected_nn,
                result.index,
                result.z_score,
            )
            return result

    @staticmethod
    def _nearest_neighbour_distances(points: npt.NDArray) -> npt.NDArray:
        """One ``k=2`` KD-tree query gives every site's nearest neighbour.

        ``query`` returns the two smallest distances per point, and the
        first is the point itself at distance 0, so the second is the
        nearest neighbour. One tree build, one pass, O(n log n) -- against
        Ripley's K's per-radius ``query_ball_point`` scan.
        """
        distances, _indices = cKDTree(points).query(points, k=2)
        return np.atleast_2d(np.asarray(distances, dtype=float))[:, 1]

    # -- Spherical statistics ----------------------------------------------

    def spherical_stats(
        self,
        vectors: npt.NDArray,
        weights: npt.NDArray | None = None,
    ) -> SphericalStatsResult:
        """Mean direction of a set of 3-D vectors on the unit sphere.

        Parameters
        ----------
        vectors:
            ``(n, 3)`` vectors. Need not be normalised; each is divided by its
            own length, because only direction is meaningful. A zero-length
            vector has no direction and is rejected.
        weights:
            Optional non-negative weights, one per vector, used both for the
            resultant and as Fisher's ``n``.

        Returns
        -------
        SphericalStatsResult

        Notes
        -----
        This is not ``stratigraphy/directional.py`` with a third column
        added. That module works on the unit *circle*, where the mean is
        ``atan2(sum sin, sum cos)`` of one scalar angle per observation;
        here the mean is the normalised 3-vector sum

            S = sum_i w_i u_i,   R = |S|,   R-bar = R / sum(w)

        and the divisor of R-bar is ``n``, not ``n - 1``. Identical unit
        vectors give ``R = n`` and ``R-bar = 1``; an ``n - 1`` normalisation
        caps R-bar at ``n / (n - 1) > 1`` and quietly rescales every other
        sample, so the tests assert that exact reduction.

        ``kappa`` is Fisher's concentration with the 1953 small-sample
        correction. The 95 % confidence cone is solved from the exact
        von Mises-Fisher cdf rather than Fisher's large-kappa approximation
        ``cos(w) = 1 + ln(0.05) / kappa``, which is the only regime these
        samples are *not* in.

        Axial data (a direction and its reverse being the same measurement)
        are not handled here; ``stratigraphy/directional.py`` covers axial
        data in 2-D, and doing it on S^2 needs the double-angle treatment
        described there.
        """
        with self._lock:
            vecs = validate_data_array(
                vectors,
                allow_nan=False,
                allow_inf=False,
                name="vectors",
                preserve_dimensions=True,
            )
            if vecs.ndim != 2 or vecs.shape[1] != 3:
                raise MatrixDimensionError(
                    "spherical_stats needs (n, 3) vectors: the mean direction on "
                    f"the 2-sphere is a 3-vector, got shape {vecs.shape}. For "
                    "planar or 1-D direction data use "
                    "stratigraphy.directional.DirectionalAnalyzer instead.",
                    details={"shape": tuple(int(x) for x in vecs.shape)},
                )
            n = int(vecs.shape[0])
            if n < 3:
                raise DataValidationError(
                    f"spherical_stats needs at least 3 vectors for Fisher's small-sample kappa correction, got {n}",
                    details={"n_vectors": n, "required": 3},
                )
            lengths = np.linalg.norm(vecs, axis=1)
            if np.any(lengths <= 0):
                raise DataValidationError(
                    f"spherical_stats: {int(np.sum(lengths <= 0))} vector(s) have "
                    "zero length and therefore no direction",
                    details={"n_zero_length": int(np.sum(lengths <= 0))},
                )

            if weights is None:
                w = np.ones(n, dtype=float)
            else:
                w = np.asarray(weights, dtype=float).ravel()
                if w.size != n:
                    raise MatrixDimensionError(
                        f"spherical_stats: got {n} vectors but {w.size} weights; there must be one weight per vector",
                        details={"n_vectors": n, "n_weights": int(w.size)},
                    )
                if not np.all(np.isfinite(w)) or np.any(w < 0):
                    raise DataValidationError(
                        "spherical_stats: weights must be finite and non-negative",
                        details={"n_bad": int(np.sum(~np.isfinite(w) | (w < 0)))},
                    )
            effective_n = float(np.sum(w))
            if effective_n <= 2.0:
                raise DataValidationError(
                    "spherical_stats: Fisher's kappa correction divides by "
                    f"(n - 2), so the effective sample size sum(weights) must "
                    f"exceed 2; got {effective_n:.4f}",
                    details={"effective_n": effective_n, "required_gt": 2.0},
                )

            unit = vecs / lengths[:, np.newaxis]
            resultant = np.sum(w[:, np.newaxis] * unit, axis=0)
            r_length = float(np.linalg.norm(resultant))
            r_bar = r_length / effective_n
            # Whether the *mean direction* is defined at all: the vector sum
            # is zero, so every direction is equally the mean. This is a
            # numerical test, not a significance test -- a set of nearly
            # cancelling vectors defines a direction perfectly well, it is
            # just a poorly concentrated one, and ``kappa`` and the cone are
            # what say so.
            concentrated = r_bar > _MIN_MEAN_CONCENTRATION

            if concentrated:
                mean_direction: npt.NDArray | None = resultant / r_length
                azimuth = float(np.degrees(np.arctan2(mean_direction[1], mean_direction[0])) % 360.0)
                polar = float(np.degrees(np.arccos(np.clip(mean_direction[2], -1.0, 1.0))))
            else:
                # R is at the numerical floor: the vector sum is zero and the
                # mean direction is genuinely undefined, not merely noisy.
                mean_direction = None
                azimuth = float("nan")
                polar = float("nan")

            if r_bar >= _PERFECT_CONCENTRATION:
                # Every vector identical: the vMF limit has kappa -> inf, and
                # the formula is 0/0 rather than a number.
                kappa = float("inf")
            else:
                kappa_uncorrected = effective_n * r_bar * (1.0 - r_bar**2) / (1.0 - r_bar)
                kappa = (2.0 * (effective_n - 1.0) - kappa_uncorrected) / (effective_n - 2.0)

            cone_rad = _confidence_cone(kappa)
            if kappa <= 0.0:
                self._logger.warning(
                    "spherical_stats: kappa = %.4f <= 0, so the data show no "
                    "preferred direction; the reported cone is the uniform "
                    "limit, not a confidence interval",
                    kappa,
                )

            result = SphericalStatsResult(
                mean_direction=mean_direction,
                mean_azimuth_deg=azimuth,
                mean_polar_deg=polar,
                resultant_length=r_length,
                mean_resultant_length=r_bar,
                kappa=float(kappa),
                cone_half_angle_deg=float(np.degrees(cone_rad)),
                cone_half_angle_rad=float(cone_rad),
                n_vectors=n,
                effective_n=effective_n,
                concentrated=bool(concentrated),
            )
            self._last_spherical = result
            self._last_result = result
            self._logger.info(
                "spherical_stats completed: R=%.4f, R-bar=%.4f, kappa=%.4f, cone=%.2f deg",
                result.resultant_length,
                result.mean_resultant_length,
                result.kappa,
                result.cone_half_angle_deg,
            )
            return result

    # -- Results -----------------------------------------------------------

    @property
    def last_result(self) -> Any:
        """Most recent result of any kind, or None."""
        with self._lock:
            return self._last_result

    @property
    def last_morans_i(self) -> MoransIResult | None:
        """Most recent Moran's I result, or None."""
        with self._lock:
            return self._last_morans_i

    @property
    def last_interpolation(self) -> GridInterpolationResult | None:
        """Most recent interpolation result, or None."""
        with self._lock:
            return self._last_interpolation

    @property
    def last_nearest_neighbour(self) -> NearestNeighbourResult | None:
        """Most recent nearest-neighbour result, or None."""
        with self._lock:
            return self._last_nearest_neighbour

    @property
    def last_spherical(self) -> SphericalStatsResult | None:
        """Most recent spherical statistics result, or None."""
        with self._lock:
            return self._last_spherical
