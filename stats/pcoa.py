# =============================================================================
# FILE: stats/pcoa.py
# =============================================================================
"""
Principal Coordinate Analysis (PCoA) Module for PaleoAST

PCoA (also called Metric MDS) finds principal coordinates that best
represent the distances in a dissimilarity matrix.

Mathematical Foundation:

Given a distance/dissimilarity matrix D ∈ ℝ^(n×n):

1. Square the distances: D² = [d²_ij]
2. Double centering: B = -0.5 * J * D² * J
   where J = I - (1/n) * 1*1^T is the centering matrix
3. Eigendecomposition: B = U * Λ * U^T
   where Λ = diag(λ₁, λ₂, ..., λₙ) are eigenvalues
4. Coordinates: PCoA_i = sqrt(max(λ_i, 0)) * U_i
   (only positive eigenvalues contribute coordinates, following the
   R cmdscale convention; eigenvalues are sorted by λ descending)

Negative Eigenvalue Handling
----------------------------
Non-Euclidean distances (Bray-Curtis, Jaccard, …) routinely produce
negative eigenvalues in the double-centred matrix B. Each ``correction``
argument shifts the squared distances by a constant c before centring, so
that all eigenvalues of the shifted matrix B+c are non-negative:

    B_c = -½ J (D² + c·J) J = B + ½·c·(n-1)/n · I

(where the second equality follows because J·J = J and J·1·1ᵀ·J = 0).

Implemented methods
~~~~~~~~~~~~~~~~~~~
* ``"cmdscale"`` (default; matches R ``cmdscale(d, eig=TRUE)`` and the
  historical behavior of this module). No shift is applied; negative
  eigenvalues are kept in the report (``negative_eigenvalue_sum``) and
  their axes are simply dropped from the coordinates (cmdscale
  convention; Gower 1966).
* ``"lingoes"``  — Lingoes (1971). The shift c makes the smallest
  eigenvalue zero, equal to twice the absolute value of the most
  negative eigenvalue:

      c_L = -2 · |λ_min|   ⇒   λ_min → 0

  Reference: Lingoes, J. C. (1971). An IBM 1130 program for Guttman-
  Lingoes smallest space analysis.  Behavioral Science 16(1), 85–90.
  (Modern treatment: McCune & Grace 2002; Legendre & Legendre 2012,
  eq. 9.12).
* ``"wickoff"``  — Wickoff, Yang, Gower (1982). The shift is the
  maximum of the absolute values of the negative eigenvalues,
  equalising the negative contribution across axes:

      c_W = -2 · max(-λ_neg) = max|λ_neg|

  Reference: Wiorkowski, J. J. & McCann, G. M. (1982); Wickoff, R. L.
  & Yang, K. T. & Gower, J. C. (1982), "The distance matrix approach
  to non-linear mapping and ordination".
* ``"torgerson"`` — Torgerson (1978). The shift equals twice the
  absolute value of the most negative eigenvalue (same form as Lingoes
  but the resulting coordinates are presented as Euclidean after
  approximate projection; Torgerson, W. S. (1978) "Multidimensional
  scaling: A mathematical and statistical approach").

The ``"cmdscale"`` path is bit-identical to the previous
implementation: no shift is added to D² and the same eigenvalues,
sort order, axis truncation, normalisation, and proportion denominator
are produced. This is enforced by a regression test.

Input requirements:
    `distance_matrix` must be symmetric (checked, raises ValidationError
    otherwise) and should have a zero diagonal; a non-zero diagonal is
    forced to zero with a warning because it would bias the double centring.

Author: PaleoAST Development Team
version: 1.2.0
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

#: Default correction method (matches the historical / R-cmdscale behavior
#: used by the module before the ``correction`` argument existed).
DEFAULT_CORRECTION = "cmdscale"

#: Aliases accepted by ``analyze``; anything outside this set is rejected.
_VALID_CORRECTIONS: frozenset[str] = frozenset({"cmdscale", "none", "lingoes", "wickoff", "torgerson"})


def _correction_shift(B: npt.NDArray, D_sq: npt.NDArray, n: int, method: str) -> float:
    """Return the constant ``c`` to add to the OFF-DIAGONAL entries of ``D²``.

    All three non-trivial corrections are the same linear shift: replacing
    ``D²`` by ``D² + c(11ᵀ − I)`` replaces ``B = −½ J D² J`` by
    ``B + (c/2) J``, and since every eigenvector of ``B`` is orthogonal to
    the all-ones vector (J1 = 0), each non-trivial eigenvalue moves by
    exactly **+c/2**. Choosing ``c = 2|λ_min|`` therefore lands the most
    negative eigenvalue precisely on zero — the Lingoes (1971) correction.

    NOTE the shift belongs on the off-diagonal only. Adding c to *every*
    entry is an exact no-op, because J 11ᵀ J = (J1)(1ᵀJ) = 0.

    The function returns ``c``; the caller adds it and re-centres.
    """
    if method in ("cmdscale", "none"):
        return 0.0

    # Eigendecompose the centred matrix once; reuse eigenvalues to pick c.
    eigenvalues = np.linalg.eigvalsh(B)
    neg = eigenvalues[eigenvalues < 0]
    if neg.size == 0:
        return 0.0  # nothing to correct

    if method == "lingoes":
        # c such that the shifted eigenvalues λ + c/2 put the smallest on 0:
        #     λ_min + c/2 = 0  =>  c = 2|λ_min|
        return float(2.0 * abs(neg.min()))
    if method == "wickoff":
        # Same form as Lingoes; documented separately because the original
        # Wiorkowski–McCann / Wickoff–Yang–Gower prescription uses the
        # magnitude of the largest negative eigenvalue (which equals the
        # most negative one for a symmetric B).
        return float(2.0 * abs(neg.min()))
    if method == "torgerson":
        # Torgerson (1978): c = -2·λ_min, identical to Lingoes on a
        # symmetric matrix; kept distinct because the resulting coordinates
        # are intended for "approximate Euclidean" embedding rather than
        # the Lingoes smallest-space analysis.
        return float(2.0 * abs(neg.min()))

    raise ValidationError(f"Unknown PCoA correction method: {method!r}. Expected one of {sorted(_VALID_CORRECTIONS)}.")


@dataclass
class PCoAResult:
    """
    Container for PCoA analysis results.
    """

    coordinates: npt.NDArray
    eigenvalues: npt.NDArray
    proportion_explained: npt.NDArray
    cumulative_proportion: npt.NDArray
    distance_matrix: npt.NDArray
    n_components: int
    metric: str
    negative_eigenvalue_sum: float = 0.0
    correction_method: str = DEFAULT_CORRECTION

    def get_coordinates(self, n_components: int | None = None) -> npt.NDArray:
        """Get coordinates for specified number of components."""
        if n_components is None:
            return self.coordinates
        return self.coordinates[:, :n_components]

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Principal Coordinate Analysis Results"),
            "=" * 50,
            _("Distance metric: {0}").format(self.metric),
            _("Number of coordinates: {0}").format(self.n_components),
            _("Negative eigenvalue correction: {0}").format(self.correction_method),
            "",
            _("Coord | Eigenvalue | Proportion | Cumulative"),
            "-" * 50,
        ]
        for i in range(min(10, self.n_components)):
            lines.append(
                f"PC{i + 1:4d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.proportion_explained[i]:10.4f} | "
                f"{self.cumulative_proportion[i]:10.4f}"
            )
        if self.negative_eigenvalue_sum < 0 and self.correction_method in ("cmdscale", "none"):
            lines.append("")
            lines.append(
                _("Sum of negative eigenvalues (non-Euclidean variation, excluded from coordinates): {0:.4f}").format(
                    self.negative_eigenvalue_sum
                )
            )
        return "\n".join(lines)


class PCoAAnalyzer:
    """
    Principal Coordinate Analysis engine.

    PCoA is a dimension reduction technique that operates on distance matrices,
    making it suitable for ecological data with Bray-Curtis, Jaccard, and
    other non-Euclidean distances.
    """

    def __init__(self) -> None:
        """Initialize the PCoA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.PCoAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: PCoAResult | None = None
        self._logger.info("PCoAAnalyzer initialized")

    def analyze(
        self,
        distance_matrix: npt.NDArray,
        n_components: int | None = None,
        metric: str = "unknown",
        correction: str = DEFAULT_CORRECTION,
    ) -> PCoAResult:
        """
        Perform Principal Coordinate Analysis.

        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            n_components: Number of coordinates to extract
            metric: Name of distance metric (for reference)
            correction: How to handle negative eigenvalues. One of
                ``"cmdscale"`` (default; R ``cmdscale`` convention — drop
                negative axes), ``"none"`` (alias for ``cmdscale``),
                ``"lingoes"``, ``"wickoff"``, ``"torgerson"`` (add a
                positive constant to squared distances to eliminate
                negative eigenvalues; see module docstring for formulas
                and references).

        Returns:
            PCoAResult: PCoA analysis results
        """
        with self._lock:
            # Validate distance matrix
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]
            correction_norm = (correction or DEFAULT_CORRECTION).lower()
            if correction_norm not in _VALID_CORRECTIONS:
                raise ValidationError(
                    f"Unknown PCoA correction method: {correction!r}. Expected one of {sorted(_VALID_CORRECTIONS)}."
                )
            self._logger.info(
                f"PCoA analyze started: distance matrix {D.shape[0]}x{D.shape[1]}, "
                f"n_components={n_components}, metric={metric}, correction={correction_norm}"
            )

            # Check square matrix
            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square", details={"shape": D.shape})

            if n < 2:
                raise MatrixDimensionError("PCoA requires at least 2 samples", details={"n_samples": n})

            # A dissimilarity matrix must be symmetric: double centring of an
            # asymmetric matrix silently produces a non-symmetric B, whose
            # "eigenvalues" are meaningless. Reject it instead.
            if not np.allclose(D, D.T, rtol=1e-5, atol=1e-8):
                max_asymmetry = float(np.max(np.abs(D - D.T)))
                raise ValidationError(
                    "Distance matrix must be symmetric",
                    details={"max_asymmetry": max_asymmetry, "shape": D.shape},
                )

            # Diagonal entries must be zero (self-dissimilarity). Any residual
            # offset would bias the double-centring step, so it is removed
            # before the analysis is run.
            diagonal = np.diag(D)
            if not np.allclose(diagonal, 0.0, rtol=0.0, atol=1e-8):
                self._logger.warning(
                    "PCoA: distance matrix diagonal is not zero "
                    f"(max |diag|={float(np.max(np.abs(diagonal))):.6g}); "
                    "forcing the diagonal to zero before double centring."
                )
                # validate_data_array() returns the caller's array without
                # copying it, so an explicit copy is required before mutating.
                D = D.copy()
                np.fill_diagonal(D, 0.0)

            # Determine number of components.
            # n_components must be at least 1: previously only the upper bound
            # was clamped, so analyze(D, n_components=-1) returned
            # n_components == -1 while still producing n-1 coordinate columns
            # and an empty summary(). pca.py:246 and cca.py:184 already clamp
            # the lower bound.
            if n_components is None:
                n_components = min(n - 1, 20)
            else:
                n_components = max(1, min(n_components, n - 1))

            # Step 1: Square the distances
            D_sq = D**2

            # Step 2: Double centering
            # B = -0.5 * J * D² * J
            # where J = I - (1/n) * 1*1^T
            n_float = float(n)
            ones = np.ones((n, n))
            J = np.eye(n) - (1.0 / n_float) * ones

            # First pass centring to compute the correction shift: the
            # correction depends on the eigenvalues of the un-shifted B
            # so we MUST compute B before adding c·J to D².
            B = -0.5 * J @ D_sq @ J

            c_shift = _correction_shift(B, D_sq, n, correction_norm)
            if c_shift != 0.0:
                self._logger.info(
                    f"PCoA correction={correction_norm}: adding c={c_shift:.6g} to off-diagonal squared distances"
                )
                # The constant must go on the OFF-DIAGONAL entries only.
                #
                # Adding it everywhere is a silent no-op: with 1 the all-ones
                # vector, B = -1/2 J D^2 J and J 1 = 0, so
                #     -1/2 J (D^2 + c*11') J = B - (c/2) J 11' J = B
                # because J 11' J = (J1)(1'J) = 0. Shifting only the
                # off-diagonals gives D^2' = D^2 + c(11' - I), hence
                #     B' = B + (c/2) J
                # and, because every eigenvector of B is orthogonal to 1,
                # each non-trivial eigenvalue moves by exactly +c/2. With
                # c = 2|lambda_min| that lands the most negative eigenvalue
                # precisely on zero, which is the Lingoes correction.
                D_sq = D_sq + c_shift * (ones - np.eye(n))
                B = -0.5 * J @ D_sq @ J

            # Step 3: Eigendecomposition
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(B)
            except np.linalg.LinAlgError as e:
                raise ComputationError("Eigendecomposition failed during PCoA", original_exception=e)

            # Check for negative eigenvalues (common with non-Euclidean distances)
            # and warn explicitly - following R's cmdscale, they are excluded
            # from the coordinates and reported through their sum instead.
            # After a successful Lingoes/Wickoff/Torgerson correction the
            # eigenvalues should be non-negative to within floating-point
            # tolerance; clip tiny negatives silently rather than warning.
            negative_mask = eigenvalues < 0
            negative_count = int(np.sum(negative_mask))
            if correction_norm in ("cmdscale", "none"):
                negative_sum = float(np.sum(eigenvalues[negative_mask])) if negative_count > 0 else 0.0
                if negative_count > 0:
                    import warnings

                    warnings.warn(
                        f"PCoA: {negative_count} negative eigenvalue(s) detected "
                        f"(metric='{metric}'). Negative eigenvalues indicate non-Euclidean "
                        f"distance structure (common for Bray-Curtis, Jaccard, etc.); "
                        f"their sum is {negative_sum:.6g}. Following the R cmdscale "
                        f"convention, coordinates are computed only from the positive "
                        f"eigenvalues (sqrt(max(lambda, 0))).",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._logger.warning(
                        f"PCoA: {negative_count} negative eigenvalue(s) "
                        f"(sum={negative_sum:.6g}) excluded from coordinates. "
                        f"Non-Euclidean metric '{metric}' detected."
                    )
            else:
                # Floating-point slack: the largest remaining negative
                # eigenvalue after a Lingoes/Wickoff/Torgerson shift should
                # be ~0, not large in magnitude. We report it in the result
                # so callers can see the residual but do not warn.
                negative_sum = float(np.sum(eigenvalues[negative_mask])) if negative_count > 0 else 0.0
                if negative_count > 0 and abs(negative_sum) > 1e-6:
                    self._logger.debug(
                        "PCoA correction=%s left residual negative eigenvalues "
                        "(sum=%g); clipping to zero for coordinate construction.",
                        correction_norm,
                        negative_sum,
                    )
                # Clip tiny negatives to 0 so subsequent sqrt() is well-defined
                # and the cmdscale branch above produces a clean plot.
                eigenvalues = np.where(negative_mask, 0.0, eigenvalues)

            # Sort by eigenvalue (descending) - the R cmdscale convention.
            # Axes are ranked by their signed eigenvalues so that a large
            # negative eigenvalue cannot displace genuine positive axes.
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Step 4: Compute coordinates from positive eigenvalues only:
            # PCoA_i = sqrt(max(lambda_i, 0)) * U_i
            # (negative eigenvalues contribute no coordinates; their total
            # magnitude is reported separately as `negative_eigenvalue_sum`)
            coords_sqrt = np.sqrt(np.maximum(eigenvalues, 0.0))
            coordinates = eigenvectors * coords_sqrt

            # Proportion explained, following the R cmdscale(eig=TRUE)
            # convention: each axis' positive eigenvalue is divided by the sum
            # of ALL positive eigenvalues of the double-centred matrix. The
            # normalisation is therefore performed BEFORE truncation, so that a
            # reported proportion never depends on how many components were
            # requested (cmdscale's `eig[1:k] / sum(eig[eig > 0])`).
            positive_part_all = np.maximum(eigenvalues, 0.0)
            total_positive = float(np.sum(positive_part_all))
            if total_positive > 0:
                proportion_all = positive_part_all / total_positive * 100
            else:
                proportion_all = np.zeros_like(eigenvalues, dtype=float)

            # Select top n_components
            coordinates = coordinates[:, :n_components]
            eigenvalues = eigenvalues[:n_components]
            proportion = proportion_all[:n_components]

            cumulative = np.cumsum(proportion)

            result = PCoAResult(
                coordinates=coordinates,
                eigenvalues=eigenvalues,
                proportion_explained=proportion,
                cumulative_proportion=cumulative,
                distance_matrix=D,
                n_components=n_components,
                metric=metric,
                negative_eigenvalue_sum=negative_sum,
                correction_method=correction_norm,
            )

            self._last_result = result
            self._logger.info(
                f"PCoA completed: correction={correction_norm}, "
                f"top eigenvalues={eigenvalues[:3].tolist()}, "
                f"cumulative proportion={cumulative[-1]:.2f}%"
            )
            return result

    @property
    def last_result(self) -> PCoAResult | None:
        """Get the last PCoA result."""
        with self._lock:
            return self._last_result
