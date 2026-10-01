# =============================================================================
# FILE: morphometrics/relative_warps.py
# =============================================================================
"""
Relative Warps Analysis Module for PaleoAST

Relative Warps Analysis performs PCA on Procrustes-aligned coordinates
to extract the major axes of shape variation.

Mathematical Foundation:

Given N Procrustes-aligned configurations Y₁, Y₂, ..., Yₙ ∈ ℝ^k:
    k = n_landmarks × n_dimensions (flattened)

1. Flatten each configuration to a vector
   y_i = vec(Y_i) ∈ ℝ^k

2. Compute covariance matrix
   S = (1/(n-1)) * Σ(y_i - ȳ)(y_i - ȳ)'

3. PCA on covariance matrix
   S * v_j = λ_j * v_j

4. Relative Warps (RW_j):
   RW_j = y * v_j

Relative warps are analogous to principal components in traditional PCA
but applied to shape space.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


@dataclass
class RelativeWarpsResult:
    """
    Container for Relative Warps analysis results.
    """

    relative_warps: npt.NDArray  # RW scores (n_specimens, n_components)
    eigenvalues: npt.NDArray
    explained_variance: npt.NDArray
    cumulative_variance: npt.NDArray
    eigenvectors: npt.NDArray  # Shape change vectors
    mean_shape: npt.NDArray  # Mean Procrustes configuration
    n_components: int
    n_landmarks: int
    n_dims: int
    # The λ^α exponent the scores in ``relative_warps`` were weighted with.
    # Recorded so :meth:`RelativeWarpsAnalyzer.get_shape_at_warp` can invert
    # the same weighting instead of reconstructing an unweighted PCA score.
    alpha: float = 0.0

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Relative Warps Analysis Results"),
            "=" * 50,
            _("Number of specimens: {0}").format(self.relative_warps.shape[0]),
            _("Number of landmarks: {0}").format(self.n_landmarks),
            _("Dimensionality: {0}D").format(self.n_dims),
            _("Number of components: {0}").format(self.n_components),
            "",
            _("RW  | Eigenvalue | Variance % | Cumulative"),
            "-" * 50,
        ]

        for i in range(min(10, self.n_components)):
            lines.append(
                f"RW{i + 1:2d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.explained_variance[i]:9.2f} | "
                f"{self.cumulative_variance[i]:10.2f}"
            )

        return "\n".join(lines)


class RelativeWarpsAnalyzer:
    """
    Relative Warps Analysis engine.

    Performs PCA on Procrustes-aligned landmark configurations
    to extract major patterns of shape variation.
    """

    def __init__(self) -> None:
        """Initialize the Relative Warps analyzer."""
        self._logger = logging.getLogger(f"{__name__}.RelativeWarpsAnalyzer")
        self._logger.info("RelativeWarpsAnalyzer initialized")
        self._lock = threading.RLock()
        self._last_result: RelativeWarpsResult | None = None

    def analyze(
        self,
        aligned_configurations: npt.NDArray,
        n_components: int | None = None,
        alpha: float = 0.0,
    ) -> RelativeWarpsResult:
        """
        Perform Relative Warps Analysis (Bookstein 1991).

        Parameters:
            aligned_configurations: 3D array from GPA (n_specimens, n_landmarks, n_dims)
            n_components: Number of relative warps to extract
            alpha: Weighting exponent applied to each partial-warp eigenvalue
                when scoring the warps:

                    RW_k = λ_k^α · (centred · v_k)

                * α =  0  →  standard PCA (uniform weighting; default and
                  bit-identical to the legacy implementation)
                * α = -1  →  "uniform" warps (Bookstein 1991; small-scale
                  shape features amplified)
                * α = +1  →  "affine-dominated" warps (large-scale shape
                  features amplified)

                The eigenvalues themselves are NOT reweighted — only the
                *score* of each warp is, which is the convention in
                Bookstein (1991) and Walker & Oyana (2001).

        Returns:
            RelativeWarpsResult: Relative Warps analysis results
        """
        with self._lock:
            # Validate input
            if aligned_configurations.ndim != 3:
                raise ComputationError("Aligned configurations must be 3D (n_specimens, n_landmarks, n_dims)")

            n_specimens, n_landmarks, n_dims = aligned_configurations.shape

            # A single specimen has no between-specimen variation: the
            # (n_specimens - 1) divisor below is 0, so every eigenvalue came
            # back NaN/inf and the completion log indexed an empty array.
            if n_specimens < 2:
                raise ComputationError(
                    f"At least 2 specimens are required for Relative Warps analysis, got {n_specimens}"
                )

            self._logger.info(
                f"Relative Warps analysis started: n_specimens={n_specimens}, "
                f"n_landmarks={n_landmarks}, n_dimensions={n_dims}, "
                f"alpha={alpha}"
            )

            # Determine number of components
            max_components = min(n_specimens - 1, n_landmarks * n_dims)
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)

            # Flatten configurations
            # Shape: (n_specimens, n_landmarks * n_dims)
            flattened = aligned_configurations.reshape(n_specimens, n_landmarks * n_dims)

            # Compute mean shape
            mean_shape = np.mean(aligned_configurations, axis=0)

            # Center the data
            flattened_centered = flattened - mean_shape.flatten()

            # PCA via SVD (more numerically stable than eigendecomposition)
            try:
                _U, singular_values, Vt = np.linalg.svd(flattened_centered, full_matrices=False)
            except np.linalg.LinAlgError as e:
                raise ComputationError("SVD failed during Relative Warps analysis", original_exception=e)

            # Eigenvalues from singular values
            all_eigenvalues = (singular_values**2) / (n_specimens - 1)

            # Select top components
            eigenvalues = all_eigenvalues[:n_components]
            eigenvectors = Vt[:n_components].T

            # Compute relative warps (projections), weighted by λ_k^α.
            # α = 0 must reproduce the legacy behaviour bit-for-bit: the
            # legacy code was a plain PCA score ``flattened_centered @
            # eigenvectors``.  Going through ``np.power(λ, 0)`` gives an
            # all-ones vector and a multiply by 1.0 — bit-identical in
            # IEEE 754 — so we don't need a special branch, but we DO
            # guard against pathological α inputs (NaN eigenvalue, α
            # non-finite) by returning the unweighted scores.
            if alpha == 0.0:
                relative_warps = flattened_centered @ eigenvectors
            else:
                # λ_k^α: weight zero eigenvalues to 1.0 to avoid 0^(-1)
                # blowing up; the corresponding axis' score will still be
                # zero because the centred data is orthogonal to a
                # zero-eigenvalue direction.
                safe_eigs = np.where(eigenvalues > 0, eigenvalues, 1.0)
                weights = np.power(safe_eigs, alpha)
                relative_warps = (flattened_centered @ eigenvectors) * weights

            # Compute explained variance against the full variance, not only
            # the retained components. Otherwise a truncated result always
            # reports 100% cumulative variance.
            total_variance = np.sum(all_eigenvalues)
            if total_variance > 0:
                explained_variance = (eigenvalues / total_variance) * 100
            else:
                explained_variance = np.zeros_like(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance)
            self._logger.info(
                f"Relative Warps analysis completed: {n_components} components, "
                f"PC1 variance={explained_variance[0]:.2f}%, "
                f"cumulative={cumulative_variance[-1]:.2f}%"
            )

            result = RelativeWarpsResult(
                relative_warps=relative_warps,
                eigenvalues=eigenvalues,
                explained_variance=explained_variance,
                cumulative_variance=cumulative_variance,
                eigenvectors=eigenvectors,
                mean_shape=mean_shape,
                n_components=n_components,
                n_landmarks=n_landmarks,
                n_dims=n_dims,
                alpha=float(alpha),
            )

            self._last_result = result
            return result

    @staticmethod
    def _warp_weight(eigenvalue: float, alpha: float) -> float:
        """Weight :meth:`analyze` applies to a single warp's score.

        Mirrors the λ_k^α branch of :meth:`analyze` exactly, including its
        ``alpha == 0`` fast path and its substitution of 1.0 for a
        non-positive eigenvalue (so 0^α never becomes 0 or inf).
        """
        if alpha == 0.0:
            return 1.0
        safe_eigenvalue = eigenvalue if eigenvalue > 0 else 1.0
        return float(np.power(safe_eigenvalue, alpha))

    def get_shape_at_warp(
        self,
        result: RelativeWarpsResult | None = None,
        warp_number: int = 0,
        warp_score: float = 3.0,
        alpha: float | None = None,
    ) -> npt.NDArray:
        """
        Reconstruct shape at a specific position along a relative warp.

        Parameters:
            result: Relative warps result. If None, uses last result.
            warp_number: Which relative warp (0-indexed)
            warp_score: Score along the warp, on the same scale as the
                ``relative_warps`` column reported for ``warp_number``
                (i.e. already weighted by λ^α). To place a shape at
                ``n`` standard deviations of that warp, pass
                ``n * result.relative_warps[:, warp_number].std()``.
            alpha: Weighting exponent to invert. Defaults to the exponent
                the analysis was run with, which is recorded on the result.

        Returns:
            npt.NDArray: Reconstructed configuration
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No Relative Warps result available")

        if warp_number >= result.n_components:
            raise ComputationError(f"Warp number {warp_number} exceeds available components")

        # Start from mean shape
        shape = result.mean_shape.flatten()

        # Add contribution from specified warp.
        #
        # ``analyze`` scores a warp as RW_k = λ_k^α · (centred · v_k), so a
        # perturbation of amplitude A along the unit eigenvector v_k scores
        # λ_k^α · A. The amplitude that realises ``warp_score`` is therefore
        # ``warp_score / λ_k^α``. The previous ``warp_score * sqrt(λ_k)``
        # inverted nothing: it rebuilt the *unweighted* PCA score, so with
        # α != 0 the returned configuration sat at 0.36x (α = -1) / 2.75x
        # (α = +1) the score the function claims to reconstruct.
        eigenvector = result.eigenvectors[:, warp_number]
        weight = self._warp_weight(result.eigenvalues[warp_number], result.alpha if alpha is None else alpha)

        if not np.isfinite(weight) or weight == 0.0:
            # A degenerate axis carries no score information (``analyze``
            # reports ~0 for a zero eigenvalue, and an extreme α can
            # underflow the weight to 0), so the only configuration
            # consistent with any requested score is the mean shape.
            self._logger.warning(
                f"Warp {warp_number} has a non-invertible weight {weight!r}; returning the mean shape"
            )
            amplitude = 0.0
        else:
            amplitude = warp_score / weight

        shape = shape + eigenvector * amplitude

        # Reshape to configuration
        return shape.reshape(result.n_landmarks, result.n_dims)

    @property
    def last_result(self) -> RelativeWarpsResult | None:
        """Get the last Relative Warps result."""
        with self._lock:
            return self._last_result
