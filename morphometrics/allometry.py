# =============================================================================
# FILE: morphometrics/allometry.py
# =============================================================================
"""
Allometry and Morphological Integration Analysis for PaleoAST

Implements two related analyses:

1. Allometry (Size-Shape Relationship)
   Analyzes the relationship between centroid size and shape using
   multivariate regression of Procrustes coordinates on log centroid size.

   Klingenberg, C.P. (2016). Nature Reviews Genetics, 17(4), 207-223.

2. Morphological Integration (2B-PLS)
   Two-Block Partial Least Squares analysis to measure协方差 between
   two sets of shape variables.

   Rohlf, F.J. & Corti, M. (2000). Systematic Biology, 49(4), 740-753.

Mathematical Framework:
==============================================================================

Allometry:
    Centroid Size: CS = sqrt(sum_{i=1}^{k} sum_{j=1}^{m} x_{ij}²)

    Log-linear regression: Y = Xβ + ε
    where Y = Procrustes coordinates (flattened)
          X = [1, log(CS)] design matrix
          β = regression coefficients

    Isometry test: H₀: all allometric coefficients = 0
    F-test comparing full model vs intercept-only

2B-PLS:
    Cross-block covariance: C_AB = (1/(n-1)) * X_A' * X_B

    SVD: C_AB = U * S * V'
    PLS scores: A = X_A * U, B = X_B * V

    RV coefficient: measure of integration between blocks

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class AllometryResult:
    """
    Container for allometry analysis results.

    Attributes:
        centroid_sizes: Centroid size for each specimen
        log_centroid_sizes: Log-transformed centroid sizes
        regression_coefficients: Shape change per unit log(size)
        regression_intercept: Intercept of regression
        r_squared: Proportion of shape variance explained by size
        f_statistic: F-statistic for isometry test
        isometry_pvalue: P-value for isometry test
        residuals: Residual shape variation after removing size effect
        predicted_shapes: Predicted shape at mean log(size)
        n_specimens: Number of specimens
        n_landmarks: Number of landmarks
        n_dims: Number of dimensions
    """

    centroid_sizes: npt.NDArray[np.float64]
    log_centroid_sizes: npt.NDArray[np.float64]
    regression_coefficients: npt.NDArray[np.float64]
    regression_intercept: npt.NDArray[np.float64]
    r_squared: float
    f_statistic: float
    isometry_pvalue: float
    residuals: npt.NDArray[np.float64]
    predicted_shapes: npt.NDArray[np.float64]
    n_specimens: int
    n_landmarks: int
    n_dims: int

    def summary(self) -> str:
        """Generate summary text."""
        if self.isometry_pvalue < 0.001:
            sig = "***"
        elif self.isometry_pvalue < 0.01:
            sig = "**"
        elif self.isometry_pvalue < 0.05:
            sig = "*"
        else:
            sig = ""
        mean_cs = np.mean(self.centroid_sizes)
        min_cs = np.min(self.centroid_sizes)
        max_cs = np.max(self.centroid_sizes)
        return (
            f"{_('Allometry Analysis')}\n"
            f"{'=' * 50}\n"
            f"{_('Specimens: {0}, Landmarks: {1}, Dimensions: {2}').format(self.n_specimens, self.n_landmarks, self.n_dims)}\n"
            f"{_('R²: {0}').format(f'{self.r_squared:.4f}')}\n"
            f"{_('Isometry test: F={0:.4f}, p={1:.4f} {2}').format(self.f_statistic, self.isometry_pvalue, sig)}\n"
            f"{_('Mean centroid size: {0}').format(f'{mean_cs:.4f}')}\n"
            f"{_('Size range: {0:.4f} to {1:.4f}').format(min_cs, max_cs)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "centroid_sizes": self.centroid_sizes.tolist(),
            "log_centroid_sizes": self.log_centroid_sizes.tolist(),
            "regression_coefficients": self.regression_coefficients.tolist(),
            "regression_intercept": self.regression_intercept.tolist(),
            "r_squared": self.r_squared,
            "f_statistic": self.f_statistic,
            "isometry_pvalue": self.isometry_pvalue,
            "residuals": self.residuals.tolist(),
            "predicted_shapes": self.predicted_shapes.tolist(),
            "n_specimens": self.n_specimens,
            "n_landmarks": self.n_landmarks,
            "n_dims": self.n_dims,
            "summary": self.summary(),
        }


@dataclass
class PLSResult:
    """
    Container for 2-Block Partial Least Squares analysis results.

    Attributes:
        singular_values: PLS singular values
        covariance_explained: Percentage of covariance explained per component
        cumulative_covariance: Cumulative covariance explained
        left_scores: PLS scores for block A (n_specimens, n_components)
        right_scores: PLS scores for block B (n_specimens, n_components)
        pls_loadings_left: Loadings for block A
        pls_loadings_right: Loadings for block B
        rv_coefficients: RV coefficient per component
        integration_index: Mean absolute RV coefficient (overall integration)
        n_components: Number of PLS components
        n_specimens: Number of specimens
    """

    singular_values: npt.NDArray[np.float64]
    covariance_explained: npt.NDArray[np.float64]
    cumulative_covariance: npt.NDArray[np.float64]
    left_scores: npt.NDArray[np.float64]
    right_scores: npt.NDArray[np.float64]
    pls_loadings_left: npt.NDArray[np.float64]
    pls_loadings_right: npt.NDArray[np.float64]
    rv_coefficients: npt.NDArray[np.float64]
    integration_index: float
    n_components: int
    n_specimens: int

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Two-Block Partial Least Squares Analysis')}\n",
            f"{'=' * 50}\n",
            f"{_('Number of specimens: {0}').format(self.n_specimens)}\n",
            f"{_('Number of PLS components: {0}').format(self.n_components)}\n",
            f"{_('Overall integration index (mean RV): {0:.4f}').format(self.integration_index)}\n",
            "",
            f"{_('RV Coefficients by component:')}\n",
        ]
        for i, rv in enumerate(self.rv_coefficients):
            lines.append(f"  {i + 1}: {rv:.4f}")

        lines.append("")
        lines.append(f"{_('Covariance explained:')}")
        for i, (cov, cum) in enumerate(zip(self.covariance_explained, self.cumulative_covariance, strict=False)):
            lines.append(f"  {i + 1}: {cov:.2f}% (cumulative: {cum:.2f}%)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "singular_values": self.singular_values.tolist(),
            "covariance_explained": self.covariance_explained.tolist(),
            "cumulative_covariance": self.cumulative_covariance.tolist(),
            "left_scores": self.left_scores.tolist(),
            "right_scores": self.right_scores.tolist(),
            "pls_loadings_left": self.pls_loadings_left.tolist(),
            "pls_loadings_right": self.pls_loadings_right.tolist(),
            "rv_coefficients": self.rv_coefficients.tolist(),
            "integration_index": self.integration_index,
            "n_components": self.n_components,
            "n_specimens": self.n_specimens,
            "summary": self.summary(),
        }


# =============================================================================
# Allometry Analyzer
# =============================================================================


class AllometryAnalyzer:
    """
    Analyzes allometric relationship between size and shape.

    Performs multivariate regression of Procrustes coordinates on
    log-transformed centroid size to detect and quantify allometric
    shape variation.

    Example:
        >>> from morphometrics import GPAAnalyzer, AllometryAnalyzer
        >>> gpa = GPAAnalyzer()
        >>> gpa_result = gpa.align(configurations)
        >>> allometry = AllometryAnalyzer()
        >>> result = allometry.analyze_allometry(gpa_result.aligned_configurations)
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize allometry analyzer."""
        self._logger = logging.getLogger(f"{__name__}.AllometryAnalyzer")
        self._lock = threading.RLock()
        self._last_result: AllometryResult | None = None

    @property
    def last_result(self) -> AllometryResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze_allometry(
        self,
        aligned_configurations: npt.NDArray,
        n_components: int | None = None,
    ) -> AllometryResult:
        """
        Analyze relationship between centroid size and shape.

        Parameters:
            aligned_configurations: 3D array (n_specimens, n_landmarks, n_dims)
                                  from GPAResult.aligned_configurations
            n_components: Number of PCs to use as shape variables (default: all)

        Returns:
            AllometryResult with regression coefficients and statistics

        Raises:
            ValidationError: If input data is invalid
        """
        with self._lock:
            self._logger.info(f"Analyzing allometry for shape {aligned_configurations.shape}")

            # Validate input
            if aligned_configurations.ndim != 3:
                raise ValidationError(_("Aligned configurations must be 3D array (specimens, landmarks, dims)"))

            n_specimens, n_landmarks, n_dims = aligned_configurations.shape

            if n_specimens < 3:
                raise ValidationError(_("Need at least 3 specimens for allometry analysis"))

            # Step 1: Compute centroid size for each specimen
            centroid_sizes = self._compute_centroid_sizes(aligned_configurations)
            log_cs = np.log(centroid_sizes)

            # Step 2: Flatten configurations to 2D
            # Shape: (n_specimens, n_landmarks * n_dims)
            flattened = aligned_configurations.reshape(n_specimens, n_landmarks * n_dims)

            # Step 3: Center the shape data
            mean_shape = np.mean(flattened, axis=0)
            shape_centered = flattened - mean_shape

            # Step 4: Determine which columns to use (optional PCA reduction)
            if n_components is not None and n_components < shape_centered.shape[1]:
                # Use PCA to reduce dimensionality
                pca_matrix = shape_centered.T @ shape_centered / (n_specimens - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(pca_matrix)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx[:n_components]]
                shape_reduced = shape_centered @ eigenvectors
            else:
                shape_reduced = shape_centered

            # Step 5: Multivariate regression - Shape ~ log(CS)
            # Design matrix: [1, log(CS)]
            X = np.column_stack([np.ones(n_specimens), log_cs])

            # Solve least squares: β = (X'X)^(-1) X'y
            # Using numpy lstsq for numerical stability
            beta, _residuals_mat, _rank, _s = np.linalg.lstsq(X, shape_reduced, rcond=None)

            intercept = beta[0]
            coefficients = beta[1]

            # Step 6: Compute predicted shapes and residuals
            predicted = X @ beta
            residuals = shape_reduced - predicted

            # Step 7: R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(shape_reduced**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Step 8: F-test for isometry (coefficients == 0)
            # Compare full model vs intercept-only model
            X_null = np.ones((n_specimens, 1))
            beta_null, _, _, _ = np.linalg.lstsq(X_null, shape_reduced, rcond=None)
            predicted_null = X_null @ beta_null
            ss_null = np.sum((shape_reduced - predicted_null) ** 2)

            # Degrees of freedom
            df1 = shape_reduced.shape[1]  # number of shape variables
            df2 = n_specimens - 2  # residual df

            if ss_res > 0 and df2 > 0:
                f_statistic = ((ss_null - ss_res) / df1) / (ss_res / df2)
                # P-value from F-distribution
                isometry_pvalue = 1.0 - stats.f.cdf(f_statistic, df1, df2)
            else:
                f_statistic = 0.0
                isometry_pvalue = 1.0

            # Step 9: Transform predictions back to full shape space if needed
            if n_components is not None and n_components < n_landmarks * n_dims:
                predicted_full = predicted @ eigenvectors.T + mean_shape
            else:
                predicted_full = predicted + mean_shape

            result = AllometryResult(
                centroid_sizes=centroid_sizes,
                log_centroid_sizes=log_cs,
                regression_coefficients=coefficients,
                regression_intercept=intercept,
                r_squared=float(r_squared),
                f_statistic=float(f_statistic),
                isometry_pvalue=float(isometry_pvalue),
                residuals=residuals,
                predicted_shapes=predicted_full,
                n_specimens=n_specimens,
                n_landmarks=n_landmarks,
                n_dims=n_dims,
            )

            self._last_result = result
            self._logger.info(f"Allometry: R²={r_squared:.4f}, F={f_statistic:.4f}, p={isometry_pvalue:.4f}")
            return result

    def _compute_centroid_sizes(self, configurations: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Compute centroid size for each specimen.

        CS = sqrt(sum_{i=1}^{k} sum_{j=1}^{m} x_{ij}²)

        Parameters:
            configurations: 3D array (n_specimens, n_landmarks, n_dims)

        Returns:
            1D array of centroid sizes (n_specimens,)
        """
        n_specimens = configurations.shape[0]
        centroid_sizes = np.zeros(n_specimens)

        for i in range(n_specimens):
            # Flatten to 1D and compute norm
            centroid_sizes[i] = np.sqrt(np.sum(configurations[i] ** 2))

        return centroid_sizes


# =============================================================================
# Integration Analyzer (2B-PLS)
# =============================================================================


class IntegrationAnalyzer:
    """
    Analyzes morphological integration using Two-Block Partial Least Squares.

    2B-PLS finds pairs of axes that maximize协方差 between two blocks
    of shape variables, providing a measure of morphological integration.

    Example:
        >>> pls = IntegrationAnalyzer()
        >>> result = pls.analyze_pls(block_a, block_b)
        >>> print(f"Integration index: {result.integration_index:.4f}")
    """

    def __init__(self) -> None:
        """Initialize integration analyzer."""
        self._logger = logging.getLogger(f"{__name__}.IntegrationAnalyzer")
        self._lock = threading.RLock()
        self._last_result: PLSResult | None = None

    @property
    def last_result(self) -> PLSResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze_pls(
        self,
        block_a: npt.NDArray,
        block_b: npt.NDArray,
        n_components: int | None = None,
    ) -> PLSResult:
        """
        Perform Two-Block Partial Least Squares analysis.

        Parameters:
            block_a: First block of shape variables (n_specimens, n_vars_a)
            block_b: Second block of shape variables (n_specimens, n_vars_b)
            n_components: Number of PLS components (default: min(n_vars_a, n_vars_b, n_specimens-1))

        Returns:
            PLSResult with PLS scores, loadings, and integration metrics

        Raises:
            ValidationError: If input data is invalid
        """
        with self._lock:
            self._logger.info(f"PLS analysis: block_a {block_a.shape}, block_b {block_b.shape}")

            # Validate inputs
            if block_a.ndim != 2 or block_b.ndim != 2:
                raise ValidationError(_("Both blocks must be 2D arrays"))

            n_specimens_a, n_vars_a = block_a.shape
            n_specimens_b, n_vars_b = block_b.shape

            if n_specimens_a != n_specimens_b:
                raise ValidationError(_("Both blocks must have same number of specimens"))

            if n_specimens_a < 3:
                raise ValidationError(_("Need at least 3 specimens for PLS analysis"))

            # Determine number of components
            max_comp = min(n_vars_a, n_vars_b, n_specimens_a - 1)
            if n_components is None:
                n_components = max_comp
            n_components = min(n_components, max_comp)

            # Center both blocks
            block_a_centered = block_a - np.mean(block_a, axis=0)
            block_b_centered = block_b - np.mean(block_b, axis=0)

            # Cross-block covariance matrix
            C_ab = (1.0 / (n_specimens_a - 1)) * block_a_centered.T @ block_b_centered

            # SVD of cross-covariance
            U, singular_values, Vt = np.linalg.svd(C_ab, full_matrices=False)

            # PLS scores
            pls_scores_left = block_a_centered @ U[:, :n_components]
            pls_scores_right = block_b_centered @ Vt[:n_components, :].T

            # RV coefficients per component
            rv_coefficients = np.zeros(n_components)
            for i in range(n_components):
                if np.std(pls_scores_left[:, i]) > 0 and np.std(pls_scores_right[:, i]) > 0:
                    rv_coefficients[i] = np.corrcoef(pls_scores_left[:, i], pls_scores_right[:, i])[0, 1]
                else:
                    rv_coefficients[i] = 0.0

            # Overall integration index (mean absolute RV)
            integration_index = float(np.mean(np.abs(rv_coefficients)))

            # Covariance explained
            total_variance = np.sum(singular_values**2)
            covariance_explained = (
                singular_values[:n_components] ** 2 / total_variance * 100
                if total_variance > 0
                else np.zeros(n_components)
            )
            cumulative_covariance = np.cumsum(covariance_explained)

            result = PLSResult(
                singular_values=singular_values[:n_components],
                covariance_explained=covariance_explained,
                cumulative_covariance=cumulative_covariance,
                left_scores=pls_scores_left,
                right_scores=pls_scores_right,
                pls_loadings_left=U[:, :n_components],
                pls_loadings_right=Vt[:n_components, :].T,
                rv_coefficients=rv_coefficients,
                integration_index=integration_index,
                n_components=n_components,
                n_specimens=n_specimens_a,
            )

            self._last_result = result
            self._logger.info(f"PLS: integration index = {integration_index:.4f}")
            return result

    def divide_configuration_into_blocks(
        self,
        aligned_configurations: npt.NDArray,
        division: str = "anterior_posterior",
        random_seed: int | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Divide landmark configurations into two blocks for PLS analysis.

        Parameters:
            aligned_configurations: 3D array (n_specimens, n_landmarks, n_dims)
            division: How to divide landmarks

                - ``"anterior_posterior"``: split landmarks into two
                  contiguous groups at the midpoint.
                - ``"size_matched"``: split the *landmark columns* at
                  the midpoint (same as anterior_posterior on the
                  flattened columns; kept for API compatibility). The
                  previous implementation first partitioned *specimens*
                  by centroid size and then threw that partition away
                  and re-sliced by columns — the size-based partition
                  was dead code that produced blocks with mismatched
                  specimen counts, which PLS cannot consume. The dead
                  code has been removed.
                - ``"random"``: random permutation of landmarks, split
                  at the midpoint.
            random_seed: Optional seed for the ``"random"`` division.
                The previous implementation hard-coded
                ``np.random.seed(42)`` which silently reseeds the
                *global* numpy RNG — a side effect that contaminated
                every downstream stochastic operation. Use a local
                :class:`numpy.random.Generator` instead so the global
                RNG state is left untouched.

        Returns:
            ``(block_a, block_b)`` tuple of 2D arrays.

        Raises:
            ValidationError: If division method is invalid.
        """
        if aligned_configurations.ndim != 3:
            raise ValidationError(_("Aligned configurations must be 3D array"))

        n_specimens, n_landmarks, n_dims = aligned_configurations.shape
        flattened = aligned_configurations.reshape(n_specimens, n_landmarks * n_dims)

        if division == "anterior_posterior":
            # Simple split at midpoint
            mid = n_landmarks // 2
            block_a = flattened[:, : mid * n_dims]
            block_b = flattened[:, mid * n_dims :]

        elif division == "size_matched":
            # Split landmark columns at the midpoint. (The previous
            # implementation computed a specimen-level size partition
            # and then discarded it; that dead code is removed here.)
            mid_cols = (n_landmarks * n_dims) // 2
            block_a = flattened[:, :mid_cols]
            block_b = flattened[:, mid_cols:]

        elif division == "random":
            # Use a local Generator so the global numpy RNG is not
            # perturbed as a side effect.
            rng = np.random.default_rng(random_seed)
            indices = rng.permutation(n_landmarks)
            mid = n_landmarks // 2
            # Select first half landmarks and second half landmarks
            first_half = indices[:mid]
            second_half = indices[mid:]
            # Convert landmark indices to flattened column indices (each landmark spans n_dims columns)
            block_a_cols = np.concatenate([first_half * n_dims + d for d in range(n_dims)])
            block_b_cols = np.concatenate([second_half * n_dims + d for d in range(n_dims)])
            block_a = flattened[:, np.sort(block_a_cols)]
            block_b = flattened[:, np.sort(block_b_cols)]

        else:
            raise ValidationError(_("Unknown division method: {0}").format(division))

        return block_a, block_b
