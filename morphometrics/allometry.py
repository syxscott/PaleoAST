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
        r_squared: Proportion of shape variance explained by size. ``NaN``
            when the regression is undefined, i.e. the centroid sizes are
            constant across specimens (see
            :meth:`AllometryAnalyzer.analyze_allometry`).
        f_statistic: F-statistic for isometry test
        isometry_pvalue: P-value for isometry test
        residuals: Residual shape variation after removing size effect
        predicted_shapes: Predicted shape at mean log(size)
        n_specimens: Number of specimens
        n_landmarks: Number of landmarks
        n_dims: Number of dimensions
        cac_scores: Common Allometric Component scores — projection of each
            specimen onto the unit allometric direction (Mitteroecker et al.
            2004; geomorph ``plotAllometry(method="CAC")``)
        cac_variance: Fraction of total Procrustes variance carried by the CAC
        rsc_scores: Residual Shape Component scores — PCA of the shapes after
            the allometric direction is removed
        rsc_proportion: Variance proportions of the reported RSC axes
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
    cac_scores: npt.NDArray[np.float64] | None = None
    cac_variance: float | None = None
    rsc_scores: npt.NDArray[np.float64] | None = None
    rsc_proportion: npt.NDArray[np.float64] | None = None

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
        degenerate = not np.isfinite(self.r_squared)
        if degenerate:
            r2_line = _("R²: undefined (centroid size is identical for every specimen)")
            isometry_line = _("Isometry test: F={0:.4f}, p={1:.4f} {2}").format(
                self.f_statistic, self.isometry_pvalue, sig
            )
            note = (
                "\n"
                + _(
                    "Note: log(centroid size) is constant, so the size-shape regression "
                    "has no explanatory power and R² is undefined (not 0). Supply the "
                    "PRE-GPA centroid sizes via analyze_allometry(centroid_sizes=...), "
                    "e.g. GPAResult.centroid_sizes, when the shapes come from GPA."
                )
                + "\n"
            )
        else:
            r2_line = _("R²: {0}").format(f"{self.r_squared:.4f}")
            isometry_line = _("Isometry test: F={0:.4f}, p={1:.4f} {2}").format(
                self.f_statistic, self.isometry_pvalue, sig
            )
            note = ""
        mean_cs = np.mean(self.centroid_sizes)
        min_cs = np.min(self.centroid_sizes)
        max_cs = np.max(self.centroid_sizes)
        return (
            f"{_('Allometry Analysis')}\n"
            f"{'=' * 50}\n"
            f"{_('Specimens: {0}, Landmarks: {1}, Dimensions: {2}').format(self.n_specimens, self.n_landmarks, self.n_dims)}\n"
            f"{r2_line}\n"
            f"{isometry_line}\n"
            f"{_('Mean centroid size: {0}').format(f'{mean_cs:.4f}')}\n"
            f"{_('Size range: {0:.4f} to {1:.4f}').format(min_cs, max_cs)}"
            f"{note}"
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
            "cac_scores": None if self.cac_scores is None else self.cac_scores.tolist(),
            "cac_variance": self.cac_variance,
            "rsc_scores": None if self.rsc_scores is None else self.rsc_scores.tolist(),
            "rsc_proportion": None if self.rsc_proportion is None else self.rsc_proportion.tolist(),
            "summary": self.summary(),
        }


@dataclass
class PLSResult:
    """
    Container for 2-Block Partial Least Squares analysis results.

    Attributes:
        singular_values: PLS singular values (cross-covariance SVD)
        covariance_explained: Percentage of cross-block covariance explained per component
        cumulative_covariance: Cumulative covariance explained
        left_scores: PLS scores for block A (n_specimens, n_components)
        right_scores: PLS scores for block B (n_specimens, n_components)
        pls_loadings_left: Loadings for block A
        pls_loadings_right: Loadings for block B
        pls_correlations: Correlation between paired PLS score vectors per
            component (Rohlf & Corti 2000).  Formerly mislabelled
            ``rv_coefficients`` — these are score correlations, not Escoufier
            RV coefficients.
        integration_index: Correlation of the first pair of PLS axes (r₁),
            the standard two-block integration estimate (geomorph
            ``integration.test`` r.pls).
        rv_coefficient: Escoufier's RV coefficient between the two centred
            blocks (overall multivariate association), or None.
        pls1_pvalue: Two-sided permutation p-value for |r₁| (None when no
            permutations were run).
        pls1_z: Effect size of r₁ relative to the permutation null —
            (r_obs − mean(r_rand)) / sd(r_rand) (geomorph 3.0.4+ Z).
        random_correlations: Permutation distribution of r₁ (None unless
            permutations were run).
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
    pls_correlations: npt.NDArray[np.float64]
    integration_index: float
    n_components: int
    n_specimens: int
    rv_coefficient: float | None = None
    pls1_pvalue: float | None = None
    pls1_z: float | None = None
    random_correlations: npt.NDArray[np.float64] | None = None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Two-Block Partial Least Squares Analysis')}\n",
            f"{'=' * 50}\n",
            f"{_('Number of specimens: {0}').format(self.n_specimens)}\n",
            f"{_('Number of PLS components: {0}').format(self.n_components)}\n",
            f"{_('Integration index (PLS1 correlation r1): {0:.4f}').format(self.integration_index)}\n",
        ]
        if self.rv_coefficient is not None:
            lines.append(f"{_('Escoufier RV coefficient: {0:.4f}').format(self.rv_coefficient)}\n")
        if self.pls1_pvalue is not None:
            lines.append(f"{_('Permutation test of r1: p={0:.4f}, Z={1:.2f}').format(self.pls1_pvalue, self.pls1_z)}\n")
        lines.append("")
        lines.append(f"{_('PLS score correlations by component:')}\n")
        for i, r in enumerate(self.pls_correlations):
            lines.append(f"  {i + 1}: {r:.4f}")

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
            "pls_correlations": self.pls_correlations.tolist(),
            "integration_index": self.integration_index,
            "rv_coefficient": self.rv_coefficient,
            "pls1_pvalue": self.pls1_pvalue,
            "pls1_z": self.pls1_z,
            "random_correlations": None if self.random_correlations is None else self.random_correlations.tolist(),
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
        >>> result = allometry.analyze_allometry(
        ...     gpa_result.aligned_configurations,
        ...     centroid_sizes=gpa_result.centroid_sizes,
        ... )
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
        centroid_sizes: npt.NDArray | None = None,
    ) -> AllometryResult:
        """
        Analyze relationship between centroid size and shape.

        Parameters:
            aligned_configurations: 3D array (n_specimens, n_landmarks, n_dims)
                                  from GPAResult.aligned_configurations
            n_components: Number of PCs to use as shape variables (default: all)
            centroid_sizes: Optional PRE-GPA centroid size per specimen
                (n_specimens,), e.g. ``GPAResult.centroid_sizes``. Required
                whenever ``aligned_configurations`` is GPA output, because
                GPA scales every specimen to a common centroid size: the
                sizes recomputed from such an array are all equal, so
                log(CS) is a constant regressor and the regression is
                undefined. When ``None`` (default) the sizes are recomputed
                from ``aligned_configurations`` as before, which is correct
                only for arrays that are not size-normalised.

        Returns:
            AllometryResult with regression coefficients and statistics.
            ``r_squared`` is ``NaN`` when log(CS) is constant, because the
            variance explained by size is then undefined rather than 0.

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

            # Step 1: Centroid size for each specimen.  When the caller
            # supplies the pre-GPA sizes (the only informative choice for
            # GPA output) use them; otherwise fall back to recomputing from
            # the array as before.
            if centroid_sizes is None:
                centroid_sizes = self._compute_centroid_sizes(aligned_configurations)
            else:
                centroid_sizes = self._validate_centroid_sizes(centroid_sizes, n_specimens)
            log_cs = np.log(centroid_sizes)
            # A constant regressor makes the design matrix rank-deficient:
            # log(CS) explains nothing, so R² is undefined and the F-test
            # can only return F=0, p=1.  GPA does not produce *exactly*
            # equal sizes -- it leaves a residue around 1e-16 -- so the
            # test is relative: a spread that small is float noise, not
            # biology (real specimens differ in size by a factor of ~2).
            log_scale = max(float(np.max(np.abs(log_cs))), 1.0)
            regressor_degenerate = bool(np.ptp(log_cs) <= 1e-12 * log_scale)
            if regressor_degenerate:
                self._logger.warning(
                    "Centroid size is constant across specimens, so the size-shape "
                    "regression is undefined; supply pre-GPA sizes via "
                    "centroid_sizes=GPAResult.centroid_sizes"
                )

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

            # Step 7: R-squared (still computed in the reduced space
            # because that is where the regression was fitted).
            # A constant regressor yields whatever the rank-deficient
            # lstsq happens to return, i.e. a number that can look like a
            # real effect size while being pure floating-point noise
            # (0.6, 0.0, -2e-16 ... depending on the data). Report NaN so
            # the caller cannot mistake it for variance explained.
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(shape_reduced**2)
            if regressor_degenerate or ss_tot <= 0:
                r_squared = float("nan")
            else:
                r_squared = 1 - (ss_res / ss_tot)

            # Step 8: F-test for isometry (coefficients == 0).
            # The isometry hypothesis is "shape does not depend on log(CS)
            # at all", which is a statement about the FULL shape space —
            # not about the PCA-reduced axes.  Therefore the F-statistic
            # MUST be computed on the FULL shape space: SS in the full
            # shape and df1 = full shape dimension.  The previous
            # implementation used ``df1 = shape_reduced.shape[1]``, which
            # silently truncated the test to the retained PCs and was
            # inconsistent with the function's docstring.
            full_shape_dim = n_landmarks * n_dims
            # Step 9 (reordered): we need predicted_full to compute the
            # full-space SS.  The back-projection uses the orthonormal
            # PCA loadings, so the back-projected regression reproduces
            # the reduced regression on the retained axes and zeros on
            # the discarded ones.
            if n_components is not None and n_components < full_shape_dim:
                predicted_full = predicted @ eigenvectors.T + mean_shape
            else:
                predicted_full = predicted + mean_shape

            X_null = np.ones((n_specimens, 1))
            beta_null, _, _, _ = np.linalg.lstsq(X_null, shape_centered, rcond=None)
            predicted_null_full = X_null @ beta_null + mean_shape
            ss_null_full = float(np.sum((flattened - predicted_null_full) ** 2))
            ss_res_full = float(np.sum((flattened - predicted_full) ** 2))

            df1 = full_shape_dim
            df2 = n_specimens - 2  # residual df

            if ss_res_full > 0 and df2 > 0:
                f_statistic = ((ss_null_full - ss_res_full) / df1) / (ss_res_full / df2)
                # P-value from F-distribution
                isometry_pvalue = 1.0 - stats.f.cdf(f_statistic, df1, df2)
            else:
                f_statistic = 0.0
                isometry_pvalue = 1.0

            # Step 10: Common Allometric Component (CAC) and Residual Shape
            # Components (RSC) in the FULL shape space (Mitteroecker et al.
            # 2004; geomorph plotAllometry): unit allometric direction
            # e = a/||a|| with a = Yc'xc / (xc'xc); CAC scores = Yc e; RSC =
            # PCA of Yc after removing the allometric direction.
            Yc = flattened - mean_shape
            xc = log_cs - np.mean(log_cs)
            xzx = float(xc @ xc)
            total_shape_var = float(np.sum(Yc**2))
            cac_scores = cac_variance = rsc_scores = rsc_proportion = None
            if xzx > 0 and total_shape_var > 0:
                a_vec = Yc.T @ xc / xzx
                a_norm = float(np.linalg.norm(a_vec))
                if a_norm > np.finfo(float).eps:
                    e = a_vec / a_norm
                    cac_proj = Yc @ e
                    cac_scores = cac_proj
                    cac_variance = float(np.sum(cac_proj**2) / total_shape_var)
                    resid_full = Yc - np.outer(cac_proj, e)
                    G = resid_full.T @ resid_full / (n_specimens - 1)
                    evals, evecs = np.linalg.eigh(G)
                    order = np.argsort(evals)[::-1]
                    evals = np.clip(evals[order], 0.0, None)
                    evecs = evecs[:, order]
                    q = int(min(5, np.count_nonzero(evals > 1e-12), n_specimens - 1))
                    if q > 0:
                        rsc_scores = resid_full @ evecs[:, :q]
                        rsc_var = float(np.sum(resid_full**2))
                        rsc_proportion = evals[:q] / rsc_var if rsc_var > 0 else np.zeros(q)

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
                cac_scores=cac_scores,
                cac_variance=cac_variance,
                rsc_scores=rsc_scores,
                rsc_proportion=rsc_proportion,
            )

            self._last_result = result
            self._logger.info(f"Allometry: R²={r_squared:.4f}, F={f_statistic:.4f}, p={isometry_pvalue:.4f}")
            return result

    @staticmethod
    def _validate_centroid_sizes(
        centroid_sizes: npt.NDArray,
        n_specimens: int,
    ) -> npt.NDArray[np.float64]:
        """
        Validate externally supplied centroid sizes.

        A wrong length silently misaligns specimens against their shapes and
        a non-positive or non-finite size produces ``log(<= 0)``, so both are
        rejected here rather than inside the regression.

        Parameters:
            centroid_sizes: Candidate sizes, e.g. ``GPAResult.centroid_sizes``
            n_specimens: Number of specimens in the shape array

        Returns:
            1D float array of centroid sizes (n_specimens,)

        Raises:
            ValidationError: If the length is wrong, or any value is not
                finite and strictly positive
        """
        sizes = np.asarray(centroid_sizes, dtype=np.float64).ravel()

        if sizes.size != n_specimens:
            raise ValidationError(
                _("Centroid sizes must have one value per specimen: got {0} for {1} specimens").format(
                    sizes.size, n_specimens
                )
            )

        if not np.all(np.isfinite(sizes)):
            raise ValidationError(_("Centroid sizes must all be finite numbers"))

        if np.any(sizes <= 0):
            raise ValidationError(_("Centroid sizes must all be greater than zero (log size is undefined otherwise)"))

        return sizes

    def _compute_centroid_sizes(self, configurations: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Compute centroid size for each specimen using the Bookstein (1991)
        definition:

            CS = sqrt( sum_i ||x_i - centroid||^2 )

        where ``centroid`` is the mean landmark location of the specimen.

        The previous implementation used the Frobenius norm
        ``sqrt(sum(x^2))`` which is the distance from the *origin*, not
        from the specimen's own centroid. The two coincide only when the
        data happen to be centred at the origin, so any allometry /
        PLS / integration analysis built on the old sizes was biased
        toward specimens whose landmarks happened to be farther from
        (0, 0).

        Parameters:
            configurations: 3D array (n_specimens, n_landmarks, n_dims)

        Returns:
            1D array of centroid sizes (n_specimens,)
        """
        n_specimens = configurations.shape[0]
        centroid_sizes = np.zeros(n_specimens)

        for i in range(n_specimens):
            centroid = configurations[i].mean(axis=0)
            diff = configurations[i] - centroid
            centroid_sizes[i] = np.sqrt(np.sum(diff**2))

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
        permutations: int = 0,
        seed: int | None = None,
    ) -> PLSResult:
        """
        Perform Two-Block Partial Least Squares analysis.

        Parameters:
            block_a: First block of shape variables (n_specimens, n_vars_a)
            block_b: Second block of shape variables (n_specimens, n_vars_b)
            n_components: Number of PLS components (default: min(n_vars_a, n_vars_b, n_specimens-1))
            permutations: If > 0, run a permutation test of the PLS1
                correlation r₁ by shuffling block B specimens
                (Rohlf & Corti 2000 / geomorph ``integration.test``).
            seed: RNG seed for the permutations.

        Returns:
            PLSResult with PLS scores, loadings, score correlations, the
            Escoufier RV coefficient and (optionally) the r₁ permutation
            p-value and Z effect size.

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

            # Per-component correlation between paired PLS score vectors
            # (Rohlf & Corti 2000).  These were previously mislabelled
            # "rv_coefficients"; the true Escoufier RV is computed below.
            pls_correlations = np.zeros(n_components)
            for comp_idx in range(n_components):
                if np.std(pls_scores_left[:, comp_idx]) > 0 and np.std(pls_scores_right[:, comp_idx]) > 0:
                    pls_correlations[comp_idx] = np.corrcoef(
                        pls_scores_left[:, comp_idx], pls_scores_right[:, comp_idx]
                    )[0, 1]

            # Integration index: the PLS1 correlation r₁ (geomorph r.pls)
            integration_index = float(pls_correlations[0])

            # Escoufier's RV coefficient (overall block association):
            # RV = ||S12||² / sqrt(||S11||² * ||S22||²)  (Frobenius norms)
            S11 = block_a_centered.T @ block_a_centered
            S22 = block_b_centered.T @ block_b_centered
            denom_rv = np.sqrt(np.sum(S11**2) * np.sum(S22**2))
            rv_coefficient = float(np.sum(C_ab**2) * (n_specimens_a - 1) ** 2 / denom_rv) if denom_rv > 0 else 0.0

            # Permutation test of r₁: shuffle block B specimens, recompute
            # the leading singular vectors and the score correlation.
            pls1_pvalue: float | None = None
            pls1_z: float | None = None
            random_correlations: npt.NDArray | None = None
            if permutations and permutations > 0:
                rng = np.random.default_rng(seed)
                r_obs = integration_index
                rand_rs = np.zeros(permutations)
                for step in range(permutations):
                    perm_idx = rng.permutation(n_specimens_a)
                    Yp = block_b_centered[perm_idx]
                    C_p = block_a_centered.T @ Yp
                    U_p, _sv, Vt_p = np.linalg.svd(C_p, full_matrices=False)
                    xs = block_a_centered @ U_p[:, 0]
                    ys = Yp @ Vt_p[0, :]
                    if np.std(xs) > 0 and np.std(ys) > 0:
                        rand_rs[step] = np.corrcoef(xs, ys)[0, 1]
                random_correlations = rand_rs
                count = int(np.sum(np.abs(rand_rs) >= abs(r_obs) - 1e-15))
                pls1_pvalue = (1.0 + count) / (1.0 + permutations)
                sd_rand = float(np.std(rand_rs, ddof=1))
                pls1_z = (r_obs - float(np.mean(rand_rs))) / sd_rand if sd_rand > 0 else 0.0

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
                pls_correlations=pls_correlations,
                integration_index=integration_index,
                n_components=n_components,
                n_specimens=n_specimens_a,
                rv_coefficient=rv_coefficient,
                pls1_pvalue=pls1_pvalue,
                pls1_z=pls1_z,
                random_correlations=random_correlations,
            )

            self._last_result = result
            self._logger.info(f"PLS: r1 = {integration_index:.4f}, RV = {rv_coefficient:.4f}")
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
