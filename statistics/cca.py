# =============================================================================
# FILE: statistics/cca.py
# =============================================================================
"""
Canonical Correspondence Analysis (CCA) and Redundancy Analysis (RDA) Module
for PaleoAST.

Constrained ordination methods that relate species composition to environmental
variables.

Mathematical Foundation:

RDA (Redundancy Analysis):
    Y_centered = Y - Y_bar (center species data)
    X_centered = X - X_bar (center environmental data)
    Q = X @ inv(X^T X) @ X^T (projection matrix)
    M = Y^T Q Y (cross-covariance matrix)
    M = U Λ U^T (eigenvalue decomposition)
    scores = Y_centered @ U

CCA (Canonical Correspondence Analysis):
    Uses chi-square distance instead of Euclidean
    Similar derivation with weighted averages

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class CCAResult:
    """
    Container for CCA/RDA analysis results.

    Attributes:
        site_scores: Sample scores (n_samples, n_components)
        species_scores: Species scores (n_species, n_components)
        biplot_scores: Environmental variable arrows (n_env, n_components)
        eigenvalues: Eigenvalues for each axis
        proportion_explained: Variance explained by each axis (%)
        cumulative_proportion: Cumulative variance explained (%)
        method: 'rda' or 'cca'
        n_components: Number of constrained axes
        n_samples: Number of samples
        n_species: Number of species/variables
        n_env: Number of environmental variables
        species_names: Names of species/variables
        env_names: Names of environmental variables
        inertia: Total inertia (for CCA) or total variance (for RDA)
        constrained_variance: Variance explained by constrained axes
    """

    site_scores: npt.NDArray
    species_scores: npt.NDArray
    biplot_scores: npt.NDArray
    eigenvalues: npt.NDArray
    proportion_explained: npt.NDArray
    cumulative_proportion: npt.NDArray
    method: str
    n_components: int
    n_samples: int
    n_species: int
    n_env: int
    species_names: list[str]
    env_names: list[str]
    inertia: float
    constrained_variance: float

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Constrained Ordination Analysis')}",
            "=" * 50,
            f"{_('Method: {0}').format(self.method.upper())}",
            f"{_('Constrained axes: {0}').format(self.n_components)}",
            f"{_('Samples: {0}').format(self.n_samples)}",
            f"{_('Species: {0}').format(self.n_species)}",
            f"{_('Environmental variables: {0}').format(self.n_env)}",
            "",
            f"{_('Eigenvalue | Proportion | Cumulative')}",
            "-" * 45,
        ]

        for i in range(self.n_components):
            lines.append(
                f"AX{i + 1:6d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.proportion_explained[i]:9.2f}% | "
                f"{self.cumulative_proportion[i]:9.2f}%"
            )

        lines.append("")
        lines.append(f"{_('Total constrained variance: {0}%').format(f'{self.constrained_variance:.2f}')}")
        lines.append(f"{_('Inertia (total): {0:.4f}').format(self.inertia)}")

        return "\n".join(lines)


class CCAAnalyzer:
    """
    Canonical Correspondence Analysis (CCA) and Redundancy Analysis (RDA) analyzer.

    CCA/RDA are constrained ordination methods that relate a species
    composition matrix to environmental variables.

    CCA is designed for count data (e.g., species abundances) and uses
    chi-square distance.

    RDA is designed for continuous data and uses Euclidean distance.
    """

    def __init__(self) -> None:
        """Initialize the CCA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.CCAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: CCAResult | None = None
        self._logger.info("CCAAnalyzer initialized")

    def analyze(
        self,
        Y: npt.NDArray,
        X: npt.NDArray,
        n_components: int | None = None,
        method: str = "cca",
        species_names: list[str] | None = None,
        env_names: list[str] | None = None,
    ) -> CCAResult:
        """
        Perform CCA or RDA analysis.

        Parameters:
            Y: Species abundance matrix (n_samples, n_species)
            X: Environmental variable matrix (n_samples, n_env)
            n_components: Number of constrained axes to extract
            method: 'cca' for Canonical Correspondence Analysis,
                   'rda' for Redundancy Analysis
            species_names: Names of species/variables
            env_names: Names of environmental variables

        Returns:
            CCAResult: CCA/RDA analysis results
        """
        with self._lock:
            # Validate input
            Y_arr = validate_data_array(Y, allow_nan=False, name="species_matrix")
            X_arr = validate_data_array(X, allow_nan=False, name="env_matrix")

            n_samples, n_species = Y_arr.shape
            n_env = X_arr.shape[1]

            self._logger.info(
                f"CCA/RDA analyze started: {n_samples} samples, {n_species} species, "
                f"{n_env} env variables, method={method}, n_components={n_components}"
            )

            if X_arr.shape[0] != n_samples:
                raise MatrixDimensionError("Environmental matrix must have same number of samples as species matrix")

            # Determine number of components
            max_components = min(n_samples - 1, n_species, n_env)
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)

            if n_components < 1:
                raise MatrixDimensionError("Cannot perform constrained ordination: insufficient dimensions")

            # Perform analysis based on method
            if method == "cca":
                result = self._analyze_cca(Y_arr, X_arr, n_components, species_names, env_names)
            else:
                result = self._analyze_rda(Y_arr, X_arr, n_components, species_names, env_names)

            self._last_result = result
            self._logger.info(
                f"CCA/RDA completed: method={result.method}, constrained_variance={result.constrained_variance:.2f}%"
            )
            return result

    def _solve_XtX(
        self,
        XtX: npt.NDArray,
        method: str = "cca",
        ridge_lambda: float = 1e-8,
        cond_threshold: float = 1e10,
    ) -> npt.NDArray:
        """
        Solve X'X beta = X'y for the constrained ordination.

        Uses ridge-regularized least squares when X'X is near-singular
        (ill-conditioned), which commonly occurs with collinear environmental
        variables. This approach follows the numerical practices of R's vegan
        package (see ?vegan::cca, which uses qr() decomposition).

        Parameters:
            XtX: The X'X matrix (n_env, n_env)
            method: 'cca' or 'rda' (used in warning messages)
            ridge_lambda: Ridge regularization parameter (default 1e-8).
                Added to diagonal: X'X + lambda*I
            cond_threshold: Condition number threshold for warning (default 1e10).

        Returns:
            XtX_inv: The inverse (or regularized inverse) of XtX

        References:
            - ter Braak (1986) Ecology 67:1167-1176
            - Legendre & Legendre (2012) Numerical Ecology, 3rd ed., Elsevier
            - R package vegan::cca() source code
        """
        cond = np.linalg.cond(XtX)
        if cond > cond_threshold:
            self._logger.warning(
                f"{method.upper()}: X'X condition number = {cond:.2e} > {cond_threshold:.0e}. "
                f"Environmental matrix is ill-conditioned (collinear variables?). "
                f"Applying ridge regularization (lambda={ridge_lambda})."
            )
            # Ridge regularization: X'X + lambda*I, then solve via lstsq
            XtX_ridge = XtX + ridge_lambda * np.eye(XtX.shape[0])
            # Solve (X'X + lambda*I) @ XtX_inv = I using lstsq
            identity = np.eye(XtX.shape[0])
            XtX_inv, *_ = np.linalg.lstsq(XtX_ridge, identity, rcond=None)
        else:
            # Well-conditioned: use standard inverse for efficiency
            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                # Fallback to ridge-regularized lstsq if inv fails
                self._logger.warning(
                    f"{method.upper()}: np.linalg.inv(XtX) failed. "
                    f"Falling back to ridge-regularized lstsq."
                )
                XtX_ridge = XtX + ridge_lambda * np.eye(XtX.shape[0])
                identity = np.eye(XtX.shape[0])
                XtX_inv, *_ = np.linalg.lstsq(XtX_ridge, identity, rcond=None)
        return XtX_inv

    def _analyze_rda(
        self,
        Y: npt.NDArray,
        X: npt.NDArray,
        n_components: int,
        species_names: list[str] | None,
        env_names: list[str] | None,
    ) -> CCAResult:
        """
        Perform Redundancy Analysis (RDA).

        Mathematical Steps:
            1. Center both Y and X
            2. Compute Q = X(X'X)^-1 X' (projection matrix)
            3. Compute M = Y'QY (cross-covariance)
            4. Eigendecomposition: M = UΛU'
            5. Scores: Y_c * U
        """
        n_samples, n_species = Y.shape
        n_env = X.shape[1]

        # Step 1: Center the data
        Y_centered = Y - Y.mean(axis=0)
        X_centered = X - X.mean(axis=0)

        # Step 2: Compute projection matrix Q = X(X'X)^-1 X'
        # Use ridge-regularized lstsq for numerical stability when X'X is
        # near-singular (collinear environmental variables). This follows
        # the approach used in R's vegan package (qr() decomposition).
        XtX = X_centered.T @ X_centered
        XtX_inv = self._solve_XtX(XtX, method="rda")

        Q = X_centered @ XtX_inv @ X_centered.T

        # Step 3: Compute cross-covariance matrix M = Y'QY
        M = Y_centered.T @ Q @ Y_centered

        # Step 4: Eigendecomposition of M
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort by eigenvalues (descending) and take top n_components
        sorted_indices = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Ensure positive eigenvalues (numerical stability).
        # Warn the caller when eigenvalues have been clipped — silent
        # truncation hides near-singular environmental matrices that
        # the user should know about.
        clipped = eigenvalues < 1e-10
        if np.any(clipped):
            n_clipped = int(np.sum(clipped))
            self._logger.warning(
                f"CCA: {n_clipped} eigenvalue(s) clipped to 1e-10 — "
                "near-singular environmental matrix; check for collinear variables."
            )
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Step 5: Compute scores
        # Site scores (sample scores)
        site_scores = Y_centered @ eigenvectors

        # Species scores (weighted by eigenvalues)
        species_scores = eigenvectors * np.sqrt(eigenvalues)

        # Biplot scores (environmental variable scores)
        # env_effect = X_centered' * site_scores
        biplot_scores = X_centered.T @ site_scores

        # Normalize biplot scores
        scale_factor = np.sqrt(np.sum(biplot_scores**2, axis=0))
        scale_factor[scale_factor == 0] = 1
        biplot_scores = biplot_scores / scale_factor * np.sqrt(eigenvalues)

        # Compute proportions
        # eigenvalues are on sum-of-squares scale (not divided by n), so use total SS
        total_inertia = np.sum(Y_centered**2)  # Total sum of squares of Y

        proportion_explained = (eigenvalues / total_inertia) * 100 if total_inertia > 0 else np.zeros_like(eigenvalues)
        cumulative_proportion = np.cumsum(proportion_explained)

        # Default names
        if species_names is None:
            species_names = [f"Species_{i + 1}" for i in range(n_species)]
        if env_names is None:
            env_names = [f"Env_{i + 1}" for i in range(n_env)]

        constrained_variance = np.sum(proportion_explained)

        return CCAResult(
            site_scores=site_scores,
            species_scores=species_scores,
            biplot_scores=biplot_scores,
            eigenvalues=eigenvalues,
            proportion_explained=proportion_explained,
            cumulative_proportion=cumulative_proportion,
            method="rda",
            n_components=n_components,
            n_samples=n_samples,
            n_species=n_species,
            n_env=n_env,
            species_names=species_names,
            env_names=env_names,
            inertia=total_inertia,
            constrained_variance=constrained_variance,
        )

    def _analyze_cca(
        self,
        Y: npt.NDArray,
        X: npt.NDArray,
        n_components: int,
        species_names: list[str] | None,
        env_names: list[str] | None,
    ) -> CCAResult:
        """
        Perform Canonical Correspondence Analysis (CCA).

        Mathematical Steps:
            1. Compute chi-square weights (row and column totals)
            2. Center using chi-square standardization
            3. Similar to RDA but with weighted calculations
        """
        n_samples, n_species = Y.shape
        n_env = X.shape[1]

        # Ensure non-negative data for CCA
        if np.any(Y < 0):
            raise ComputationError("CCA requires non-negative abundance data")

        # Compute row and column totals for chi-square weights
        row_totals = Y.sum(axis=1, keepdims=True)
        col_totals = Y.sum(axis=0, keepdims=True)
        grand_total = Y.sum()

        # Avoid division by zero
        row_totals[row_totals == 0] = 1
        col_totals[col_totals == 0] = 1
        grand_total = max(grand_total, 1e-10)

        # Chi-square standardization (ter Braak 1986, Ecology 67:1167-1176).
        #
        # The canonical formulation uses unscaled counts Y directly:
        #   expected[i,k] = (row_total_i * col_total_k) / grand_total
        #   Q[i,k]        = (Y[i,k] - expected[i,k]) / sqrt(expected[i,k])
        #
        # IMPORTANT: Chi-square distance requires positive expected values.
        # When expected == 0, the contribution to chi-square distance is
        # mathematically undefined (0/0 yields no contribution when both
        # observed==0 and expected==0, but is infinite when observed>0).
        # We mark zero-expectation positions with NaN to exclude them from
        # distance computations, preventing the spurious structure that
        # arises from substituting an artificial value like 1.0.
        #
        # References:
        #   - ter Braak (1986) Ecology 67:1167-1176
        #   - Legendre & Legendre (2012) Numerical Ecology, 3rd ed., Elsevier
        p_row = row_totals / grand_total  # (n_samples, 1)
        p_col = col_totals / grand_total  # (1, n_species)
        expected = (row_totals @ col_totals) / grand_total  # (n_samples, n_species)
        # Compute standardized residuals only where expected > 0.
        # Where expected == 0: set to NaN (undefined contribution).
        # This correctly handles both cases:
        #   - observed == 0, expected == 0: contribution = 0/0 = NaN (excluded)
        #   - observed > 0, expected == 0: contribution = observed/sqrt(0) -> Inf (excluded)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_std = np.where(expected > 0, (Y - expected) / np.sqrt(expected), np.nan)

        # Center the environmental matrix
        X_centered = X - X.mean(axis=0)

        # Compute weights matrix for samples
        w = row_totals.flatten() / grand_total

        # Weighted centering for Y
        Y_weighted = Y_std * w[:, np.newaxis]

        # Compute projection matrix Q = X(X'X)^-1 X' (weighted)
        # Use ridge-regularized lstsq for numerical stability when X'X is
        # near-singular (collinear environmental variables).
        XtX = X_centered.T @ X_centered
        XtX_inv = self._solve_XtX(XtX, method="cca")

        Q = X_centered @ XtX_inv @ X_centered.T

        # Compute cross-covariance with weights
        M = Y_weighted.T @ Q @ Y_weighted

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort and select top components
        sorted_indices = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Ensure positive eigenvalues. Warn when any clipping happens so
        # silent numerical issues (e.g. collinear env variables) are
        # surfaced to the user instead of hidden behind a 1e-10 floor.
        clipped = eigenvalues < 1e-10
        if np.any(clipped):
            n_clipped = int(np.sum(clipped))
            self._logger.warning(
                f"CCA: {n_clipped} eigenvalue(s) clipped to 1e-10 — "
                "near-singular environmental matrix; check for collinear variables."
            )
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Compute scores
        site_scores = Y_weighted @ eigenvectors
        species_scores = eigenvectors * np.sqrt(eigenvalues)

        # Biplot scores
        biplot_scores = X_centered.T @ site_scores
        scale_factor = np.sqrt(np.sum(biplot_scores**2, axis=0))
        scale_factor[scale_factor == 0] = 1
        biplot_scores = biplot_scores / scale_factor * np.sqrt(eigenvalues)

        # Total inertia (chi-square distance).
        # Use nansum to handle NaN values where expected == 0.
        total_inertia = np.nansum(Y_std**2)

        # Proportions
        proportion_explained = (eigenvalues / total_inertia) * 100 if total_inertia > 0 else np.zeros_like(eigenvalues)
        cumulative_proportion = np.cumsum(proportion_explained)

        # Default names
        if species_names is None:
            species_names = [f"Species_{i + 1}" for i in range(n_species)]
        if env_names is None:
            env_names = [f"Env_{i + 1}" for i in range(n_env)]

        constrained_variance = np.sum(proportion_explained)

        return CCAResult(
            site_scores=site_scores,
            species_scores=species_scores,
            biplot_scores=biplot_scores,
            eigenvalues=eigenvalues,
            proportion_explained=proportion_explained,
            cumulative_proportion=cumulative_proportion,
            method="cca",
            n_components=n_components,
            n_samples=n_samples,
            n_species=n_species,
            n_env=n_env,
            species_names=species_names,
            env_names=env_names,
            inertia=total_inertia,
            constrained_variance=constrained_variance,
        )

    @property
    def last_result(self) -> CCAResult | None:
        """Get the last CCA/RDA result."""
        with self._lock:
            return self._last_result
