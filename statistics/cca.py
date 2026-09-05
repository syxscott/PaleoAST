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

CCA (Canonical Correspondence Analysis, ter Braak 1986):
    Chi-square standardization of the relative abundance matrix P = Y/N:
        S = (P - r c^T) / sqrt(r c^T),  r, c = row/column fractions of P
    The constraints enter through a row-weighted regression of S on the
    centered environmental matrix (hat matrix H = Xt (Xt' Dr Xt)^-1 Xt' Dr),
    and the constrained eigenvalues are those of
        D_c^{-1/2} S_hat^T D_r S_hat D_c^{-1/2}
    (equivalently, squared singular values of Dr^{1/2} S_hat D_c^{-1/2}).
    Total inertia is the chi-square inertia sum_ij (p_ij - r_i c_j)^2/(r_i c_j).

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
            import warnings as _warnings
            _warnings.warn(
                f"{method.upper()}: X'X condition number = {cond:.2e} > {cond_threshold:.0e}. "
                f"Environmental matrix is ill-conditioned (collinear variables?). "
                f"Applying ridge regularization (lambda={ridge_lambda}).",
                stacklevel=2,
            )
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
            import warnings as _warnings
            _warnings.warn(
                f"CCA: {n_clipped} eigenvalue(s) clipped to 1e-10 — near-singular environmental matrix; check for collinear variables.",
                stacklevel=2,
            )
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
        Perform Canonical Correspondence Analysis (CCA) after ter Braak (1986).

        Mathematical Steps (ter Braak 1986; Legendre & Legendre 2012, ch. 11):

            Let P = Y / grand_total be the relative-abundance matrix
            (n_samples x n_species), r = P.sum(axis=1) the row fractions and
            c = P.sum(axis=0) the column fractions (both sum to 1). The
            expected abundance under row/column independence is the outer
            product E = r c^T.

            1. Standardized residuals (chi-square standardization):

                   S = (P - r c^T) / sqrt(r c^T)      (element-wise)

               Total inertia is the chi-square inertia of the table:

                   TI = sum_ij S_ij^2 = sum_ij (P_ij - r_i c_j)^2 / (r_i c_j)

            2. Row-weighted regression of S on the environmental matrix.
               X is centered with the row weights (X_tilde = X - r^T X) and
               the hat matrix of the weighted least squares fit is

                   H = X_tilde (X_tilde' D_r X_tilde)^{-1} X_tilde' D_r,
                   D_r = diag(r)

               giving the fitted residuals S_hat = H S.

            3. Constrained eigenproblem: the constrained (canonical)
               eigenvalues lambda_k are the eigenvalues of the symmetric
               matrix

                   B = D_c^{-1/2} S_hat' D_r S_hat D_c^{-1/2},
                   D_c = diag(c)

               computed here (with identical results and better numerical
               stability) as the squared singular values of
               D_r^{1/2} S_hat D_c^{-1/2}.

            4. Scores. Scaling convention (ter Braak's weighted-averaging
               scores, mutually consistent):

                   species_scores[:, k] = v_k / sqrt(c)
                       (v_k = right singular vector;
                        sum_j c_j species_jk^2 = 1)
                   site_scores[:, k] = sqrt(lambda_k) * u_k / sqrt(r)
                       (u_k = left singular vector;
                        sum_i r_i site_ik^2 = lambda_k, i.e.
                        site_scores = S_hat @ species_scores: the site score
                        is the abundance-weighted average of the species
                        scores with weights P / r)
                   biplot_scores: row-weighted covariances of X_tilde with
                       the site scores, normalized to unit length per axis
                       and scaled by sqrt(lambda_k) (biplot arrow scaling).

            5. proportion_explained[k] = 100 * lambda_k / TI, and
               constrained_variance = sum_k proportion_explained[k].

        Rows (samples) or columns (species) of Y with zero totals have zero
        chi-square weight and carry no information: they are dropped from the
        computation with a warning (they are never replaced by pseudo-totals
        such as 1). Their entries in site_scores / species_scores are
        reported as NaN so the output arrays keep the shape of the input.

        References:
            - ter Braak, C.J.F. (1986) Canonical correspondence analysis: a
              new eigenvector technique for multivariate direct gradient
              analysis. Ecology 67:1167-1176.
            - Legendre & Legendre (2012) Numerical Ecology, 3rd ed., Elsevier.
        """
        n_samples_orig, n_species_orig = Y.shape
        n_env = X.shape[1]

        # Ensure non-negative data for CCA
        if np.any(Y < 0):
            raise ComputationError("CCA requires non-negative abundance data")

        grand_total = float(Y.sum())
        if grand_total <= 0:
            raise ComputationError("CCA requires at least one positive abundance value")

        # ------------------------------------------------------------------
        # Drop zero-total rows/columns before computing (ter Braak 1986).
        # Their chi-square weight r_i = 0 (or c_j = 0) is exactly zero, so
        # they contribute nothing to the inertia; substituting a pseudo-total
        # of 1 (the old behavior) invented samples/species and distorted the
        # expected values E = r c^T of every remaining cell.
        # ------------------------------------------------------------------
        keep_rows = Y.sum(axis=1) > 0
        keep_cols = Y.sum(axis=0) > 0
        n_dropped_rows = int(np.sum(~keep_rows))
        n_dropped_cols = int(np.sum(~keep_cols))
        if n_dropped_rows or n_dropped_cols:
            import warnings as _warnings
            msg = (
                f"CCA: dropped {n_dropped_rows} sample row(s) and "
                f"{n_dropped_cols} species column(s) with zero totals "
                f"(zero chi-square weight); their scores are reported as NaN."
            )
            _warnings.warn(msg, stacklevel=2)
            self._logger.warning(msg)
            Y = Y[keep_rows][:, keep_cols]
            X = X[keep_rows, :]
            if species_names is not None:
                species_names = [nm for nm, keep in zip(species_names, keep_cols) if keep]
            grand_total = float(Y.sum())

        n_samples, n_species = Y.shape

        # Clamp the number of axes to what is extractable after dropping
        max_components = min(n_samples - 1, n_species, n_env)
        n_components = min(n_components, max_components)
        if n_components < 1:
            raise MatrixDimensionError(
                "Cannot perform CCA: insufficient dimensions after dropping zero-total rows/columns"
            )

        # ------------------------------------------------------------------
        # Step 1: standardized residuals and total (chi-square) inertia.
        # r and c sum to 1, so expected = r c^T > 0 element-wise and S is
        # finite everywhere (no NaN masking needed).
        # ------------------------------------------------------------------
        P = Y / grand_total
        r = P.sum(axis=1)  # (n,) row fractions, sums to 1
        c = P.sum(axis=0)  # (m,) column fractions, sums to 1
        expected = np.outer(r, c)
        S = (P - expected) / np.sqrt(expected)

        # Chi-square inertia: sum (p - r c)^2 / (r c) = sum S^2
        total_inertia = float(np.sum(S**2))

        # ------------------------------------------------------------------
        # Step 2: row-weighted regression of S on the centered environment.
        # X_tilde = X - r^T X is the D_r-weighted centering (sum r = 1).
        # ------------------------------------------------------------------
        X_tilde = X - r @ X
        XtX = X_tilde.T @ (r[:, np.newaxis] * X_tilde)  # X' D_r X
        XtX_inv = self._solve_XtX(XtX, method="cca")
        # S_hat = X_tilde (X' D_r X)^{-1} X_tilde' D_r S
        S_hat = X_tilde @ (XtX_inv @ (X_tilde.T @ (r[:, np.newaxis] * S)))

        # ------------------------------------------------------------------
        # Step 3: constrained eigenproblem.
        # A = D_r^{1/2} S_hat D_c^{-1/2}; its squared singular values are the
        # eigenvalues of D_c^{-1/2} S_hat' D_r S_hat D_c^{-1/2} (ter Braak 1986).
        # ------------------------------------------------------------------
        A = np.sqrt(r)[:, np.newaxis] * S_hat / np.sqrt(c)[np.newaxis, :]
        U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
        all_lambdas = singular_values**2
        order = np.argsort(all_lambdas)[::-1][:n_components]
        eigenvalues = all_lambdas[order]

        # Ensure non-negative eigenvalues. Warn when any clipping happens so
        # silent numerical issues (e.g. collinear env variables) are
        # surfaced to the user instead of hidden behind a 1e-10 floor.
        clipped = eigenvalues < 1e-10
        if np.any(clipped):
            n_clipped = int(np.sum(clipped))
            import warnings as _warnings
            _warnings.warn(
                f"CCA: {n_clipped} eigenvalue(s) clipped to 1e-10 — "
                "near-singular environmental matrix; check for collinear variables.",
                stacklevel=2,
            )
            self._logger.warning(
                f"CCA: {n_clipped} eigenvalue(s) clipped to 1e-10 — "
                "near-singular environmental matrix; check for collinear variables."
            )
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # ------------------------------------------------------------------
        # Step 4: scores (documented scaling convention, see docstring).
        # V = right singular vectors, one column per retained axis.
        # ------------------------------------------------------------------
        V = Vt[order, :].T  # (n_species, n_components)
        species_scores = V / np.sqrt(c)[:, np.newaxis]
        # site scores = sqrt(lambda) * u / sqrt(r)  (== S_hat @ species_scores)
        site_scores = np.sqrt(eigenvalues)[np.newaxis, :] * (U[:, order] / np.sqrt(r)[:, np.newaxis])

        # Biplot scores: row-weighted covariance of X_tilde with site scores,
        # unit-normalized per axis and scaled by sqrt(lambda).
        biplot_scores = X_tilde.T @ (r[:, np.newaxis] * site_scores)
        scale_factor = np.sqrt(np.sum(biplot_scores**2, axis=0))
        scale_factor[scale_factor == 0] = 1
        biplot_scores = biplot_scores / scale_factor * np.sqrt(eigenvalues)

        # ------------------------------------------------------------------
        # Step 5: proportions of the chi-square inertia explained per axis.
        # ------------------------------------------------------------------
        proportion_explained = (eigenvalues / total_inertia) * 100 if total_inertia > 0 else np.zeros_like(eigenvalues)
        cumulative_proportion = np.cumsum(proportion_explained)

        # Default names (full input dimensions)
        if species_names is None:
            species_names = [f"Species_{i + 1}" for i in range(n_species_orig)]
        if env_names is None:
            env_names = [f"Env_{i + 1}" for i in range(n_env)]

        # Report scores at the shape of the input data; dropped zero-total
        # rows/columns get NaN (no meaningful score exists for them).
        site_scores_full = np.full((n_samples_orig, n_components), np.nan)
        site_scores_full[keep_rows, :] = site_scores
        species_scores_full = np.full((n_species_orig, n_components), np.nan)
        species_scores_full[keep_cols, :] = species_scores

        constrained_variance = np.sum(proportion_explained)

        return CCAResult(
            site_scores=site_scores_full,
            species_scores=species_scores_full,
            biplot_scores=biplot_scores,
            eigenvalues=eigenvalues,
            proportion_explained=proportion_explained,
            cumulative_proportion=cumulative_proportion,
            method="cca",
            n_components=n_components,
            n_samples=n_samples_orig,
            n_species=n_species_orig,
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
