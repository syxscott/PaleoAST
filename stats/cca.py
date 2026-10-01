# =============================================================================
# FILE: stats/cca.py
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
        f_statistic: Overall F-statistic (pseudo-F for CCA, classical F for
            RDA) for the full constrained model
        p_value: Permutation-based p-value for the overall model
        f_per_axis: Per-axis F-statistics (length n_components)
        p_per_axis: Per-axis permutation-based p-values (length n_components)
        wilks_lambda: Wilks' lambda = product(1 - r_k^2) for the constrained
            axes (close to 0 for strong relationships, 1 for none)
        n_permutations: Number of permutations used for the p-values
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
    f_statistic: float = 0.0
    p_value: float = 1.0
    f_per_axis: npt.NDArray | None = None
    p_per_axis: npt.NDArray | None = None
    wilks_lambda: float = 1.0
    n_permutations: int = 0

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
        lines.append(
            f"{_('Total constrained variance: {0}%').format(f'{self.constrained_variance:.2f}')}"
        )
        lines.append(f"{_('Inertia (total): {0:.4f}').format(self.inertia)}")
        lines.append("")
        lines.append(f"{_('Overall F: {0:.4f}').format(self.f_statistic)}")
        lines.append(f"{_('Overall p-value: {0:.4f}').format(self.p_value)}")
        lines.append(f"{_('Wilks lambda: {0:.4f}').format(self.wilks_lambda)}")
        if self.f_per_axis is not None and len(self.f_per_axis) > 0:
            lines.append("")
            lines.append(f"{_('Per-axis F / p:')}")
            for i in range(self.n_components):
                p = self.p_per_axis[i] if self.p_per_axis is not None else 1.0
                lines.append(f"  AX{i + 1}: F = {self.f_per_axis[i]:.4f}, p = {p:.4f}")

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
        n_permutations: int = 999,
        random_seed: int | None = None,
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
            n_permutations: Number of permutations for the significance
                test. The default (999) matches ``vegan::anova.cca``. Set
                to 0 to skip the permutation test (still returns F/Wilks).
            random_seed: Optional seed for the permutation RNG so the
                p-value is reproducible. When supplied, a local
                ``np.random.default_rng(seed)`` is used so the global
                numpy state is unaffected.

        Returns:
            CCAResult: CCA/RDA analysis results, including F-statistic,
                p-value, per-axis F and p, and Wilks lambda.

        Notes:
            Permutations shuffle **Y rows** (the response) -- not X. CCA
            permutes the abundance table, not the environmental table; this
            is the convention of ``vegan::anova.cca(..., by='marginal')``
            and Legendre & Legendre (2012, §11.4).
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

            # Permutation-based significance test (shuffles Y rows).
            f_overall, f_per_axis, wilks_lambda, p_overall, p_per_axis = self._permutation_test(
                Y_arr, X_arr, n_components, method,
                n_permutations=n_permutations, random_seed=random_seed,
            )
            result.f_statistic = float(f_overall)
            result.f_per_axis = np.asarray(f_per_axis, dtype=float)
            result.p_per_axis = np.asarray(p_per_axis, dtype=float)
            result.p_value = float(p_overall)
            result.wilks_lambda = float(wilks_lambda)
            result.n_permutations = int(n_permutations) if n_permutations > 0 else 0

            self._last_result = result
            self._logger.info(
                f"CCA/RDA completed: method={result.method}, "
                f"constrained_variance={result.constrained_variance:.2f}%, "
                f"F={result.f_statistic:.4f}, p={result.p_value:.4f}"
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
                    f"{method.upper()}: np.linalg.inv(XtX) failed. Falling back to ridge-regularized lstsq."
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
                species_names = [nm for nm, keep in zip(species_names, keep_cols, strict=False) if keep]
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

    def _permutation_test(
        self,
        Y: npt.NDArray,
        X: npt.NDArray,
        n_components: int,
        method: str,
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> tuple[float, npt.NDArray, float, float, npt.NDArray]:
        """
        Permutation-based significance test for CCA / RDA.

        Implementation notes
        --------------------
        - Permutations shuffle the **Y rows** (the response), not the X
          rows. This is the convention of ``vegan::anova.cca(..., by='marginal')``
          and Legendre & Legendre (2012, §11.4). Permuting X instead would
          test a different null (a null on the environment, not on the
          species/response) and is a common bug in ecological software.
        - Test statistic = trace of the constrained eigenvalues (sum of the
          top-q constrained eigenvalues). For RDA this is the constrained
          sum of squares; for CCA the constrained chi-square inertia.
        - Overall F is the classical ratio (constrained SS / q) / (residual
          SS / (n - 1 - q)) for RDA, and an analogous trace-based F for CCA.
        - Wilks lambda = prod(1 - lambda_k / inertia) where inertia is the
          total (chi-square or SS) inertia and lambda_k the constrained
          eigenvalues. lambda is in (0, 1]; small lambda means strong
          relationship.
        - Uses a local ``np.random.default_rng`` when ``random_seed`` is
          supplied, otherwise warns and falls back to the global state
          (with no reproducibility).

        Returns
        -------
        (f_overall, f_per_axis, wilks_lambda, p_overall, p_per_axis)

        The two per-axis arrays are as long as the number of axes actually
        computed, which can be fewer than the requested ``n_components`` when
        zero-total rows/columns were dropped; that is the same count
        ``CCAResult.n_components`` reports, so its ``summary()`` stays in
        range. (The degenerate early return above keeps the requested length.)
        """
        n_samples = Y.shape[0]
        rng: np.random.Generator | np.random.RandomState
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            import warnings as _warnings

            _warnings.warn(
                "CCA: no ``random_seed`` supplied; the permutation p-values "
                "use the global ``np.random`` state and are not "
                "reproducible across runs. Pass ``random_seed=`` to make "
                "the result deterministic.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._logger.warning(
                "CCA: no random_seed supplied; p-values use global np.random "
                "state and are not reproducible."
            )
            rng = np.random

        # Compute the observed F (overall + per-axis), the constrained SS,
        # the residual SS, and Wilks lambda, by re-using the same eigen
        # decomposition that _analyze_rda / _analyze_cca just produced.
        # To avoid recomputing scores and proportions for every perm,
        # we extract just enough (eigenvalues, total inertia) here.
        observed_eigenvalues, total_inertia, total_SS, n_q, n_eff = self._constrained_eigenvalues(
            Y, X, n_components, method
        )
        if len(observed_eigenvalues) == 0 or total_inertia <= 0:
            return (
                0.0,
                np.zeros(n_components),
                1.0,
                1.0,
                np.ones(n_components),
            )

        # Overall F and per-axis F for the observed data
        f_overall, f_per_axis, wilks_lambda = self._F_from_eigenvalues(
            observed_eigenvalues, total_inertia, total_SS, n_eff, n_q
        )

        # Permutation null distribution.
        #
        # ``n_q`` -- NOT the requested ``n_components`` -- is the number of
        # axes that were actually computed: ``_constrained_eigenvalues``
        # drops zero-total rows/columns first and clamps the count to
        # min(n_rows-1, n_cols, n_env) *after* that drop. Allocating and
        # looping over the requested count therefore broadcast a shorter
        # ``f_pa`` into a wider row and raised
        # "could not broadcast input array from shape (7,) into shape (8,)"
        # for any table with an all-zero species column (a perfectly normal
        # abundance table). The per-axis arrays now carry the effective count,
        # which is what ``CCAResult.n_components`` already reports because
        # ``_analyze_cca`` applies the same clamp.
        n_axes = int(n_q)
        f_overall_perm = np.zeros(n_permutations)
        f_per_axis_perm = np.zeros((n_permutations, n_axes))
        for i in range(n_permutations):
            perm_idx = rng.permutation(n_samples)
            Y_perm = Y[perm_idx]
            perm_eigs, _, _, _, _ = self._constrained_eigenvalues(
                Y_perm, X, n_components, method
            )
            if len(perm_eigs) == 0:
                f_overall_perm[i] = 0.0
                f_per_axis_perm[i, :] = 0.0
                continue
            # `total_inertia` is the *observed* one on purpose, and it is worth
            # saying why because it reads like an oversight. For CCA the total
            # inertia is the chi-square divergence
            #   I = sum(P^2 / E) - 1,  P = Y / T,  E = outer(r, c)
            # and sum(P^2 / E) = sum_i (1/r_i) * sum_j Y_ij^2 / c_j. Permuting
            # Y's rows moves each (r_i, row_i) pair together, so that per-row
            # factor travels with its row and I is unchanged -- measured at
            # ~1e-16 relative difference over 30 permutations on tables that
            # are plain, sparse-with-zeros, and dominated by a single row.
            #
            # The constrained eigenvalues *do* change, which is the whole point
            # of the permutation; the denominator they are measured against
            # does not, so recomputing it per permutation would be wasted work
            # and would invite a subtly different bug.
            f_p, f_pa, _ = self._F_from_eigenvalues(
                perm_eigs, total_inertia, total_SS, n_eff, n_q
            )
            f_overall_perm[i] = f_p
            # A permutation can lose a further axis (its zero-total structure
            # may differ from the observed table's); fit instead of assigning
            # so a shorter vector cannot break the broadcast. Missing axes
            # count as 0.0, the same convention the ``len(perm_eigs) == 0``
            # branch above uses for a degenerate permutation.
            f_pa_aligned = np.zeros(n_axes)
            n_fill = min(len(f_pa), n_axes)
            f_pa_aligned[:n_fill] = f_pa[:n_fill]
            f_per_axis_perm[i, :] = f_pa_aligned

        # p-values (add-one correction: (1 + #{perm >= obs}) / (1 + n_perm))
        n_ge_overall = int(np.sum(f_overall_perm >= f_overall))
        p_overall = float((1 + n_ge_overall) / (1 + n_permutations))
        p_per_axis = np.array(
            [(1 + int(np.sum(f_per_axis_perm[:, k] >= f_per_axis[k]))) / (1 + n_permutations)
             for k in range(n_axes)],
            dtype=float,
        )

        return float(f_overall), f_per_axis, float(wilks_lambda), float(p_overall), p_per_axis

    def _constrained_eigenvalues(
        self,
        Y: npt.NDArray,
        X: npt.NDArray,
        n_components: int,
        method: str,
    ) -> tuple[npt.NDArray, float, float, int, int]:
        """
        Compute the constrained eigenvalues and the relevant inertias/SS
        for one (Y, X) pair (no score recomputation, just enough for the
        permutation test).

        Returns
        -------
        eigenvalues : the top-q constrained eigenvalues (length n_components)
        total_inertia : for CCA, the chi-square inertia; for RDA, the
            sum of squares of Y (same as ``total_SS``)
        total_SS : sum-of-squares of Y (RDA only; for CCA returned as
            ``np.nan`` and not used)
        n_q : number of constrained axes actually computed (may be less
            than the requested n_components when X has very few columns)
        n_eff : number of sample rows the quantities above were computed
            from. This is smaller than the input row count when CCA dropped
            zero-total samples, and it is what the residual degrees of
            freedom must use.
        """
        if method == "rda":
            Y_centered = Y - Y.mean(axis=0)
            X_centered = X - X.mean(axis=0)
            XtX = X_centered.T @ X_centered
            try:
                XtX_inv = self._solve_XtX(XtX, method="rda")
            except Exception:
                return np.array([]), 0.0, 0.0, 0, int(Y.shape[0])
            Q = X_centered @ XtX_inv @ X_centered.T
            M = Y_centered.T @ Q @ Y_centered
            eigenvalues, _ = np.linalg.eigh(M)
            # Sort descending
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[:n_components], 1e-10)
            total_SS = float(np.sum(Y_centered**2))
            return eigenvalues, total_SS, total_SS, len(eigenvalues), int(Y.shape[0])

        # CCA
        if np.any(Y < 0):
            # Non-negative data is required for CCA. If a permutation ever
            # produced negatives (impossible: we only permute rows), fall
            # back to zeros to avoid raising mid-test.
            return np.array([]), 0.0, 0.0, 0, int(Y.shape[0])
        # Drop zero-total rows and columns BEFORE doing anything --
        # otherwise the chi-square standardization divides by 0 and
        # the SVD explodes with NaN/Inf. Same convention as
        # ``_analyze_cca``.
        keep_rows = Y.sum(axis=1) > 0
        keep_cols = Y.sum(axis=0) > 0
        if not np.all(keep_rows) or not np.all(keep_cols):
            Y = Y[keep_rows][:, keep_cols]
            X = X[keep_rows, :]
        if Y.shape[0] < 2 or Y.shape[1] == 0 or X.shape[0] < 2:
            return np.array([]), 0.0, 0.0, 0, int(Y.shape[0])
        # Clamp n_components to what is actually available
        n_components_eff = min(n_components, min(Y.shape[0] - 1, Y.shape[1], X.shape[1]))
        if n_components_eff < 1:
            return np.array([]), 0.0, 0.0, 0, int(Y.shape[0])

        grand_total = float(Y.sum())
        if grand_total <= 0:
            return np.array([]), 0.0, 0.0, 0, int(Y.shape[0])
        P = Y / grand_total
        r = P.sum(axis=1)
        c = P.sum(axis=0)
        expected = np.outer(r, c)
        # Avoid /0 where r_i = 0 or c_j = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (P - expected) / np.sqrt(expected)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        total_inertia = float(np.sum(S**2))
        if total_inertia <= 0:
            return np.array([]), total_inertia, np.nan, 0, int(Y.shape[0])

        X_tilde = X - r @ X
        XtX = X_tilde.T @ (r[:, np.newaxis] * X_tilde)
        try:
            XtX_inv = self._solve_XtX(XtX, method="cca")
        except Exception:
            return np.array([]), total_inertia, np.nan, 0, int(Y.shape[0])
        S_hat = X_tilde @ (XtX_inv @ (X_tilde.T @ (r[:, np.newaxis] * S)))

        A = np.sqrt(r)[:, np.newaxis] * S_hat / np.sqrt(c)[np.newaxis, :]
        try:
            _U, singular_values, _Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            # Degenerate permuted table; treat as zero contribution so the
            # test still has a well-defined distribution.
            return np.array([]), total_inertia, np.nan, 0, int(Y.shape[0])
        all_lambdas = singular_values**2
        eigenvalues = np.sort(all_lambdas)[::-1][:n_components_eff]
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return eigenvalues, total_inertia, np.nan, len(eigenvalues), int(Y.shape[0])

    def _F_from_eigenvalues(
        self,
        eigenvalues: npt.NDArray,
        total_inertia: float,
        total_SS: float,
        n_samples: int,
        n_components_actual: int,
    ) -> tuple[float, npt.NDArray, float]:
        """
        Compute (overall F, per-axis F, Wilks lambda) from constrained
        eigenvalues and total inertia/SS.

        RDA: F_overall = (SS_constrained / q) / (SS_residual / (n - 1 - q))
        CCA: F_overall = (inertia_constrained / q) / (inertia_residual / (n - 1 - q))

        Wilks lambda = prod(1 - lambda_k / total_inertia).
        """
        n_components_actual = max(int(n_components_actual), 1)
        constrained = float(np.sum(eigenvalues))
        if total_inertia <= 0:
            return 0.0, np.zeros(n_components_actual), 1.0
        residual = max(float(total_inertia) - constrained, 0.0)
        q = n_components_actual
        df_residual = max(n_samples - 1 - q, 1)
        f_overall = (constrained / q) / (residual / df_residual) if residual > 0 else 0.0

        # Per-axis F (marginal test: SS_k vs MS_residual)
        f_per_axis = eigenvalues * df_residual / residual if residual > 0 else np.zeros(q)

        # Wilks lambda
        # r_k^2 = lambda_k / total_inertia; truncated to [0, 1]
        ratios = np.clip(eigenvalues / total_inertia, 0.0, 1.0)
        wilks_lambda = float(np.prod(1.0 - ratios))

        return float(f_overall), f_per_axis, wilks_lambda

    @property
    def last_result(self) -> CCAResult | None:
        """Get the last CCA/RDA result."""
        with self._lock:
            return self._last_result
