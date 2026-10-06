# =============================================================================
# FILE: stats/lda.py
# =============================================================================
"""
Linear Discriminant Analysis / Canonical Variate Analysis (LDA/CVA)

Uses scikit-learn's LinearDiscriminantAnalysis as the computation backend.

Mathematical Foundation:

LDA finds projection vectors that maximize the ratio of between-class
scatter to within-class scatter:

    maximize  w^T S_B w / w^T S_W w

where:
    S_B = Σ_k n_k (μ_k - μ)(μ_k - μ)^T   (between-class scatter)
    S_W = Σ_k Σ_{i∈C_k} (x_i - μ_k)(x_i - μ_k)^T   (within-class scatter)

The solution is obtained by solving the generalized eigenvalue problem:
    S_W^{-1} S_B w = λ w

Reference: Fisher (1936) "The use of multiple measurements in
taxonomic problems." Annals of Eugenics, 7, 179-188.

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
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


def _compute_canonical_eigenvalues(
    data: npt.NDArray, groups: npt.NDArray, n_components: int
) -> npt.NDArray:
    """
    Compute the canonical roots of the generalized eigenproblem
        S_W^{-1} S_B w = lambda w
    (Fisher 1936), where S_B is the between-class scatter and S_W the
    within-class scatter.

    These are the numbers R's ``lda::lda`` reports as the canonical
    eigenvalues; they are unbounded (larger == better separation), unlike
    the [0, 1] explained-variance proportions sklearn exposes.

    Returns an empty array when S_W is singular (so the canonical
    problem is undefined) -- callers should treat that as "not
    available" rather than as zero.
    """
    _n_samples, n_vars = data.shape
    overall_mean = data.mean(axis=0)
    unique_groups = np.unique(groups)

    # Between-class scatter: S_B = Σ_k n_k (mu_k - mu)(mu_k - mu)'
    S_B = np.zeros((n_vars, n_vars))
    for g in unique_groups:
        in_g = groups == g
        n_k = int(np.sum(in_g))
        if n_k == 0:
            continue
        mu_k = data[in_g].mean(axis=0)
        d = (mu_k - overall_mean).reshape(-1, 1)
        S_B += n_k * (d @ d.T)

    # Within-class scatter: S_W = Σ_k Σ_{i in C_k} (x_i - mu_k)(x_i - mu_k)'
    S_W = np.zeros((n_vars, n_vars))
    for g in unique_groups:
        in_g = groups == g
        if int(np.sum(in_g)) < 2:
            continue
        mu_k = data[in_g].mean(axis=0)
        diff = data[in_g] - mu_k
        S_W += diff.T @ diff

    # S_W may be singular when n_classes > 1 but the data lies in a
    # lower-dimensional subspace (e.g. only 2 unique samples per class).
    # Fall back to a small ridge to keep the eigenproblem well-posed.
    cond = np.linalg.cond(S_W)
    if not np.isfinite(cond) or cond > 1e10:
        S_W = S_W + 1e-8 * np.eye(n_vars)

    try:
        # Solve S_W^{-1} S_B w = lambda w via the symmetric generalized
        # eigenproblem formulation: cholesky of S_W, reduce to the
        # symmetric S_W^{-1/2} S_B S_W^{-1/2} problem.
        L = np.linalg.cholesky(S_W)
        # L L' = S_W  =>  L^{-1} S_B L^{-T} is the symmetric reduction
        Linv = np.linalg.inv(L)
        M = Linv @ S_B @ Linv.T
        M = 0.5 * (M + M.T)  # enforce symmetry
        roots = np.linalg.eigvalsh(M)
        # Sort descending and take the top n_components (canonical roots
        # can be negative when S_B is indefinite, e.g. when there are
        # fewer samples than variables -- clip to 0 for downstream use).
        roots = np.sort(roots)[::-1][:n_components]
        roots = np.maximum(roots, 0.0)
        return roots
    except np.linalg.LinAlgError:
        # S_W truly singular; canonical roots are not defined.
        return np.array([], dtype=float)


@dataclass
class LDAResult:
    """
    Container for LDA/CVA results.

    Attributes:
        scores: LD score matrix (n_samples x n_components)
        loadings: Discriminant coefficient matrix (n_variables x n_components)
        explained_variance_ratio: Proportion of between-class variance
            explained (alias for ``eigenvalue_proportions``; kept for
            backward compatibility).
        eigenvalue_proportions: Proportion of between-class variance
            explained by each LD axis. Lies in [0, 1]. Sum across axes
            is at most 1.
        eigenvalues: **DEPRECATED NAME** -- kept for backward
            compatibility, but now holds the same numbers as
            ``eigenvalue_proportions`` (i.e. the [0, 1] explained-
            variance proportion), NOT the canonical roots of
            S_W^{-1} S_B. New code should read ``eigenvalue_proportions``
            or, for the actual canonical roots, ``eigenvalues_canonical``.
        eigenvalues_canonical: The true canonical roots of the
            generalized eigenvalue problem S_W^{-1} S_B w = lambda w
            (Fisher 1936). These are the numbers R's ``lda::lda``
            reports as ``$scaling`` eigenvalues -- unbounded, larger
            means more separation. Empty array if S_W was singular.
        confusion_matrix: Classification confusion matrix
        accuracy: Cross-validated classification accuracy
        n_classes: Number of classes
        n_samples: Number of samples
        class_labels: Unique class labels
        means: Class means in LD space
        coef: Raw LDA coefficients
        groups: Group assignments for each sample
    """

    scores: npt.NDArray
    loadings: npt.NDArray
    explained_variance_ratio: npt.NDArray
    eigenvalue_proportions: npt.NDArray
    eigenvalues: npt.NDArray  # alias for eigenvalue_proportions (see note)
    eigenvalues_canonical: npt.NDArray
    confusion_matrix: npt.NDArray
    accuracy: float
    n_classes: int
    n_samples: int
    class_labels: list
    means: npt.NDArray
    coef: npt.NDArray
    groups: npt.NDArray

    def summary(self) -> str:
        lines = [
            _("Linear Discriminant Analysis (LDA / CVA)"),
            "=" * 50,
            f"{_('Classes')}: {self.n_classes}, {_('Samples')}: {self.n_samples}",
            f"{_('Cross-validated accuracy')}: {self.accuracy:.2%}",
            "",
            f"{'LD':<6} {'Eigenvalue':>12} {'Var. Explained':>15} {'Cumulative':>12}",
            "-" * 50,
        ]
        cum = 0.0
        for i, (ev, vr) in enumerate(
            zip(self.eigenvalues, self.explained_variance_ratio, strict=False)
        ):
            cum += vr
            lines.append(f"LD{i + 1:<4} {ev:>12.4f} {vr:>14.2%} {cum:>11.2%}")

        lines.append("")
        lines.append(_("Confusion Matrix:"))
        lines.append(str(self.confusion_matrix))
        return "\n".join(lines)


class LDAAnalyzer:
    """
    LDA/CVA analysis engine.

    Wraps scikit-learn's LinearDiscriminantAnalysis with
    paleontological data conventions.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.LDAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: LDAResult | None = None

    def analyze(
        self,
        data: npt.NDArray,
        groups: list[int],
        n_components: int | None = None,
        variable_names: list[str] | None = None,
        cv_folds: int = 5,
    ) -> LDAResult:
        """
        Perform LDA/CVA analysis.

        Parameters:
            data: Data matrix (n_samples x n_variables)
            groups: Group/class label for each sample
            n_components: Number of LD components (default: min(n_classes-1, n_vars))
            variable_names: Names for variables (for loadings)
            cv_folds: Number of cross-validation folds for accuracy

        Returns:
            LDAResult
        """
        with self._lock:
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from sklearn.model_selection import cross_val_predict, cross_val_score
            except ImportError:
                raise ComputationError("scikit-learn is required for LDA. Install with: pip install scikit-learn")

            data = validate_data_array(data, name="data")
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_samples, n_vars = data.shape
            groups = np.array(groups)

            if len(groups) != n_samples:
                raise ComputationError(f"Group length ({len(groups)}) must match n_samples ({n_samples})")

            # Filter out samples with NaN
            valid_mask = ~np.isnan(data).any(axis=1)
            data_clean = data[valid_mask]
            groups_clean = groups[valid_mask]

            # Sentinel values like -1 only make sense for numeric labels.
            # Applying ``groups_clean >= 0`` to string labels raised a raw
            # numpy _UFuncNoLoopError, so the controller's error formatting
            # never saw a ComputationError.
            if np.issubdtype(np.asarray(groups_clean).dtype, np.number):
                grouped_mask = groups_clean >= 0
            else:
                # np.array rather than np.ones/np.full with dtype=bool: numpy's
                # type stubs bind both of those to numeric ScalarT, so a bool
                # mask type-errors on numpy 2.x. np.array has no such bound.
                grouped_mask = np.array(
                    [True] * groups_clean.shape[0], dtype=bool
                )
            data_grouped = data_clean[grouped_mask]
            groups_grouped = groups_clean[grouped_mask]

            # Relabel to 0..k-1. np.bincount() assumes contiguous 0-based
            # integers, so groups=[2,4,5,7] produced
            # [0,0,1,0,1,1,1,1] whose min is 0, which drove the CV-fold count
            # to 0 and made the code report RESUBSTITUTION accuracy that
            # summary() then printed as "cross-validated accuracy".
            unique_classes = sorted(set(groups_grouped))
            n_classes = len(unique_classes)
            if n_classes < 2:
                raise ComputationError("LDA requires at least 2 classes")
            index_of = {c: i for i, c in enumerate(unique_classes)}
            groups_codes = np.array([index_of[g] for g in groups_grouped], dtype=int)

            if n_components is None:
                n_components = min(n_classes - 1, n_vars)

            n_components = min(n_components, n_classes - 1, n_vars)

            self._logger.info(
                f"LDA: {data_grouped.shape[0]} samples, {n_vars} vars, {n_classes} classes, {n_components} components"
            )

            # Fit LDA
            lda = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
            # sklearn's SVD solver reaches for the leading singular value to
            # pick its numerical rank, and it is zero when a class has no
            # within-class spread at all. The rank filter then keeps nothing
            # and it indexes an empty array, surfacing as
            #   IndexError: index 0 is out of bounds for axis 0 with size 0
            # from inside sklearn, with nothing to say what the user did
            # wrong. Constant columns, or classes that are constant within
            # themselves, are a real and understandable thing to hand a
            # discriminant analysis, so they get a message instead.
            spread = data_grouped.astype(float, copy=True)
            spread -= spread.mean(axis=0, keepdims=True)
            within_class_spread = spread.std(axis=0)
            degenerate = np.flatnonzero(within_class_spread <= 0.0)
            if degenerate.size:
                names = [str(v) for v in np.asarray(variable_names)[degenerate]] if variable_names is not None else [
                    f"variable {i}" for i in degenerate
                ]
                raise ComputationError(
                    f"LDA cannot proceed: {', '.join(names[:5])}"
                    f"{' ...' if names and len(names) > 5 else ''} has zero variance across "
                    f"all samples, so the within-class scatter matrix is singular. "
                    f"Remove the constant variable(s) or supply data with spread."
                )
            try:
                scores = lda.fit_transform(data_grouped, groups_grouped)
            except IndexError as exc:  # pragma: no cover - defensive
                raise ComputationError(
                    f"LDA could not fit a discriminant model to this data: {exc}. "
                    f"This usually means a class has no within-class variation."
                ) from exc

            # Loadings (coefficients)
            loadings = lda.scalings_[:, :n_components]

            # Eigenvalues and explained variance.
            # sklearn's LDA only exposes ``explained_variance_ratio_``
            # (the [0, 1] proportion of between-class variance per LD
            # axis). Historically PaleoAST stored this same array under
            # both ``explained_variance_ratio`` and ``eigenvalues`` --
            # but ``eigenvalues`` is the canonical R/Fisher term for the
            # roots of the generalized eigenproblem S_W^{-1} S_B w = lambda w,
            # which has nothing to do with a [0, 1] proportion.
            # We now:
            #   1. Add a new field ``eigenvalue_proportions`` carrying the
            #      sklearn quantity (the [0, 1] explained-variance ratio).
            #   2. Keep ``eigenvalues`` as an alias of the same array for
            #      backward compatibility (and document the change).
            #   3. Compute and expose the actual generalized eigenproblem
            #      roots under ``eigenvalues_canonical``.
            explained_var = lda.explained_variance_ratio_
            eigenvalue_proportions = explained_var.copy()
            # Backward-compatibility alias -- documented as holding the
            # [0, 1] proportions, NOT the canonical roots.
            eigenvalues = eigenvalue_proportions.copy()

            # Compute the true canonical roots lambda_k from the
            # generalized eigenvalue problem S_W^{-1} S_B w = lambda w.
            # We compute S_W and S_B directly from the data; degenerate
            # cases (S_W singular, n_components > min(n_classes-1, n_vars))
            # leave ``eigenvalues_canonical`` as an empty array.
            eigenvalues_canonical = _compute_canonical_eigenvalues(
                data_grouped, groups_codes, n_components
            )

            # Class means in LD space
            class_means = lda.transform(lda.means_)

            # Confusion matrix via cross-validation
            lda_full = LinearDiscriminantAnalysis(solver="svd")
            cv_folds_actual = min(cv_folds, int(np.bincount(groups_codes).min()))
            if cv_folds_actual >= 2:
                try:
                    cv_scores = cross_val_score(lda_full, data_grouped, groups_grouped, cv=cv_folds_actual)
                    accuracy = float(np.mean(cv_scores))
                except Exception:
                    # Fallback: training accuracy
                    lda_full.fit(data_grouped, groups_grouped)
                    accuracy = float(lda_full.score(data_grouped, groups_grouped))
            else:
                lda_full.fit(data_grouped, groups_grouped)
                accuracy = float(lda_full.score(data_grouped, groups_grouped))

            # Confusion matrix via cross-validated predictions
            if cv_folds_actual >= 2:
                try:
                    predictions = cross_val_predict(lda_full, data_grouped, groups_grouped, cv=cv_folds_actual)
                except Exception:
                    lda_full.fit(data_grouped, groups_grouped)
                    predictions = lda_full.predict(data_grouped)
            else:
                lda_full.fit(data_grouped, groups_grouped)
                predictions = lda_full.predict(data_grouped)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            for true, pred in zip(groups_grouped, predictions, strict=False):
                cm[class_to_idx[true], class_to_idx[pred]] += 1

            result = LDAResult(
                scores=scores,
                loadings=loadings,
                explained_variance_ratio=explained_var,
                eigenvalue_proportions=eigenvalue_proportions,
                eigenvalues=eigenvalues,
                eigenvalues_canonical=eigenvalues_canonical,
                confusion_matrix=cm,
                accuracy=accuracy,
                n_classes=n_classes,
                n_samples=data_grouped.shape[0],
                class_labels=unique_classes,
                means=class_means,
                coef=lda.coef_,
                groups=groups_grouped,
            )

            self._last_result = result
            self._logger.info(f"LDA complete: accuracy={accuracy:.2%}, {n_components} components")
            return result

    @property
    def last_result(self) -> LDAResult | None:
        with self._lock:
            return self._last_result
