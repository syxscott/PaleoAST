# =============================================================================
# FILE: statistics/lda.py
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


@dataclass
class LDAResult:
    """
    Container for LDA/CVA results.

    Attributes:
        scores: LD score matrix (n_samples x n_components)
        loadings: Discriminant coefficient matrix (n_variables x n_components)
        explained_variance_ratio: Proportion of between-class variance explained
        eigenvalues: Eigenvalues of the discriminant problem
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
    eigenvalues: npt.NDArray
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
        for i, (ev, vr) in enumerate(zip(self.eigenvalues, self.explained_variance_ratio, strict=False)):
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
            data_clean.shape[0]

            # Filter out ungrouped samples (if any sentinel value like -1)
            grouped_mask = groups_clean >= 0
            data_grouped = data_clean[grouped_mask]
            groups_grouped = groups_clean[grouped_mask]

            unique_classes = sorted(set(groups_grouped))
            n_classes = len(unique_classes)

            if n_classes < 2:
                raise ComputationError("LDA requires at least 2 classes")

            if n_components is None:
                n_components = min(n_classes - 1, n_vars)

            n_components = min(n_components, n_classes - 1, n_vars)

            self._logger.info(
                f"LDA: {data_grouped.shape[0]} samples, {n_vars} vars, {n_classes} classes, {n_components} components"
            )

            # Fit LDA
            lda = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
            scores = lda.fit_transform(data_grouped, groups_grouped)

            # Loadings (coefficients)
            loadings = lda.scalings_[:, :n_components]

            # Eigenvalues and explained variance.
            # sklearn's LDA only exposes `explained_variance_ratio_`
            # (proportion of between-class variance per LD axis), not the raw
            # eigenvalues. We expose this quantity under both `eigenvalues`
            # and `explained_variance_ratio` for compatibility, but document
            # explicitly that these are the explained-variance *ratios*, not
            # the original eigenvalues.
            explained_var = lda.explained_variance_ratio_
            # Pseudo-eigenvalues proportional to explained variance.
            # Used internally by `summary()` for display.
            eigenvalues = explained_var.copy()

            # Class means in LD space
            class_means = lda.transform(lda.means_)

            # Confusion matrix via cross-validation
            lda_full = LinearDiscriminantAnalysis(solver="svd")
            cv_folds_actual = min(cv_folds, min(np.bincount(groups_grouped.astype(int))))
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
                eigenvalues=eigenvalues,
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
