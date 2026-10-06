# =============================================================================
# FILE: stats/detriding.py
# =============================================================================
"""
Detrended correspondence analysis (DCA).

WHAT IT IS FOR
~~~~~~~~~~~~~~~
Correspondence analysis is an unconstrained ordination: it
finds the axes of a contingency table that best separate the samples, and
it has a well known habit of ordering them by a dominant gradient. The
first axis captures the environmental gradient, the second captures the
*variation along* it (a size gradient, say), the third a further
oscillation, and so on. That is useful when a single gradient is the
point and misleading when a fossil assemblage is read for several
independent signals, because those signals end up spread down the axis
list rather than at the top of it.

DCA removes the predictable part of each axis before ranking, so the
axes that survive are the ones a gradient does not already explain. The
ordering then reflects independent structure rather than gradient
magnitude (Legendre & Legendre 2012, §11.5.2; ter Braak 1986).

HOW IT IS COMPUTED
~~~~~~~~~~~~~~~~~~
1. Standardise the contingency table to relative frequencies, then to
   chi-square distances::

       P = F / sum(F)
       r = row masses, c = column masses
       S = (P - r c^T) / sqrt(r c^T)

2. Take the SVD ``S = U diag(sigma) V^T``. The CA eigenvalues are
   ``lambda_k = sigma_k^2 / sum(F)`` and the row principal coordinates are
   ``Y[:, k] = U[:, k] * sigma_k / sqrt(r)``.

3. Detrend. Axis 1 is regressed on the supplied gradient (or left alone
   when none is given); each later axis is regressed on the earlier
   *detrended* ones. Only the part of an axis that the regression does
   not explain survives, so the multiplier is the residual variance

       retained_k = 1 - R^2_k

   and the detrended eigenvalue is ``lambda_k * retained_k``, with the
   detrended score scaled by ``sqrt(retained_k)``.

   This is a deflation, not the regression's variance inflation factor
   ``1 / (1 - R^2)``. An axis completely explained by the gradient has
   ``R^2 = 1``, no independent variance left, and a detrended eigenvalue
   of zero -- which is the whole point of the method. Such an axis is
   kept in the output (dropping it would renumber every later axis) with
   a zero eigenvalue, and logged.

4. Re-rank by the detrended eigenvalues. Axis *a* stays axis *a* in the
   output positions; only their *order of importance* changes, which is
   what a caller comparing "which axis matters" needs.

The CA step is implemented here rather than imported because
ecology/paleoenv.py's CA exists to reconstruct a paleoenvironment, and
coupling an ordination's detrending to that module's internal helpers
would tie both to the same private code.

References
----------
Legendre, P. & Legendre, L. 2012. Numerical Ecology, 3rd ed. Elsevier, §11.
ter Braak, C. J. F. 1986. Canonical correspondence analysis: a
    discussion of its properties and use. Vegetatio 66: 121-140.
Hill, M. O. 1974. Correspondence analysis: an aid in interpreting
    multivariate data. Applied Statistics 23: 215-231.
Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError

logger = logging.getLogger(__name__)

# Smallest retained variance worth reporting. Below this an axis has no
_RETAINED_CEILING = 1e-12


@dataclass
class CorrespondenceResult:
    """Raw correspondence analysis of a contingency table.

    Attributes:
        row_scores: Row principal coordinates, (n_rows x n_components).
        column_scores: Column principal coordinates.
        eigenvalues: CA eigenvalues in descending order.
        inertia: Total inertia, the sum of the eigenvalues.
        n_rows / n_columns: Table dimensions.
        explained_ratio: Each eigenvalue as a fraction of total inertia.
    """

    row_scores: npt.NDArray
    column_scores: npt.NDArray
    eigenvalues: npt.NDArray
    inertia: float
    n_rows: int
    n_columns: int

    @property
    def explained_ratio(self) -> npt.NDArray:
        """Share of total inertia carried by each axis."""
        if self.inertia <= 0:
            return np.zeros_like(self.eigenvalues)
        return self.eigenvalues / self.inertia

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Correspondence Analysis"),
            "=" * 50,
            f"{_('Rows')}: {self.n_rows}, {_('columns')}: {self.n_columns}",
            f"{_('Total inertia')}: {self.inertia:.6f}",
        ]
        for i, (ev, ratio) in enumerate(
            zip(self.eigenvalues, self.explained_ratio, strict=True), start=1
        ):
            lines.append(f"  {_('Axis {0}').format(i)}: {ev:.6f} ({ratio * 100:.2f}%)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "eigenvalues": [float(e) for e in self.eigenvalues],
            "inertia": float(self.inertia),
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "explained_ratio": [float(x) for x in self.explained_ratio],
        }


@dataclass
class DetrendedCAResult:
    """Detrended correspondence analysis.

    Attributes:
        row_scores: Detrended row coordinates, same column order as
            ``ca``, so column *k* is still axis *k* -- only their
            importance has been re-evaluated.
        column_scores: CA column coordinates, carried through unchanged.
        ca_eigenvalues: The untrended eigenvalues, kept for comparison.
        detrended_eigenvalues: ``lambda_k * retained_k``. Always less than
            or equal to the CA eigenvalue of the same axis.
        retained_variance: The multiplier ``1 - R^2_k`` per axis. 1.0
            where nothing was removed, 0.0 where an axis carried no
            independent variance at all.
        r_squared: The R-squared of each detrending regression, i.e. how
            much of that axis the gradient (or earlier axes) already
            explained.
        axis_order: Indices of the axes in descending detrended
            eigenvalue, so a caller can say "the first *independent* axis"
            without re-sorting.
        n_rows / n_columns: Table dimensions.
        gradient: The gradient used for axis 1, or None.
    """

    row_scores: npt.NDArray
    column_scores: npt.NDArray
    ca_eigenvalues: npt.NDArray
    detrended_eigenvalues: npt.NDArray
    retained_variance: npt.NDArray
    r_squared: npt.NDArray
    axis_order: npt.NDArray
    n_rows: int
    n_columns: int
    gradient: npt.NDArray | None = None
    _ca_result: CorrespondenceResult | None = field(default=None, repr=False)

    @property
    def inertia(self) -> float:
        """Sum of the detrended eigenvalues."""
        return float(np.sum(self.detrended_eigenvalues))

    def first_independent_axis(self) -> int:
        """1-based index of the axis with the largest detrended eigenvalue."""
        return int(self.axis_order[0]) + 1

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Detrended Correspondence Analysis"),
            "=" * 50,
            f"{_('Rows')}: {self.n_rows}, {_('columns')}: {self.n_columns}",
            f"{_('Gradient detrended on axis 1')}: "
            + (
                _("yes")
                if self.gradient is not None
                else _("no (axis 1 kept as-is)")
            ),
            "",
            f"{_('Axis'):>5}  {'CA eigenvalue':>14}  {'retained':>9}  "
            f"{'R-squared':>10}  {'Detrended':>12}",
        ]
        for i in range(len(self.ca_eigenvalues)):
            lines.append(
                f"{i + 1:>5}  {self.ca_eigenvalues[i]:>14.6f}  "
                f"{self.retained_variance[i]:>9.4f}  "
                f"{self.r_squared[i]:>10.4f}  "
                f"{self.detrended_eigenvalues[i]:>12.6f}"
            )
        lines.append("")
        lines.append(
            f"{_('Most important independent axis')}: "
            f"{_('axis {0}').format(self.first_independent_axis())}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "ca_eigenvalues": [float(e) for e in self.ca_eigenvalues],
            "detrended_eigenvalues": [float(e) for e in self.detrended_eigenvalues],
            "retained_variance": [float(v) for v in self.retained_variance],
            "r_squared": [float(r) for r in self.r_squared],
            "axis_order": [int(a) for a in self.axis_order],
            "first_independent_axis": self.first_independent_axis(),
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "used_gradient": self.gradient is not None,
        }


def _regression_r_squared(y: npt.NDArray, x: npt.NDArray) -> float:
    """R^2 of y regressed on the columns of x, with an intercept.

    Returns 0.0 when x carries no usable variation, which retains everything and
    therefore "nothing was removed" -- the right answer for a constant
    gradient.
    """
    design = np.column_stack([x, np.ones(len(y))])
    # A column that is constant (or collinear with the intercept) makes the
    # normal equations singular; rank-check first rather than let lstsq pick
    # an arbitrary answer.
    if np.linalg.matrix_rank(design) < design.shape[1]:
        usable = x.std(axis=0) > 0
        if not np.any(usable):
            return 0.0
        x = x[:, usable]
        design = np.column_stack([x, np.ones(len(y))])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return 0.0

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    r_squared = 1.0 - ss_res / ss_tot
    # R^2 can exceed 1 by a hair through floating point; clamp rather than
    # hand a negative retained variance to the caller.
    return float(min(max(r_squared, 0.0), 1.0))


class DetrendedCAAnalyzer:
    """Correspondence analysis with gradient detrending."""

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.DetrendedCAAnalyzer")
        self._last_ca: CorrespondenceResult | None = None
        self._last_dca: DetrendedCAResult | None = None

    def _validate_table(self, table: npt.NDArray) -> npt.NDArray:
        """Check the table is a non-negative contingency table."""
        F = np.asarray(table, dtype=float)
        if F.ndim != 2:
            raise MatrixDimensionError(
                f"DCA: the contingency table must be 2-dimensional, got "
                f"{F.ndim} dimensions",
                details={"shape": tuple(int(x) for x in F.shape)},
            )
        if F.shape[0] < 2 or F.shape[1] < 2:
            raise MatrixDimensionError(
                f"DCA: the contingency table needs at least 2 rows and 2 "
                f"columns, got {F.shape[0]}x{F.shape[1]}",
                details={"shape": (int(F.shape[0]), int(F.shape[1]))},
            )
        if not np.all(np.isfinite(F)):
            raise ValidationError("DCA: the contingency table contains NaN or Inf")
        if np.any(F < 0):
            n_neg = int(np.sum(F < 0))
            raise ValidationError(
                f"DCA: a contingency table holds counts and cannot contain "
                f"negatives; found {n_neg}",
                details={"n_negative": n_neg},
            )
        if float(np.sum(F)) <= 0:
            raise ComputationError(
                "DCA: the contingency table sums to zero, so its "
                "chi-square distances are undefined"
            )
        return F

    def correspondence(
        self, table: npt.NDArray, n_components: int | None = None
    ) -> CorrespondenceResult:
        """Run an unconstrained correspondence analysis.

        Parameters
        ----------
        table:
            Non-negative contingency table, (n_rows x n_columns).
        n_components:
            How many axes to keep. Defaults to every non-zero axis,
            bounded by ``min(n_rows, n_columns) - 1``.

        Returns
        -------
        CorrespondenceResult
        """
        F = self._validate_table(table)
        total = float(np.sum(F))
        P = F / total
        r = P.sum(axis=1)          # row masses
        c = P.sum(axis=0)          # column masses
        expected = np.outer(r, c)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (P - expected) / np.sqrt(expected)
        # A zero row or column mass makes the corresponding expected value
        # zero and the residual NaN. Such a row or column holds no
        # information; drop it rather than carry NaN into the SVD.
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        r_valid = r > 0
        c_valid = c > 0
        S = S[np.ix_(r_valid, c_valid)]
        r_used = r[r_valid]
        c_used = c[c_valid]

        if S.size == 0:
            raise ComputationError(
                "DCA: every row or column of the table is zero, so there is "
                "no structure to analyse"
            )

        U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
        eigenvalues = (sigma**2) / total
        inertia = float(np.sum(eigenvalues))

        max_axes = min(S.shape[0], S.shape[1]) - 1
        if n_components is None:
            n_keep = max_axes
        else:
            n_keep = min(int(n_components), max_axes)
        if n_keep < 1:
            raise ValidationError(
                f"DCA: n_components must be at least 1, got {n_components}"
            )

        # Row principal coordinates: U_k * sigma_k / sqrt(r). Column
        # coordinates are the symmetric construction on the right singular
        # vectors from the same decomposition.
        row_scores = (U[:, :n_keep] * sigma[:n_keep]) / np.sqrt(r_used)[:, None]
        column_scores = (Vt[:n_keep, :].T * sigma[:n_keep]) / np.sqrt(c_used)[:, None]

        result = CorrespondenceResult(
            row_scores=row_scores,
            column_scores=column_scores,
            eigenvalues=eigenvalues[:n_keep],
            inertia=inertia,
            n_rows=int(r_used.size),
            n_columns=int(c_used.size),
        )
        self._last_ca = result
        self._logger.info(
            "CA complete: %dx%d table, inertia=%.6f", result.n_rows, result.n_columns, inertia
        )
        return result

    def analyze(
        self,
        table: npt.NDArray,
        gradient: npt.NDArray | None = None,
        n_components: int | None = None,
    ) -> DetrendedCAResult:
        """Run DCA.

        Parameters
        ----------
        table:
            Non-negative contingency table.
        gradient:
            One value per row of ``table`` -- the environmental variable
            whose linear effect should be removed from axis 1. Without it,
            axis 1 is left as CA produced it and only axes 2 and up are
            detrended, which is the standard treatment when no single
            dominant gradient is known.
        n_components:
            How many axes to keep.

        Returns
        -------
        DetrendedCAResult
        """
        ca = self.correspondence(table, n_components=n_components)
        F = self._validate_table(table)

        row_scores = np.asarray(ca.row_scores, dtype=float)
        n_axes = row_scores.shape[1]

        g: npt.NDArray | None = None
        if gradient is not None:
            g = np.asarray(gradient, dtype=float).ravel()
            if g.size != F.shape[0]:
                raise MatrixDimensionError(
                    f"DCA: the gradient needs one value per row ({F.shape[0]}), "
                    f"got {g.size}",
                    details={"rows": int(F.shape[0]), "gradient": int(g.size)},
                )
            if not np.all(np.isfinite(g)):
                raise ValidationError("DCA: the gradient contains NaN or Inf")
            if np.ptp(g) == 0:
                # A constant gradient explains nothing. Detrending against
                # it would be a no-op with an undefined regression, so say
                # so rather than quietly proceeding.
                raise ValidationError(
                    "DCA: the gradient is constant, so it cannot be used to "
                    "detrend axis 1. Supply a varying gradient, or none."
                )
            # Rows dropped for being all-zero were dropped by
            # correspondence(); the gradient has to be trimmed the same way
            # or it no longer lines up with the scores.
            if g.size != ca.n_rows:
                g = g[_valid_row_mask(F)]

        retained_all = np.ones(n_axes, dtype=float)
        r_squared = np.zeros(n_axes, dtype=float)
        detrended_scores = np.zeros_like(row_scores)
        detrended_eigen = ca.eigenvalues.copy()
        degenerate: list[int] = []

        for k in range(n_axes):
            if k == 0:
                if g is None:
                    # Nothing to detrend axis 1 against: it IS the gradient.
                    # Leave it untouched rather than inventing a null.
                    detrended_scores[:, 0] = row_scores[:, 0]
                    continue
                predictors = g[:, None]
            else:
                # Regress on the earlier *detrended* axes, so each axis is
                # made independent of what the previous ones already carry.
                predictors = detrended_scores[:, :k]

            r2 = _regression_r_squared(row_scores[:, k], predictors)
            r_squared[k] = r2
            # What survives detrending is the residual, whose variance is
            # (1 - R^2) of the original's. An eigenvalue is proportional to
            # that variance, so the multiplier is (1 - R^2) -- a DEFLATION.
            # Multiplying by the regression's variance inflation factor
            # 1/(1 - R^2) would instead enlarge every axis the gradient
            # already explains, which is the opposite of the point: an axis
            # fully explained by the gradient has no independent variance
            # left, and its detrended eigenvalue must go to zero.
            retained = 1.0 - r2
            if retained <= _RETAINED_CEILING:
                # Numerically nothing left. Keep the axis in the output --
                # dropping it would shift every later index -- but give it a
                # zero eigenvalue so it sorts last, and say so.
                retained = 0.0
                degenerate.append(k + 1)
                self._logger.warning(
                    "DCA: axis %d is completely explained by %s; its "
                    "detrended eigenvalue is 0",
                    k + 1,
                    "the gradient" if k == 0 else "the preceding axes",
                )
            detrended_eigen[k] = ca.eigenvalues[k] * retained
            retained_all[k] = retained
            detrended_scores[:, k] = row_scores[:, k] * np.sqrt(retained)

        if degenerate:
            self._logger.info(
                "DCA: %d axis/axes carried no independent variance: %s",
                len(degenerate),
                ", ".join(str(a) for a in degenerate),
            )

        axis_order = np.argsort(-detrended_eigen, kind="stable")
        result = DetrendedCAResult(
            row_scores=detrended_scores,
            column_scores=ca.column_scores,
            ca_eigenvalues=ca.eigenvalues,
            detrended_eigenvalues=detrended_eigen,
            retained_variance=retained_all,
            r_squared=r_squared,
            axis_order=axis_order,
            n_rows=ca.n_rows,
            n_columns=ca.n_columns,
            gradient=g,
            _ca_result=ca,
        )
        self._last_dca = result
        self._logger.info(
            "DCA complete: %d axes, leading independent axis = %d",
            n_axes,
            result.first_independent_axis(),
        )
        return result

    @property
    def last_ca(self) -> CorrespondenceResult | None:
        """Most recent correspondence analysis, or None."""
        return self._last_ca

    @property
    def last_result(self) -> DetrendedCAResult | None:
        """Most recent DCA, or None."""
        return self._last_dca


def _valid_row_mask(table: npt.NDArray) -> npt.NDArray:
    """Rows with a non-zero mass, i.e. those correspondence() keeps."""
    P = table / float(np.sum(table))
    return P.sum(axis=1) > 0
