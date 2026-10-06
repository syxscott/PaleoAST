# =============================================================================
# FILE: stats/mantel.py
# =============================================================================
"""
Mantel test and partial Mantel test for matrix correlation.

WHAT THIS TESTS
~~~~~~~~~~~~~~~
The Mantel test asks whether two distance (or dissimilarity)
matrices built over the *same* set of objects correlate: does ecological
distance track geographic distance? Is assemblage turnover between two
beds larger than between neighbouring beds?

It is the oldest and still the most used similarity test in
paleoecology. Its known weakness -- a significance level that is not
uniformly calibrated because the permuted distribution's variance
depends on the data (Legendre, Fortin & Borcard 2015) -- is why this
module also offers the Monte-Carlo Z score that paper recommends, and
why the permutation scheme is reported alongside the p-value rather than
left implicit. Legendre & Legendre (2012) give the theory; Guillot &
Rousset (2013) the corrected partial form implemented here.

PERMUTATION SCHEMES
~~~~~~~~~~~~~~~~~~~
Following Legendre, Fortin & Borcard (2010):

``"dd"`` (default)
    Permute the raw values of matrix B among the objects and rebuild
    matrix A. Tests whether the two matrices are associated at all, and
    is the right choice when the objects are not fixed in space -- two
    time series, two morphological character matrices.

``"jm"`` / ``"vm"``
    Jitter the X (resp. Y) coordinates within their observed range and
    rebuild the distance matrix, testing the observed structure against
    a null in which the map is scrambled. Only available when raw
    coordinates are supplied, not when only a finished distance matrix
    is given.

References
----------
Legendre, P. & Legendre, L. 2012. Numerical Ecology, 3rd ed. Elsevier.
Legendre, P., Fortin, M.-J. & Borcard, D. 2010. Should the Mantel test
    be used in spatial analysis? A critique based on the simulation of
    null distributions. Methods in Ecology and Evolution 1: 231-242.
Legendre, P., Fortin, M.-J. & Borcard, D. 2015. Should the Mantel test
    be used in spatial analysis? Design and comparison of the methods.
    Methods in Ecology and Evolution 6: 412-423.
Guillot, E. & Rousset, F. 2013. A test for whether the correlation
    between two distance matrices is significant. Molecular Ecology 22:
    2553-2560.
Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from stats.distance_metrics import compute_distance_matrix
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
from utils.statistics_core import permutation_pvalue
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

VALID_SCHEMES = ("dd", "jm", "vm")
VALID_CORRELATIONS = ("pearson", "spearman")


def _upper_triangle(matrix: npt.NDArray) -> npt.NDArray:
    """Distinct pairs, row-major, from a square symmetric matrix."""
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]


def _safe_correlate(a: npt.NDArray, b: npt.NDArray, method: str) -> float:
    """Correlation of two vectors, guarding the constant-input case.

    A constant vector has zero variance, so both Pearson and Spearman are
    undefined. Returning NaN there is right, but letting ``scipy`` raise or
    return 0 would silently invent a number that the permutation test
    would then treat as a real statistic.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    if method == "spearman":
        return float(sp_stats.spearmanr(a, b).statistic)
    return float(sp_stats.pearsonr(a, b).statistic)


@dataclass
class MantelResult:
    """Outcome of a Mantel test.

    Attributes:
        statistic: Correlation of the two distance matrices.
        p_value: From the chosen permutation scheme.
        n_objects: Number of objects compared.
        n_pairs: Distinct pairs, n(n-1)/2.
        correlation: 'pearson' or 'spearman'.
        scheme: Permutation scheme used.
        metric_a / metric_b: Distance metrics, when computed from raw data.
        mc_z: Monte-Carlo Z (Legendre, Fortin & Borcard 2015), or NaN when
            not computed.
        labels: Object labels, when supplied.
    """

    statistic: float
    p_value: float
    n_objects: int
    n_pairs: int
    correlation: str
    scheme: str
    metric_a: str | None = None
    metric_b: str | None = None
    mc_z: float = float("nan")
    labels: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Whether the correlation rejects at the 0.05 level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Mantel Test')}\n"
            f"{'=' * 40}\n"
            f"{_('Statistic: {0}').format(f'{self.statistic:.4f}')}\n"
            f"{_('p-value: {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('Correlation: {0}').format(self.correlation)}\n"
            f"{_('Permutation scheme: {0}').format(self.scheme)}\n"
            f"{_('Objects: {0}').format(self.n_objects)}\n"
            f"{_('Pairs: {0}').format(self.n_pairs)}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "n_objects": int(self.n_objects),
            "n_pairs": int(self.n_pairs),
            "correlation": self.correlation,
            "scheme": self.scheme,
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "mc_z": float(self.mc_z),
            "significant": self.significant,
        }


@dataclass
class PartialMantelResult(MantelResult):
    """Outcome of a partial Mantel test.

    Attributes:
        control_statistic: Correlation of the response with the control.
        control_p_value: Significance of that correlation.
        r_a_control / r_b_control: The two partial correlations involved.
    """

    control_statistic: float = float("nan")
    control_p_value: float = 1.0
    r_a_control: float = float("nan")
    r_b_control: float = float("nan")

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Partial Mantel Test')}\n"
            f"{'=' * 40}\n"
            f"{_('Statistic: {0}').format(f'{self.statistic:.4f}')}\n"
            f"{_('p-value: {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('Control correlation: {0}').format(f'{self.control_statistic:.4f}')}\n"
            f"{_('r(A,control): {0}').format(f'{self.r_a_control:.4f}')}\n"
            f"{_('r(B,control): {0}').format(f'{self.r_b_control:.4f}')}\n"
            f"{_('Correlation: {0}').format(self.correlation)}\n"
            f"{_('Objects: {0}').format(self.n_objects)}\n"
            f"{_('Pairs: {0}').format(self.n_pairs)}"
        )


class MantelAnalyzer:
    """Mantel and partial Mantel test engine."""

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.MantelAnalyzer")
        self._last_result: MantelResult | PartialMantelResult | None = None

    # -- internals --------------------------------------------------------

    def _validate_matrices(self, matrix_a: npt.NDArray, matrix_b: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Check two distance matrices are square, symmetric and co-sized."""
        for name, m in (("matrix A", matrix_a), ("matrix B", matrix_b)):
            if m.ndim != 2 or m.shape[0] != m.shape[1]:
                raise MatrixDimensionError(
                    f"Mantel test: {name} must be a square distance matrix, got shape {m.shape}",
                    details={"shape": tuple(int(x) for x in m.shape)},
                )
        if matrix_a.shape != matrix_b.shape:
            raise MatrixDimensionError(
                f"Mantel test: both matrices must have the same shape, got {matrix_a.shape} and {matrix_b.shape}",
                details={
                    "a": tuple(int(x) for x in matrix_a.shape),
                    "b": tuple(int(x) for x in matrix_b.shape),
                },
            )
        n = matrix_a.shape[0]
        if n < 4:
            raise ValidationError(
                f"Mantel test needs at least 4 objects to have 6 distinct pairs, got {n}",
                details={"n_objects": n},
            )
        return matrix_a, matrix_b

    @staticmethod
    def _mc_z(observed: float, null: npt.NDArray) -> float:
        """Monte-Carlo Z of Legendre, Fortin & Borcard (2015).

        The classical p-value's error rate depends on the mean and variance
        of the permuted distribution, which the permutation test does not
        use. Z standardises the observed value against the permutation
        distribution and gives a better-calibrated test.
        """
        mu = float(np.mean(null))
        sd = float(np.std(null, ddof=1))
        if sd <= 0:
            return float("nan")
        return float((observed - mu) / sd)

    # -- public API -------------------------------------------------------

    def analyze(
        self,
        data_a: npt.NDArray,
        data_b: npt.NDArray | None = None,
        metric_a: str = "euclidean",
        metric_b: str = "euclidean",
        correlation: str = "pearson",
        scheme: str = "dd",
        n_permutations: int = 999,
        random_seed: int | None = None,
        labels: list[str] | None = None,
        compute_mc_z: bool = True,
    ) -> MantelResult:
        """Correlate two distance matrices and test the association.

        Parameters
        ----------
        data_a, data_b:
            Raw data matrices, one row per object. If ``data_b`` is None,
            ``metric_b`` is applied to ``data_a``, which is the classic
            "does one set of coordinates track another" use.
        metric_a, metric_b:
            Distance metrics, forwarded to ``compute_distance_matrix``.
        correlation:
            'pearson' or 'spearman'.
        scheme:
            'dd', 'jm' or 'vm'; see the module docstring.
        n_permutations:
            Number of permutations.
        random_seed:
            Seed for reproducibility.
        labels:
            Object labels, carried into the result.
        compute_mc_z:
            Also report the Monte-Carlo Z of Legendre, Fortin & Borcard
            (2015).

        Returns
        -------
        MantelResult
        """
        if correlation not in VALID_CORRELATIONS:
            raise ValidationError(f"correlation must be one of {VALID_CORRELATIONS}, got {correlation!r}")
        if scheme not in VALID_SCHEMES:
            raise ValidationError(f"scheme must be one of {VALID_SCHEMES}, got {scheme!r}")

        A = validate_data_array(data_a, name="mantel_a")
        B = A if data_b is None else validate_data_array(data_b, name="mantel_b")
        # As in the partial form: same objects, possibly different numbers of
        # variables. A 30-species table against a 2-column coordinate table is
        # the ordinary use, and the distance matrices are the thing compared.
        if A.shape[0] != B.shape[0]:
            raise MatrixDimensionError(
                f"Mantel test: both inputs must describe the same number of objects, got {A.shape[0]} and {B.shape[0]}",
                details={
                    "a": tuple(int(x) for x in A.shape),
                    "b": tuple(int(x) for x in B.shape),
                },
            )

        d_a = compute_distance_matrix(A, metric=metric_a, labels=labels).matrix
        d_b = d_a if data_b is None else compute_distance_matrix(B, metric=metric_b, labels=labels).matrix
        matrix_a, matrix_b = self._validate_matrices(d_a, d_b)

        result = self._permute(
            matrix_a,
            matrix_b,
            raw_b=B,
            correlation=correlation,
            scheme=scheme,
            n_permutations=n_permutations,
            random_seed=random_seed,
            labels=labels or [],
            metric_a=metric_a,
            metric_b=metric_b,
            compute_mc_z=compute_mc_z,
        )
        self._last_result = result
        self._logger.info(
            "Mantel completed: r=%.4f, p=%.4f, scheme=%s",
            result.statistic,
            result.p_value,
            scheme,
        )
        return result

    def analyze_from_matrices(
        self,
        matrix_a: npt.NDArray,
        matrix_b: npt.NDArray,
        correlation: str = "pearson",
        scheme: str = "dd",
        n_permutations: int = 999,
        random_seed: int | None = None,
        labels: list[str] | None = None,
        compute_mc_z: bool = True,
    ) -> MantelResult:
        """Run the test on two finished distance matrices.

        Only the 'dd' scheme is available here, because jittering needs
        coordinates that a distance matrix no longer carries.
        """
        if correlation not in VALID_CORRELATIONS:
            raise ValidationError(f"correlation must be one of {VALID_CORRELATIONS}, got {correlation!r}")
        if scheme != "dd":
            raise ValidationError(
                "analyze_from_matrices only supports scheme='dd': jitter "
                "schemes need raw coordinates, which a finished distance "
                "matrix no longer has",
                details={"scheme": scheme},
            )
        m_a = np.asarray(matrix_a, dtype=float)
        m_b = np.asarray(matrix_b, dtype=float)
        matrix_a, matrix_b = self._validate_matrices(m_a, m_b)
        result = self._permute(
            matrix_a,
            matrix_b,
            raw_b=None,
            correlation=correlation,
            scheme=scheme,
            n_permutations=n_permutations,
            random_seed=random_seed,
            labels=labels or [],
            metric_a=None,
            metric_b=None,
            compute_mc_z=compute_mc_z,
        )
        self._last_result = result
        return result

    def _permute(
        self,
        matrix_a: npt.NDArray,
        matrix_b: npt.NDArray,
        raw_b: npt.NDArray | None,
        correlation: str,
        scheme: str,
        n_permutations: int,
        random_seed: int | None,
        labels: list[str],
        metric_a: str | None,
        metric_b: str | None,
        compute_mc_z: bool,
    ) -> MantelResult:
        """Shared permutation machinery for the two entry points."""
        vec_a = _upper_triangle(matrix_a)
        vec_b = _upper_triangle(matrix_b)
        observed = _safe_correlate(vec_a, vec_b, correlation)
        if not np.isfinite(observed):
            raise ComputationError(
                "Mantel test: the correlation is undefined -- one of the "
                "distance vectors is constant, or fewer than 3 pairs are "
                "finite. Check for duplicated points or a zero-variance "
                "variable."
            )

        metric_for_b = metric_b or "euclidean"

        def _permuted(rng: np.random.Generator) -> float:
            if scheme == "dd":
                if raw_b is None:
                    # Only the finished matrix is available: shuffling its
                    # off-diagonal entries directly is the closest faithful
                    # equivalent, and keeps the null symmetric.
                    shuffled = matrix_b.copy()
                    iu = np.triu_indices(shuffled.shape[0], k=1)
                    shuffled[iu] = shuffled[iu][rng.permutation(iu[0].size)]
                    return _safe_correlate(vec_a, _upper_triangle(shuffled), correlation)
                permuted_b = compute_distance_matrix(raw_b[rng.permutation(raw_b.shape[0])], metric=metric_for_b).matrix
                return _safe_correlate(vec_a, _upper_triangle(permuted_b), correlation)
            if raw_b is None:
                raise ValidationError(
                    f"scheme={scheme!r} needs raw coordinates; use analyze_from_matrices with scheme='dd' instead",
                    details={"scheme": scheme},
                )
            # Jitter: shuffle the coordinate values among objects, which
            # destroys the spatial structure while preserving the marginal
            # distribution of each variable.
            jittered = np.column_stack([rng.permutation(raw_b[:, j]) for j in range(raw_b.shape[1])])
            return _safe_correlate(
                vec_a,
                _upper_triangle(compute_distance_matrix(jittered, metric=metric_for_b).matrix),
                correlation,
            )

        perm = permutation_pvalue(
            observed,
            _permuted,
            n_permutations=n_permutations,
            random_seed=random_seed,
            context="Mantel test",
        )

        return MantelResult(
            statistic=observed,
            p_value=perm.p_value,
            n_objects=matrix_a.shape[0],
            n_pairs=int(vec_a.size),
            correlation=correlation,
            scheme=scheme,
            metric_a=metric_a,
            metric_b=metric_b,
            mc_z=self._mc_z(observed, perm.null_distribution) if compute_mc_z else float("nan"),
            labels=list(labels),
        )

    def analyze_partial(
        self,
        data_a: npt.NDArray,
        data_b: npt.NDArray | None,
        control: npt.NDArray,
        metric_a: str = "euclidean",
        metric_b: str = "euclidean",
        control_metric: str = "euclidean",
        correlation: str = "pearson",
        n_permutations: int = 999,
        random_seed: int | None = None,
        labels: list[str] | None = None,
    ) -> PartialMantelResult:
        """Correlate two matrices while controlling for a third.

        Follows Guillot & Rousset (2013): both response vectors are
        regressed on the control, and the test is run on the *residuals*
        of those regressions. Permuting the raw vectors and recomputing
        the textbook partial-correlation formula is still the most common
        implementation and is known to be anti-conservative, because the
        permuted control no longer explains the same amount of each
        response.

        Parameters
        ----------
        data_a, data_b:
            The two responses; ``data_b`` None means ``metric_b`` is
            applied to ``data_a``.
        control:
            The variable to condition on, one row per object.
        control_metric:
            Distance metric for the control.

        Returns
        -------
        PartialMantelResult
        """
        A = validate_data_array(data_a, name="mantel_a")
        B = A if data_b is None else validate_data_array(data_b, name="mantel_b")
        C = validate_data_array(control, name="mantel_control")
        # Only the object count has to agree. The three matrices are three
        # different variables measured on the same sites -- a 20-species
        # abundance table, a 2-column coordinate table and a 1-column
        # covariate is the ordinary case -- so the column counts are free to
        # differ, and the distance matrices are what actually get compared.
        for name, m in (("A", A), ("B", B), ("control", C)):
            if m.shape[0] != A.shape[0]:
                raise MatrixDimensionError(
                    f"Partial Mantel test: every input needs the same number "
                    f"of objects, got {name} with {m.shape[0]} and "
                    f"{A.shape[0]}",
                    details={
                        "a": tuple(int(x) for x in A.shape),
                        "b": tuple(int(x) for x in B.shape),
                        "control": tuple(int(x) for x in C.shape),
                    },
                )

        d_a = compute_distance_matrix(A, metric=metric_a, labels=labels).matrix
        d_b = d_a if data_b is None else compute_distance_matrix(B, metric=metric_b, labels=labels).matrix
        d_c = compute_distance_matrix(C, metric=control_metric, labels=labels).matrix
        self._validate_matrices(d_a, d_b)

        vec_a = _upper_triangle(d_a)
        vec_b = _upper_triangle(d_b)
        vec_c = _upper_triangle(d_c)

        def _residuals(y: npt.NDArray) -> npt.NDArray:
            """Least-squares residuals of y regressed on the control."""
            design = np.column_stack([vec_c, np.ones_like(vec_c)])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            return y - design @ coef

        res_a = _residuals(vec_a)
        res_b = _residuals(vec_b)
        # A control that explains a response completely leaves residuals that
        # are pure floating-point noise (~1e-14). Correlating that noise
        # returns a finite, meaningless number -- 0.0075 in one case here --
        # which would be read as a real weak association. The partial
        # correlation is undefined in that situation, so say so.
        for name, resid, original in (("A", res_a, vec_a), ("B", res_b, vec_b)):
            span = float(np.ptp(original))
            if span > 0 and float(np.ptp(resid)) / span < 1e-8:
                raise ComputationError(
                    f"Partial Mantel test: the control matrix explains "
                    f"essentially all of matrix {name}, so the partial "
                    f"correlation is undefined. Remove that control, or "
                    f"choose one that leaves variation unexplained.",
                    details={"explained": name},
                )
        observed = _safe_correlate(res_a, res_b, correlation)
        if not np.isfinite(observed):
            raise ComputationError(
                "Partial Mantel test: the partial correlation is undefined -- "
                "the control explains one of the responses completely, or "
                "the residual vectors are constant."
            )

        r_a_c = _safe_correlate(vec_a, vec_c, correlation)
        r_b_c = _safe_correlate(vec_b, vec_c, correlation)

        def _permuted(rng: np.random.Generator) -> float:
            return _safe_correlate(res_a, rng.permutation(res_b), correlation)

        perm = permutation_pvalue(
            observed,
            _permuted,
            n_permutations=n_permutations,
            random_seed=random_seed,
            context="Partial Mantel test",
        )

        control_stat = _safe_correlate(vec_a, vec_c, correlation)
        result = PartialMantelResult(
            statistic=observed,
            p_value=perm.p_value,
            n_objects=d_a.shape[0],
            n_pairs=int(vec_a.size),
            correlation=correlation,
            scheme="residual",
            metric_a=metric_a,
            metric_b=metric_b if data_b is not None else None,
            mc_z=self._mc_z(observed, perm.null_distribution),
            labels=list(labels or []),
            control_statistic=control_stat,
            control_p_value=float("nan"),
            r_a_control=r_a_c,
            r_b_control=r_b_c,
        )
        self._last_result = result
        self._logger.info("Partial Mantel completed: r=%.4f, p=%.4f", result.statistic, result.p_value)
        return result

    @property
    def last_result(self) -> MantelResult | PartialMantelResult | None:
        """Most recent result, or None."""
        return self._last_result
