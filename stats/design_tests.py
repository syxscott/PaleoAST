# =============================================================================
# FILE: stats/design_tests.py
# =============================================================================
"""
Design-based hypothesis tests that ``stats/univariate.py`` does not cover.

WHAT THIS MODULE ADDS
~~~~~~~~~~~~~~~~~~~~~
``stats/univariate.py`` implements the one-way analyses: one-way ANOVA,
Kruskal-Wallis, t-tests, Mann-Whitney and normality. Three designs that
paleontological data actually arrive in are missing from it entirely, and
each of them is a question a reviewer will ask:

``two_way_anova``
    A full factorial: does the lithology affect the measurement, does the
    basin affect it, and -- the one that gets botched -- do they interact?

``repeated_measures_anova``
    The same specimen, or the same population, measured several times
    (before/after a perturbation, across observers, across CT slices). The
    observations within a subject are not independent, so a one-way ANOVA
    over the pooled data is wrong; its F is inflated by a factor of k.

``intraclass_correlation``
    Repeatability: how much of the total variance is between specimens
    rather than within the measurement error of one specimen? Reported with
    the interval, without which the coefficient cannot be used.

A contingency-table chi-square with expected counts and Cramer's V is
included because it is the categorical counterpart of the same question
and costs one scipy call plus a result container.

WHY HAND-ROLLED RATHER THAN statsmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``statsmodels`` 0.14 is installed and would do all of this
(``statsmodels.stats.anova.anova_lm`` with a ``typ=2`` design matrix,
``statsmodels.stats.anova.AnovaRM``, ``pingouin``-style ICC helpers), but
no module under ``stats/`` imports it today, and every ANOVA in this
codebase spells its own sums of squares out of ``scipy.stats.f_oneway`` plus
explicit arithmetic. Adopting one heavy general linear-model path in a
single file would give ``stats/`` two incompatible provenances, make the
df conventions differ between neighbouring modules, and put a
straight-line fit inside what is otherwise an exact-arithmetic package.
So the decompositions here are written by hand, in the same style as
``stats/univariate.py``, and cross-checked in
``tests/stats/test_design_tests.py`` against identities derived
independently -- against ``scipy.stats.f.sf`` for every p-value, against
a closed-form SS decomposition written in the test, and against
variance-component ratios. Agreement with R's ``aov``, ``car::Anova`` and
``irr::icc`` is the intended external reference; the closed forms below are
the textbook ones those packages implement.

TWO-WAY ANOVA: THE SUM-OF-SQUARES DECOMPOSITION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a balanced-or-unbalanced a x b layout with cell means ``m_ij`` and
cell sizes ``n_ij`` (N = sum n_ij, grand mean ``M``):

    SS_total = sum_over_all (y - M)^2
    SS_A     = sum_i   n_i.   (m_i. - M)^2            df_A   = a - 1
    SS_B     = sum_j   n_.j   (m_.j - M)^2            df_B   = b - 1
    SS_AB    = sum_ij  n_ij  (m_ij - m_i. - m_.j + M)^2
                                                  df_AB  = (a - 1)(b - 1)
    SS_error = sum_ij  n_ij (y - m_ij)^2             df_err = N - a*b

The four add up to SS_total exactly; the decomposition is a projection
onto mutually orthogonal spaces, so it holds for unbalanced designs too.
The interaction contrast ``m_ij - m_i. - m_.j + M`` is the whole point of
fitting the interaction: it is the part of a cell that neither factor's
marginal mean predicts, and a design that omits it silently folds that
part back into the error term.

Every F in this design has the *residual* mean square in its denominator:

    F_A  = (SS_A  / df_A)  / (SS_error / df_error)
    F_B  = (SS_B  / df_B)  / (SS_error / df_error)
    F_AB = (SS_AB / df_AB) / (SS_error / df_error)

Using ``MS_A + MS_B`` or ``MS_B + MS_AB`` in a denominator -- the common
"sequential"/Type-III confusion -- is the classic bug this docstring
exists to prevent. Type I sums of squares are used throughout (they are
the orthogonal-projection quantities above), so a balanced design
reproduces R's ``aov(y ~ a*b)`` exactly.

On an *unbalanced* design those four terms no longer add up to
SS_total, because the weighted marginal means are not orthogonal: on the
2x2 with cell sizes 2, 2, 3, 2 they sum to 538.67 while SS_total is
531.56. The terms are therefore built as differences of nested weighted
fits to the cell means -- null, then +A, then +A+B, then +A:B -- which is
the Type I decomposition and always closes exactly. Balanced designs are
unaffected, because the nested additive fit collapses to
``m_i. + m_.j - M`` there; unbalanced ones need the least-squares
additive fit. An unbalanced design is reported as ``balanced = False``
rather than quietly handled, because Type I terms on unbalanced data are
order-dependent by construction.

Effect sizes, both reported per term:

    eta^2     = SS_term / SS_total                      (classical)
    partial   = SS_term / (SS_term + SS_error) = F*df1 / (F*df1 + df2)

The partial form is obtained from ``stats.univariate.partial_eta_squared``,
whose signature ``(F, df_between, df_error)`` fits this design exactly. The
classical form is computed here rather than through
``stats.univariate.eta_squared`` because that helper evaluates
``F*df1 / (F*df1 + df2)``, which in a factorial design is SS_term /
(SS_term + SS_residual) -- the *partial* quantity -- and not SS_term /
SS_total. The two coincide in a one-way design and differ in every
multi-factor one, which is precisely where the distinction matters.

PERMUTATION SCHEME AND ITS ASSUMPTION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The permutation test shuffles the *response values* over the whole
design, keeping the factor labels fixed, and recomputes each term's F.
This is a "permutation of the response", i.e. it assumes free
exchangeability of the observations: every assignment of the observed
values to the design cells is equally plausible under the null.

That assumption is the design's, not the test's, and it is stronger than
a one-way ANOVA needs in only one direction: if the observations are not
exchangeable across cells, the reported p-value is invalid. The
permutation is stated rather than buried because a factorial design is
exactly the case where exchangeability is an explicit modelling
assumption -- a subject measured at several time points (use
``repeated_measures_anova``), a stratigraphically ordered set of samples
where depth is confounded with treatment, or cells with different
measurement effort. Permuting *within* cells instead would respect the
cell structure and be conservative, never anti-conservative; that scheme
is not implemented here, so the reported interval assumes exchangeability.

REPEATED-MEASURES ANOVA: EVERY DEGREE OF FREEDOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n subjects per group, g groups, k repeated measurements per subject,
balanced, with the subject as the random effect:

    df_total       = g*n*k - 1
    df_groups      = g - 1                       SS_groups     (between subjects)
    df_subjects    = g*(n - 1)                   SS_subjects   (between subjects)
    df_measurement = k - 1                       SS_measurement (within subjects)
    df_interaction = (g - 1)*(k - 1)             SS_interaction (within subjects)
    df_error       = g*(n - 1)*(k - 1)           SS_error      (within subjects)

    (g-1) + g(n-1) + (k-1) + (g-1)(k-1) + g(n-1)(k-1) = g*n*k - 1

The two strata are the whole point, and each has its own denominator:

    F_groups      = (SS_groups / df_groups) / (SS_subjects / df_subjects)
    F_measurement = (SS_measurement / df_measurement) / (SS_error / df_error)
    F_interaction = (SS_interaction / df_interaction) / (SS_error / df_error)

Note ``df_error = g*(n-1)*(k-1)`` and not ``n*(g-1)*(k-1)``. The
between-subject stratum pools the n-1 subject degrees of freedom inside
*each* of the g groups, which gives g*(n-1) subject df and, mirrored,
g*(n-1) error df. Writing ``n*(g-1)*(k-1)`` leaves the total df short by
``(g-1)*(n-1)*(k-1)`` and still produces F values of a plausible size --
the classic "wrong denominator without looking wrong".

An unbalanced design is rejected rather than repaired: a subject with
fewer than k measurements has no within-subject mean for the missing
cells, and filling it in by averaging silently changes both strata. See
``repeated_measures_anova`` for what is refused and why.

INTRACLASS CORRELATION
~~~~~~~~~~~~~~~~~~~~~~
For a balanced two-way table of n targets (rows) by k raters or repeats
(columns), from the two-way ANOVA:

    MS_targets = k * sum_i  (y_i.  - M)^2 / (n - 1)
    MS_raters  = n * sum_j  (y_.j  - M)^2 / (k - 1)
    MS_error   = (SS_total - SS_targets - SS_raters) / (n-1)(k-1)

    ICC(2,1) = (MS_targets - MS_error)
               / (MS_targets + (k-1)*MS_error + k*(MS_raters - MS_error)/n)
    ICC(3,1) = (MS_targets - MS_error)
               / (MS_targets + (k-1)*MS_error)

ICC(2,1) is McGraw and Wong's (1996) "two-way random effects, absolute
agreement, single measurement": a systematic rater offset counts against
reliability, which is what a fossil measured by three technicians should
report. ICC(3,1) is "two-way mixed effects, consistency, single
measurement": raters may disagree by a constant offset and the
coefficient ignores it. Both equal the definitional variance ratio

    ICC = sigma^2_targets / (sigma^2_targets + sigma^2_error  [+ sigma^2_raters])

because MS_targets = (sigma^2_error + k*sigma^2_t) * chi^2/(n-1),
MS_raters = (sigma^2_error + n*sigma^2_c) * chi^2/(k-1) and
MS_error = sigma^2_error * chi^2/((n-1)(k-1)), each an exact scale
multiple of an independent central chi-square. That identity is what the
tests assert.

CONFIDENCE INTERVAL
~~~~~~~~~~~~~~~~~~~
The interval is built from the F distribution, and it is exact
for this table rather than asymptotic. With R = MS_targets/MS_error the
non-central parameter drops out -- the ratio of two mean squares is

    R = theta * F(n-1, (n-1)(k-1)),  theta = 1 + k*A,  A = sigma^2_t/sigma^2_e

so inverting the two-sided F test of ``theta`` gives

    theta_low  = R / F(1 - alpha/2, n-1, (n-1)(k-1))
    theta_high = R / F(alpha/2,     n-1, (n-1)(k-1))

whose coverage is exactly ``1 - alpha``, and A = (theta - 1)/k maps
monotonically onto the ICC scale. ICC(3,1) uses this interval directly.
ICC(2,1) also needs the rater component, ``B = sigma^2_c/sigma^2_e``,
estimated from ``(MS_raters - MS_error)/MS_error`` in exactly the same
way; the two mean squares share MS_error, so their intervals are not
independent and are combined at ``alpha/2`` each (Bonferroni), which
keeps joint coverage at or above 1 - alpha at the cost of width.
``tests/stats/test_design_tests.py`` verifies both by coverage
simulation against a known ICC rather than by comparing this module to
itself.

References
----------
Winer, B. J. 1971. Statistical Principles in Experimental Design, 2nd ed.
    McGraw-Hill. The reference for balanced factorial designs; the source
    of the sum-of-squares decomposition above.
Howell, S. 2013. Fundamental Statistics for the Behavioral Sciences, 8th
    ed. Cengage. Repeated-measures ANOVA, the two strata and their
    denominators.
Shrout, P. E. & Fleiss, J. L. 1979. Intraclass correlations: uses in
    assessing reproducibility. Psychological Bulletin 86: 420-428.
McGraw, K. O. & Wong, S. P. 1996. Forming inferences about some
    intraclass correlation coefficients. Psychological Methods 1: 30-46.
    The naming of the ICC forms implemented here.
Cohen, J. 1988. Statistical Power Analysis for the Behavioral Sciences,
    2nd ed. Lawrence Erlbaum. Effect-size benchmarks.
Faraone, T. V. 2001. Multitiered meta-analysis of the intraclass
    correlation coefficient. Psychological Methods 6: 54-67. On which ICC
    form suits which design.
Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from stats.univariate import partial_eta_squared
from utils.exceptions import (
    ComputationError,
    DataValidationError,
    MatrixDimensionError,
    ValidationError,
)
from utils.statistics_core import permutation_pvalue

logger = logging.getLogger(__name__)

__all__ = [
    "VALID_ICC_FORMS",
    "ANOVATerm",
    "ContingencyResult",
    "DesignTestAnalyzer",
    "ICCResult",
    "RepeatedMeasuresResult",
    "TwoWayANOVAResult",
]

VALID_ICC_FORMS = ("2,1", "3,1")
_NAN = float("nan")


# =============================================================================
# Input normalisation
# =============================================================================


def _as_label_array(labels: Sequence[Any], n_rows: int, name: str) -> npt.NDArray:
    """Return one factor label per row of the response.

    A short label array silently truncates the data (every design test in
    ``stats/univariate.py`` indexes the response with a per-row label, so
    the tail of the data set just disappears) and a long one raises a bare
    ``IndexError`` from inside the loop. Both are refused here, matching
    ``stats.univariate._check_group_length``.

    Raises
    ------
    MatrixDimensionError
        If the label count does not match the response length.
    DataValidationError
        If a label is missing (NaN), because a missing factor level is a
        data error rather than an extra level.
    """
    array = np.asarray(labels, dtype=object).ravel()
    if array.size != n_rows:
        raise MatrixDimensionError(
            f"{name}: got {array.size} labels for {n_rows} observations; a "
            f"factor must supply exactly one label per observation",
            details={"n_labels": int(array.size), "n_observations": int(n_rows)},
        )
    for position, label in enumerate(array):
        if isinstance(label, float) and np.isnan(label):
            raise DataValidationError(
                f"{name}: observation {position} has no factor label. "
                f"Every observation must be assigned to a level.",
                details={"position": int(position)},
            )
    return array


def _as_1d_response(
    values: npt.NDArray, name: str, allow_nan: bool = True
) -> npt.NDArray:
    """Flatten the response to one finite float per row.

    A two-dimensional input is accepted only when it carries a single
    variable (``n x 1``), which is the shape a one-variable column read
    from a data table arrives in. A wider matrix is refused rather than
    flattened: silently pooling several variables into one response turns
    a typing mistake into a wrong answer.

    Non-finite rows are dropped (and logged) for the designs that tolerate
    an incomplete cell, and passed through for the designs that do not.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise MatrixDimensionError(
                f"{name}: expected one response column, got an array of shape "
                f"{array.shape}. Pass a single column or a 1-D array; "
                f"pooling several variables into one response would "
                f"silently change the analysis.",
                details={"shape": tuple(int(x) for x in array.shape)},
            )
        array = array.ravel()
    elif array.ndim != 1:
        raise MatrixDimensionError(
            f"{name}: expected a 1-D response, got an array of shape "
            f"{array.shape}",
            details={"shape": tuple(int(x) for x in array.shape)},
        )
    if array.size == 0:
        raise DataValidationError(f"{name}: no observations supplied")

    if not allow_nan:
        if not np.all(np.isfinite(array)):
            bad = int(np.sum(~np.isfinite(array)))
            raise DataValidationError(
                f"{name}: {bad} observation(s) are missing or infinite. "
                f"This design needs one value per observation; drop the "
                f"incomplete rows explicitly so the choice is visible.",
                details={"n_invalid": bad, "n_observations": int(array.size)},
            )
        return array

    if not np.all(np.isfinite(array)):
        bad = int(np.sum(~np.isfinite(array)))
        logger.warning("%s: dropping %d missing/infinite observation(s)", name, bad)
        array = array[np.isfinite(array)]
    return array


def _ordered_levels(labels: npt.NDArray) -> list[Any]:
    """Distinct factor levels in order of first appearance."""
    return list(dict.fromkeys(labels.tolist()))


def _f_sf(f_statistic: float, df1: int, df2: int) -> float:
    """Upper-tail p-value of the F distribution, guarding df <= 0."""
    if df1 <= 0 or df2 <= 0 or not np.isfinite(f_statistic):
        return _NAN
    if f_statistic < 0:
        return _NAN
    return float(sp_stats.f.sf(f_statistic, df1, df2))


# =============================================================================
# Results
# =============================================================================


@dataclass
class ANOVATerm:
    """One tested term of an ANOVA, with everything needed to report it.

    Attributes:
        name: Term label, e.g. ``"A:B"`` or ``"measurement"``.
        ss: Sum of squares.
        df: Degrees of freedom of the numerator.
        ms: ``ss / df``, NaN when ``df == 0``.
        f_statistic: ``ms`` over its denominator mean square.
        p_value: Upper-tail p-value from the F distribution.
        permutation_p_value: From permuting the response, NaN when no
            permutation test was requested.
        eta_squared: ``ss / ss_total`` -- the classical effect size.
        partial_eta_squared: ``ss / (ss + ss_error)``, the effect size of
            the term against the error term alone.
    """

    name: str
    ss: float
    df: int
    ms: float
    f_statistic: float
    p_value: float
    permutation_p_value: float = _NAN
    eta_squared: float = 0.0
    partial_eta_squared: float = 0.0

    @property
    def significant(self) -> bool:
        """Whether the term rejects at the 0.05 level."""
        return np.isfinite(self.p_value) and self.p_value < 0.05

    def summary(self) -> str:
        """Generate summary text for one row of the ANOVA table."""
        permutation = (
            ""
            if not np.isfinite(self.permutation_p_value)
            else _(", perm p = {0}").format(f"{self.permutation_p_value:.4f}")
        )
        return (
            f"{self.name:<18} SS = {self.ss:>12.4f}  df = {self.df:>4}  "
            f"MS = {self.ms:>10.4f}  F = {self.f_statistic:>8.4f}  "
            f"p = {self.p_value:.4f}{permutation}  "
            f"eta2 = {self.eta_squared:.4f}  eta2p = {self.partial_eta_squared:.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of one term."""
        return {
            "name": self.name,
            "ss": float(self.ss),
            "df": int(self.df),
            "ms": float(self.ms),
            "f_statistic": float(self.f_statistic),
            "p_value": float(self.p_value),
            "permutation_p_value": float(self.permutation_p_value),
            "eta_squared": float(self.eta_squared),
            "partial_eta_squared": float(self.partial_eta_squared),
            "significant": self.significant,
        }


@dataclass
class TwoWayANOVAResult:
    """Outcome of a two-way factorial ANOVA with interaction.

    Attributes:
        term_a / term_b / term_interaction: The three tested terms.
        ss_total / df_total: Total variation, ``N - 1``.
        ss_error / df_error: Residual, ``N - a*b``.
        grand_mean: Mean of every retained observation.
        n_obs: Number of observations the decomposition used.
        n_levels_a / n_levels_b: Level counts of the two factors.
        n_missing: Non-finite observations dropped before the analysis.
        balanced: Whether every cell holds the same number of observations.
        n_permutations: Permutations actually run.
        random_seed: Seed used, or ``None``.
    """

    term_a: ANOVATerm
    term_b: ANOVATerm
    term_interaction: ANOVATerm
    ss_total: float
    df_total: int
    ss_error: float
    df_error: int
    grand_mean: float
    n_obs: int
    n_levels_a: int
    n_levels_b: int
    n_missing: int = 0
    balanced: bool = True
    n_permutations: int = 0
    random_seed: int | None = None

    @property
    def terms(self) -> tuple[ANOVATerm, ANOVATerm, ANOVATerm]:
        """The three terms, in reporting order."""
        return (self.term_a, self.term_b, self.term_interaction)

    @property
    def significant_terms(self) -> list[str]:
        """Names of the terms that reject at the 0.05 level."""
        return [t.name for t in self.terms if t.significant]

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Two-Way ANOVA"),
            "=" * 72,
            _("Factors: {0} ({1} levels) x {2} ({3} levels)").format(
                self.term_a.name, self.n_levels_a, self.term_b.name, self.n_levels_b
            ),
            _("Observations: {0}, design: {1}").format(self.n_obs, "balanced" if self.balanced else "unbalanced"),
            _("Grand mean: {0}").format(f"{self.grand_mean:.6g}"),
            "",
            f"{'term':<18} {'SS':>12} {'df':>4} {'MS':>10} {'F':>8} {'p':>8}",
            "-" * 72,
        ]
        for term in self.terms:
            lines.append(term.summary())
        lines.append(
            f"{_('error'):<18} {self.ss_error:>12.4f} {self.df_error:>4} "
            f"{self.ss_error / self.df_error if self.df_error else _NAN:>10.4f}"
        )
        lines.append(f"{_('total'):<18} {self.ss_total:>12.4f} {self.df_total:>4}")
        lines.append("")
        lines.append(
            _("Permutation of the response (free exchangeability assumed): "
               "{0} permutation(s), seed {1}").format(self.n_permutations, self.random_seed)
        )
        if self.significant_terms:
            lines.append(_("Significant terms: {0}").format(", ".join(self.significant_terms)))
        else:
            lines.append(_("No term rejects at the 0.05 level."))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "term_a": self.term_a.to_dict(),
            "term_b": self.term_b.to_dict(),
            "term_interaction": self.term_interaction.to_dict(),
            "ss_total": float(self.ss_total),
            "df_total": int(self.df_total),
            "ss_error": float(self.ss_error),
            "df_error": int(self.df_error),
            "grand_mean": float(self.grand_mean),
            "n_obs": int(self.n_obs),
            "n_levels_a": int(self.n_levels_a),
            "n_levels_b": int(self.n_levels_b),
            "n_missing": int(self.n_missing),
            "balanced": bool(self.balanced),
            "significant_terms": self.significant_terms,
            "n_permutations": int(self.n_permutations),
            "random_seed": self.random_seed,
        }


@dataclass
class RepeatedMeasuresResult:
    """Outcome of a balanced repeated-measures ANOVA.

    Between-subject stratum: ``term_groups`` against ``term_subjects``.
    Within-subject stratum: ``term_measurement`` and ``term_interaction``
    against ``term_error``. The two strata are reported together because
    one df total without the other is how a repeated-measures analysis
    gets misread.

    Attributes:
        term_groups: Between-subject group effect.
        term_subjects: Subject-within-group variation, the denominator of
            the group effect. Its ``f_statistic`` is NaN by construction.
        term_measurement: Within-subject measurement (treatment) effect.
        term_interaction: Measurement x group interaction.
        term_error: Within-subject error, denominator of both
            within-subject effects.
        ss_total / df_total: Total variation, ``g*n*k - 1``.
        n_subjects: Distinct subjects.
        n_groups: Distinct groups.
        n_per_group: Subjects per group.
        n_measurements: Repeated measurements per subject.
        grand_mean: Mean of every observation.
    """

    term_groups: ANOVATerm
    term_subjects: ANOVATerm
    term_measurement: ANOVATerm
    term_interaction: ANOVATerm
    term_error: ANOVATerm
    ss_total: float
    df_total: int
    n_subjects: int
    n_groups: int
    n_per_group: int
    n_measurements: int
    grand_mean: float = 0.0

    @property
    def tested_terms(self) -> tuple[ANOVATerm, ANOVATerm, ANOVATerm]:
        """The three terms that have an F test."""
        return (self.term_groups, self.term_measurement, self.term_interaction)

    @property
    def significant_terms(self) -> list[str]:
        """Names of the tested terms that reject at the 0.05 level."""
        return [t.name for t in self.tested_terms if t.significant]

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Repeated-Measures ANOVA"),
            "=" * 72,
            _("Design: {0} subjects x {1} measurements in {2} group(s) "
               "({3} per group)").format(
                self.n_subjects, self.n_measurements, self.n_groups, self.n_per_group
            ),
            _("Grand mean: {0}").format(f"{self.grand_mean:.6g}"),
            "",
            _("Between-subject stratum:"),
            f"  {self.term_groups.summary()}",
            f"  {self.term_subjects.summary()}",
            _("Within-subject stratum:"),
            f"  {self.term_measurement.summary()}",
            f"  {self.term_interaction.summary()}",
            f"  {self.term_error.summary()}",
            "",
            f"{_('Total'):<18} SS = {self.ss_total:.4f}  df = {self.df_total}",
        ]
        if self.significant_terms:
            lines.append(_("Significant terms: {0}").format(", ".join(self.significant_terms)))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "term_groups": self.term_groups.to_dict(),
            "term_subjects": self.term_subjects.to_dict(),
            "term_measurement": self.term_measurement.to_dict(),
            "term_interaction": self.term_interaction.to_dict(),
            "term_error": self.term_error.to_dict(),
            "ss_total": float(self.ss_total),
            "df_total": int(self.df_total),
            "n_subjects": int(self.n_subjects),
            "n_groups": int(self.n_groups),
            "n_per_group": int(self.n_per_group),
            "n_measurements": int(self.n_measurements),
            "significant_terms": self.significant_terms,
        }


@dataclass
class ICCResult:
    """Outcome of an intraclass correlation coefficient.

    Attributes:
        form: ``"2,1"`` (absolute agreement) or ``"3,1"`` (consistency).
        icc: The coefficient.
        ci_lower / ci_upper: Confidence interval, clipped to [-1, 1].
        confidence: Level the interval was built at.
        f_statistic: ``MS_targets / MS_error``, the quantity the interval
            inverts.
        ms_targets / ms_raters / ms_error: The two-way mean squares.
        df_error: ``(n-1)(k-1)``.
        variance_components: Moment estimates of the variance components,
            with any negative estimate reported as 0 (see ``summary``).
        n_targets / n_raters: Table dimensions.
        interval_method: How the interval was constructed.
    """

    form: str
    icc: float
    ci_lower: float
    ci_upper: float
    confidence: float
    f_statistic: float
    ms_targets: float
    ms_raters: float
    ms_error: float
    df_error: int
    variance_components: dict[str, float]
    n_targets: int
    n_raters: int
    interval_method: str = ""

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        return self.ci_lower > 0.0

    def summary(self) -> str:
        """Generate summary text."""
        label = (
            _("ICC(2,1) -- two-way random, absolute agreement, single")
            if self.form == "2,1"
            else _("ICC(3,1) -- two-way mixed, consistency, single")
        )
        components = ", ".join(f"{k} = {v:.4g}" for k, v in self.variance_components.items())
        negative_note = (
            _("\nNote: at least one variance component was estimated below zero and "
               "reported as 0; the ICC point estimate is left unclipped because "
               "that bias is part of the estimator.")
            if any(v == 0.0 for v in self.variance_components.values())
            else ""
        )
        interval = _("95% CI: [{0}, {1}]").format(
            f"{self.ci_lower:.4f}", f"{self.ci_upper:.4f}"
        ).replace("95%", f"{round(self.confidence * 100)}%")
        mean_squares = _("MS targets = {0}, MS raters = {1}, MS error = {2}").format(
            f"{self.ms_targets:.4g}", f"{self.ms_raters:.4g}", f"{self.ms_error:.4g}"
        )
        return (
            f"{_('Intraclass Correlation')}\n"
            f"{'=' * 40}\n"
            f"{label}\n"
            f"{_('ICC: {0}').format(f'{self.icc:.4f}')}\n"
            f"{interval}\n"
            f"{_('Targets: {0}, raters: {1}').format(self.n_targets, self.n_raters)}\n"
            f"{mean_squares}\n"
            f"{_('Variance components: {0}').format(components)}{negative_note}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "form": self.form,
            "icc": float(self.icc),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "confidence": float(self.confidence),
            "f_statistic": float(self.f_statistic),
            "ms_targets": float(self.ms_targets),
            "ms_raters": float(self.ms_raters),
            "ms_error": float(self.ms_error),
            "df_error": int(self.df_error),
            "variance_components": {k: float(v) for k, v in self.variance_components.items()},
            "n_targets": int(self.n_targets),
            "n_raters": int(self.n_raters),
            "significant": self.significant,
        }


@dataclass
class ContingencyResult:
    """Outcome of a contingency-table chi-square test.

    Attributes:
        chi_square / p_value / dof: The test.
        cramers_v: ``sqrt(chi_square / (N * (min(r, c) - 1)))``.
        expected: Expected counts, row by row.
        min_expected: Smallest expected count; below 5 the chi-square
            approximation is weak.
        standardized_residuals: ``(observed - expected)/sqrt(expected)``.
        n_obs: Table total.
        labels_rows / labels_cols: Axis labels.
    """

    chi_square: float
    p_value: float
    dof: int
    cramers_v: float
    expected: list[list[float]]
    standardized_residuals: list[list[float]]
    min_expected: float
    n_obs: int
    labels_rows: list[str] = field(default_factory=list)
    labels_cols: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Whether the table rejects independence at the 0.05 level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Generate summary text."""
        warning = (
            _("\nWarning: a cell has an expected count below 5; the chi-square "
               "approximation is unreliable here.")
            if self.min_expected < 5.0
            else ""
        )
        totals = _("N: {0}, smallest expected count: {1}").format(
            self.n_obs, f"{self.min_expected:.2f}"
        )
        return (
            f"{_('Contingency Chi-Square')}\n"
            f"{'=' * 40}\n"
            f"{_('Chi-square: {0}').format(f'{self.chi_square:.4f}')}\n"
            f"{_('Degrees of freedom: {0}').format(self.dof)}\n"
            f"{_('p-value: {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('Cramer V: {0}').format(f'{self.cramers_v:.4f}')}\n"
            f"{totals}{warning}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "chi_square": float(self.chi_square),
            "p_value": float(self.p_value),
            "dof": int(self.dof),
            "cramers_v": float(self.cramers_v),
            "expected": [[float(v) for v in row] for row in self.expected],
            "standardized_residuals": [[float(v) for v in row] for row in self.standardized_residuals],
            "min_expected": float(self.min_expected),
            "n_obs": int(self.n_obs),
            "significant": self.significant,
        }


# =============================================================================
# Interval helpers
# =============================================================================


def _f_ratio_interval(
    observed: float, df1: int, df2: int, alpha: float
) -> tuple[float, float]:
    """Interval for a ratio of two independent mean squares.

    ``MS1 = m1 * chi2(df1)/df1`` and ``MS2 = m2 * chi2(df2)/df2``
    independently, so ``MS1/MS2 = (m1/m2) * F(df1, df2)`` and inverting
    the two-sided F test of ``m1/m2 = theta`` gives the interval below.
    Its coverage is exactly ``1 - alpha``, not approximately, because
    ``MS1/MS2 / theta`` *is* an F variate.
    """
    if not np.isfinite(observed) or observed < 0 or df1 <= 0 or df2 <= 0:
        return (0.0, float("inf"))
    if observed == 0:
        return (0.0, float("inf"))
    lower = observed / float(sp_stats.f.ppf(1.0 - alpha / 2.0, df1, df2))
    upper = observed / float(sp_stats.f.ppf(alpha / 2.0, df1, df2))
    return (lower, upper)


def _component_bounds(
    observed_ratio: float, df1: int, df2: int, alpha: float, multiplier: float
) -> tuple[float, float]:
    """Interval for ``A = sigma^2_between / sigma^2_error``, floored at 0.

    The variance ratio is read off a mean-square ratio as
    ``A = (theta - 1) / multiplier`` with ``theta`` from
    ``_f_ratio_interval``; a negative lower bound is floored at 0 because
    a variance cannot be negative, which widens the interval rather than
    narrowing it.
    """
    lower, upper = _f_ratio_interval(observed_ratio, df1, df2, alpha)
    a_lower = max((lower - 1.0) / multiplier, 0.0)
    a_upper = max((upper - 1.0) / multiplier, 0.0)
    return (a_lower, a_upper)


# =============================================================================
# Analyzer
# =============================================================================


class DesignTestAnalyzer:
    """Engine for factorial, repeated-measures and reliability tests.

    One class rather than three: the three analyses share the input
    normalisation above, the same effect-size helpers, the same result
    containers and the same ``last_result`` slot, so splitting them would
    mean three copies of the label validation rather than three analyses.
    This mirrors ``MantelAnalyzer``, which likewise holds both the plain
    and the partial Mantel test.
    """

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.DesignTestAnalyzer")
        self._lock = threading.RLock()
        self._last_result: TwoWayANOVAResult | RepeatedMeasuresResult | ICCResult | ContingencyResult | None = None

    @property
    def last_result(self) -> TwoWayANOVAResult | RepeatedMeasuresResult | ICCResult | ContingencyResult | None:
        """The most recent result this analyzer produced."""
        return self._last_result

    # -- two-way ANOVA ----------------------------------------------------

    def two_way_anova(
        self,
        values: npt.NDArray,
        factor_a: Sequence[Any],
        factor_b: Sequence[Any],
        *,
        factor_a_name: str = "A",
        factor_b_name: str = "B",
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> TwoWayANOVAResult:
        """Test two factors and their interaction.

        Parameters
        ----------
        values:
            One response per observation. A ``(n, 1)`` array is accepted;
            wider matrices are refused.
        factor_a, factor_b:
            One label per observation.
        factor_a_name, factor_b_name:
            Labels for the report, e.g. ``"lithology"``, ``"basin"``.
        n_permutations:
            Permutations of the response used for the permutation
            p-values. ``0`` reports only the F-distribution p-values.
        random_seed:
            Seed for the permutation test.

        Returns
        -------
        TwoWayANOVAResult

        Raises
        ------
        MatrixDimensionError
            If a label array does not match the response length, or the
            response is a matrix rather than a single column.
        DataValidationError
            If a label is missing, or dropping non-finite values leaves a
            cell empty.
        ValidationError
            If a factor has fewer than two levels, or the design leaves
            no residual degrees of freedom (fewer than two observations
            per cell).
        ComputationError
            If every retained value is identical.
        """
        with self._lock:
            raw = np.asarray(values, dtype=float)
            n_input = int(raw.shape[0]) if raw.ndim > 1 else int(raw.size)
            labels_a = _as_label_array(factor_a, n_input, factor_a_name)
            labels_b = _as_label_array(factor_b, n_input, factor_b_name)
            response = _as_1d_response(raw, factor_a_name)
            finite = np.isfinite(raw.ravel())
            n_missing = n_input - int(np.sum(finite))
            if n_missing:
                labels_a = labels_a[finite]
                labels_b = labels_b[finite]

            levels_a = _ordered_levels(labels_a)
            levels_b = _ordered_levels(labels_b)
            for levels, name in ((levels_a, factor_a_name), (levels_b, factor_b_name)):
                if len(levels) < 2:
                    raise ValidationError(
                        f"Two-way ANOVA needs at least 2 levels of {name}, got "
                        f"{len(levels)}: {levels}. A factor with one level has no "
                        f"degrees of freedom to test.",
                        details={"factor": name, "n_levels": len(levels)},
                    )

            if np.ptp(response) == 0:
                raise ComputationError(
                    f"Two-way ANOVA is undefined for a constant response "
                    f"(every value is {response[0]}): there is no variation to "
                    f"decompose. Check whether the column was selected correctly.",
                    details={"constant_value": float(response[0])},
                )

            ss, df, counts = _two_way_sums_of_squares(response, labels_a, labels_b, levels_a, levels_b)
            n_obs = int(response.size)
            n_a, n_b = len(levels_a), len(levels_b)
            df_error = n_obs - n_a * n_b
            if df_error <= 0:
                empty = sorted(
                    f"{a} x {b}" for a in levels_a for b in levels_b if counts[a][b] == 0
                )
                raise DataValidationError(
                    f"Two-way ANOVA needs at least 2 observations per cell to "
                    f"estimate the residual: {n_obs} observations in {n_a}x{n_b} "
                    f"cells leaves {df_error} residual df. Dropping non-finite "
                    f"values emptied: {', '.join(empty) if empty else 'no cell'}.",
                    details={
                        "n_obs": n_obs,
                        "n_cells": n_a * n_b,
                        "df_error": df_error,
                        "empty_cells": empty,
                    },
                )

            ms_error = ss["error"] / df_error
            permutation_p = _two_way_permutation_p(
                response,
                labels_a,
                labels_b,
                levels_a,
                levels_b,
                factor_a_name,
                factor_b_name,
                n_permutations,
                random_seed,
            )

            terms: dict[str, ANOVATerm] = {}
            for key, df_term, name in (
                ("A", df["A"], factor_a_name),
                ("B", df["B"], factor_b_name),
                ("AB", df["AB"], f"{factor_a_name}:{factor_b_name}"),
            ):
                ms = ss[key] / df_term if df_term else _NAN
                f_stat = ms / ms_error if (np.isfinite(ms) and ms_error > 0) else _NAN
                terms[key] = ANOVATerm(
                    name=name,
                    ss=float(ss[key]),
                    df=int(df_term),
                    ms=float(ms),
                    f_statistic=float(f_stat),
                    p_value=_f_sf(float(f_stat), df_term, df_error),
                    permutation_p_value=permutation_p.get(key, _NAN),
                    eta_squared=float(ss[key] / ss["total"]) if ss["total"] > 0 else 0.0,
                    partial_eta_squared=(
                        float(partial_eta_squared(f_stat, df_term, df_error))
                        if (np.isfinite(f_stat) and df_term > 0)
                        else 0.0
                    ),
                )

            result = TwoWayANOVAResult(
                term_a=terms["A"],
                term_b=terms["B"],
                term_interaction=terms["AB"],
                ss_total=float(ss["total"]),
                df_total=n_obs - 1,
                ss_error=float(ss["error"]),
                df_error=df_error,
                grand_mean=float(np.mean(response)),
                n_obs=n_obs,
                n_levels_a=n_a,
                n_levels_b=n_b,
                n_missing=n_missing,
                balanced=bool(len({c for row in counts.values() for c in row.values()}) == 1),
                n_permutations=int(n_permutations),
                random_seed=random_seed,
            )
            self._last_result = result
            self._logger.info(
                "Two-way ANOVA completed: N=%d, F(%s)=%.4f, F(%s)=%.4f, F(%s)=%.4f",
                result.n_obs,
                result.term_a.name,
                result.term_a.f_statistic,
                result.term_b.name,
                result.term_b.f_statistic,
                result.term_interaction.name,
                result.term_interaction.f_statistic,
            )
            return result

    # -- repeated measures ------------------------------------------------

    def repeated_measures_anova(
        self,
        values: npt.NDArray,
        subject: Sequence[Any],
        group: Sequence[Any],
        *,
        measurement: Sequence[Any] | None = None,
        group_name: str = "group",
        subject_name: str = "subject",
        measurement_name: str = "measurement",
    ) -> RepeatedMeasuresResult:
        """Analyse a balanced repeated-measures design.

        Parameters
        ----------
        values:
            One response per subject-measurement observation.
        subject:
            Label per observation; identifies the repeated subject.
        group:
            Between-subject label per observation.
        measurement:
            Which of the repeated measurements each observation is. When
            omitted, the measurements are taken positionally: the i-th
            occurrence of a subject in ``values`` is that subject's i-th
            measurement. Pass the label whenever the rows are not already
            in per-subject time order.
        group_name, subject_name, measurement_name:
            Labels for the report.

        Returns
        -------
        RepeatedMeasuresResult

        Raises
        ------
        MatrixDimensionError
            If a label array does not match the response length, if the
            two factor labels disagree about the design, or if a subject
            has an incomplete measurement series. An unbalanced design is
            refused rather than repaired: averaging the measurements a
            subject does have silently changes both strata, and the
            resulting F values look entirely ordinary.
        ValidationError
            If there are fewer than two subjects, groups or measurements.
        DataValidationError
            If a label is missing or a response value is not finite.
        ComputationError
            If every value is identical.
        """
        with self._lock:
            raw = np.asarray(values, dtype=float)
            n_input = int(raw.shape[0]) if raw.ndim > 1 else int(raw.size)
            subjects = _as_label_array(subject, n_input, subject_name)
            groups = _as_label_array(group, n_input, group_name)
            response = _as_1d_response(raw, subject_name, allow_nan=False)

            if np.ptp(response) == 0:
                raise ComputationError(
                    f"Repeated-measures ANOVA is undefined for a constant "
                    f"response (every value is {response[0]}): there is no "
                    f"variation to decompose.",
                    details={"constant_value": float(response[0])},
                )

            subject_levels = _ordered_levels(subjects)
            group_levels = _ordered_levels(groups)
            if len(subject_levels) < 2:
                raise ValidationError(
                    f"Repeated-measures ANOVA needs at least 2 subjects, got "
                    f"{len(subject_levels)}",
                    details={"n_subjects": len(subject_levels)},
                )

            subject_group: dict[Any, Any] = {}
            for s, g in zip(subjects, groups, strict=True):
                if s in subject_group and subject_group[s] != g:
                    raise MatrixDimensionError(
                        f"{subject_name} {s!r} appears in both {group_name} "
                        f"{subject_group[s]!r} and {group_name} {g!r}; a subject "
                        f"cannot belong to two groups.",
                        details={"subject": str(s)},
                    )
                subject_group[s] = g

            if measurement is None:
                # Positional fallback: the i-th occurrence of a subject is
                # its i-th measurement. Only valid when the rows are
                # already ordered per subject, which is why the label can
                # be supplied instead.
                occurrence: dict[Any, int] = {}
                measurement_labels = np.empty(response.size, dtype=object)
                for i, s in enumerate(subjects):
                    measurement_labels[i] = occurrence.get(s, 0)
                    occurrence[s] = int(measurement_labels[i]) + 1
            else:
                measurement_labels = _as_label_array(measurement, n_input, measurement_name)

            measurement_levels = _ordered_levels(measurement_labels)
            n_measurements = len(measurement_levels)
            if n_measurements < 2:
                raise ValidationError(
                    f"Repeated-measures ANOVA needs at least 2 measurements per "
                    f"subject, got {n_measurements}",
                    details={"n_measurements": n_measurements},
                )

            expected = set(measurement_levels)
            incomplete = sorted(
                str(s)
                for s in subject_levels
                if set(measurement_labels[subjects == s]) != expected
            )
            if incomplete:
                raise MatrixDimensionError(
                    f"Repeated-measures ANOVA requires every subject to carry "
                    f"all {n_measurements} measurements; {len(incomplete)} "
                    f"subject(s) do not: {', '.join(incomplete[:10])}. Complete "
                    f"the series or drop those subjects explicitly -- averaging "
                    f"what a subject has would change both strata without "
                    f"changing the row count.",
                    details={
                        "n_incomplete": len(incomplete),
                        "expected_measurements": n_measurements,
                        "incomplete": incomplete[:10],
                    },
                )

            # Order observations subject-major so each subject's k
            # measurements occupy one contiguous block, in the order the
            # measurement levels were first seen, then reshape.
            subject_position = {s: i for i, s in enumerate(subject_levels)}
            measurement_position = {m: j for j, m in enumerate(measurement_levels)}
            keys = np.asarray(
                [subject_position[s] for s in subjects], dtype=int
            ) * n_measurements + np.asarray(
                [measurement_position[m] for m in measurement_labels], dtype=int
            )
            matrix = response[np.argsort(keys, kind="stable")].reshape(
                len(subject_levels), n_measurements
            )
            group_by_subject = np.asarray(
                [subject_group[s] for s in subject_levels], dtype=object
            )
            sizes = np.asarray([np.sum(group_by_subject == g) for g in group_levels], dtype=int)
            if np.unique(sizes).size != 1:
                raise MatrixDimensionError(
                    f"{group_name} sizes differ "
                    f"({dict(zip(group_levels, sizes.tolist(), strict=True))}); this "
                    f"implementation requires a balanced design. An unbalanced "
                    f"repeated-measures design needs the univariate or "
                    f"multivariate missing-data machinery, not a Type I "
                    f"decomposition.",
                    details={
                        "group_sizes": {
                            str(k): int(v) for k, v in zip(group_levels, sizes, strict=True)
                        },
                        "group_sizes_per_group": int(sizes[0]) if sizes.size else 0,
                    },
                )
            n_per_group = int(sizes[0])
            g_count, k_count = len(group_levels), n_measurements

            ss, df = _repeated_measures_sums_of_squares(matrix, group_by_subject, n_per_group)
            grand = float(np.mean(response))

            def _term(name: str, key: str, denominator_ss: str, denominator_df: int) -> ANOVATerm:
                df_num = df[key]
                ms = ss[key] / df_num if df_num else _NAN
                ms_den = ss[denominator_ss] / denominator_df if denominator_df > 0 else _NAN
                f_stat = ms / ms_den if (np.isfinite(ms) and np.isfinite(ms_den) and ms_den > 0) else _NAN
                return ANOVATerm(
                    name=name,
                    ss=float(ss[key]),
                    df=int(df_num),
                    ms=float(ms),
                    f_statistic=float(f_stat),
                    p_value=_f_sf(float(f_stat), df_num, denominator_df),
                    eta_squared=float(ss[key] / ss["total"]) if ss["total"] > 0 else 0.0,
                    partial_eta_squared=(
                        float(partial_eta_squared(f_stat, df_num, denominator_df))
                        if (np.isfinite(f_stat) and df_num > 0 and denominator_df >= 0)
                        else 0.0
                    ),
                )

            result = RepeatedMeasuresResult(
                term_groups=_term(group_name, "groups", "subjects", df["subjects"]),
                term_subjects=ANOVATerm(
                    name=f"{subject_name}(within {group_name})",
                    ss=float(ss["subjects"]),
                    df=int(df["subjects"]),
                    ms=float(ss["subjects"] / df["subjects"]) if df["subjects"] else _NAN,
                    f_statistic=_NAN,
                    p_value=_NAN,
                    eta_squared=float(ss["subjects"] / ss["total"]) if ss["total"] > 0 else 0.0,
                    partial_eta_squared=0.0,
                ),
                term_measurement=_term(
                    measurement_name, "measurement", "error", df["error"]
                ),
                term_interaction=_term(
                    f"{measurement_name}:{group_name}", "interaction", "error", df["error"]
                ),
                term_error=ANOVATerm(
                    name=f"{subject_name}:{measurement_name} (error)",
                    ss=float(ss["error"]),
                    df=int(df["error"]),
                    ms=float(ss["error"] / df["error"]) if df["error"] else _NAN,
                    f_statistic=_NAN,
                    p_value=_NAN,
                ),
                ss_total=float(ss["total"]),
                df_total=g_count * n_per_group * k_count - 1,
                n_subjects=len(subject_levels),
                n_groups=g_count,
                n_per_group=n_per_group,
                n_measurements=k_count,
                grand_mean=grand,
            )
            self._last_result = result
            self._logger.info(
                "Repeated-measures ANOVA completed: %d subjects x %d measurements "
                "in %d groups; F(%s)=%.4f, F(%s)=%.4f",
                result.n_subjects,
                result.n_measurements,
                result.n_groups,
                result.term_groups.name,
                result.term_groups.f_statistic,
                result.term_measurement.name,
                result.term_measurement.f_statistic,
            )
            return result

    # -- intraclass correlation -------------------------------------------

    def intraclass_correlation(
        self,
        values: npt.NDArray,
        targets: Sequence[Any] | None = None,
        groups: Sequence[Any] | None = None,
        *,
        form: str = "2,1",
        confidence: float = 0.95,
    ) -> ICCResult:
        """Measure how much of the variation is between targets.

        Parameters
        ----------
        values:
            Either a ``(n_targets, n_raters)`` matrix of measurements, or a
            long 1-D response accompanied by ``targets`` and ``groups``.
        targets:
            Target label per observation (long form) or per row (matrix
            form, labels only).
        groups:
            Rater label per observation; required in the long form.
        form:
            ``"2,1"`` absolute agreement or ``"3,1"`` consistency.
        confidence:
            Level of the interval.

        Returns
        -------
        ICCResult

        Raises
        ------
        ValidationError
            For an unknown ``form`` or a confidence level outside (0, 1).
        MatrixDimensionError
            If the long form does not give exactly one measurement per
            target-rater pair, or the table would have fewer than two
            rows or columns.
        DataValidationError
            If a measurement is missing.
        ComputationError
            If the table carries no variation at all.
        """
        with self._lock:
            if form not in VALID_ICC_FORMS:
                raise ValidationError(
                    f"form must be one of {VALID_ICC_FORMS}, got {form!r}",
                    details={"form": form},
                )
            if not 0.0 < confidence < 1.0:
                raise ValidationError(
                    f"confidence must lie strictly between 0 and 1, got {confidence}",
                    details={"confidence": confidence},
                )

            matrix, n_targets, n_raters = self._icc_table(values, targets, groups)

            if np.ptp(matrix) == 0:
                raise ComputationError(
                    f"Intraclass correlation is undefined for a constant table "
                    f"(every value is {matrix.flat[0]}): there is no variation "
                    f"to attribute.",
                    details={"constant_value": float(matrix.flat[0])},
                )

            grand = float(np.mean(matrix))
            row_means = np.mean(matrix, axis=1)
            col_means = np.mean(matrix, axis=0)
            ss_targets = float(n_raters * np.sum((row_means - grand) ** 2))
            ss_raters = float(n_targets * np.sum((col_means - grand) ** 2))
            ss_total = float(np.sum((matrix - grand) ** 2))
            df_error = (n_targets - 1) * (n_raters - 1)
            ss_error = ss_total - ss_targets - ss_raters
            if ss_error < 0 and ss_error > -1e-9 * max(ss_total, 1.0):
                ss_error = 0.0
            ms_targets = ss_targets / (n_targets - 1)
            ms_raters = ss_raters / (n_raters - 1)
            ms_error = ss_error / df_error

            if ms_error <= 0:
                # No residual variance left: the moment estimator has no
                # sampling error to bound, so the interval collapses onto
                # the point estimate. Reported rather than silently given
                # an arbitrary width.
                icc = self._icc_value(form, ms_targets, ms_raters, ms_error, n_targets, n_raters)
                self._last_result = ICCResult(
                    form=form,
                    icc=float(icc),
                    ci_lower=float(icc),
                    ci_upper=float(icc),
                    confidence=float(confidence),
                    f_statistic=float("inf") if ms_targets > 0 else _NAN,
                    ms_targets=float(ms_targets),
                    ms_raters=float(ms_raters),
                    ms_error=float(ms_error),
                    df_error=int(df_error),
                    variance_components={
                        "targets": float(max((ms_targets - ms_error) / n_raters, 0.0)),
                        "raters": float(max((ms_raters - ms_error) / n_targets, 0.0)),
                        "error": float(max(ms_error, 0.0)),
                    },
                    n_targets=int(n_targets),
                    n_raters=int(n_raters),
                    interval_method=_(
                        "degenerate: zero residual variance, the estimate has no "
                        "sampling error"
                    ),
                )
                return self._last_result

            alpha = 1.0 - confidence
            df1 = n_targets - 1
            if form == "3,1":
                a_lower, a_upper = _component_bounds(
                    ms_targets / ms_error, df1, df_error, alpha, float(n_raters)
                )
                ci_lower, ci_upper = a_lower / (1.0 + a_lower), a_upper / (1.0 + a_upper)
                interval_method = _(
                    "exact F inversion of MS_targets/MS_error, alpha = 0.05 split "
                    "into two tails"
                )
            else:
                # Two mean squares share MS_error, so their intervals are not
                # independent; each is built at alpha/2 and combined
                # (Bonferroni), which keeps joint coverage at or above
                # 1 - alpha at the cost of width.
                a_lower, a_upper = _component_bounds(
                    ms_targets / ms_error, df1, df_error, alpha / 2.0, float(n_raters)
                )
                b_lower, b_upper = _component_bounds(
                    (ms_raters - ms_error) / ms_error,
                    n_raters - 1,
                    df_error,
                    alpha / 2.0,
                    float(n_targets),
                )
                ci_lower = a_lower / (a_lower + b_upper + 1.0)
                ci_upper = a_upper / (a_upper + b_lower + 1.0)
                interval_method = _(
                    "exact F inversion of both mean-square ratios, Bonferroni at "
                    "alpha/2 because they share MS_error"
                )

            icc = self._icc_value(form, ms_targets, ms_raters, ms_error, n_targets, n_raters)
            result = ICCResult(
                form=form,
                icc=float(icc),
                ci_lower=float(np.clip(ci_lower, -1.0, 1.0)),
                ci_upper=float(np.clip(ci_upper, -1.0, 1.0)),
                confidence=float(confidence),
                f_statistic=float(ms_targets / ms_error),
                ms_targets=float(ms_targets),
                ms_raters=float(ms_raters),
                ms_error=float(ms_error),
                df_error=int(df_error),
                variance_components={
                    "targets": float(max((ms_targets - ms_error) / n_raters, 0.0)),
                    "raters": float(max((ms_raters - ms_error) / n_targets, 0.0)),
                    "error": float(ms_error),
                },
                n_targets=int(n_targets),
                n_raters=int(n_raters),
                interval_method=interval_method,
            )
            self._last_result = result
            self._logger.info(
                "ICC(%s) completed: n=%d, k=%d, ICC=%.4f, CI=[%.4f, %.4f]",
                form,
                result.n_targets,
                result.n_raters,
                result.icc,
                result.ci_lower,
                result.ci_upper,
            )
            return result

    def _icc_table(
        self,
        values: npt.NDArray,
        targets: Sequence[Any] | None,
        groups: Sequence[Any] | None,
    ) -> tuple[npt.NDArray, int, int]:
        """Build the ``(n_targets, n_raters)`` table from either input form."""
        array = np.asarray(values, dtype=float)

        if array.ndim == 1:
            if targets is None or groups is None:
                raise ValidationError(
                    "A long-form ICC needs both ``targets`` and ``groups``: one "
                    "label per observation saying which target was measured and "
                    "by whom.",
                    details={"has_targets": targets is not None, "has_groups": groups is not None},
                )
            labels_t = _as_label_array(targets, array.size, "targets")
            labels_g = _as_label_array(groups, array.size, "groups")
            bad = int(np.sum(~np.isfinite(array)))
            if bad:
                raise DataValidationError(
                    f"Intraclass correlation: {bad} measurement(s) are missing or "
                    f"infinite; the table must be complete.",
                    details={"n_invalid": bad},
                )
            t_levels = _ordered_levels(labels_t)
            g_levels = _ordered_levels(labels_g)
            table = np.full((len(t_levels), len(g_levels)), np.nan)
            t_index = {level: i for i, level in enumerate(t_levels)}
            g_index = {level: j for j, level in enumerate(g_levels)}
            for value, t_label, g_label in zip(array, labels_t, labels_g, strict=True):
                row, col = t_index[t_label], g_index[g_label]
                if np.isfinite(table[row, col]):
                    raise MatrixDimensionError(
                        f"Intraclass correlation: target {t_label!r} was measured "
                        f"more than once by {g_label!r}; a target-rater cell must "
                        f"hold exactly one measurement.",
                        details={"target": str(t_label), "group": str(g_label)},
                    )
                table[row, col] = value
            matrix = table
        elif array.ndim == 2:
            matrix = array
            if not np.all(np.isfinite(matrix)):
                bad = int(np.sum(~np.isfinite(matrix)))
                raise DataValidationError(
                    f"Intraclass correlation: {bad} cell(s) are missing or "
                    f"infinite; the table must be complete.",
                    details={"n_invalid": bad},
                )
            if targets is not None:
                labels = _as_label_array(targets, matrix.shape[0], "targets")
                if len(_ordered_levels(labels)) != matrix.shape[0]:
                    raise ValidationError(
                        "targets must be unique per row of the measurement matrix",
                        details={"n_rows": int(matrix.shape[0])},
                    )
        else:
            raise MatrixDimensionError(
                f"Intraclass correlation: expected a (targets x raters) matrix or "
                f"a 1-D response, got shape {array.shape}",
                details={"shape": tuple(int(x) for x in array.shape)},
            )

        n_targets, n_raters = int(matrix.shape[0]), int(matrix.shape[1])
        if n_targets < 2 or n_raters < 2:
            raise MatrixDimensionError(
                f"Intraclass correlation needs at least 2 targets and 2 raters "
                f"to have a residual mean square, got a {n_targets}x{n_raters} "
                f"table.",
                details={"n_targets": n_targets, "n_raters": n_raters},
            )
        if np.ptp(matrix) == 0:
            raise ComputationError(
                f"Intraclass correlation is undefined for a constant table "
                f"(every value is {matrix.flat[0]})",
                details={"constant_value": float(matrix.flat[0])},
            )
        return matrix, n_targets, n_raters

    @staticmethod
    def _icc_value(
        form: str, ms_targets: float, ms_raters: float, ms_error: float, n_targets: int, n_raters: int
    ) -> float:
        """The two implemented ICC forms."""
        if form == "3,1":
            denominator = ms_targets + (n_raters - 1) * ms_error
        else:
            denominator = (
                ms_targets
                + (n_raters - 1) * ms_error
                + n_raters * (ms_raters - ms_error) / n_targets
            )
        if denominator == 0:
            return _NAN
        return float((ms_targets - ms_error) / denominator)

    # -- contingency chi-square -------------------------------------------

    def contingency_chi_square(
        self,
        table: npt.NDArray,
        *,
        labels_rows: Sequence[str] | None = None,
        labels_cols: Sequence[str] | None = None,
    ) -> ContingencyResult:
        """Test independence in a contingency table.

        Parameters
        ----------
        table:
            Observed counts, one row per category of the first variable.
        labels_rows, labels_cols:
            Axis labels for the report.

        Returns
        -------
        ContingencyResult

        Raises
        ------
        MatrixDimensionError
            If the table is not 2-D with at least two rows and columns.
        DataValidationError
            If a count is negative or not finite.
        ValidationError
            If the table totals zero.
        """
        with self._lock:
            observed = np.asarray(table, dtype=float)
            if observed.ndim != 2 or observed.shape[0] < 2 or observed.shape[1] < 2:
                raise MatrixDimensionError(
                    f"Contingency chi-square needs a table of at least 2x2, got "
                    f"shape {observed.shape}",
                    details={"shape": tuple(int(x) for x in observed.shape)},
                )
            if not np.all(np.isfinite(observed)) or np.any(observed < 0):
                raise DataValidationError(
                    "Contingency counts must be finite and non-negative",
                    details={"min": float(np.min(observed))},
                )
            total = float(np.sum(observed))
            if total <= 0:
                raise ValidationError(
                    "Contingency table totals zero; there is nothing to test",
                    details={"total": total},
                )

            row_totals = np.sum(observed, axis=1, keepdims=True)
            col_totals = np.sum(observed, axis=0, keepdims=True)
            expected = row_totals @ col_totals / total
            with np.errstate(divide="ignore", invalid="ignore"):
                chi_square = float(np.sum((observed - expected) ** 2 / expected))
            dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
            p_value = float(sp_stats.chi2.sf(chi_square, dof))
            cramers_v = float(np.sqrt(chi_square / (total * (min(observed.shape) - 1))))
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals = (observed - expected) / np.sqrt(expected)

            result = ContingencyResult(
                chi_square=chi_square,
                p_value=p_value,
                dof=int(dof),
                cramers_v=cramers_v,
                expected=[[float(v) for v in row] for row in expected],
                standardized_residuals=[[float(v) for v in row] for row in residuals],
                min_expected=float(np.min(expected)),
                n_obs=int(total),
                labels_rows=[str(v) for v in labels_rows] if labels_rows is not None else [],
                labels_cols=[str(v) for v in labels_cols] if labels_cols is not None else [],
            )
            self._last_result = result
            self._logger.info(
                "Contingency chi-square completed: chi2=%.4f, p=%.4f, V=%.4f",
                result.chi_square,
                result.p_value,
                result.cramers_v,
            )
            return result


# =============================================================================
# Decompositions
# =============================================================================


def _additive_fit(cell_means: npt.NDArray, cells: npt.NDArray) -> npt.NDArray:
    """Weighted least-squares additive fit ``mu + alpha_i + beta_j``.

    Fitted to the *cell means* with the cell sizes as weights, which is
    the additive model restricted to the observed design. The dummy
    design is rank deficient (the intercept and the two sets of dummies
    are collinear), but the fitted values are unique, and ``lstsq``
    returns them regardless of which minimum-norm coefficients it picks.

    For a balanced design this returns ``m_i. + m_.j - M`` exactly; it is
    only needed because the weighted marginal means are not orthogonal
    when the cells hold different numbers of observations.
    """
    n_a, n_b = cell_means.shape
    index = np.arange(n_a * n_b)
    design = np.zeros((n_a * n_b, 1 + n_a + n_b), dtype=float)
    design[:, 0] = 1.0
    design[index, 1 + index // n_b] = 1.0
    design[index, 1 + n_a + index % n_b] = 1.0
    root = np.sqrt(cells.ravel().astype(float))
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], cell_means.ravel() * root, rcond=None)
    return (design @ coefficients).reshape(n_a, n_b)


def _two_way_sums_of_squares(
    response: npt.NDArray,
    labels_a: npt.NDArray,
    labels_b: npt.NDArray,
    levels_a: list[Any],
    levels_b: list[Any],
) -> tuple[dict[str, float], dict[str, int], dict[Any, dict[Any, int]]]:
    """Decompose the response into A, B, A:B and residual.

    The three terms are built as differences of *nested* weighted fits to
    the cell means, so they always add up to ``SS_total`` exactly::

        SS_null  = sum n_ij (m_ij - M)^2            one mean for every cell
        SS_a     = sum n_ij (m_ij - m_i.)^2         + row means
        SS_ab    = sum n_ij (m_ij - m_hat_ij)^2     + additive fit m_hat
        SS_A     = SS_null - SS_a
        SS_B     = SS_a   - SS_ab
        SS_AB    = SS_ab
        SS_error = sum (y - m_ij)^2                + the cell means themselves

    For a balanced design the additive fit is ``m_i. + m_.j - M``, the
    interaction contrast is exactly ``m_ij - m_i. - m_.j + M`` and these
    are the textbook closed forms written in the module docstring. For an
    unbalanced design the weighted marginal means are *not* orthogonal,
    so the simple contrast does not close -- it returns a sum that differs
    from ``SS_total`` by a non-zero amount (7.13 on the 2x2 with cell
    sizes 2, 2, 3, 2 in the tests) -- and the least-squares additive fit
    has to be solved instead. That is the Type I decomposition R's
    ``aov(y ~ a*b)`` reports.

    Everything is written as an explicit sum of squares rather than taken
    as a leftover, so the terms can be checked against hand arithmetic.
    """
    n_a, n_b = len(levels_a), len(levels_b)
    cells = np.empty((n_a, n_b), dtype=int)
    sums = np.empty((n_a, n_b))
    counts: dict[Any, dict[Any, int]] = {a: {} for a in levels_a}
    for j, b_level in enumerate(levels_b):
        for i, a_level in enumerate(levels_a):
            mask = (labels_a == a_level) & (labels_b == b_level)
            n_here = int(np.sum(mask))
            cells[i, j] = n_here
            counts[a_level][b_level] = n_here
            sums[i, j] = float(np.sum(response[mask])) if n_here else 0.0

    # An empty cell has no mean; it contributes nothing to any term, and
    # its residual df is negative, which the caller rejects.
    cell_means = np.where(cells > 0, sums / np.maximum(cells, 1), 0.0)
    weighted_cells = cell_means * cells
    row_means = weighted_cells.sum(axis=1) / cells.sum(axis=1)
    grand = float(np.sum(response) / response.size)

    weights = cells.astype(float).ravel()
    flat_means = cell_means.ravel()
    fitted_additive = _additive_fit(cell_means, cells)
    flat_fitted = fitted_additive.ravel()

    ss_null = float(np.sum(weights * (flat_means - grand) ** 2))
    ss_a_model = float(np.sum(weights * (flat_means - np.repeat(row_means, n_b)) ** 2))
    ss_ab_model = float(np.sum(weights * (flat_means - flat_fitted) ** 2))
    ss = {
        "total": float(np.sum((response - grand) ** 2)),
        "A": ss_null - ss_a_model,
        "B": ss_a_model - ss_ab_model,
        "AB": ss_ab_model,
    }
    within_cells = np.zeros((n_a, n_b))
    for j, b_level in enumerate(levels_b):
        for i, a_level in enumerate(levels_a):
            if cells[i, j]:
                mask = (labels_a == a_level) & (labels_b == b_level)
                within_cells[i, j] = float(np.sum((response[mask] - cell_means[i, j]) ** 2))
    ss["error"] = float(np.sum(within_cells))

    df = {
        "A": n_a - 1,
        "B": n_b - 1,
        "AB": (n_a - 1) * (n_b - 1),
        "error": int(response.size - n_a * n_b),
    }
    return ss, df, counts


def _two_way_permutation_p(
    response: npt.NDArray,
    labels_a: npt.NDArray,
    labels_b: npt.NDArray,
    levels_a: list[Any],
    levels_b: list[Any],
    name_a: str,
    name_b: str,
    n_permutations: int,
    random_seed: int | None,
) -> dict[str, float]:
    """Permutation p-values for the three terms.

    The response is permuted across the whole design -- a "permutation of
    the response" -- which assumes the observations are freely
    exchangeable across cells. See the module docstring: that assumption
    is the design's, and it is what makes the reported p-value valid.
    """
    if n_permutations <= 0:
        return {"A": _NAN, "B": _NAN, "AB": _NAN}

    ss, df, _ = _two_way_sums_of_squares(response, labels_a, labels_b, levels_a, levels_b)
    ms_error = ss["error"] / df["error"] if df["error"] else _NAN
    observed = {
        key: (ss[key] / df[key]) / ms_error for key in ("A", "B", "AB") if df[key] and ms_error > 0
    }

    def _permuted_term(rng: np.random.Generator) -> dict[str, float]:
        shuffled = _two_way_sums_of_squares(
            response[rng.permutation(response.size)], labels_a, labels_b, levels_a, levels_b
        )[0]
        ms_error_shuffled = shuffled["error"] / df["error"] if df["error"] else _NAN
        if not np.isfinite(ms_error_shuffled) or ms_error_shuffled <= 0:
            return {key: _NAN for key in observed}
        return {
            key: (shuffled[key] / df[key]) / ms_error_shuffled
            for key in observed
        }

    def _statistic(key: str) -> Any:
        def _inner(rng: np.random.Generator) -> float:
            return float(_permuted_term(rng).get(key, _NAN))

        return _inner

    context = f"two-way ANOVA ({name_a} x {name_b})"
    p_values: dict[str, float] = {}
    for key in ("A", "B", "AB"):
        if key not in observed:
            p_values[key] = _NAN
            continue
        p_values[key] = float(
            permutation_pvalue(
                float(observed[key]),
                _statistic(key),
                n_permutations=n_permutations,
                random_seed=random_seed,
                context=context,
            ).p_value
        )
    return p_values


def _repeated_measures_sums_of_squares(
    matrix: npt.NDArray, group_by_subject: npt.NDArray, n_per_group: int
) -> tuple[dict[str, float], dict[str, int]]:
    """Decompose a balanced repeated-measures design.

    ``matrix`` is ``(n_subjects, n_measurements)`` with subjects ordered
    by group. The between-subject stratum is a one-way ANOVA on the
    subject means; the within-subject stratum is a two-way ANOVA
    (group x measurement) on the subject-centred data, which zeroes the
    group and subject terms and leaves measurement, group x measurement
    and the subject x measurement error.

    Degrees of freedom:

        groups      g - 1
        subjects    g*(n - 1)
        measurement k - 1
        interaction (g - 1)*(k - 1)
        error       g*(n - 1)*(k - 1)

    which sum to ``g*n*k - 1``. The error df is ``g*(n-1)*(k-1)`` and not
    ``n*(g-1)*(k-1)``: the between-subject stratum pools the n-1 subject
    df inside each of the g groups.
    """
    n_subjects, n_measurements = matrix.shape
    grand = float(np.mean(matrix))
    subject_means = np.mean(matrix, axis=1)

    group_levels = _ordered_levels(group_by_subject)
    group_means = np.asarray(
        [float(np.mean(subject_means[group_by_subject == g])) for g in group_levels]
    )
    ss_total = float(np.sum((matrix - grand) ** 2))
    ss_groups = float(n_measurements * n_per_group * np.sum((group_means - grand) ** 2))
    ss_subjects = float(n_measurements * np.sum((subject_means - grand) ** 2))
    ss_subjects_within = ss_subjects - ss_groups

    centred = matrix - subject_means[:, None]
    measurement_means = np.mean(centred, axis=0)
    cell_group_means = np.vstack(
        [np.mean(centred[group_by_subject == g], axis=0) for g in group_levels]
    )
    ss_measurement = float(n_subjects * np.sum(measurement_means**2))
    # Each group x measurement cell holds n_per_group subject means, and
    # the group means of the subject-centred data already sum to zero over
    # measurements, so the interaction contrast needs no further correction.
    ss_interaction = float(
        n_per_group * np.sum((cell_group_means - measurement_means) ** 2)
    )
    ss_error = float(np.sum(centred**2) - ss_measurement - ss_interaction)

    g_count = len(group_levels)
    df = {
        "groups": g_count - 1,
        "subjects": g_count * (n_per_group - 1),
        "measurement": n_measurements - 1,
        "interaction": (g_count - 1) * (n_measurements - 1),
        "error": g_count * (n_per_group - 1) * (n_measurements - 1),
    }
    ss = {
        "total": ss_total,
        "groups": ss_groups,
        "subjects": ss_subjects_within,
        "measurement": ss_measurement,
        "interaction": ss_interaction,
        "error": ss_error,
    }
    return ss, df
