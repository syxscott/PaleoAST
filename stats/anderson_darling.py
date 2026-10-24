# =============================================================================
# FILE: stats/anderson_darling.py
# =============================================================================
"""
The Anderson-Darling k-sample test: are these groups the same population?

WHAT THIS IS FOR
~~~~~~~~~~~~~~~~
``stats/univariate.py`` answers "are the groups different?" with location
tests -- a t-test, a Mann-Whitney, a Kruskal-Wallis.  All three are blind to
a difference in *spread* at matched location, and all three are blind to a
difference in *shape* that leaves the mean and the median alone.  The
Anderson-Darling k-sample test asks the question directly: is there ONE
population behind all of these samples, or several?

It is a rank test, so it needs no assumption about the shape of the
population and no normality.  It is an *omnibus* test, and that is its
point: it is effective against a change of scale at matched location, which
is exactly where the Kruskal-Wallis test is weak.

For a palaeontologist the typical question is "can I pool these two
sections?", and pooling two distributions that only share a mean produces a
summary that describes neither.

THE STATISTIC
~~~~~~~~~~~~~
Scholz & Stephens (1987) generalised Darling's two-sample statistic to k
samples.  With ``n_i`` observations in group ``i``, ``n = sum(n_i)``, and the
pooled sample's distinct values ``z(1) < ... < z(L)`` carrying multiplicities
``h_j``:

    H_j  = (pooled count below z_j) + h_j / 2
    F_ij = (group i count below z_j) + h_ij / 2

the tie-adjusted statistic is

    ADK = (n - 1) / n^2 * sum_i [ 1/n_i * sum_j
              h_j (n F_ij - n_i H_j)^2 / (H_j (n - H_j) - n h_j / 4) ]

and the version that ignores ties is

    ADKn = 1 / n * sum_i [ 1/n_i * sum_{j=1}^{L-1}
              h_j (n F_ij - n_i H_j)^2 / (H_j (n - H_j)) ]

Under the null hypothesis the mean of ADK is ``k - 1`` and its variance is

    Var(ADK) = (a n^3 + b n^2 + c n + d) / ((n - 1)(n - 2)(n - 3))

    S = sum_i 1/n_i
    T = sum_{i=1}^{n-1} 1/i
    g = sum_{i=1}^{n-2} (1/(n - i)) * sum_{j=i+1}^{n-1} 1/j

    a = (4g - 6)(k - 1) + (10 - 6g) S
    b = (2g - 4) k^2 + 8 T k + (2g - 14 T - 4) S - 8 T + 4g - 6
    c = (6 T + 2g - 2) k^2 + (4 T - 4g + 6) k + (2 T - 6) S + 4 T
    d = (2 T + 6) k^2 - 4 T k

The statistic is then standardised,

    ADKs = (ADK - (k - 1)) / sqrt(Var(ADK))

which removes most of its dependence on the sample size.

THE P-VALUE
~~~~~~~~~~~
The p-value comes from an extrapolation-interpolation procedure built on
the critical values Scholz & Stephens give for k = 2, 3, 4, ... .  The
standardised statistic is compared against

    qnt = b0 + b1 / sqrt(k - 1) + b2 / (k - 1)

for the five levels alpha = 0.25, 0.10, 0.05, 0.025, 0.01; the four
neighbouring points are joined by a cubic spline and the value is mapped
through a logistic function to a probability.  Outside the tabulated range
the spline is extrapolated, which is what lets an extreme statistic still
produce a usable probability instead of a missing one.

WHAT WAS CHECKED AGAINST WHAT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The worked example is the one published with the method (Scholz & Stephens
1987, p. 922): the smoothness of a certain kind of paper, measured by four
independent laboratories, eight measurements each, 32 observations and 3
ties.  The published values are

    mean of the rank statistic            3
    standard deviation of the rank statistic   1.2037664
    statistic, ties not adjusted           8.3558723
    standardised statistic, ties not adjusted  4.4492622
    probability, ties not adjusted          0.0022547
    statistic, ties adjusted               8.3926093
    standardised statistic, ties adjusted  4.4797806
    probability, ties adjusted              0.0021700

and this module reproduces all of them (see
``tests/golden/test_anderson_darling.py``).

References
~~~~~~~~~~
* Scholz, F. W. & Stephens, M. A. (1987). K-Sample Anderson-Darling Tests.
  *Journal of the American Statistical Association*, 82(399), 918-924.
* Anderson, T. W. & Darling, D. A. (1952). Asymptotic theory of certain
  goodness of fit criteria based on stochastic processes. *Annals of
  Mathematical Statistics*, 23(2), 193-212.
* Darling, D. A. (1957). The Kolmogorov-Smirnov, Cramer-von Mises tests.
  *Annals of Mathematical Statistics*, 28(3), 823-838.
* Pettitt, A. N. (1976). A two-sample Anderson-Darling rank statistic.
  *Biometrika*, 63(1), 161-168.
* Trujillo-Ortiz, A., Hernandez-Walls, R., Barba-Rojo, K., Cupul-Magana, L. &
  Zavala-Garcia, R. C. (2007). AnDarksamtest: Anderson-Darling k-sample
  procedure to test the hypothesis that the populations of the drawn groups
  are identical. *MATLAB File Exchange*.
  The MATLAB implementation is GPL-free but its licence does not cover a
  derivative of this repository, so the method was written from the paper
  and the published worked example instead of from that file; the two agree
  to the digits printed above.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: The smallest number of observations the test accepts in total.
MIN_TOTAL_OBSERVATIONS = 8

#: The smallest number of observations the test accepts in one group.
MIN_GROUP_OBSERVATIONS = 2

#: The coefficients of the critical-value approximation, one column per
#: significance level (0.25, 0.10, 0.05, 0.025, 0.01).
_CRITICAL_B0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326], dtype=float)
_CRITICAL_B1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822], dtype=float)
_CRITICAL_B2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396], dtype=float)

#: The values the critical-value spline is fitted through, one per
#: significance level.  They are the logistic transform of the level, so the
#: spline is fitted in log-odds space.
_LOGIT_LEVELS = np.array([1.0986123, 2.1972246, 2.9444390, 3.6635616, 4.5951199], dtype=float)


@dataclass
class AndersonDarlingKResult:
    """Outcome of a k-sample Anderson-Darling test.

    Attributes:
        statistic: The rank statistic adjusted for ties, ``ADK``.
        statistic_no_ties: The rank statistic that ignores ties, ``ADKn``.
        standardized: ``ADK`` standardised by its own mean and standard
            deviation under the null.
        standardized_no_ties: The same for ``ADKn``.
        p_value: The probability attached to the tie-adjusted statistic.
        p_value_no_ties: The probability attached to the statistic that
            ignores ties.
        mean: The mean of the statistic under the null, ``k - 1``.
        std: Its standard deviation under the null.
        n_groups: How many groups were compared.
        group_sizes: How many observations each group held.
        n_total: How many observations in all.
        n_ties: How many observations shared their value with another one.
        n_distinct: How many distinct values the pooled sample held.
        alpha: The significance level the decision was made at.
        significant: Whether the groups were judged to come from different
            populations at that level.
    """

    statistic: float
    statistic_no_ties: float
    standardized: float
    standardized_no_ties: float
    p_value: float
    p_value_no_ties: float
    mean: float
    std: float
    n_groups: int
    group_sizes: tuple[int, ...] = ()
    n_total: int = 0
    n_ties: int = 0
    n_distinct: int = 0
    alpha: float = 0.05
    significant: bool = False
    group_labels: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Anderson-Darling k-Sample Test"),
            "=" * 50,
            _("Number of samples") + f": {self.n_groups}",
            _("Sample sizes") + f": {' '.join(str(size) for size in self.group_sizes)}",
            _("Total number of observations") + f": {self.n_total}",
            _("Number of ties (identical values)") + f": {self.n_ties}",
            _("Mean of the Anderson-Darling rank statistic") + f": {self.mean:g}",
            f"ADK = {self.statistic:.7f}, standardized = {self.standardized:.7f}",
            f"ADKn = {self.statistic_no_ties:.7f}, standardized = {self.standardized_no_ties:.7f}",
            _("Probability") + f": {self.p_value:.7f} (ties adjusted), {self.p_value_no_ties:.7f} (ties ignored)",
            f"{_('Significant at')} {self.alpha:g}: {_('Yes') if self.significant else _('No')}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "statistic": float(self.statistic),
            "statistic_no_ties": float(self.statistic_no_ties),
            "standardized": float(self.standardized),
            "standardized_no_ties": float(self.standardized_no_ties),
            "p_value": float(self.p_value),
            "p_value_no_ties": float(self.p_value_no_ties),
            "mean": float(self.mean),
            "std": float(self.std),
            "n_groups": int(self.n_groups),
            "group_sizes": [int(size) for size in self.group_sizes],
            "n_total": int(self.n_total),
            "n_ties": int(self.n_ties),
            "n_distinct": int(self.n_distinct),
            "alpha": float(self.alpha),
            "significant": bool(self.significant),
            "group_labels": list(self.group_labels),
        }


def _safe_divide(denominator: npt.NDArray) -> npt.NDArray:
    """Divide by a denominator that can be zero at the pooled endpoints.

    ``H_j (n - H_j) - n h_j / 4`` is zero when the pooled sample holds a
    single distinct value; the numerator there is zero too, so the term is
    0/0.  Returning a large divisor there turns it into a zero term instead
    of a NaN, which keeps the arithmetic finite without changing any other
    term.
    """
    safe = np.where(np.abs(denominator) > 0.0, denominator, 1.0)
    return safe


def _spline_probability(standardized: float, n_groups: int) -> float:
    """The probability attached to a standardised statistic.

    The five tabulated significance levels are turned into critical values
    for the given number of groups, the four nearest ones are joined by a
    cubic spline in log-odds space, and the observed statistic is mapped
    through the logistic function.  The spline is extrapolated outside the
    tabulated range, so an extreme statistic still yields a probability.
    """
    if n_groups < 2:
        raise ValidationError(
            _("the k-sample test needs at least two groups; got {0}").format(n_groups),
            details={"n_groups": int(n_groups)},
        )
    root = float(np.sqrt(n_groups - 1))
    critical = _CRITICAL_B0 + _CRITICAL_B1 / root + _CRITICAL_B2 / (n_groups - 1)
    if standardized <= critical[2]:
        knots_x = critical[:4]
        knots_y = _LOGIT_LEVELS[:4]
    else:
        knots_x = critical[1:5]
        knots_y = _LOGIT_LEVELS[1:5]
    spline = CubicSpline(knots_x, knots_y, extrapolate=True)
    log_odds = float(spline(standardized))
    # Guard the exponential: a very extreme statistic would otherwise
    # overflow and turn into inf, which would make the logistic 0.0.
    log_odds = float(np.clip(log_odds, -700.0, 700.0))
    return float(1.0 / (1.0 + np.exp(log_odds)))


def _null_moments(n_total: int, sizes: npt.NDArray, n_groups: int) -> tuple[float, float]:
    """The mean and standard deviation of the statistic under the null."""
    mean = float(n_groups - 1)
    if n_total < 4:
        raise ValidationError(
            _(
                "the Anderson-Darling k-sample test needs at least 4 observations "
                "in total to standardise the statistic; got {0}"
            ).format(n_total),
            details={"n_total": int(n_total)},
        )
    harmonic = float(np.sum(1.0 / np.arange(1, n_total)))
    g = 0.0
    for i in range(1, n_total - 1):
        g += float(np.sum(1.0 / np.arange(i + 1, n_total))) / (n_total - i)
    s = float(np.sum(1.0 / sizes))
    k = float(n_groups)
    a = (4.0 * g - 6.0) * (k - 1.0) + (10.0 - 6.0 * g) * s
    b = (
        (2.0 * g - 4.0) * k**2
        + 8.0 * harmonic * k
        + (2.0 * g - 14.0 * harmonic - 4.0) * s
        - 8.0 * harmonic
        + 4.0 * g
        - 6.0
    )
    c = (
        (6.0 * harmonic + 2.0 * g - 2.0) * k**2
        + (4.0 * harmonic - 4.0 * g + 6.0) * k
        + (2.0 * harmonic - 6.0) * s
        + 4.0 * harmonic
    )
    d = (2.0 * harmonic + 6.0) * k**2 - 4.0 * harmonic * k
    denominator = float((n_total - 1) * (n_total - 2) * (n_total - 3))
    variance = (a * n_total**3 + b * n_total**2 + c * n_total + d) / denominator
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValidationError(
            _(
                "the null variance of the Anderson-Darling statistic is not "
                "positive for this design (k = {0}, n = {1}); the test cannot "
                "be standardised"
            ).format(n_groups, n_total),
            details={"n_groups": int(n_groups), "n_total": int(n_total), "variance": float(variance)},
        )
    return mean, float(np.sqrt(variance))


def anderson_darling_ksample(
    values: npt.NDArray,
    groups: npt.NDArray,
    alpha: float = 0.05,
) -> AndersonDarlingKResult:
    """Test whether several samples come from one common population.

    Parameters:
        values: The observations, in any order.
        groups: The group each observation belongs to, the same length as
            ``values``.  The labels may be numbers or strings; they are
            sorted, so the group order does not depend on the order the
            observations arrive in.
        alpha: The significance level the decision is made at.

    Returns:
        AndersonDarlingKResult: The statistic, its standardised form, the
        probability, and the sizes of the groups that went into it.

    Raises:
        ValidationError: If the two inputs differ in length, if either holds
            a non-finite value, if there are fewer than two groups, if a
            group is too small, or if ``alpha`` is outside ``(0, 1)``.

    Examples:
        >>> import numpy as np
        >>> values = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        >>> groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        >>> result = anderson_darling_ksample(values, groups)
        >>> result.n_groups
        2
    """
    observations = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(groups).ravel()

    if observations.size != labels.size:
        raise ValidationError(
            _("values and groups must have the same length; got {0} and {1}").format(observations.size, labels.size),
            details={"n_values": int(observations.size), "n_groups": int(labels.size)},
        )
    if observations.size < MIN_TOTAL_OBSERVATIONS:
        raise ValidationError(
            _("the Anderson-Darling k-sample test needs at least {0} observations; got {1}").format(
                MIN_TOTAL_OBSERVATIONS, observations.size
            ),
            details={"n_values": int(observations.size), "minimum_required": MIN_TOTAL_OBSERVATIONS},
        )
    if not np.all(np.isfinite(observations)):
        raise ValidationError(
            _("values must all be finite; the test cannot rank a missing observation"),
            details={"n_non_finite": int(np.sum(~np.isfinite(observations)))},
        )
    if np.any(np.asarray(labels, dtype=object) is None):  # pragma: no cover - defensive
        raise ValidationError(_("groups must not hold None"), details={})

    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValidationError(
            _("alpha must lie in (0, 1); got {0}").format(alpha),
            details={"alpha": alpha},
        )

    unique_labels, group_index = np.unique(labels, return_inverse=True)
    n_groups = int(unique_labels.size)
    if n_groups < 2:
        raise ValidationError(
            _("the Anderson-Darling k-sample test needs at least two groups; got {0}").format(n_groups),
            details={"n_groups": n_groups},
        )

    sizes = np.array([int(np.sum(group_index == g)) for g in range(n_groups)], dtype=float)
    too_small = [int(unique_labels[g]) for g in range(n_groups) if sizes[g] < MIN_GROUP_OBSERVATIONS]
    if too_small:
        raise ValidationError(
            _("every group needs at least {0} observations; these have fewer: {1}").format(
                MIN_GROUP_OBSERVATIONS, too_small
            ),
            details={"minimum_required": MIN_GROUP_OBSERVATIONS, "groups": too_small},
        )

    n_total = int(observations.size)

    # The pooled sample's distinct values and their multiplicities.
    distinct, pooled_counts = np.unique(observations, return_counts=True)
    n_distinct = int(distinct.size)
    if n_distinct < 2:
        raise ValidationError(
            _(
                "every observation has the same value, so they cannot be ranked; "
                "the Anderson-Darling k-sample test needs at least two distinct values"
            ),
            details={"n_values": int(observations.size), "n_distinct": n_distinct},
        )
    # H_j: the pooled count below z_j plus half the count equal to z_j.
    h_j = pooled_counts.astype(float)
    h_joint = np.cumsum(h_j) - h_j / 2.0
    # The same, ignoring ties: the pooled count at or below z_j.
    h_joint_no_ties = np.cumsum(h_j)[:-1]

    adjusted_total = 0.0
    no_ties_total = 0.0
    for g in range(n_groups):
        group_values = observations[group_index == g]
        n_i = float(group_values.size)
        # The counts of group g at each distinct pooled value.
        counts_at = np.array([float(np.sum(group_values == value)) for value in distinct], dtype=float)
        # F_ij: the group count below z_j plus half the count equal to z_j.
        f_ij = np.cumsum(counts_at) - counts_at / 2.0
        numerator = (n_total * f_ij - n_i * h_joint) ** 2
        denominator = h_joint * (n_total - h_joint) - n_total * h_j / 4.0
        adjusted_total += float(np.sum(h_j * numerator / _safe_divide(denominator))) / n_i

        # The version that ignores ties runs over the first L - 1 values.
        f_ij_no_ties = np.cumsum(counts_at)[:-1]
        numerator_no_ties = (n_total * f_ij_no_ties - n_i * h_joint_no_ties) ** 2
        denominator_no_ties = h_joint_no_ties * (n_total - h_joint_no_ties)
        no_ties_total += float(np.sum(h_j[:-1] * numerator_no_ties / _safe_divide(denominator_no_ties))) / n_i

    adk = (n_total - 1.0) / n_total**2 * adjusted_total
    adkn = 1.0 / n_total * no_ties_total

    mean, std = _null_moments(n_total, sizes, n_groups)
    adks = (adk - mean) / std
    adksn = (adkn - mean) / std

    p_adjusted = _spline_probability(adks, n_groups)
    p_no_ties = _spline_probability(adksn, n_groups)

    result = AndersonDarlingKResult(
        statistic=float(adk),
        statistic_no_ties=float(adkn),
        standardized=float(adks),
        standardized_no_ties=float(adksn),
        p_value=p_adjusted,
        p_value_no_ties=p_no_ties,
        mean=mean,
        std=std,
        n_groups=n_groups,
        group_sizes=tuple(int(size) for size in sizes),
        n_total=n_total,
        n_ties=int(n_total - n_distinct),
        n_distinct=n_distinct,
        alpha=level,
        significant=bool(p_adjusted < level),
        group_labels=tuple(str(label) for label in unique_labels.tolist()),
    )
    logger.info(
        "Anderson-Darling k-sample: k=%d, n=%d, ADK=%.6f, p=%.6f",
        result.n_groups,
        result.n_total,
        result.statistic,
        result.p_value,
    )
    return result
