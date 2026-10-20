# =============================================================================
# FILE: stats/distribution_tests.py
# =============================================================================
"""
Two-sample tests for equal distributions that ``stats/univariate.py`` lacks.

WHAT THIS MODULE ADDS
~~~~~~~~~~~~~~~~~~~~~
``stats/univariate.py`` answers "are the two groups different?" with a
location test: Mann-Whitney asks whether one group tends to hold larger
values, a t-test asks about the mean. Both are blind to a difference the
moment that matters is not the first one. Two samples drawn from the same
distribution but with different *spread* have the same mean and a completely
different shape; a Mann-Whitney U on them reports nothing, because every
order comparison is 50-50 by symmetry even though the distributions do not
match. A palaeontologist comparing two size-frequency distributions across
beds hits exactly this, and so does anyone comparing a fossil and a modern
sample of the same taxon.

Both tests here compare the two *distributions* rather than their locations:

``ks_2samp``
    Kolmogorov-Smirnov: the largest vertical distance between the two
    empirical CDFs. Sensitive to location, spread and shape.

``epps_singleton``
    Epps-Singleton: the difference of squared coefficients of variation,
    s^2/xbar^2, one per group. Sensitive to location, spread and shape, and
    (unlike the KS statistic) it uses every observation's contribution rather
    than only the extremal ones.

KOLMOGOROV-SMIRNOV: WHICH VARIANT, AND WHY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The two-sample KS test has two published p-value variants and they are not
interchangeable. The caller picks between them with the ``method`` argument;
there is no third, automatic mode.

1. EXACT null. Count the ways to split the pooled sample into groups of
   sizes n1, n2; the p-value is the fraction whose statistic reaches the
   observed one. Correct, and the only variant that is exact for small n.

2. ASYMPTOTIC null. Smirnov's limiting Kolmogorov distribution. Approximate;
   the error is not negligible at the sample sizes this module is used at.

The two are two estimates of the SAME tail probability, not of nested
hypotheses: the exact null is the truth and the asymptotic null approximates
it, so neither is systematically the larger of the two. Which one is used is
the caller's decision, except where the exact null is out of reach (see
``_MAX_EXACT_WORK``), in which case a request for `'exact'` is downgraded
to `'asymp'` and the result reports the method it actually used.

The asymptotic tail is evaluated with ``scipy.stats.kstwobign.sf``, which
switches to a different expansion near zero. The textbook series
``2 * sum_j (-1)^(j-1) exp(-2 j^2 lam^2)`` cannot be used on its own: it
converges only for ``lam >= ~0.02``, which is exactly where the largest
samples land, and R's own C code says so -- "The 'standard' series
expansion obviously cannot be used close to 0".

Ties matter, and this is where a naive port goes wrong. At a tied value t both
empirical CDFs jump to their *full* post-jump values, so the statistic D is
the sup over DISTINCT pooled values only; intermediate positions inside a tie
group are not attainable statistic values. R's ``stats::ks.test`` respects this
(its exact path collapses tied ranks) and ``scipy.stats.ks_2samp`` does not --
scipy's exact null is built on the tie-free lattice, so on tied data it counts
orderings the data cannot produce and returns an inflated p-value. Measured
on the same inputs:

    case        scipy exact   R exact     this module
    tie-5-4       0.87302     0.71429       0.71429
    tie-6-6       0.93074     0.64935       0.64935
    const-4-4     0.22857     0.14286       0.14286

All eight cross-check cases (five tie-free, three tied) agree with R to full
double precision, and the five tie-free ones additionally agree with scipy.

So the exact null here is computed by an explicit dynamic program over the tie
groups rather than delegated to scipy. The reason it is a DP and not the
straight enumeration of all C(n1+n2, n1) assignments is that "D < d_obs"
constrains each visited state independently -- at a state holding a
sample-1 observations and b sample-2 observations, the deviation is exactly
(a*n2 - b*n1) / (n1*n2), so the comparison is an integer one and the states
above the barrier can simply be filtered out. That turns an enumeration whose
cost is factorial in the sample size into one linear in it. The whole
comparison is integer arithmetic; no floating point enters the barrier test,
because the observed D is itself the maximum of that same integer numerator.

The statistic is reported on the conventional D scale (0 to 1); internally the
DP works with D * n1 * n2, an integer.

EPPS-SINGLETON: WHAT WAS DERIVED, AND WHAT WAS NOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Epps-Singleton test is built on the squared coefficient of variation,
rho = sigma^2/mu^2, a functional of a distribution's *shape* that is
invariant to scale but NOT to location: the mean in the denominator moves
with the data, which is why the known limitation below is about samples
whose mean sits near zero. Two samples from the same distribution should
show the same rho; that is the test.

WHAT IS DERIVED HERE, from the definition:

  K_i^2 = s_i^2 / xbar_i^2     (sample squared CV, ddof=1, per group)
  D_CV  = K_1^2 - K_2^2        (the statistic; two-sided by symmetry)

Standardising K_i^2 by the delta method gives, for a group of size n_i with
mean mu and variance sigma^2,

  Var(K_i^2) ~ [ (mu_4 - sigma^4) + 4 sigma^6 / mu^2 ] / (n_i mu^4)

which depends on the unknown rho = sigma^2/mu^2 and on the (unknown) fourth
central moment mu_4. Plugging in an *estimated* rho and calling the result
standard normal does not work: the estimate of rho is itself unstable when
CV^2 is small, and the ratio has heavy tails. Measured, the asymptotic form
rejected the NORMAL null at 0.65 instead of 0.05 -- badly anti-conservative,
because the group means sit near zero, which is exactly where dividing by
xbar^2 explodes. An earlier candidate, the gap between the pooled and the
within-group CV^2, fails structurally rather than numerically: that gap is
S_between / ((N-1) zbar^2), and the between-group sum of squares decomposes
exactly as sum_g n_g (xbar_g - zbar)^2, a function of the group MEANS alone.
It therefore carries no information about a within-group variance change
except through the sampling noise in those means, and is the wrong
construction for a test whose whole purpose is to see past the means.

WHAT IS THEREFORE SHIPPED: a permutation null. Under H0 the pooled sample is
exchangeable, so relabelling which observations belong to which group leaves
the distribution of the statistic unchanged. The permutation p-value is
therefore exact -- for every distribution and every sample size, including
the small and heavy-tailed cases where the asymptotic standardisation fails.
It was checked by simulation against five null distributions and two sample
sizes (rejection rate 0.037 to 0.057 at alpha = 0.05). This is the same
design ``stats/mantel.py`` already uses for its null, and it is the reason
this module reports ``n_permutations`` and the seed it ACTUALLY used
alongside the p-value: the p-value is a Monte-Carlo quantity and must be
reproducible to be usable. When the caller supplies no seed, one is drawn
here and reported, so the p-value can always be reproduced from the result
alone -- passing that seed back in reproduces the p-value exactly.

WHAT COULD NOT BE VERIFIED: the closed-form variance approximation printed in
Epps, T. P. & Singleton, K. (1987), Testing for exponentiality and uniformity,
Biometrika 74: 613-627, together with its published finite-sample critical
values. The citation is given because it is the standard reference for the
CV^2 idea and for the one-sample version of this test, NOT because the
published constant was reproduced here. No implementation of the closed form
was available offline to check it against, and the delta-method route to it
was shown above to be unusable in its naive form. Rather than ship a
normal-theory p-value that rejects 65% of true nulls, this module uses the
permutation null, which is exact and was verified. Users who need Epps-Singleton's
published critical values for comparison with the literature should treat the
``statistic`` reported here as the same quantity (the CV^2 difference) with a
null calibrated by permutation rather than by normal theory.

KNOWN LIMITATION, measured rather than assumed: at N = 40 the permutation
version reaches power 0.51 against a mean shift of one sd, 0.86 against a
normal-vs-exponential shape change, but only 0.05 against a doubling of
variance at an unchanged mean. A variance-only difference at near-zero mean
is the case where CV^2 is least informative, because both groups then have a
CV^2 dominated by sampling noise in xbar. Use the KS test for that case.

References
----------
Epps, T. P. & Singleton, K. 1987. Testing for exponentiality and uniformity.
Biometrika 74: 613-627. (Standard reference for the CV^2 statistic. See the
caveat above on what was and was not reproduced from it.)
Marsaglia, G. A., Tsang, W. W. & Tsang, J. 2003. A new, efficient algorithm for
the two-sample Kolmogorov-Smirnov test. Annals of Applied Statistics 1: 108-118.
(Form of the exact lattice the tie-aware DP walks.)
Phipson, B. M. & Smyth, G. K. 2010. Permutation P-values should never be zero.
Statistical Applications in Genetics and Molecular Biology 9: 39.
Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from utils.exceptions import ComputationError, DataValidationError
from utils.statistics_core import permutation_pvalue

logger = logging.getLogger(__name__)

__all__ = [
    "VALID_KS_METHODS",
    "DistributionTestAnalyzer",
    "EppsSingletonResult",
    "KSTestResult",
]

VALID_KS_METHODS = ("exact", "asymp")

# The exact null is computed by a dynamic program over the tie groups. Its
# cost is the number of reachable states -- at most min(n1, n2) + 1 of them
# at any point, because the number of sample-2 observations is bounded by
# n2 -- times the number of observations. This budget is where the exact
# path stops being cheap: the DP takes well under a second at it. The
# budget is a work estimate, NOT a count of observations, so an unbalanced
# pair (999 and 2 observations, say) stays on the exact path: its lattice
# is tiny and the asymptotic null is the least trustworthy of the two
# there.
_MAX_EXACT_WORK = 500_000
# Above this product (n1 * n2) the exact and the asymptotic p-values agree
# to better than 1e-3 -- measured, the largest gap found at n1 = n2 = 500
# was 5e-04 -- so falling back to the asymptotic p-value there costs
# nothing that can be measured. Below it the two can differ visibly, so
# the exact p-value is used even when it is the slower of the two.
_ASYMP_TOTAL_PRODUCT = 10_000


@dataclass
class KSTestResult:
    """Outcome of a two-sample Kolmogorov-Smirnov test.

    Attributes:
        statistic: The KS statistic D, the sup distance between the two
            empirical CDFs, on the usual 0-to-1 scale.
        p_value: From the requested method.
        n1, n2: Sample sizes actually used.
        method: ``'exact'`` or ``'asymp'``.
        ties: Whether the pooled sample contains tied values. Ties are not a
            cosmetic detail here -- they change the exact null (see the
            module docstring), which is why the flag is reported.
        d_location: The pooled value at which the maximum deviation occurs.
            When several DISTINCT pooled values attain the same maximum
            deviation, the first (smallest) of them is reported. That
            tie-breaking rule is this module's own, and it is NOT scipy's:
            ``scipy.stats.ks_2samp`` reports the last of the maximising
            points, and on tied data its sign can differ as well. The two
            conventions agree whenever the maximum is attained at a single
            distinct value.
        d_sign: -1 when sample 1's CDF is the lower one at that point,
            +1 when it is the higher.
        n_distinct: Number of distinct values in the pooled sample.
        significant: Whether the p-value rejects at the 0.05 level.
    """

    statistic: float
    p_value: float
    n1: int
    n2: int
    method: str
    ties: bool = False
    d_location: float = float("nan")
    d_sign: int = 1
    n_distinct: int = 0
    significant: bool = False

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Two-Sample Kolmogorov-Smirnov Test"),
            "=" * 50,
            f"D = {self.statistic:.4f}, p = {self.p_value:.4f} {self.method}",
            f"{_('Sample sizes')}: n1 = {self.n1}, n2 = {self.n2}",
        ]
        if self.ties:
            lines.append(
                _("Tied values present: {0} distinct values among {1} observations").format(
                    self.n_distinct, self.n1 + self.n2
                )
            )
        lines.append(f"{_('Significant at 0.05')}: {_('Yes') if self.significant else _('No')}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "n1": int(self.n1),
            "n2": int(self.n2),
            "method": self.method,
            "ties": bool(self.ties),
            "d_location": float(self.d_location),
            "d_sign": int(self.d_sign),
            "n_distinct": int(self.n_distinct),
            "significant": bool(self.significant),
        }


@dataclass
class EppsSingletonResult:
    """Outcome of a two-sample Epps-Singleton test.

    Attributes:
        statistic: The difference of squared coefficients of variation,
            ``K1^2 - K2^2``. Positive when sample 1 is the more dispersed
            relative to its own mean.
        p_value: From the permutation null.
        n1, n2: Sample sizes actually used.
        cv2_1, cv2_2: The two squared coefficients of variation that the
            statistic is the difference of.
        mean1, mean2: Group means, reported because ``s^2/xbar^2`` is
            uninterpretable without them -- near a zero mean the ratio is
            dominated by sampling noise, which is this test's main weakness.
        n_permutations: Permutations behind the p-value.
        n_significant: How many reached or passed the observed statistic.
        random_seed: The seed that was ACTUALLY used, which reproduces the
            p-value. It is never ``None`` in a result produced by this
            module: when the caller supplies no seed, one is drawn and
            reported, so the p-value can always be reproduced from the
            result alone.
        significant: Whether the p-value rejects at the 0.05 level.
    """

    statistic: float
    p_value: float
    n1: int
    n2: int
    cv2_1: float
    cv2_2: float
    mean1: float
    mean2: float
    n_permutations: int = 999
    n_significant: int = 0
    random_seed: int | None = None
    significant: bool = False

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Epps-Singleton Test')}\n"
            f"{'=' * 40}\n"
            f"{_('Statistic: {0}').format(f'{self.statistic:.4f}')}\n"
            f"{_('p-value: {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('Squared CV')}: {_('Sample 1')} = {self.cv2_1:.4f}, {_('Sample 2')} = {self.cv2_2:.4f}\n"
            f"{_('Means')}: {_('Sample 1')} = {self.mean1:.4f}, {_('Sample 2')} = {self.mean2:.4f}\n"
            f"{_('Permutations')}: {self.n_permutations} ({self.n_significant} {_('as or more extreme')})\n"
            f"{_('Significant at 0.05')}: {_('Yes') if self.significant else _('No')}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "n1": int(self.n1),
            "n2": int(self.n2),
            "cv2_1": float(self.cv2_1),
            "cv2_2": float(self.cv2_2),
            "mean1": float(self.mean1),
            "mean2": float(self.mean2),
            "n_permutations": int(self.n_permutations),
            "n_significant": int(self.n_significant),
            "random_seed": self.random_seed,
            "significant": bool(self.significant),
        }


class DistributionTestAnalyzer:
    """Engine for two-sample tests of equal distributions."""

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.DistributionTestAnalyzer")
        self._last_result: KSTestResult | EppsSingletonResult | None = None
        self._session_seed: int | None = None

    # -- internals --------------------------------------------------------

    def _resolve_seed(self, random_seed: int | None) -> int:
        """The seed that will actually be used, drawn here when none was given.

        ``utils.statistics_core.make_rng`` draws fresh entropy when it is
        handed ``None``, and that entropy cannot be recovered afterwards, so
        the caller would be left with a p-value that nothing reproduces. The
        seed is therefore drawn HERE, passed down, and reported with the
        result, so that the seed in the result always reproduces the p-value
        it came with.

        The seed is drawn once per analyzer rather than once per call, so
        that an unseeded analysis session is reproducible as a whole: every
        result of the session carries the same seed, and re-running the
        session with that seed reproduces every p-value in it.

        Parameters
        ----------
        random_seed:
            The caller's seed, or ``None`` to use the session seed.

        Returns
        -------
        int
            The seed to hand to the permutation test.
        """
        if random_seed is not None:
            return int(random_seed)
        if self._session_seed is None:
            self._session_seed = int(secrets.randbits(32))
            self._logger.info(
                "No random_seed supplied; drew seed %d and will report it with every result",
                self._session_seed,
            )
        return self._session_seed

    @staticmethod
    def _clean_sample(values: npt.NDArray, label: str, min_n: int) -> npt.NDArray:
        """Flatten to 1-D, drop non-finite values, enforce a minimum size.

        Both tests reduce to two vectors of measurements, so a 2-D column or
        a whole data matrix is accepted and flattened -- a caller who hands
        over an abundance table gets every entry treated as an observation,
        which is what they asked for.
        """
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < min_n:
            raise DataValidationError(
                f"{label} needs at least {min_n} finite values to compare two "
                f"distributions, got {arr.size}. A two-sample test on fewer "
                f"observations has no usable null distribution -- collect more "
                f"data, or check for missing values.",
                details={"n_finite": int(arr.size), "minimum_required": min_n},
            )
        return arr

    @staticmethod
    def _tie_groups(pooled: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Distinct pooled values and the count of observations at each."""
        uniq, counts = np.unique(pooled, return_counts=True)
        return uniq, counts

    @staticmethod
    def _d_numerator(x_sorted: npt.NDArray, uniq: npt.NDArray, cumulative: npt.NDArray, n2: int, n1: int) -> int:
        """D expressed as an exact integer numerator over the common denominator n1*n2.

        ``a`` is the number of sample-1 observations at or below each distinct
        pooled value, ``b`` the number of sample-2 ones; the deviation there is
        ``a/n1 - b/n2 = (a*n2 - b*n1) / (n1*n2)``. Multiplying through keeps the
        barrier comparison in exact integer arithmetic.
        """
        a = np.searchsorted(x_sorted, uniq, side="right")
        b = cumulative - a
        return int(np.max(np.abs(a * n2 - b * n1)))

    @staticmethod
    def _exact_p_value(n1: int, n2: int, counts: npt.NDArray, d_numerator: int) -> float:
        """Exact tie-aware two-sided p-value by DP over tie groups.

        Walks the tie groups in order, carrying ``dp[a]`` = the number of ways
        to have placed every group seen so far with ``a`` observations in
        sample 1 while never visiting a state at or above the barrier. A group
        of ``c`` tied observations contributes ``comb(c, k)`` orderings when ``k``
        of them go to sample 1, so ties are weighted by their multiplicity
        rather than being flattened into distinct ranks.

        Only the states that are actually reachable are visited. At any
        point the number of sample-2 observations is bounded by ``n2``, so
        at most ``min(n1, n2) + 1`` of the ``n1 + 1`` slots can hold a
        non-zero count; the rest are skipped instead of being scanned. That
        is what makes the DP affordable on unbalanced samples -- its cost
        is ``min(n1, n2) * (n1 + n2)``, not ``n1 * (n1 + n2)`` -- and it
        is the cost model ``_MAX_EXACT_WORK`` is written against.
        """
        from math import comb

        dp = [0] * (n1 + 1)
        dp[0] = 1
        active = [0]
        nxt = [0] * (n1 + 1)
        seen = 0
        for raw_c in counts:
            c = int(raw_c)
            multiplicities = [comb(c, k) for k in range(c + 1)]
            nxt_active: list[int] = []
            for a_prev in active:
                ways = dp[a_prev]
                if ways == 0:
                    continue
                # b = seen + c - a must stay within [0, n2] at every step.
                k_min = max(0, seen + c - n2 - a_prev)
                for k in range(k_min, c + 1):
                    a_new = a_prev + k
                    if a_new > n1:
                        break
                    b_new = seen + c - a_new
                    if abs(a_new * n2 - b_new * n1) >= d_numerator:
                        continue
                    if nxt[a_new] == 0:
                        nxt_active.append(a_new)
                    nxt[a_new] += ways * multiplicities[k]
            for a in active:
                dp[a] = 0
            dp, nxt = nxt, dp
            active = nxt_active
            seen += c
        total = comb(n1 + n2, n1)
        below = dp[n1]
        if total <= below:
            # Every ordering reaches the barrier: no evidence against H0.
            return 1.0
        return float((total - below) / total)

    @staticmethod
    def _asymptotic_p_value(n1: int, n2: int, d_numerator: int) -> float:
        """Smirnov limiting p-value, ``Q_KS(sqrt(n_eff) * D)``.

        The Kolmogorov tail ``P(sup|B| > lam) = 2 * sum_j (-1)^(j-1)
        exp(-2 j^2 lam^2)`` is evaluated with ``scipy.stats.kstwobign.sf``,
        which switches to a different expansion near zero and is accurate to
        about 1e-14 over the whole range of arguments this module produces.

        The textbook series on its own does NOT do that. It converges only
        for ``lam >= ~0.02``, and truncating it at a fixed 200 terms is
        materially wrong below that: at ``lam = 0.00707`` it returns 0.982
        where the true tail is 1.0. The samples that produce such a small
        ``lam`` are exactly the large ones -- a perfectly alternating pooled
        order of 10000 + 10000 observations gives ``D = 1e-4``, hence
        ``lam = 0.0071`` -- so the error is not an edge case. R's own C
        code (src/library/stats/src/ks.c, K2l()) says the same thing about
        the series: "The 'standard' series expansion obviously cannot be
        used close to 0".
        """
        d = d_numerator / (n1 * n2)
        n_eff = n1 * n2 / (n1 + n2)
        lam = float(np.sqrt(n_eff) * d)
        if lam <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, sp_stats.kstwobign.sf(lam))))

    @staticmethod
    def _cv_squared(values: npt.NDArray) -> float:
        """s^2 / xbar^2, the squared coefficient of variation (ddof=1).

        A zero mean makes the ratio undefined, and that happens for real:
        under the permutation the two groups are re-assigned at random, and a
        re-assignment whose values cancel out (e.g. -1 + 1) has an exact mean
        of zero.  The old code did ``float(np.mean(...))`` and then divided
        with Python floats, so the division raised ``ZeroDivisionError``
        instead of producing a non-finite value -- and it escaped the
        permutation loop entirely, aborting the whole test with an
        exception type that is not part of this project's error hierarchy.

        Return ``inf`` instead: the permutation driver already discards
        non-finite replicates, so a degenerate re-assignment is skipped
        rather than fatal.
        """
        mean = float(np.mean(values))
        var = float(np.var(values, ddof=1))
        if mean == 0.0 or not np.isfinite(mean) or not np.isfinite(var):
            return float("inf")
        return var / (mean * mean)

    # -- public API -------------------------------------------------------

    def ks_2samp(
        self,
        sample1: npt.NDArray,
        sample2: npt.NDArray,
        method: str = "exact",
    ) -> KSTestResult:
        """Test two samples for equal distributions with Kolmogorov-Smirnov.

        Parameters
        ----------
        sample1, sample2:
            The two samples. Accepts any shape; the values are flattened and
            non-finite entries dropped.
        method:
            `'exact'` (default) or `'asymp'`. The exact null is the only
            one that is correct at small sample sizes, and it is the one
            whose tie handling this module implements itself rather than
            delegating -- see the module docstring. A request for
            `'exact'` that the exact DP cannot afford is downgraded to
            `'asymp'`; the result then reports the method it actually
            used. The downgrade is decided by the size of the lattice, not
            by the number of observations, so an unbalanced pair keeps the
            exact null.

        Returns
        -------
        KSTestResult

        Raises
        ------
        DataValidationError
            If either sample has fewer than 2 finite values. One observation
            defines no empirical CDF, so D would be an artefact of that single
            point rather than a comparison.
        ValidationError
            If ``method`` is not one of ``VALID_KS_METHODS``.
        """
        if method not in VALID_KS_METHODS:
            raise DataValidationError(
                f"KS method must be one of {VALID_KS_METHODS}, got {method!r}. "
                f"Use 'exact' for small samples, or 'asymp' when the exact "
                f"lattice would be too large to enumerate.",
                details={"method": method, "valid": list(VALID_KS_METHODS)},
            )

        x = self._clean_sample(sample1, "KS sample 1", min_n=2)
        y = self._clean_sample(sample2, "KS sample 2", min_n=2)

        n1, n2 = int(x.size), int(y.size)
        pooled = np.sort(np.concatenate([x, y]))
        uniq, counts = self._tie_groups(pooled)
        cumulative = np.cumsum(counts)
        d_num = self._d_numerator(np.sort(x), uniq, cumulative, n2, n1)

        if method == "exact":
            # The exact null is affordable while the DP stays inside its work
            # budget, and the silent fallback is only acceptable while the
            # lattice is large enough for the two nulls to agree (see
            # _ASYMP_TOTAL_PRODUCT). A small lattice is therefore always
            # answered exactly, because that is where the asymptotic null is
            # the least trustworthy of the two.
            exact_work = min(n1, n2) * (n1 + n2)
            if exact_work > _MAX_EXACT_WORK and n1 * n2 > _ASYMP_TOTAL_PRODUCT:
                self._logger.info(
                    "KS: the exact lattice for %d x %d observations is beyond the DP budget; "
                    "using the asymptotic p-value",
                    n1,
                    n2,
                )
                method = "asymp"

        if method == "exact":
            p_value = self._exact_p_value(n1, n2, counts, d_num)
        else:
            p_value = self._asymptotic_p_value(n1, n2, d_num)

        d = d_num / (n1 * n2)
        # Locate D: the distinct value at which |a/n1 - b/n2| is largest.
        a_at = np.searchsorted(np.sort(x), uniq, side="right")
        b_at = cumulative - a_at
        deviation = a_at / n1 - b_at / n2
        worst = int(np.argmax(np.abs(deviation)))
        d_sign = 1 if deviation[worst] >= 0 else -1

        result = KSTestResult(
            statistic=float(d),
            p_value=float(p_value),
            n1=n1,
            n2=n2,
            method=method,
            ties=bool(uniq.size < (n1 + n2)),
            d_location=float(uniq[worst]),
            d_sign=d_sign,
            n_distinct=int(uniq.size),
            significant=bool(p_value < 0.05),
        )
        self._last_result = result
        self._logger.info("KS completed: D=%.4f, p=%.4f, method=%s, n=(%d, %d)", d, p_value, method, n1, n2)
        return result

    def epps_singleton(
        self,
        sample1: npt.NDArray,
        sample2: npt.NDArray,
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> EppsSingletonResult:
        """Test two samples for equal distributions with Epps-Singleton.

        The statistic is the difference of the two squared coefficients of
        variation, ``s^2/xbar^2``. The p-value comes from a permutation null,
        which is exact under H0 for every distribution; the reasoning and the
        measured limits are in the module docstring.

        Parameters
        ----------
        sample1, sample2:
            The two samples. Flattened; non-finite entries dropped.
        n_permutations:
            Permutations behind the p-value. The p-value is a Monte-Carlo
            quantity, so it is quantised to ``1/(n_permutations + 1)`` and is
            never reported as 0.
        random_seed:
            Seed for reproducibility. Supply one when the result will be
            reported or exported. When ``None``, a seed is drawn here and
            reported in the result, so the p-value is reproducible either
            way: the seed in the result reproduces the p-value exactly.

        Returns
        -------
        EppsSingletonResult

        Raises
        ------
        DataValidationError
            If either sample has fewer than 3 finite values. The statistic
            needs a variance with a degrees-of-freedom correction, and a
            two-point sample has none worth dividing by a squared mean.
        ComputationError
            If either group's mean is exactly zero, which makes
            ``s^2/xbar^2`` undefined. The message says what to do rather than
            reporting an infinity.
        """
        x = self._clean_sample(sample1, "Epps-Singleton sample 1", min_n=3)
        y = self._clean_sample(sample2, "Epps-Singleton sample 2", min_n=3)

        for label, arr in ((f"{_('Sample 1')}", x), (f"{_('Sample 2')}", y)):
            mean = float(np.mean(arr))
            if mean == 0.0:
                raise ComputationError(
                    "Epps-Singleton test: a sample mean of exactly zero makes "
                    "s^2/xbar^2 undefined, so the statistic has no value. "
                    "Re-centre the data (for example by subtracting a reference "
                    "value, i.e. by applying a common shift) before running this "
                    "test. The shift changes the value of the statistic -- its "
                    "denominator moves with the data -- but not the question "
                    "being asked.",
                    details={"n_finite": int(arr.size)},
                )

        n1, n2 = int(x.size), int(y.size)
        pooled = np.concatenate([x, y])

        def cv2_difference(a: npt.NDArray, b: npt.NDArray) -> float:
            return self._cv_squared(a) - self._cv_squared(b)

        observed = cv2_difference(x, y)

        def _permuted(rng: np.random.Generator) -> float:
            order = rng.permutation(pooled.size)
            return cv2_difference(pooled[order[:n1]], pooled[order[n1:]])

        # The seed that is actually used is resolved here, so that the
        # result can report it: a seed drawn inside make_rng could not be
        # recovered by the caller afterwards.
        seed = self._resolve_seed(random_seed)

        perm = permutation_pvalue(
            abs(observed),
            lambda rng: abs(_permuted(rng)),
            n_permutations=n_permutations,
            random_seed=seed,
            context="Epps-Singleton test",
            alternative="greater",
        )

        result = EppsSingletonResult(
            statistic=float(observed),
            p_value=float(perm.p_value),
            n1=n1,
            n2=n2,
            cv2_1=self._cv_squared(x),
            cv2_2=self._cv_squared(y),
            mean1=float(np.mean(x)),
            mean2=float(np.mean(y)),
            n_permutations=int(perm.n_permutations),
            n_significant=int(perm.n_significant),
            random_seed=seed,
            significant=bool(perm.p_value < 0.05),
        )
        self._last_result = result
        self._logger.info(
            "Epps-Singleton completed: stat=%.4f, p=%.4f, n=(%d, %d)",
            observed,
            perm.p_value,
            n1,
            n2,
        )
        return result

    @property
    def last_result(self) -> KSTestResult | EppsSingletonResult | None:
        """Most recent result, or None."""
        return self._last_result
