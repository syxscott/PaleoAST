# =============================================================================
# FILE: ecology/diversity_extra.py
# =============================================================================
"""
Diversity sampling standardisation and between-group tests (the PAST 4
Diversity menu).

WHAT THIS MODULE IS FOR
~~~~~~~~~~~~~~~~~~~~~~~
``ecology/diversity.py`` answers "how diverse is this sample".
This module answers the three questions that come next, all of which need a
*fair comparison* between samples that are not equally complete:

1. **SQS** (:func:`sqs_subsample`) -- standardise every sample to the same
   number of individuals (the *quorum*) before comparing, by taking taxa in
   decreasing abundance order and rotating through them until the quorum is
   reached. This is the palaeontological standard.
2. **Sample-coverage rarefaction** (:func:`sample_rarefaction_tao`) -- the
   sample (incidence) analogue of individual rarefaction, driven by how much
   of the community the sampling actually covered rather than by how many
   individuals it happened to collect.
3. **Diversity profiles** (:func:`compute_diversity_profile`) -- every index
   as a function of sampling depth, which is what shows whether a difference
   between two samples is a real signal or just a difference in how hard they
   were dug.

and the two tests that compare diversity *between* groups:

4. **Diversity t test** (:func:`diversity_t_test`) and **diversity
   permutation test** (:func:`diversity_permutation_test`).

SQS IN DETAIL
~~~~~~~~~~~~~
SQS is *not* random. Given a sample with abundances ``a_1 >= a_2 >= ... >= a_S``
and a quorum ``q``, it walks the taxa in that fixed decreasing-abundance order
and takes **one individual at a time from each taxon in turn**, cycling back
around, until exactly ``q`` individuals have been collected. A taxon that runs
out of individuals is skipped and the cycle continues with the next.

Two properties follow, and both are asserted in the tests because they are
what makes SQS usable as a standardisation:

* **It is invariant to the order the taxa arrive in.** Two rows of the same
  table permuted column-wise give the same subsample. A method that depended on
  column order would silently depend on how the user sorted their spreadsheet.
* **Observed richness is monotone non-decreasing in the quorum.** A larger
  quorum never loses a taxon it would have kept. This is the property that
  makes a diversity profile interpretable as a curve rather than as noise.

The third property is the one this method is *for*: SQS preserves the
sample's relative structure rather than flattening it. Subsampling
``[10, 10, 10]`` to ``q = 6`` gives ``[2, 2, 2]`` and
``[28, 1, 1]`` gives ``[4, 1, 1]`` -- the even sample stays even and the
dominated one stays dominated. It is **not** true that equal-quorum
subsampling gives equal indices to samples with different evenness; that is
the opposite of what it does.

SAMPLE COVERAGE AND MAO'S TAU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ecology/rarefaction.py`` rarefies over *individuals*. Its sample-based
counterpart models the *incidence* matrix (samples x taxa) and is driven by
**sample coverage** ``C``, the proportion of the community that the survey has
already detected. Following Chao & Jost (2012), the incidence coverage of an
``m``-sample survey with ``f1`` taxa seen in exactly one sample and ``f2`` seen
in exactly two is

.. math::

    \\hat C = 1 - \\frac{f_1}{m}\\left[
        \\left(1 - \\frac{f_2}{f_1+1}\\right)\\frac{m-1}{m-2}
        + \\frac{f_2}{(f_1+1)(m-2)} \\right]

**Mao's tau** (:func:`sample_coverage`) is the resulting estimate of unseen
diversity, ``tau = m (1 - C)``, with the accompanying unseen-species estimate
``f0 = f1 (1 - C) / C``. Neither is a rediscovery of the Chao1 estimator: they
answer "how much did we miss" in *sample* units rather than in *individual*
units, which is the right question when the sampling unit is a bed, a core
section or a time slice rather than a specimen.

The rarefaction curve itself (:func:`sample_rarefaction_tao`) interpolates
below the observed number of samples and extrapolates above it, both anchored
at the observed richness.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It does not reimplement Shannon, Simpson, Pielou, Margalef, Fisher's alpha or
Chao1. Those live in :mod:`ecology.diversity` and are called from here, so
there is exactly one definition of each index in the codebase. A second
implementation of Shannon would be a second thing to drift.

PROVENANCE OF THE TWO ESTIMATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQS here is deterministic and is pinned by property rather than by
comparison with a reference implementation: the tests assert that the quorum
is always reached exactly, that a full-size quorum is the identity, that the
subsample is invariant to input column order, and that observed richness is
monotone non-decreasing in the quorum.

The coverage estimator is Chao & Jost (2012) eq. 5 for incidence data, and
is checked against a value derived by hand in the tests (six samples, f1 = 2,
f2 = 1 gives C = 0.694444...). **It has not been cross-validated against R's
``iNEXT``,** which is what ``tests/cross_validation/test_vs_inext.py`` does
for Chao1: ``iNEXT`` is installed on the cross-validation CI job but is not
installed in the development environment this module was written in, so no
live comparison was possible here. ``iNEXT`` does not implement SQS, so the
SQS side has no reference to compare against in any case.

References
----------
Alroy, M. (2010). Sampling standardization and subsampling in paleoecology.
    In: *Quantitative Methods in Paleobiology*. Yale University Press.
Good, L. J. & Turing, R. E. (1959). The population of a 'large' animal or
    plant, and the accuracy of its census. *Biometrika* 46: 415-428.
Chao, A. & Jost, L. (2012). Coverage-based rarefaction and extrapolation:
    adapting species-based inference to incidence data. *Ecology* 93: 829-834.
Mao, C. X., Xiao, Y., Kuratov, M., Zhang, Y. D., Yang, Y., Zhang, Y.,
    Yang, J. & Zhang, K. (2015). Sample coverage-based rarefaction,
    extrapolation and prediction: how many samples are needed?
    *Journal of Integrative Biology* 10: 477-490.
Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from ecology.diversity import compute_diversity_indices
from models.diversity_result import DiversityResult
from utils.exceptions import ComputationError, StatisticalError, ValidationError
from utils.statistics_core import group_indices, permutation_pvalue
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

__all__ = [
    "INDEX_KEYS",
    "SQS_SAMPLE_NAME",
    "DiversityExtraAnalyzer",
    "DiversityProfileResult",
    "DiversityTestResult",
    "SQSResult",
    "SampleCoverageResult",
    "SampleRarefactionResult",
    "compute_diversity_profile",
    "diversity_permutation_test",
    "diversity_t_test",
    "sample_coverage",
    "sample_rarefaction_tao",
    "sqs_subsample",
]

#: Default label for a single anonymous sample.
SQS_SAMPLE_NAME = "Sample"

#: Index keys ``ecology.diversity.compute_diversity_indices`` can produce.
#: ``pielou`` is absent for S == 1 and ``margalef`` for N <= 1 or S < 2, so
#: every consumer of this list has to cope with a missing entry rather than
#: assume one.
INDEX_KEYS = ("shannon", "simpson", "pielou", "margalef", "fisher_alpha", "chao1")


# =============================================================================
# Input normalisation
# =============================================================================


def _as_aligned_counts(values: npt.NDArray, name: str) -> npt.NDArray:
    """Integer abundance counts in the *input* column order.

    Shares ``ecology.diversity``'s contract: raw counts only. A
    relative-abundance vector would silently become a count vector, so the
    integrality test is on the *values*, not on their sum -- fractions that
    happen to total 1.0 pass a sum-integrality check and are still wrong.

    Zero and negative entries are kept as zeros rather than dropped, so the
    returned array has one entry per input column and column ``i`` still
    refers to input column ``i``. Dropping them would silently renumber the
    taxa: anything built from the shortened vector (a subsample, a profile
    curve) could no longer be matched back to the columns the caller
    supplied.

    Parameters
    ----------
    values : array-like
        Raw counts for one sample.
    name : str
        Label used in the error messages.

    Returns
    -------
    numpy.ndarray
        Integer counts aligned with the input columns.

    Raises
    ------
    ValidationError
        If the input has no positive counts, or any positive count is
        fractional.
    """
    arr = validate_data_array(values, allow_nan=False, name=name).flatten()
    positive = arr[arr > 0]
    if positive.size == 0:
        raise ValidationError(
            f"{name}: no positive abundances found; diversity indices are undefined for an empty sample",
            details={"name": name},
        )
    non_integer = ~np.isclose(positive, np.round(positive), rtol=0.0, atol=1e-9)
    if np.any(non_integer):
        offenders = np.unique(positive[non_integer])[:5]
        raise ValidationError(
            f"{name}: diversity sampling requires integer abundance counts, but got "
            f"fractional values {list(offenders)}. Pass raw counts, not a "
            "relative-abundance or percentage vector.",
            details={"name": name, "offenders": list(map(float, offenders))},
        )
    aligned = np.zeros(arr.size, dtype=int)
    aligned[arr > 0] = np.round(positive).astype(int)
    return aligned


def _as_counts(values: npt.NDArray, name: str) -> npt.NDArray:
    """Positive integer abundance counts, flattened, in the order given.

    Zero and negative entries are dropped, so the result is *shorter* than
    the input whenever the input contains any. Use
    :func:`_as_aligned_counts` when one entry per input column is needed.

    Parameters
    ----------
    values : array-like
        Raw counts for one sample.
    name : str
        Label used in the error messages.

    Returns
    -------
    numpy.ndarray
        The positive counts, in input order.

    Raises
    ------
    ValidationError
        If the input has no positive counts, or any positive count is
        fractional.
    """
    aligned = _as_aligned_counts(values, name)
    return aligned[aligned > 0]


def _as_incidence(values: npt.NDArray) -> npt.NDArray:
    """Binary presence/absence incidence matrix, one row per sample.

    Values above 0 are taken as presence, so an abundance table also works:
    for a coverage estimator what matters is *whether* a taxon occurred in a
    sample, not how many times. Rows that are entirely absent are kept -- a
    sample that found nothing is still a sample -- so the row count the caller
    passes is the row count the estimator sees.

    A 1-D input is rejected rather than guessed at. ``validate_data_array``
    reshapes a 1-D array to ``(n, 1)``, which would read a presence vector
    such as ``[1, 1, 0]`` as "three samples of one taxon" instead of "one
    sample of three taxa" -- a different survey, with a different coverage.
    The dimensionality is therefore captured *before* the reshape (by asking
    the validator to preserve it), so the check below is reachable.

    Parameters
    ----------
    values : array-like, shape (n_samples, n_taxa)
        Presence/absence matrix.

    Returns
    -------
    numpy.ndarray
        0/1 matrix, one row per sample.

    Raises
    ------
    ValidationError
        If the input is not 2-D, or has no rows or no columns.
    """
    arr = validate_data_array(values, allow_nan=False, name="incidence", dtype=float, preserve_dimensions=True)
    if arr.ndim != 2:
        raise ValidationError(
            f"incidence must be a 2-D samples x taxa matrix, got {arr.ndim}-d. "
            "Reshape it explicitly: a 1-D vector is ambiguous (one sample of many "
            "taxa, or many samples of one taxon?).",
            details={"shape": tuple(int(x) for x in arr.shape), "ndim": int(arr.ndim)},
        )
    if arr.shape[0] == 0:
        raise ValidationError(
            "incidence has no rows; sample coverage needs at least one sample",
            details={"shape": tuple(int(x) for x in arr.shape)},
        )
    if arr.shape[1] == 0:
        raise ValidationError(
            "incidence has no columns; sample coverage needs at least one taxon",
            details={"shape": tuple(int(x) for x in arr.shape)},
        )
    return (arr > 0).astype(int)


# =============================================================================
# 1. SQS -- Shareholder Quorum Subsampling
# =============================================================================


@dataclass(frozen=True)
class SQSResult:
    """Outcome of one SQS subsample.

    Attributes:
        sample_name: Label carried through to both index sets.
        quorum: The number of individuals the sample was reduced to.
        subsample: Per-taxon counts after subsampling, aligned with the
            *input* column order: one entry per input column, zero for a
            taxon the quorum excluded and zero for a taxon that was absent
            from the input in the first place.
        raw_diversity: Indices on the original sample.
        standardized_diversity: Indices recomputed on the subsample.
        indices_recomputed: Whether the subsample could be re-analysed (it
            always can when ``quorum >= 1``, but the field keeps the
            distinction explicit rather than implied).
    """

    sample_name: str
    quorum: int
    subsample: npt.NDArray
    raw_diversity: DiversityResult
    standardized_diversity: DiversityResult | None = None
    indices_recomputed: bool = True

    @property
    def individuals(self) -> int:
        """Individuals actually retained (equals ``quorum``)."""
        return int(self.subsample.sum())

    @property
    def taxa_retained(self) -> int:
        """Taxa still present after subsampling."""
        return int(np.count_nonzero(self.subsample))

    @property
    def dropped_taxa(self) -> int:
        """Taxa the quorum removed entirely."""
        return int(self.raw_diversity.taxa_count - self.taxa_retained)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        lines = [
            f"SQS subsample: {self.sample_name}",
            "=" * 40,
            f"Quorum: {self.quorum}",
            f"Taxa: {self.raw_diversity.taxa_count} -> {self.taxa_retained} (dropped {self.dropped_taxa})",
            f"Individuals: {self.raw_diversity.individuals} -> {self.individuals}",
            "",
        ]
        if self.standardized_diversity is not None:
            std = self.standardized_diversity
            lines.append("Index              raw          SQS(q)")
            lines.append("-" * 40)
            for key in INDEX_KEYS:
                raw_value = self.raw_diversity.get(key)
                std_value = std.get(key)
                if raw_value is None and std_value is None:
                    continue
                raw_txt = "n/a" if raw_value is None else f"{raw_value:.4f}"
                std_txt = "n/a" if std_value is None else f"{std_value:.4f}"
                lines.append(f"{key:<16} {raw_txt:>10}  {std_txt:>10}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "sample_name": self.sample_name,
            "quorum": int(self.quorum),
            "subsample": [int(x) for x in np.asarray(self.subsample).ravel()],
            "individuals": self.individuals,
            "taxa_retained": self.taxa_retained,
            "taxa_dropped": self.dropped_taxa,
            "raw_diversity": self.raw_diversity.to_dict(),
            "standardized_diversity": (
                self.standardized_diversity.to_dict() if self.standardized_diversity is not None else None
            ),
        }


def sqs_subsample(abundances: npt.NDArray, quorum: int, sample_name: str = SQS_SAMPLE_NAME) -> SQSResult:
    """Reduce one sample to ``quorum`` individuals by SQS and re-index it.

    Taxa are visited in decreasing abundance order and given **one individual
    each per cycle**, cycling until exactly ``quorum`` individuals have been
    taken; a taxon that has run out is skipped. The resulting counts keep the
    input column order so a taxon can still be named.

    The indices are then recomputed on the subsample by
    :func:`ecology.diversity.compute_diversity_indices`, so a standardised
    Shannon is the *same* Shannon, not a second implementation of it.

    Parameters
    ----------
    abundances : array-like
        Per-taxon counts for one sample. Zeros and negatives are *not*
        dropped from the result: the subsample keeps one entry per input
        column, so column ``i`` of ``subsample`` is still column ``i``
        of the input, with a zero for a taxon that was absent.
    quorum : int
        Target number of individuals. Must be between 1 and the sample's
        own total -- a quorum larger than the sample cannot be reached, and
        returning a quietly truncated subsample would misreport the depth the
        indices were computed at.
    sample_name : str
        Label for the result.

    Returns
    -------
    SQSResult

    Raises
    ------
    ValidationError
        If the input has no positive counts, is fractional, or ``quorum`` is
        outside ``[1, N]``.

    Examples
    --------
    >>> import numpy as np
    >>> r = sqs_subsample(np.array([10, 10, 10]), 6)
    >>> r.subsample.tolist()
    [2, 2, 2]
    >>> r.taxa_retained
    3
    """
    aligned = _as_aligned_counts(abundances, "abundances")
    counts = aligned[aligned > 0]
    total = int(counts.sum())

    if not isinstance(quorum, (int, np.integer)) or isinstance(quorum, bool):
        raise ValidationError(
            f"SQS quorum must be a whole number of individuals, got {quorum!r}",
            details={"quorum": repr(quorum), "total_individuals": total},
        )
    quorum = int(quorum)
    if quorum < 1:
        raise ValidationError(
            f"SQS quorum must be at least 1 individual, got {quorum}",
            details={"quorum": quorum, "total_individuals": total},
        )
    if quorum > total:
        raise ValidationError(
            f"SQS quorum ({quorum}) exceeds the {total} individuals available in this "
            "sample; it cannot be reached. Use a quorum no larger than the smallest "
            "sample when standardising several samples.",
            details={"quorum": quorum, "total_individuals": total},
        )

    raw = compute_diversity_indices(counts.astype(float), sample_name)

    # Visit in decreasing abundance order. ``stable`` keeps equal-abundance
    # taxa in their input order, so the result is reproducible run to run.
    # The subsample is indexed by *input* column, so a taxon that was absent
    # from the input keeps a zero and column i of the result is still column
    # i of the input.
    order = np.argsort(-aligned, kind="stable")
    subsample = np.zeros(aligned.size, dtype=int)
    remaining = quorum

    while remaining > 0:
        for idx in order:
            if remaining <= 0:
                break
            if subsample[idx] < aligned[idx]:
                subsample[idx] += 1
                remaining -= 1

    logger.debug(
        "SQS: sample=%s, quorum=%d, taxa %d -> %d",
        sample_name,
        quorum,
        counts.size,
        int(np.count_nonzero(subsample)),
    )

    # A quorum of at least 1 always leaves at least one individual, so the
    # re-index cannot fail here; the guard exists so a future caller passing a
    # different quorum path gets a reported reason rather than a raw raise.
    try:
        standardized = compute_diversity_indices(
            subsample[subsample > 0].astype(float), f"{sample_name} (SQS q={quorum})"
        )
        recomputed = True
    except ComputationError:
        standardized = None
        recomputed = False
        logger.warning("SQS: could not recompute indices for sample=%s at quorum=%d", sample_name, quorum)

    return SQSResult(
        sample_name=sample_name,
        quorum=quorum,
        subsample=subsample,
        raw_diversity=raw,
        standardized_diversity=standardized,
        indices_recomputed=recomputed,
    )


# =============================================================================
# 2. Sample coverage and Mao's tau
# =============================================================================


@dataclass(frozen=True)
class SampleCoverageResult:
    """Sample coverage of an incidence matrix, and what it says was missed.

    Attributes:
        coverage: ``C``, the estimated proportion of the community detected,
            in ``[0, 1]``.
        tau: Mao's tau, ``m (1 - C)`` -- the estimated number of unseen
            taxa in sample units.
        unseen_species: ``f1 (1 - C) / C``, the same quantity expressed as a
            count of taxa, or ``None`` when the estimate is not finite
            (zero coverage), because the estimator divides by ``C``.
        observed_richness: Number of taxa seen at least once.
        n_samples: Rows of the incidence matrix (``m``).
        f1 / f2: Taxa seen in exactly one / exactly two samples.
        method: Label for the estimator branch actually used.
    """

    coverage: float
    tau: float
    unseen_species: float | None
    observed_richness: int
    n_samples: int
    f1: float
    f2: float
    method: str = "chao_jost_incidence"

    @property
    def asymptotic_richness(self) -> float | None:
        """Observed richness plus the estimated unseen taxa.

        ``None`` when ``unseen_species`` is ``None``, i.e. when the
        estimate is not finite.
        """
        if self.unseen_species is None:
            return None
        return float(self.observed_richness + self.unseen_species)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        unseen = self.unseen_species
        asymptotic = self.asymptotic_richness
        unseen_txt = "n/a (coverage is 0, so f0 is not estimable)" if unseen is None else f"{unseen:.4f}"
        asymptotic_txt = "n/a" if asymptotic is None else f"{asymptotic:.4f}"
        return (
            f"Sample coverage (Mao's tau)\n"
            f"{'=' * 40}\n"
            f"Samples (m): {self.n_samples}\n"
            f"Observed richness (S_obs): {self.observed_richness}\n"
            f"Singletons (f1) / doubletons (f2): {self.f1:.0f} / {self.f2:.0f}\n"
            f"Sample coverage (C): {self.coverage:.4f}\n"
            f"Mao's tau (tau = m(1-C)): {self.tau:.4f}\n"
            f"Estimated unseen species (f0): {unseen_txt}\n"
            f"Asymptotic richness (S_obs + f0): {asymptotic_txt}\n"
            f"Estimator: {self.method}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view.

        ``unseen_species`` and ``asymptotic_richness`` are ``None``
        rather than ``inf`` when the estimate is not finite, so the
        payload stays valid JSON (RFC 8259 has no ``Infinity`` literal).
        """
        return {
            "coverage": float(self.coverage),
            "tau": float(self.tau),
            "unseen_species": None if self.unseen_species is None else float(self.unseen_species),
            "observed_richness": int(self.observed_richness),
            "n_samples": int(self.n_samples),
            "f1": float(self.f1),
            "f2": float(self.f2),
            "method": self.method,
            "asymptotic_richness": self.asymptotic_richness,
        }


def sample_coverage(incidence: npt.NDArray) -> SampleCoverageResult:
    """Estimate sample coverage and Mao's tau from a samples x taxa matrix.

    Implements the incidence estimator of Chao & Jost (2012), which is the
    sample-based counterpart of the individual-based coverage already used in
    :mod:`ecology.beta_diversity`:

    .. math::

        \\hat C = 1 - \\frac{f_1}{m}\\left[
            \\left(1 - \\frac{f_2}{f_1+1}\\right)\\frac{m-1}{m-2}
            + \\frac{f_2}{(f_1+1)(m-2)} \\right]

    That expression needs ``m >= 3``. For fewer samples it is undefined, so
    this falls back to Good's estimator ``C = 1 - f1/m`` and says so in
    ``method`` rather than returning a number from a division by zero.
    ``C`` is clamped to ``[0, 1]``: with every taxon a singleton the raw
    expression goes negative, which is a real feature of the estimator (the
    survey really did miss things) rather than a bug to hide.

    The unseen-species estimate ``f0 = f1 (1 - C) / C`` divides by ``C``,
    so it is not defined when the clamped coverage is exactly 0. It is
    reported as ``None`` in that case rather than as an infinity, which
    would not survive a JSON round trip.

    Parameters
    ----------
    incidence : array-like, shape (n_samples, n_taxa)
        Presence/absence, one row per sample. Positive values count as
        presence, so an abundance table is accepted. A 1-D input is
        rejected: it is ambiguous between "one sample of many taxa" and
        "many samples of one taxon".

    Returns
    -------
    SampleCoverageResult

    Raises
    ------
    ValidationError
        If the matrix is empty, not 2-D, or 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> inc = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
    >>> sample_coverage(inc).coverage
    1.0
    """
    inc = _as_incidence(incidence)
    m = inc.shape[0]
    occurrences = inc.sum(axis=0)
    observed_richness = int(np.count_nonzero(occurrences))
    f1 = float(np.count_nonzero(occurrences == 1))
    f2 = float(np.count_nonzero(occurrences == 2))

    if f1 <= 0:
        # No singletons means nothing is known to have been missed.
        coverage = 1.0
        method = "complete_no_singletons"
    elif m < 3:
        coverage = 1.0 - f1 / m
        method = "good_fallback"
    else:
        inner = (1.0 - f2 / (f1 + 1.0)) * (m - 1.0) / (m - 2.0) + f2 / ((f1 + 1.0) * (m - 2.0))
        coverage = 1.0 - (f1 / m) * inner
        method = "chao_jost_incidence"

    coverage = float(min(1.0, max(0.0, coverage)))
    tau = float(m * (1.0 - coverage))
    # A clamped coverage of exactly 0 (every taxon a singleton) makes the
    # unseen-species estimate f0 = f1 (1 - C) / C a division by zero.
    # Reporting an infinity would put a non-finite number into the JSON
    # payload (json.dumps emits "Infinity", which RFC 8259 does not allow),
    # so the estimate is reported as None: "not estimable", not "infinitely
    # many taxa".
    unseen = float(f1 * (1.0 - coverage) / coverage) if coverage > 0 else None

    logger.debug(
        "Sample coverage: m=%d, S_obs=%d, f1=%.0f, f2=%.0f, C=%.4f, tau=%.4f",
        m,
        observed_richness,
        f1,
        f2,
        coverage,
        tau,
    )
    return SampleCoverageResult(
        coverage=coverage,
        tau=tau,
        unseen_species=unseen,
        observed_richness=observed_richness,
        n_samples=m,
        f1=f1,
        f2=f2,
        method=method,
    )


@dataclass(frozen=True)
class SampleRarefactionResult:
    """Coverage-based (Mao's tau) sample rarefaction curve.

    Attributes:
        sample_sizes: Number of samples at each point.
        expected_richness: Expected taxa at each sample size.
        coverage: The coverage the curve was anchored on.
        tau: Mao's tau at the full sample size.
        extrapolated: Which points are beyond the observed sample count.
        n_samples: The observed sample count (m0) the curve is anchored on.
        s_observed: The observed richness (S_obs) the curve is anchored on.
    """

    sample_sizes: npt.NDArray
    expected_richness: npt.NDArray
    coverage: float
    tau: float
    extrapolated: npt.NDArray
    n_samples: int = 0
    s_observed: int = -1

    @property
    def observed_richness(self) -> int:
        """Richness at the full observed sample count.

        This is the curve's anchor, S_obs, and it is the same number
        whatever grid of sample sizes the caller asked for. Reading the
        curve at its *last requested* point instead would report an
        interpolated value (for a grid that stops below m0) or an
        extrapolated one (for a grid that runs past m0), neither of which
        is the observed richness.
        """
        if self.s_observed >= 0:
            return int(self.s_observed)
        # Fallback for a result built without the anchor: read the curve at
        # the largest requested size.
        sizes = np.asarray(self.sample_sizes)
        richness = np.asarray(self.expected_richness, dtype=float)
        if sizes.size == 0:
            return 0
        hits = np.flatnonzero(sizes == sizes.max())
        return round(float(richness[hits[0]] if hits.size else richness[-1]))

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        sizes = np.asarray(self.sample_sizes)
        richness = np.asarray(self.expected_richness)
        lines = [
            "Sample rarefaction (Mao's tau)",
            "=" * 40,
            f"Sample coverage (C): {self.coverage:.4f}",
            f"Mao's tau: {self.tau:.4f}",
            f"Points: {sizes.size} ({int(np.count_nonzero(np.asarray(self.extrapolated)))} extrapolated)",
            "",
            "  samples   expected taxa",
        ]
        for size, value in zip(sizes, richness, strict=False):
            lines.append(f"  {int(size):>7}   {value:>13.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "sample_sizes": [int(x) for x in np.asarray(self.sample_sizes).ravel()],
            "expected_richness": [float(x) for x in np.asarray(self.expected_richness).ravel()],
            "coverage": float(self.coverage),
            "tau": float(self.tau),
            "extrapolated": [bool(x) for x in np.asarray(self.extrapolated).ravel()],
            "n_samples": int(self.n_samples),
            "observed_richness": int(self.observed_richness),
        }


def sample_rarefaction_tao(
    incidence: npt.NDArray,
    sample_sizes: npt.NDArray | None = None,
) -> SampleRarefactionResult:
    """Sample-coverage rarefaction over the number of *samples*.

    Distinct from ``ecology.rarefaction.compute_sample_based_rarefaction``,
    which is the exact combinatorial (Hurlbert) curve over the incidence
    matrix. That one answers "if I drew ``k`` of these samples at random, how
    many taxa would I expect?". This one answers the coverage question --
    "how many more taxa would more sampling of this kind reveal?" -- and is
    anchored on the coverage of the survey rather than on its combinatorics,
    which is what makes it usable for small, unevenly sampled records.

    The curve interpolates below the observed sample count and extrapolates
    above it, both anchored so that ``S(m) = S_obs`` exactly at the observed
    count:

    .. math::

        S(m) = S_{obs} + f_1\\frac{m - m_0}{m_0} \\quad (m \\le m_0), \\qquad
        S(m) = S_{obs} + \\frac{f_1}{\\hat C}\\frac{m - m_0}{m_0} \\quad (m > m_0)

    The curve is a function of the sample count, so the order in which the
    caller supplies the sizes does not matter: the same set of sizes gives
    the same richness at each size whether it is given in ascending or
    descending order.

    Parameters
    ----------
    incidence : array-like, shape (n_samples, n_taxa)
        Presence/absence, one row per sample. A 1-D input is rejected: it is
        ambiguous between "one sample of many taxa" and "many samples of one
        taxon".
    sample_sizes : array-like, optional
        Sample counts to evaluate. Defaults to ``1 .. m``. Values above ``m``
        are extrapolated, not clipped -- silently clipping would answer a
        different question than the one asked. The sizes must be finite and
        positive.

    Returns
    -------
    SampleRarefactionResult

    Raises
    ------
    ValidationError
        If the incidence matrix is not 2-D, or if ``sample_sizes`` is empty,
        not finite, or not positive.
    """
    inc = _as_incidence(incidence)
    cov = sample_coverage(inc)
    m0 = cov.n_samples
    f1 = cov.f1

    if sample_sizes is None:
        sizes = np.arange(1, m0 + 1, dtype=int)
    else:
        try:
            sizes = np.asarray(sample_sizes, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "sample_sizes must be an array-like of sample counts",
                details={"n_samples": m0},
                original_exception=exc,
            ) from exc
        if sizes.size == 0:
            raise ValidationError(
                "sample_sizes is empty; supply at least one sample count to evaluate",
                details={"n_samples": m0},
            )
        # A NaN or an infinity is not a point on the curve. Casting one to
        # int overflows to INT64_MIN in numpy 2.x, and that negative size
        # would then be used as m.
        if not np.all(np.isfinite(sizes)):
            raise ValidationError(
                "sample_sizes must be finite; NaN and infinite sample counts are not points on the curve",
                details={"n_samples": m0},
            )
        if np.any(sizes < 1):
            raise ValidationError(
                f"sample_sizes must be positive, got values in [{sizes.min()}, {sizes.max()}]",
                details={"min": float(sizes.min()), "max": float(sizes.max())},
            )
        sizes = np.round(sizes).astype(int)
        # Re-checked on the rounded sizes, which are the ones the curve is
        # evaluated on.
        if np.any(sizes < 1):
            raise ValidationError(
                f"sample_sizes must be positive, got values in [{sizes.min()}, {sizes.max()}]",
                details={"min": int(sizes.min()), "max": int(sizes.max())},
            )

    s_obs = float(cov.observed_richness)
    slope = f1 / m0
    extrap_slope = (f1 / cov.coverage / m0) if cov.coverage > 0 else 0.0

    # The curve is evaluated on a *sorted* copy of the requested sizes and
    # mapped back afterwards. The monotonicity guard below walks the array
    # in the order it is given, so applying it to the caller's order made
    # the result depend on that order: a descending grid was flattened to a
    # constant, and the anchor S(m0) = S_obs was destroyed with it.
    order = np.argsort(sizes, kind="stable")
    sorted_sizes = sizes[order]

    expected_sorted = np.empty(sorted_sizes.size, dtype=float)
    for i, m in enumerate(sorted_sizes):
        delta = (m - m0) / m0
        # Interpolation uses f1/m0; extrapolation uses (f1/C)/m0, which is
        # steeper by exactly 1/C -- an under-sampled survey reveals taxa
        # faster per additional sample than a well-covered one.
        expected_sorted[i] = s_obs + delta * (slope if m <= m0 else extrap_slope) * m0

    # Guard the sign of the rarefaction curve: interpolation below the
    # observed size must not invent taxa, and both branches must be
    # non-decreasing in m. The floor is the smaller of one taxon and the
    # observed richness, so the curve can never claim a taxon the survey
    # did not see.
    floor = min(1.0, s_obs)
    expected_sorted = np.maximum.accumulate(np.maximum(expected_sorted, floor))

    expected = np.empty_like(expected_sorted)
    expected[order] = expected_sorted

    return SampleRarefactionResult(
        sample_sizes=sizes,
        expected_richness=expected,
        coverage=cov.coverage,
        tau=cov.tau,
        extrapolated=sizes > m0,
        n_samples=m0,
        s_observed=int(cov.observed_richness),
    )


# =============================================================================
# 3. Diversity profiles
# =============================================================================


@dataclass(frozen=True)
class DiversityProfileResult:
    """Every index as a function of sampling depth.

    Attributes:
        sample_name: Label of the profiled sample.
        depths: Quorums (individuals) at which each point was evaluated.
        index_values: index name -> value at each depth. An index absent at
            some depth (Pielou at S=1, Margalef at N=1) is ``NaN`` there
            rather than missing, so the arrays stay aligned with ``depths``.
    """

    sample_name: str
    depths: npt.NDArray
    index_values: dict[str, npt.NDArray] = field(default_factory=dict)

    @property
    def richness(self) -> npt.NDArray:
        """Observed richness at each depth."""
        return np.asarray(self.index_values.get("richness", np.full(self.depths.size, np.nan)), dtype=float)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        depths = np.asarray(self.depths)
        lines = [
            f"Diversity profile: {self.sample_name}",
            "=" * 40,
            f"Depth range: {int(depths.min())} - {int(depths.max())} individuals",
            f"Points: {depths.size}",
            "",
        ]
        header = f"{'depth':>7}" + "".join(f"  {key:>12}" for key in sorted(self.index_values))
        lines.append(header)
        lines.append("-" * len(header))
        for i, depth in enumerate(depths):
            row = f"{int(depth):>7}"
            for key in sorted(self.index_values):
                value = float(np.asarray(self.index_values[key])[i])
                row += f"  {value:>12.4f}" if np.isfinite(value) else f"  {'n/a':>12}"
            lines.append(row)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "sample_name": self.sample_name,
            "depths": [int(x) for x in np.asarray(self.depths).ravel()],
            "index_values": {k: [float(x) for x in np.asarray(v).ravel()] for k, v in self.index_values.items()},
        }


def compute_diversity_profile(
    abundances: npt.NDArray,
    quorums: npt.NDArray | None = None,
    sample_name: str = SQS_SAMPLE_NAME,
    n_points: int = 20,
) -> DiversityProfileResult:
    """Compute every diversity index across a range of sampling depths.

    Each depth is an SQS quorum, so the profile answers the practical
    question directly: *at equal sampling effort, how do these samples
    compare?* A divergence that only appears at high depth is a
    sampling-depth artefact; one that is stable across the curve is a real
    difference.

    Parameters
    ----------
    abundances : array-like
        Per-taxon counts for one sample.
    quorums : array-like of int, optional
        Depths to evaluate. Defaults to ``n_points`` values spread over
        ``1 .. N``. Values above ``N`` are rejected, because a depth the
        sample cannot support would have to be silently clipped. The depths
        must be finite whole numbers of individuals: a fractional depth is
        rejected rather than rounded, because rounding it would report a
        profile at a depth the caller did not ask for.
    sample_name : str
        Label for the result.
    n_points : int
        Number of depths when ``quorums`` is not given.

    Returns
    -------
    DiversityProfileResult

    Raises
    ------
    ValidationError
        If the sample has no positive abundances, if ``n_points`` is below 1,
        or if ``quorums`` is empty, not finite, fractional, or outside
        ``[1, N]``.
    """
    counts = _as_counts(abundances, "abundances")
    total = int(counts.sum())

    if quorums is None:
        if n_points < 1:
            raise ValidationError(
                f"n_points must be at least 1, got {n_points}",
                details={"n_points": n_points},
            )
        sizes = np.unique(np.linspace(1, total, min(int(n_points), total)).astype(int))
    else:
        try:
            sizes = np.asarray(quorums, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "quorums must be an array-like of sampling depths",
                details={"total_individuals": total},
                original_exception=exc,
            ) from exc
        if sizes.size == 0:
            raise ValidationError(
                "quorums is empty; supply at least one sampling depth",
                details={"total_individuals": total},
            )
        # A NaN or an infinity is not a depth. A NaN compares False against
        # both bounds, so it used to slip through the range check and then
        # crash the error message that formats the range with int().
        if not np.all(np.isfinite(sizes)):
            raise ValidationError(
                "quorums must be finite; NaN and infinite sampling depths are not points on the profile",
                details={"total_individuals": total},
            )
        # A depth is a count of individuals, so a fractional one is a
        # caller error rather than something to round away silently.
        non_integer = ~np.isclose(sizes, np.round(sizes), rtol=0.0, atol=1e-9)
        if np.any(non_integer):
            offenders = np.unique(sizes[non_integer])[:5]
            raise ValidationError(
                f"quorums must be whole numbers of individuals, got fractional depths {[float(x) for x in offenders]}",
                details={"offenders": [float(x) for x in offenders], "total_individuals": total},
            )
        sizes = sizes.astype(int)
        if np.any(sizes < 1) or np.any(sizes > total):
            raise ValidationError(
                f"quorums must lie in [1, {total}] individuals, got values in [{int(sizes.min())}, {int(sizes.max())}]",
                details={"min": int(sizes.min()), "max": int(sizes.max()), "total_individuals": total},
            )
        sizes = np.unique(sizes)
        sizes = sizes[(sizes >= 1) & (sizes <= total)]
        # Checked again after filtering: an empty profile has no depths, and
        # its summary() would then fail on a zero-size array.
        if sizes.size == 0:
            raise ValidationError(
                "quorums is empty after filtering; supply at least one sampling depth",
                details={"total_individuals": total},
            )

    curves: dict[str, list[float]] = {"richness": []}
    for key in INDEX_KEYS:
        curves[key] = []

    for depth in sizes:
        result = sqs_subsample(counts, int(depth), sample_name)
        curves["richness"].append(float(result.taxa_retained))
        for key in INDEX_KEYS:
            value = None if result.standardized_diversity is None else result.standardized_diversity.get(key)
            curves[key].append(float("nan") if value is None else float(value))

    return DiversityProfileResult(
        sample_name=sample_name,
        depths=sizes,
        index_values={k: np.asarray(v, dtype=float) for k, v in curves.items()},
    )


# =============================================================================
# 4. Diversity t test and permutation test
# =============================================================================


@dataclass(frozen=True)
class DiversityTestResult:
    """Per-index comparison of diversity between groups.

    The test is applied **one index at a time** to the per-sample index
    values -- not to the index vectors as a whole. There is no joint
    distribution over "Shannon and Simpson together" to test against, so a
    single multivariate p-value would be a fiction; running one test per index
    is also what the reference tool does. The multiplicity that introduces is
    reported through ``n_indices`` rather than silently corrected for.

    Attributes:
        test: ``"ttest"`` / ``"anova"`` / ``"permutation"``.
        statistic_name: ``"t"``, ``"F"``, or the permutation statistic's name.
        statistics: index name -> test statistic.
        p_values: index name -> p-value.
        group_means: index name -> group label -> mean index value.
        group_sizes: Group label -> number of samples.
        index_names: Indices actually tested.
        groups: Group labels, in the order they were supplied.
        significant_indices: Indices with ``p < 0.05``.
        p_value_two_sided: index name -> two-sided p-value, for the
            permutation test. Empty for the parametric tests.
        n_permutations / random_seed: Permutation bookkeeping, or ``None``.
        test_per_index: Whether each index was tested separately (always
            ``True`` here; present so a caller reading the result does not
            have to know).
    """

    test: str
    statistic_name: str
    statistics: dict[str, float]
    p_values: dict[str, float]
    group_means: dict[str, dict[str, float]]
    group_sizes: dict[str, int]
    index_names: tuple[str, ...]
    groups: tuple[str, ...] = ()
    significant_indices: tuple[str, ...] = ()
    p_value_two_sided: dict[str, float] = field(default_factory=dict)
    n_permutations: int | None = None
    random_seed: int | None = None
    test_per_index: bool = True

    @property
    def alpha_level(self) -> float:
        """Significance threshold the ``significant_indices`` filter used."""
        return 0.05

    def significant(self, index: str) -> bool:
        """Whether one index rejects at the 0.05 level."""
        return bool(self.p_values.get(index, 1.0) < 0.05)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        lines = [
            f"Diversity {self.test} (per index)",
            "=" * 46,
            f"Groups ({len(self.group_sizes)}): " + ", ".join(f"{k}(n={v})" for k, v in self.group_sizes.items()),
            f"Statistic: {self.statistic_name}",
            f"Indices tested: {len(self.index_names)}",
            "",
            f"{'index':<14}{'statistic':>12}{'p-value':>12}  group means",
            "-" * 70,
        ]
        for index in self.index_names:
            stat = self.statistics.get(index, float("nan"))
            pval = self.p_values.get(index, float("nan"))
            means = self.group_means.get(index, {})
            means_txt = "  ".join(f"{k}={v:.4f}" for k, v in means.items())
            flag = "*" if self.significant(index) else " "
            lines.append(f"{index:<14}{stat:>12.4f}{pval:>12.4f}{flag} {means_txt}")
        lines.append("")
        lines.append("* = significant at p < 0.05")
        if self.n_permutations is not None:
            lines.append(f"Permutations: {self.n_permutations} (seed={self.random_seed})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "test": self.test,
            "statistic_name": self.statistic_name,
            "statistics": {k: float(v) for k, v in self.statistics.items()},
            "p_values": {k: float(v) for k, v in self.p_values.items()},
            "p_value_two_sided": {k: float(v) for k, v in self.p_value_two_sided.items()},
            "group_means": {k: dict(v) for k, v in self.group_means.items()},
            "group_sizes": {k: int(v) for k, v in self.group_sizes.items()},
            "index_names": list(self.index_names),
            "significant_indices": list(self.significant_indices),
            "test_per_index": self.test_per_index,
            "n_permutations": self.n_permutations,
            "random_seed": self.random_seed,
        }


def _index_matrix(abundance_matrix: npt.NDArray, sample_names: list[str]) -> dict[str, npt.NDArray]:
    """Per-sample index values, keeping only indices every sample produced.

    ``compute_diversity_indices`` omits Pielou when S == 1 and Margalef when
    N <= 1 or S < 2, so the set of available keys varies by row. Testing an
    index that only some samples have would compare against an undefined
    value, so such indices are dropped here and reported through the result's
    ``index_names``. A dropped index is also logged, so an index that
    silently disappears (Fisher's alpha fails to converge when every taxon is
    a singleton) is visible to the user instead of vanishing.

    Parameters
    ----------
    abundance_matrix : array-like, shape (n_samples, n_taxa)
        One row per sample. A 1-D input is read as a single sample of many
        taxa, which is what the reshape below has always intended: the
        validator would otherwise turn it into "many samples of one taxon".
    sample_names : list of str
        Labels for the samples. Empty means "generate them".

    Returns
    -------
    dict[str, numpy.ndarray]
        Index name -> per-sample values. Always contains ``"richness"``.

    Raises
    ------
    ValidationError
        If the matrix has no rows, is not 2-D, has a different number of
        names than rows, or a sample has no individuals at all.
    """
    matrix = validate_data_array(abundance_matrix, allow_nan=False, name="abundance_matrix", preserve_dimensions=True)
    if matrix.ndim == 1:
        # One sample, many taxa. The dimensionality is captured before the
        # validator's reshape, so this branch is reachable.
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim != 2:
        raise ValidationError(
            f"abundance_matrix must be a 2-D samples x taxa matrix, got {matrix.ndim}-d",
            details={"shape": tuple(int(x) for x in matrix.shape), "ndim": int(matrix.ndim)},
        )
    if matrix.shape[0] == 0:
        raise ValidationError(
            "abundance_matrix has no rows; there is nothing to compare",
            details={"shape": tuple(int(x) for x in matrix.shape)},
        )

    # Default labels have to be generated here rather than at the call site:
    # indexing `sample_names[i]` against an empty list raised IndexError for
    # every call that did not pass names, which is the common case.
    names = list(sample_names) if sample_names else [f"Sample_{i + 1}" for i in range(matrix.shape[0])]
    if len(names) != matrix.shape[0]:
        raise ValidationError(
            f"sample_names has {len(names)} name(s) but the abundance matrix has {matrix.shape[0]} row(s)",
            details={"n_names": len(names), "n_samples": int(matrix.shape[0])},
        )

    results = []
    for i in range(matrix.shape[0]):
        try:
            results.append(compute_diversity_indices(matrix[i], names[i]))
        except ComputationError as exc:
            # A sample with no individuals has no indices. The dependency
            # reports that as a computation error; from the caller's point
            # of view it is bad input, and the message says which sample.
            raise ValidationError(
                f"sample {i} ({names[i]}) has no positive abundances; diversity indices are "
                "undefined for a sample with no individuals",
                details={"sample_index": i, "sample_name": names[i]},
                original_exception=exc,
            ) from exc

    common = set(INDEX_KEYS)
    missing: dict[str, list[str]] = {}
    for res in results:
        for key in INDEX_KEYS:
            if key not in res.indices:
                missing.setdefault(key, []).append(res.sample_name)
        common &= set(res.indices)
    if missing:
        logger.warning(
            "Diversity test: index/indices %s are undefined for at least one sample and are not tested: %s",
            sorted(missing),
            {key: sample_names for key, sample_names in missing.items()},
        )
    common = [key for key in INDEX_KEYS if key in common]

    values = {key: np.asarray([res.indices[key].value for res in results], dtype=float) for key in common}
    values["richness"] = np.asarray([float(res.taxa_count) for res in results], dtype=float)
    return values


def _validate_groups(groups: npt.NDArray, n_samples: int) -> dict[Any, npt.NDArray]:
    """Map group labels to row positions, rejecting groups too small to test."""
    labels = np.asarray(groups).ravel()
    if labels.size != n_samples:
        raise ValidationError(
            f"groups has {labels.size} label(s) but the abundance matrix has {n_samples} sample(s); "
            "they must line up one label per sample.",
            details={"n_groups": int(labels.size), "n_samples": int(n_samples)},
        )
    buckets = group_indices(labels)
    if len(buckets) < 2:
        raise ValidationError(
            f"a diversity test needs at least 2 groups to compare, got {len(buckets)}",
            details={"n_groups": len(buckets)},
        )
    too_small = {str(k): int(v.size) for k, v in buckets.items() if v.size < 2}
    if too_small:
        raise StatisticalError(
            f"every group needs at least 2 samples for a test on index values; too small: {too_small}",
            details={"group_sizes": {str(k): int(v.size) for k, v in buckets.items()}},
        )
    return buckets


def diversity_t_test(
    abundance_matrix: npt.NDArray,
    groups: npt.NDArray,
    indices: npt.NDArray | None = None,
    equal_var: bool = True,
    sample_names: list[str] | None = None,
) -> DiversityTestResult:
    """Test whether diversity differs between groups, one index at a time.

    Each sample's indices are computed with
    :func:`ecology.diversity.compute_diversity_indices`, then **each index is
    tested separately across the groups**: a Student's t test for exactly two
    groups, a one-way ANOVA for three or more. The statistic is computed on
    index *values*, not on the index vector as a whole.

    Parameters
    ----------
    abundance_matrix : array-like, shape (n_samples, n_taxa)
        One row per sample. A 1-D input is read as a single sample of many
        taxa, so the group labels must then be a single label.
    groups : array-like, shape (n_samples,)
        Group label per sample.
    indices : sequence of str, optional
        Which indices to test. Defaults to every index available in every
        sample, plus ``"richness"``.
    equal_var : bool
        Pool variances (Student's t) when there are two groups. Ignored for
        more than two groups.
    sample_names : list of str, optional
        Labels for the samples.

    Returns
    -------
    DiversityTestResult

    Raises
    ------
    ValidationError
        Fewer than two groups, a label/row count mismatch, or a sample with
        no individuals.
    StatisticalError
        Any group has fewer than two samples, so its variance is undefined,
        or no requested index carries a testable difference: an index that
        is constant across the samples, or constant within every group,
        would give an undefined statistic rather than a merely
        insignificant one.
    """
    values = _index_matrix(abundance_matrix, sample_names or [])
    n_samples = len(next(iter(values.values())))
    buckets = _validate_groups(groups, n_samples)

    selected = _select_indices(values, indices, buckets)
    two_groups = len(buckets) == 2

    statistics: dict[str, float] = {}
    p_values: dict[str, float] = {}
    group_means: dict[str, dict[str, float]] = {}
    tested: list[str] = []

    for key in selected:
        per_group = [values[key][idx] for idx in buckets.values()]
        if two_groups:
            result = sp_stats.ttest_ind(per_group[0], per_group[1], equal_var=equal_var)
        else:
            result = sp_stats.f_oneway(*per_group)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            # Safety net: nothing non-finite is ever reported. An infinite
            # statistic means the pooled variance was zero -- already
            # excluded by _select_indices -- and a NaN p-value is not a
            # probability.
            logger.warning(
                "Diversity t test: index '%s' produced a non-finite statistic (%r) and is not reported",
                key,
                statistic,
            )
            continue
        tested.append(key)
        statistics[key] = statistic
        p_values[key] = p_value
        group_means[key] = {str(label): float(np.mean(vals)) for label, vals in zip(buckets, per_group, strict=False)}

    if not tested:
        raise StatisticalError(
            "no index could be tested: every one produced a non-finite statistic",
            details={"indices": list(selected)},
        )

    significant = tuple(key for key in tested if p_values[key] < 0.05)
    logger.info(
        "Diversity %s across %d group(s): %d index/indices significant",
        "t test" if two_groups else "ANOVA",
        len(buckets),
        len(significant),
    )
    return DiversityTestResult(
        test="ttest" if two_groups else "anova",
        statistic_name="t" if two_groups else "F",
        statistics=statistics,
        p_values=p_values,
        group_means=group_means,
        group_sizes={str(label): int(idx.size) for label, idx in buckets.items()},
        index_names=tuple(tested),
        groups=tuple(str(label) for label in buckets),
        significant_indices=significant,
        p_value_two_sided={},
        n_permutations=None,
        random_seed=None,
        test_per_index=True,
    )


def _zero_within_group_variance(vector: npt.NDArray, buckets: dict[Any, npt.NDArray]) -> bool:
    """Whether every group is constant, so the pooled variance would be zero.

    A two-sample t test divides by the pooled within-group variance and an
    ANOVA divides by the within-group sum of squares. Both are zero exactly
    when every group is constant, and the statistic is then infinite (or
    NaN when the group means are equal too).

    Parameters
    ----------
    vector : numpy.ndarray
        The per-sample values of one index.
    buckets : dict
        Group label -> row positions.

    Returns
    -------
    bool
        True when the index carries no within-group variance at all.
    """
    values = np.asarray(vector, dtype=float)
    return all(np.ptp(values[positions]) <= 0 for positions in buckets.values())


def _select_indices(
    values: dict[str, npt.NDArray],
    indices: npt.NDArray | None,
    buckets: dict[Any, npt.NDArray] | None = None,
) -> list[str]:
    """Validate and resolve the requested index names against what exists.

    Indices that are constant across every sample are dropped. They carry no
    information, and -- worse -- they break the test rather than quietly
    failing it: ``ttest_ind`` returns ``nan`` for zero within-group variance,
    and a ``nan`` then flows into the permutation statistic as well. In
    ``permutation_pvalue`` the ``null >= observed`` comparison is ``False``
    for every ``nan`` on either side, so ``n_ge`` is 0 and the p-value
    collapses to ``1 / (n + 1)`` -- the *most significant value the
    machinery can report*. A constant index therefore came out "significant
    at p = 0.005" purely because it had no variance to compare. Dropping it
    here is the fix; the alternative (reporting ``nan`` and letting the
    caller judge) would ship a false positive through a plausible-looking
    number.

    The same reasoning applies to an index that is constant *within every
    group* while differing between groups: the pooled variance is zero, the
    t statistic is infinite and the p-value comes back as 0.0 -- a number
    that is not a probability. Such an index is dropped too, so the t test
    and the permutation test agree on which indices are testable at all.

    Parameters
    ----------
    values : dict[str, numpy.ndarray]
        Index name -> per-sample values.
    indices : numpy.ndarray or None
        The requested index names, or None for "every available index".
    buckets : dict, optional
        Group label -> row positions. When given, indices with no
        within-group variance are dropped as well.

    Returns
    -------
    list[str]
        The indices that can actually be tested.

    Raises
    ------
    ValidationError
        If an unknown index is requested, or the requested list is empty.
    StatisticalError
        If no requested index carries any testable difference.
    """
    available = list(values)
    if indices is None:
        candidates = available
    else:
        requested = [str(name) for name in np.asarray(indices).ravel()]
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValidationError(
                f"requested index/indices {unknown} are not available for every sample; available: {available}",
                details={"unknown": unknown, "available": available},
            )
        if not requested:
            raise ValidationError(
                "indices was given but empty; pass at least one index name",
                details={"available": available},
            )
        candidates = requested

    varying = [key for key in candidates if np.ptp(np.asarray(values[key], dtype=float)) > 0]
    constant = [key for key in candidates if key not in varying]
    if constant:
        logger.info(
            "Diversity test: skipping index/indices %s -- constant across all samples, "
            "so there is no between-group difference to test",
            constant,
        )
    if not varying:
        raise StatisticalError(
            "every requested index is constant across all samples; there is no between-group difference to test",
            details={"constant_indices": constant},
        )

    degenerate: list[str] = []
    if buckets is not None:
        degenerate = [key for key in varying if _zero_within_group_variance(values[key], buckets)]
        if degenerate:
            logger.info(
                "Diversity test: skipping index/indices %s -- constant within every group, so the "
                "pooled variance is zero and the test statistic would be undefined",
                degenerate,
            )
    testable = [key for key in varying if key not in degenerate]
    if not testable:
        raise StatisticalError(
            "every requested index is constant within each group; the pooled variance is zero, "
            "so there is no testable between-group difference",
            details={"constant_indices": constant, "degenerate_indices": degenerate},
        )
    return testable


def diversity_permutation_test(
    abundance_matrix: npt.NDArray,
    groups: npt.NDArray,
    indices: npt.NDArray | None = None,
    n_permutations: int = 999,
    random_seed: int | None = None,
    sample_names: list[str] | None = None,
) -> DiversityTestResult:
    """Permutation test of the same per-index comparison as the t test.

    The statistic is recomputed after shuffling the group labels, so the null
    is "the index does not know which group it came from" -- no normality
    assumption, which is the reason to prefer it over the t test on the same
    data.

    For two groups the statistic is ``|t|`` and the permutation is two-sided
    by symmetry; for three or more it is the ANOVA ``F``, which is
    one-sided by construction.

    Parameters
    ----------
    abundance_matrix : array-like, shape (n_samples, n_taxa)
        One row per sample. A 1-D input is read as a single sample of many
        taxa, so the group labels must then be a single label.
    groups : array-like, shape (n_samples,)
        Group label per sample.
    indices : sequence of str, optional
        Which indices to test.
    n_permutations : int
        Number of label shuffles (>= 1).
    random_seed : int, optional
        Seed for reproducibility. Without one the result is not reproducible
        and ``utils.statistics_core.make_rng`` warns.
    sample_names : list of str, optional
        Labels for the samples.

    Returns
    -------
    DiversityTestResult
        With ``n_permutations``, ``random_seed`` and ``p_value_two_sided``
        populated.

    Raises
    ------
    ValidationError
        Fewer than two groups, a label/row count mismatch, or a sample with
        no individuals.
    StatisticalError
        Any group has fewer than two samples, or no requested index carries
        a testable difference -- the same indices the t test refuses, so
        the two tests agree on what is testable.
    """
    values = _index_matrix(abundance_matrix, sample_names or [])
    n_samples = len(next(iter(values.values())))
    buckets = _validate_groups(groups, n_samples)
    selected = _select_indices(values, indices, buckets)

    labels = np.asarray(groups).ravel()
    label_order = list(buckets)
    two_groups = len(buckets) == 2
    stat_name = "|t|" if two_groups else "F"

    statistics: dict[str, float] = {}
    p_values: dict[str, float] = {}
    p_two: dict[str, float] = {}
    tested: list[str] = []

    for key in selected:
        vector = values[key]

        def _statistic(rng: np.random.Generator, _v: npt.NDArray = vector) -> float:
            shuffled = rng.permutation(labels)
            groups_here = [np.flatnonzero(shuffled == label) for label in label_order]
            if two_groups:
                return float(abs(sp_stats.ttest_ind(_v[groups_here[0]], _v[groups_here[1]], equal_var=True).statistic))
            return float(sp_stats.f_oneway(*[_v[g] for g in groups_here]).statistic)

        if two_groups:
            observed = abs(
                float(
                    sp_stats.ttest_ind(
                        values[key][buckets[label_order[0]]], values[key][buckets[label_order[1]]]
                    ).statistic
                )
            )
            alternative = "greater"
        else:
            observed = float(sp_stats.f_oneway(*[vector[idx] for idx in buckets.values()]).statistic)
            alternative = "greater"

        if not np.isfinite(observed):
            # Safety net, matching the t test: an infinite statistic means
            # the pooled variance was zero, which _select_indices has
            # already excluded. Nothing non-finite is ever reported.
            logger.warning(
                "Diversity permutation test: index '%s' produced a non-finite statistic (%r) and is not reported",
                key,
                observed,
            )
            continue

        perm = permutation_pvalue(
            observed,
            _statistic,
            n_permutations=n_permutations,
            random_seed=random_seed,
            context=f"diversity permutation test ({key})",
            alternative=alternative,
        )
        tested.append(key)
        statistics[key] = float(observed)
        p_values[key] = float(perm.p_value)
        p_two[key] = float(perm.p_value_two_sided)

    if not tested:
        raise StatisticalError(
            "no index could be tested: every one produced a non-finite statistic",
            details={"indices": list(selected)},
        )

    group_means = {
        key: {str(label): float(np.mean(values[key][idx])) for label, idx in buckets.items()} for key in tested
    }
    significant = tuple(key for key in tested if p_values[key] < 0.05)
    logger.info(
        "Diversity permutation test across %d group(s), %d permutations: %d index/indices significant",
        len(buckets),
        n_permutations,
        len(significant),
    )
    return DiversityTestResult(
        test="permutation",
        statistic_name=stat_name,
        statistics=statistics,
        p_values=p_values,
        group_means=group_means,
        group_sizes={str(label): int(idx.size) for label, idx in buckets.items()},
        index_names=tuple(tested),
        groups=tuple(str(label) for label in buckets),
        significant_indices=significant,
        p_value_two_sided=p_two,
        n_permutations=n_permutations,
        random_seed=random_seed,
        test_per_index=True,
    )


# =============================================================================
# Analyzer facade
# =============================================================================


class DiversityExtraAnalyzer:
    """Single entry point for the diversity sampling and comparison tools.

    Mirrors :class:`stats.mantel.MantelAnalyzer`: every method returns a frozen
    result and records it as :attr:`last_result`.
    """

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.DiversityExtraAnalyzer")
        self._last_result: (
            SQSResult
            | SampleCoverageResult
            | SampleRarefactionResult
            | DiversityProfileResult
            | DiversityTestResult
            | None
        ) = None

    def subsample(self, abundances: npt.NDArray, quorum: int, sample_name: str = SQS_SAMPLE_NAME) -> SQSResult:
        """Run :func:`sqs_subsample`."""
        result = sqs_subsample(abundances, quorum, sample_name)
        self._last_result = result
        return result

    def coverage(self, incidence: npt.NDArray) -> SampleCoverageResult:
        """Run :func:`sample_coverage`."""
        result = sample_coverage(incidence)
        self._last_result = result
        return result

    def rarefaction(self, incidence: npt.NDArray, sample_sizes: npt.NDArray | None = None) -> SampleRarefactionResult:
        """Run :func:`sample_rarefaction_tao`."""
        result = sample_rarefaction_tao(incidence, sample_sizes)
        self._last_result = result
        return result

    def profile(
        self,
        abundances: npt.NDArray,
        quorums: npt.NDArray | None = None,
        sample_name: str = SQS_SAMPLE_NAME,
        n_points: int = 20,
    ) -> DiversityProfileResult:
        """Run :func:`compute_diversity_profile`."""
        result = compute_diversity_profile(abundances, quorums, sample_name, n_points)
        self._last_result = result
        return result

    def t_test(
        self,
        abundance_matrix: npt.NDArray,
        groups: npt.NDArray,
        indices: npt.NDArray | None = None,
        equal_var: bool = True,
        sample_names: list[str] | None = None,
    ) -> DiversityTestResult:
        """Run :func:`diversity_t_test`."""
        result = diversity_t_test(abundance_matrix, groups, indices, equal_var, sample_names)
        self._last_result = result
        return result

    def permutation_test(
        self,
        abundance_matrix: npt.NDArray,
        groups: npt.NDArray,
        indices: npt.NDArray | None = None,
        n_permutations: int = 999,
        random_seed: int | None = None,
        sample_names: list[str] | None = None,
    ) -> DiversityTestResult:
        """Run :func:`diversity_permutation_test`."""
        result = diversity_permutation_test(
            abundance_matrix, groups, indices, n_permutations, random_seed, sample_names
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> Any:
        """Most recent result from any of the methods, or ``None``."""
        return self._last_result
