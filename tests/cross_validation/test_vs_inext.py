# =============================================================================
# FILE: tests/cross_validation/test_vs_inext.py
# =============================================================================
"""
Cross-validation against R's ``iNEXT`` package.

iNEXT provides the Chao estimators that the coverage-based rarefaction
literature is written against. Comparing against it matters because the Chao1
estimator has several published variants (bias-corrected, the ``f2 == 0``
fallback, the ``n < 2m`` regime) and an implementation can be "reasonable" and
still disagree with the reference in exactly those branches.

Comparisons are live.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import as_float, r, r_vector, require

pytestmark = pytest.mark.cross_validation

R_INEXT = require("iNEXT")

# iNEXT has no `ChaoSpecies` -- that name is not in its exported namespace, which
# is why `importr("iNEXT").ChaoSpecies` raised AttributeError and
# `r("iNEXT::ChaoRichness")` then raised
# "'ChaoSpecies' is not an exported object from 'namespace:iNEXT'".
# The current entry point is `ChaoRichness`, which returns a data.frame with one
# row per estimator; the first row is the default, Chao1.
_CHAO_RICHNESS = r("iNEXT::ChaoRichness")


def _sample(abundances: np.ndarray) -> np.ndarray:
    """One sample's species abundances as a 1-row matrix, as iNEXT expects."""
    return np.atleast_2d(np.asarray(abundances, dtype=float))


def _r_single_sample(abundances: np.ndarray):
    """One sample as a **column** vector, n rows x 1 column.

    The orientation matters and was wrong. ``ChaoRichness`` counts taxa down
    the rows, so a 1 x n row vector was read as *one taxon* with a single
    sample: the fixture with abundances [10, 8, 5, 5, 3, 2, 2, 1, 1] came back
    with Observed = 1 and Chao1 = 1.0, while the textbook answer for nine
    taxa is S_obs = 9, f1 = 2, f2 = 2, so Chao1 = 9 + 4/4 = 10.0 -- which is
    what PaleoAST returns, and what the failing CI run reported as ACTUAL.

    The 10x ratio was the tell: 1.0 is one taxon's Chao1, not nine taxa's.

    ``r("t")`` is gone from here for a second reason: ``t()`` on a plain
    numeric *vector* is a no-op in R, it only transposes a matrix. It was
    never doing anything, and the explicit ``nrow``/``ncol`` below says what
    shape is actually wanted.
    """
    flat = np.asarray(abundances, dtype=float).ravel()
    return r("matrix")(r_vector(flat.tolist()), nrow=flat.size, ncol=1)


def _paleo_chao1(abundances: np.ndarray) -> float:
    from ecology.diversity import compute_diversity_indices

    return float(compute_diversity_indices(np.asarray(abundances, dtype=float)).indices["chao1"].value)


#: Column names this file reads out of an ``iNEXT::ChaoRichness`` data.frame.
#: The estimate column is ``Estimator``, not ``Est`` -- confirmed against the
#: real return value, whose names are
#: ``['Observed', 'Estimator', 'Est_s.e.', '95% Lower', '95% Upper']``.
#: ``Est_s.e.`` is a *different* quantity (the standard error), so guessing a
#: prefix here would have compared the estimate against its own error bar.
_CHAO_OBSERVED = "Observed"
_CHAO_ESTIMATE = "Estimator"


def _chao_columns(result) -> tuple[int, int]:
    """Locate the ``Observed`` and ``Estimator`` columns of a ChaoRichness result.

    Read by name rather than by position, and refuse to guess: the previous
    version of this file assumed a fixed column layout and would have compared
    the wrong numbers -- or the standard error -- without saying so. That
    refusal is also what made the ``Est``/``Estimator`` mix-up obvious instead
    of silent.
    """
    names = list(result.names)
    missing = [c for c in (_CHAO_OBSERVED, _CHAO_ESTIMATE) if c not in names]
    if missing:
        raise AssertionError(f"iNEXT::ChaoRichness result has no column(s) {missing}; it returned {names}")
    return names.index(_CHAO_OBSERVED), names.index(_CHAO_ESTIMATE)


def _r_chao(abundances: np.ndarray) -> tuple[float, float]:
    """(observed richness, Chao1 estimate) from iNEXT, for one sample."""
    result = _CHAO_RICHNESS(_r_single_sample(abundances))
    observed_col, est_col = _chao_columns(result)
    names = list(result.names)
    # Read a named column out of the data.frame and take its first element.
    # Two earlier attempts failed for the same reason: rpy2's data.frame
    # wrapper exposes neither tuple indexing (`result[0, col]` -> "Indices must
    # be integers or slices, not <class 'tuple'>") nor string indexing
    # (`result[name]` -> "Indices must be integers or slices, not <class
    # 'str'>"). ``rx2`` is the accessor that works, and it is the same one
    # this suite already uses for prcomp's sdev and rotation.
    #
    # Row 0 is the default estimator, Chao1.
    observed = as_float(result.rx2(names[observed_col])[0])
    estimate = as_float(result.rx2(names[est_col])[0])
    return observed, estimate


def _r_chao1(abundances: np.ndarray) -> float:
    """iNEXT's Chao1 estimate for one sample, as a scalar."""
    return _r_chao(abundances)[1]


def _r_observed(abundances: np.ndarray) -> float:
    """iNEXT's observed species richness for one sample."""
    return _r_chao(abundances)[0]


class TestChao1VsINEXT:
    """Verify the Chao1 richness estimator against ``iNEXT::ChaoRichness``."""

    def test_standard_case_with_singletons_and_doubletons(self):
        """f1 = 2, f2 = 1: Chao1 = S + f1^2 / (2 f2)."""
        abundances = np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        assert_allclose(
            _paleo_chao1(abundances),
            _r_chao1(abundances),
            rtol=1e-8,
            atol=1e-10,
            err_msg="Chao1 disagrees with iNEXT::ChaoRichness (f2 > 0 branch)",
        )

    def test_no_doubletons_branch(self):
        """f2 = 0 with singletons present: the published fallback branch.

        This is the branch implementations most often get wrong, and it is the
        one a hand-written test is least likely to cover.
        """
        abundances = np.array([10.0, 8.0, 3.0, 3.0, 1.0, 1.0, 1.0])
        assert_allclose(
            _paleo_chao1(abundances),
            _r_chao1(abundances),
            rtol=1e-8,
            atol=1e-10,
            err_msg="Chao1 disagrees with iNEXT::ChaoRichness (f2 == 0 branch)",
        )

    def test_no_singleton_branch(self):
        """No rare species: the estimate collapses to observed richness S."""
        abundances = np.array([10.0, 8.0, 5.0, 4.0, 3.0])
        paleo = _paleo_chao1(abundances)
        reference = _r_chao1(abundances)

        assert_allclose(paleo, reference, rtol=1e-8, atol=1e-10)
        # In this branch both must equal observed richness; assert it against R's
        # observed value so a "fix" cannot quietly reintroduce a correction term.
        r_observed = _r_observed(abundances)
        assert_allclose(paleo, r_observed, rtol=1e-8, atol=1e-10)

    def test_estimate_is_never_below_observed(self):
        """Chao1 is an upper-bound estimator: never below observed richness.

        Checked across a spread of samples against iNEXT's own ``Observed``,
        which is a property both implementations must share and which a
        comparison against a single hardcoded case would not exercise.
        """
        samples = [
            np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0]),
            np.array([6.0, 6.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        ]
        for abundances in samples:
            observed = _r_observed(abundances)
            paleo = _paleo_chao1(abundances)
            assert paleo >= observed - 1e-9, (
                f"Chao1 {paleo} is below iNEXT's observed richness {observed} for {abundances.tolist()}"
            )


class TestRarefactionVsINEXT:
    """Verify observed richness against iNEXT's sample-order-1 curve."""

    def test_observed_richness_matches(self):
        """The t=1 (observed) point of the q=0 curve equals observed richness.

        iNEXT's q=0 curve starts at the observed number of species, which is
        the one deterministic point on an otherwise interpolated curve -- the
        curve itself is model-based and depends on ``ntree`` draws, so only the
        observed endpoint is comparable without fixing R's RNG.
        """
        from ecology.diversity import compute_diversity_indices

        abundances = np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        paleo = float(compute_diversity_indices(abundances).taxa_count)

        reference = _r_observed(abundances)

        assert_allclose(
            paleo,
            reference,
            rtol=0.0,
            atol=0.0,
            err_msg="observed richness disagrees with iNEXT::ChaoRichness(Observed)",
        )
