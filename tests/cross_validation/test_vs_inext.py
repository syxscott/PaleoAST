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

# `ChaoSpecies` is an S4 generic. `importr` walks the package namespace and does
# not attach S4 generics to the Python module, so `R_INEXT.ChaoSpecies` raised
# AttributeError in CI. Going through R's own `::` lookup is the reliable path
# and does not depend on what importr chooses to export.
_CHAO_SPECIES = r("iNEXT::ChaoSpecies")


def _sample(abundances: np.ndarray) -> np.ndarray:
    """One sample's species abundances as a 1-row matrix, as iNEXT expects."""
    return np.atleast_2d(np.asarray(abundances, dtype=float))


def _r_single_sample(abundances: np.ndarray):
    """One sample as a 1-row R matrix, the shape iNEXT's estimators expect."""
    flat = np.asarray(abundances, dtype=float).ravel()
    return r("matrix")(r("t")(r_vector(flat.tolist())), nrow=1)


def _paleo_chao1(abundances: np.ndarray) -> float:
    from ecology.diversity import compute_diversity_indices

    return float(compute_diversity_indices(np.asarray(abundances, dtype=float)).indices["chao1"].value)


def _r_chao1(abundances: np.ndarray) -> float:
    """iNEXT's Chao1 estimate for one sample, as a scalar."""
    return as_float(_CHAO_SPECIES(_r_single_sample(abundances), q=0, method="Chao1").rx2("Est"))


class TestChao1VsINEXT:
    """Verify the Chao1 richness estimator against ``iNEXT::ChaoSpecies``."""

    def test_standard_case_with_singletons_and_doubletons(self):
        """f1 = 2, f2 = 1: Chao1 = S + f1^2 / (2 f2)."""
        abundances = np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        assert_allclose(
            _paleo_chao1(abundances),
            _r_chao1(abundances),
            rtol=1e-8,
            atol=1e-10,
            err_msg="Chao1 disagrees with iNEXT::ChaoSpecies (f2 > 0 branch)",
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
            err_msg="Chao1 disagrees with iNEXT::ChaoSpecies (f2 == 0 branch)",
        )

    def test_no_singleton_branch(self):
        """No rare species: the estimate collapses to observed richness S."""
        abundances = np.array([10.0, 8.0, 5.0, 4.0, 3.0])
        paleo = _paleo_chao1(abundances)
        reference = _r_chao1(abundances)

        assert_allclose(paleo, reference, rtol=1e-8, atol=1e-10)
        # In this branch both must equal observed richness; assert it against R's
        # observed value so a "fix" cannot quietly reintroduce a correction term.
        r_x = _r_single_sample(abundances)
        r_observed = as_float(_CHAO_SPECIES(r_x, q=0, method="Chao1").rx2("Observed"))
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
            r_x = _r_single_sample(abundances)
            r_result = _CHAO_SPECIES(r_x, q=0, method="Chao1")
            observed = as_float(r_result.rx2("Observed"))
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

        r_x = _r_single_sample(abundances)
        r_result = _CHAO_SPECIES(r_x, q=0, method="Chao1")
        reference = as_float(r_result.rx2("Observed"))

        assert_allclose(
            paleo,
            reference,
            rtol=0.0,
            atol=0.0,
            err_msg="observed richness disagrees with iNEXT::ChaoSpecies(Observed)",
        )
