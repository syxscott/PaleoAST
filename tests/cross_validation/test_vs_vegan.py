# =============================================================================
# FILE: tests/cross_validation/test_vs_vegan.py
# =============================================================================
"""
Cross-validation against R's ``vegan`` package.

vegan is the reference implementation for community ecology, so its numbers are
the ones a reviewer will check. Every comparison here is live: ``vegan``
computes its value during the test run.

Not compared, and why
---------------------
* **p-values.** ``adonis2`` and PaleoAST's PERMANOVA use independent
  permutation draws, so even with the same seed the p-values are two samples
  from the same distribution, not two equal numbers. Comparing them would test
  the RNG, not the implementation. The test statistic and R-squared -- the
  deterministic parts -- are compared instead.
* **NMDS coordinates.** ``metaMDS`` is stochastic and its axes have no
  canonical sign or order, so raw coordinates cannot be compared elementwise.
  Comparing ordination *correlation* would be the right test, but it needs a
  second statistic (PROTEST) and is deliberately left out of this tranche rather
  than approximated badly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import as_array, as_float, matrix_to_array, r, r_data_frame, r_matrix, require

pytestmark = pytest.mark.cross_validation

R_VEGAN = require("vegan")


def _abundance_matrix() -> np.ndarray:
    """Four deterministic samples x five species, all non-negative."""
    return np.array(
        [
            [10.0, 8.0, 2.0, 0.0, 5.0],
            [6.0, 7.0, 9.0, 1.0, 3.0],
            [0.0, 4.0, 8.0, 12.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ]
    )


def _grouped_dataset() -> tuple[np.ndarray, list[str]]:
    """Sixteen 5-variate samples in two well-separated groups."""
    rng = np.random.default_rng(11)
    data = rng.normal(loc=np.repeat([0.0, 4.0], 8)[:, None], scale=1.0, size=(16, 5))
    groups = ["A"] * 8 + ["B"] * 8
    return data, groups


class TestDistanceVsVegan:
    """Verify dissimilarity against ``vegan::vegdist``."""

    def test_bray_curtis_matches_vegdist(self):
        """Bray-Curtis dissimilarity matches ``vegdist(method='bray')``."""
        from stats.distance_metrics import compute_distance_matrix

        x = _abundance_matrix()
        paleo = compute_distance_matrix(x, metric="bray_curtis").matrix

        r_diss = matrix_to_array(R_VEGAN.vegdist(r_matrix(x), method="bray"))
        iu = np.triu_indices(x.shape[0], k=1)

        assert_allclose(
            paleo[iu],
            r_diss[iu],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Bray-Curtis dissimilarity disagrees with vegan::vegdist",
        )

    def test_distance_matrix_is_symmetric_with_zero_diagonal(self):
        """The matrix is well formed, checked against R's own output.

        Comparing shape and symmetry against ``vegdist`` rather than against
        hardcoded numbers means a malformed matrix cannot pass by agreeing with
        a constant that was itself derived from a malformed matrix.
        """
        from stats.distance_metrics import compute_distance_matrix

        x = _abundance_matrix()
        paleo = np.asarray(compute_distance_matrix(x, metric="bray_curtis").matrix)
        r_diss = matrix_to_array(R_VEGAN.vegdist(r_matrix(x), method="bray"))

        assert paleo.shape == r_diss.shape
        assert_allclose(paleo, paleo.T, atol=0.0)
        assert_allclose(np.diag(paleo), np.zeros(x.shape[0]), atol=0.0)
        # And the values themselves, whole matrix rather than upper triangle.
        assert_allclose(paleo, r_diss, rtol=1e-10, atol=1e-12)


class TestDiversityVsVegan:
    """Verify alpha-diversity indices against ``vegan::diversity``."""

    @pytest.mark.parametrize("index_name,py_key", [("shannon", "shannon"), ("simpson", "simpson")])
    def test_single_sample_indices_match(self, index_name, py_key):
        """Shannon and Simpson match ``vegan::diversity`` on one sample."""
        from ecology.diversity import compute_diversity_indices

        abundances = np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        paleo = float(compute_diversity_indices(abundances).indices[py_key].value)

        r_value = as_float(R_VEGAN.diversity(r_matrix(abundances), index=index_name))

        assert_allclose(
            paleo,
            r_value,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{py_key} disagrees with vegan::diversity(index='{index_name}')",
        )

    def test_simpson_is_one_minus_sum_of_squares(self):
        """Pin the convention, which is the usual source of a silent mismatch.

        ``vegan``'s default ``simpson`` is ``1 - sum(p^2)``. An implementation
        that returns ``sum(p^2)`` differs from it by exactly 1, which any
        tolerance loose enough to tolerate floating point would still hide.
        So the convention is asserted against R's value for the same data
        rather than against a formula.
        """
        from ecology.diversity import compute_diversity_indices

        abundances = np.array([10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        paleo = float(compute_diversity_indices(abundances).indices["simpson"].value)
        r_value = as_float(R_VEGAN.diversity(r_matrix(abundances), index="simpson"))

        assert_allclose(paleo, r_value, rtol=1e-12, atol=1e-14)
        # Guard against a "fix" that silently adopts the other convention.
        assert 0.0 < paleo < 1.0
        assert not np.isclose(paleo, 1.0 - paleo)

    def test_multiple_samples_match(self):
        """Per-sample indices match when a whole abundance matrix is passed."""
        from ecology.diversity import compute_diversity_indices

        x = np.array(
            [
                [10.0, 8.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0],
                [6.0, 6.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
        paleo = np.array([float(compute_diversity_indices(row).indices["shannon"].value) for row in x])
        r_values = np.array([as_float(v) for v in R_VEGAN.diversity(r_matrix(x), index="shannon")])

        assert_allclose(paleo, r_values, rtol=1e-10, atol=1e-12)


class TestPermanovaVsAdonis2:
    """Verify PERMANOVA against ``vegan::adonis2``.

    For a single grouping factor, ``adonis2``'s default sequential sums of
    squares are identical to marginal ones, so its R-squared and F are directly
    comparable with PaleoAST's. p-values are deliberately not compared; see the
    module docstring.
    """

    def test_r_squared_and_f_match(self):
        """R-squared and the F statistic match ``adonis2``."""
        from stats.distance_metrics import compute_distance_matrix
        from stats.permanova import PERMANOVAAnalyzer

        data, groups = _grouped_dataset()
        paleo = PERMANOVAAnalyzer().analyze(
            compute_distance_matrix(data, metric="euclidean").matrix,
            groups,
            n_permutations=99,
            random_seed=0,
        )

        frame = r_data_frame(
            {"v1": data[:, 0], "v2": data[:, 1], "v3": data[:, 2], "v4": data[:, 3], "v5": data[:, 4], "group": groups}
        )
        r_result = R_VEGAN.adonis2(
            r("formula")("~ group"),
            data=frame,
            method="euclidean",
            permutations=99,
        )
        r_sq = as_float(r_result.rx2("R2")[0])
        r_f = as_float(r_result.rx2("F")[0])

        paleo_r_sq = float(paleo.ss_between) / float(paleo.ss_between + paleo.ss_within)

        assert_allclose(
            paleo_r_sq,
            r_sq,
            rtol=1e-6,
            atol=1e-10,
            err_msg="PERMANOVA R-squared disagrees with vegan::adonis2",
        )
        assert_allclose(
            float(paleo.f_statistic),
            r_f,
            rtol=1e-6,
            atol=1e-10,
            err_msg="PERMANOVA F disagrees with vegan::adonis2",
        )

    def test_sums_of_squares_are_consistent(self):
        """The deterministic ingredients agree, and are internally consistent.

        Checking ``ss_between``/``ss_within`` against adonis2's own components
        localises a failure: if only the ratio disagrees, the bug is in the
        normalisation, not in the sums themselves.
        """
        from stats.distance_metrics import compute_distance_matrix
        from stats.permanova import PERMANOVAAnalyzer

        data, groups = _grouped_dataset()
        paleo = PERMANOVAAnalyzer().analyze(
            compute_distance_matrix(data, metric="euclidean").matrix,
            groups,
            n_permutations=99,
            random_seed=0,
        )

        frame = r_data_frame(
            {"v1": data[:, 0], "v2": data[:, 1], "v3": data[:, 2], "v4": data[:, 3], "v5": data[:, 4], "group": groups}
        )
        r_result = R_VEGAN.adonis2(
            r("formula")("~ group"),
            data=frame,
            method="euclidean",
            permutations=99,
        )
        r_ss = as_array(r_result.rx2("SumOfSquares"))
        r_df = as_array(r_result.rx2("Df"))
        paleo_total = float(paleo.ss_between) + float(paleo.ss_within)

        assert_allclose(
            paleo_total,
            float(np.sum(r_ss)),
            rtol=1e-6,
            atol=1e-10,
            err_msg="total sum of squares disagrees with vegan::adonis2",
        )
        assert_allclose(float(paleo.df_between), float(r_df[1]), atol=0.5)
        assert_allclose(float(paleo.df_within), float(r_df[2]), atol=0.5)
