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

# Imported from ._rbridge rather than from rpy2 directly: _rbridge is the only
# module that is allowed to know how to reach R, and its module-level
# importorskip is what turns "no R here" into a skip. An `import rpy2` above
# this line would raise instead, turning a skip into a collection error on
# every machine without R.
from ._rbridge import ListVector, StrVector, as_float, matrix_to_array, r, r_matrix, require

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


def _r_adonis2_table(data: np.ndarray, groups: list[str]):
    """The result table of ``vegan::adonis2`` for this dataset.

    Three things about adonis2 that the previous version of this file got wrong,
    all of them stated in the R documentation (``?adonis2``):

    * **The LHS of the formula must be a matrix.** adonis2 partitions
      *distances*, and the docs say the LHS "must be either a community data
      matrix or a dissimilarity matrix". Naming the columns one by one
      (``v1 + v2 + v3 ~ group``) makes ``model.frame`` build a *multi-column
      data frame* response instead, and adonis2 then fails trying to turn that
      into a distance matrix -- surfacing as
      ``Error in eval(YVAR, parent.frame(), environment(formula))``. The
      documented form puts a single matrix in the data frame and names it:
      ``as.formula(paste("counts_matrix", rhs, sep = " ~ "))``.
    * **The formula is evaluated in its own environment.** adonis2 resolves the
      LHS there, not in ``data``: naming a column of the data frame produced
      ``object 'spec' not found`` from ``eval(YVAR, parent.frame(),
      environment(formula))``. ``?adonis2`` says ``data`` carries "the data
      frame for the independent variables", so the community matrix has to be
      reachable by name from the formula's environment.

      Building the formula with ``as.formula("spec ~ group", env=<local env>)``
      did not work: the name is still resolved in the evaluation frame rather
      than the one supplied, and the same ``object 'spec' not found`` came
      back. The bindings therefore go into the global environment, which is
      where an rpy2 call evaluates by default, and are removed again in a
      ``finally``. The cleanup matters: leaving a stale ``spec`` bound in
      globalenv would let a later test pick up the previous dataset, which is
      the sys.modules-mock failure mode all over again.
    * **The return value is the table itself.** adonis2 returns an
      ``anova.cca`` object that *inherits from* ``data.frame`` -- the AOV
      table is the object, not a component of it. vegan's own guidance on
      the adonis2 transition is explicit: "adonis2 return object is
      essentially the same as the aov.tab element of adonis. Instead of
      object$aov.tab refer only to object." Reading ``$table`` therefore
      returns NULL, and every downstream column access then fails with
      ``TypeError: 'NULLType' object is not iterable`` -- an error that
      names neither the real problem nor where it is. Columns are ``Df``,
      ``SumOfSqs``, ``R2``, ``F`` and ``Pr(>F)``, with one row per term
      plus ``Residual`` and ``Total``.
    """
    group = r("factor")(StrVector([str(g) for g in groups]))
    frame = r("data.frame")(ListVector({"group": group}), check_names=False)

    # R's `assign`/`rm` take `envir` as an *environment object*, and the
    # default is `as.environment(parent.frame())`, which for an rpy2 call is
    # the global environment. Passing the string "globalenv" gave
    # "invalid 'envir' argument"; the default is both simpler and what the
    # formula is going to be evaluated in.
    r("assign")("spec", r_matrix(data))
    r("assign")("group", group)
    try:
        formula = r("as.formula")("spec ~ group")
        result = R_VEGAN.adonis2(formula, data=frame, method="euclidean", permutations=99)
    finally:
        r("rm")("spec", "group")
    # The result IS the aov.cca data.frame; see the docstring. Do not
    # reach for a "table" component -- adonis2 has none, and the NULL that
    # comes back turns into a TypeError two frames away from the mistake.
    return result


#: adonis2 labels the row holding the grouping term "Model", not after the
#: term. Verified against vegan 2.7.6 on the dataset below:
#: rownames(adonis2(spec ~ group, data, method="euclidean", permutations=99))
#:   -> "Model" "Residual" "Total"
#: Asking for "group" finds nothing, and the lookup helper's own message
#: then reports the three labels it did find -- which is how this was
#: found in the first place.
_MODEL_ROW = "Model"


def _r_adonis2_component(data: np.ndarray, groups: list[str], column: str, row: str) -> float:
    """One cell of the adonis2 result table, located by name.

    Rows are located by their term label and columns by name; a miss raises
    naming what was actually present. Indexing positionally is what let an
    earlier version of this file read the wrong quantity silently.
    """
    table = _r_adonis2_table(data, groups)

    columns = [str(c) for c in table.names]
    if column not in columns:
        raise AssertionError(f"adonis2 table has no column {column!r}; it has {columns}")

    labels = [str(x) for x in r("rownames")(table)]
    if row not in labels:
        raise AssertionError(f"adonis2 table has no row {row!r}; it has {labels}")

    return as_float(table[labels.index(row), columns.index(column)])


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

        r_sq = _r_adonis2_component(data, groups, "R2", _MODEL_ROW)
        r_f = _r_adonis2_component(data, groups, "F", _MODEL_ROW)

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

        paleo_total = float(paleo.ss_between) + float(paleo.ss_within)
        r_total_ss = _r_adonis2_component(data, groups, "SumOfSqs", "Total")
        r_df_between = _r_adonis2_component(data, groups, "Df", _MODEL_ROW)
        r_df_within = _r_adonis2_component(data, groups, "Df", "Residual")

        assert_allclose(
            paleo_total,
            r_total_ss,
            rtol=1e-6,
            atol=1e-10,
            err_msg="total sum of squares disagrees with vegan::adonis2",
        )
        assert_allclose(float(paleo.df_between), r_df_between, atol=0.5)
        assert_allclose(float(paleo.df_within), r_df_within, atol=0.5)
