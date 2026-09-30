"""Regression tests for ``config/imputation.py``.

That module shipped with 170 statements and zero test coverage while being the
stage that decides the values every downstream statistic is computed from, so
these tests exist to make the module's *contract* explicit rather than to chase
a percentage. Each test below pins a defect that was live in the shipped code;
the counter-proof notes say what reverting each fix does.

Covered:
  1. ``impute_knn`` accepted ``distance_metric`` and ignored it, so asking for
     manhattan silently returned euclidean neighbours.
  2. ``impute_knn(k=0)`` silently wrote 0.0 into every imputed cell -- exactly
     the "missing data becomes zero" corruption the DAT parser refuses to do.
  3. ``impute()`` validated the method only after the no-NaN early return, so
     an invalid method was accepted on a clean matrix and rejected on a dirty
     one.
  4. ``impute()`` dropped ``distance_metric`` when forwarding to KNN, which
     would have reopened defect 1 one level up.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.imputation import (
    ImputationMethod,
    analyze_missing_values,
    impute,
    impute_knn,
    impute_mean,
    impute_median,
    remove_columns_with_nan,
    remove_rows_with_nan,
)


@pytest.fixture
def dirty() -> np.ndarray:
    """A small matrix with three scattered missing cells."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(8, 4)) * 10 + 100
    data[0, 1] = np.nan
    data[3, 0] = np.nan
    data[5, 2] = np.nan
    return data


class TestKNNParametersAreHonoured:
    """Defects 1 and 2: accepted parameters that changed nothing."""

    def test_unsupported_distance_metric_raises(self, dirty):
        # Before the fix this returned a result, and the result was identical
        # to the euclidean one: euclidean == manhattan == cosine, bit for bit.
        with pytest.raises(ValueError, match="distance_metric"):
            impute_knn(dirty, k=3, distance_metric="manhattan")

    def test_euclidean_still_works(self, dirty):
        """The one supported metric must keep working."""
        result = impute_knn(dirty, k=3, distance_metric="euclidean")
        assert not np.isnan(result.data).any()

    @pytest.mark.parametrize("bad_k", [0, -1, -5])
    def test_non_positive_k_raises(self, dirty, bad_k):
        # k=0 previously produced 0.0 in every imputed cell with no error and
        # no warning. k<0 silently used "all but the last" neighbour.
        with pytest.raises(ValueError, match="k must be >= 1"):
            impute_knn(dirty, k=bad_k)

    def test_non_integer_k_raises(self, dirty):
        with pytest.raises(ValueError, match="k must be an integer"):
            impute_knn(dirty, k=2.5)  # type: ignore[arg-type]

    def test_no_imputed_cell_is_ever_silently_zero(self, dirty):
        """A value that legitimately *is* 0 must be distinguishable from a
        failure that fills 0. With k=3 no cell here may come out exactly 0.0,
        because the surrounding data is centred near 100."""
        result = impute_knn(dirty, k=3)
        imputed = [result.data[0, 1], result.data[3, 0], result.data[5, 2]]
        assert all(v != 0.0 for v in imputed), f"a missing cell became 0.0: {imputed}"


class TestDispatchValidatesTheMethod:
    """Defect 3: the same invalid input gave two different answers."""

    def test_invalid_method_rejected_on_clean_data(self):
        clean = np.ones((3, 3))
        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute(clean, method="not-a-method")  # type: ignore[arg-type]

    def test_invalid_method_rejected_on_dirty_data(self):
        dirty = np.ones((3, 3))
        dirty[0, 0] = np.nan
        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute(dirty, method="not-a-method")  # type: ignore[arg-type]

    def test_clean_data_does_not_change_validity(self):
        """The outcome must depend on the method alone, not on the data."""
        clean = np.ones((3, 3))
        dirty = np.ones((3, 3))
        dirty[0, 0] = np.nan
        for method in ImputationMethod:
            assert impute(clean, method).nan_removed == 0


class TestDispatchForwardsKnnOptions:
    """Defect 4: impute() dropped distance_metric on the way to impute_knn."""

    def test_distance_metric_is_forwarded_not_dropped(self, dirty):
        # Before the fix impute() called impute_knn(data, k=k) only, so this
        # call quietly ran euclidean despite asking for manhattan.
        with pytest.raises(ValueError, match="distance_metric"):
            impute(dirty, ImputationMethod.KNN, k=3, distance_metric="manhattan")

    def test_k_is_forwarded(self, dirty):
        with pytest.raises(ValueError, match="k must be >= 1"):
            impute(dirty, ImputationMethod.KNN, k=0)


class TestDocumentedBehaviourStillHolds:
    """The fixes must not have changed the working paths."""

    def test_mean_uses_the_column_mean(self, dirty):
        result = impute_mean(dirty)
        assert result.data[0, 1] == pytest.approx(np.nanmean(dirty[:, 1]))
        assert not np.isnan(result.data).any()
        assert result.nan_removed == 3

    def test_median_uses_the_column_median(self, dirty):
        result = impute_median(dirty)
        assert result.data[0, 1] == pytest.approx(np.nanmedian(dirty[:, 1]))
        assert not np.isnan(result.data).any()

    def test_all_nan_column_falls_back_to_zero_and_says_so(self, caplog):
        """An all-NaN column has no mean; the fallback is 0, and it is logged.

        0 is the documented fallback here -- unlike the k=0 case above, where
        0 appeared without anyone choosing it.
        """
        data = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
        with caplog.at_level("WARNING"):
            result = impute_mean(data)
        assert np.all(result.data[:, 1] == 0.0)
        assert any("all-NaN column" in r.message for r in caplog.records)

    def test_input_is_not_mutated(self, dirty):
        before = dirty.copy()
        for fn in (impute_mean, impute_median):
            fn(dirty)
        impute_knn(dirty, k=3)
        assert np.array_equal(np.isnan(dirty), np.isnan(before)), "input matrix was modified"
        assert np.array_equal(dirty[~np.isnan(before)], before[~np.isnan(before)])

    def test_remove_rows_drops_every_row_holding_a_nan(self, dirty):
        result = remove_rows_with_nan(dirty)
        assert result.rows_removed == 3
        assert result.data.shape[0] == dirty.shape[0] - 3
        assert not np.isnan(result.data).any()

    def test_remove_columns_drops_every_column_holding_a_nan(self, dirty):
        result = remove_columns_with_nan(dirty)
        assert result.columns_removed == 3
        assert result.data.shape[1] == dirty.shape[1] - 3
        assert not np.isnan(result.data).any()

    def test_report_counts_match_the_matrix(self, dirty):
        report = analyze_missing_values(dirty)
        assert report.total_nan == 3
        assert report.rows_with_nan == 3
        assert report.cols_with_nan == 3
        assert report.nan_proportion == pytest.approx(3 / dirty.size)
        # The fixture puts NaN at rows 0, 3 and 5.
        assert list(report.nan_by_row) == [1, 0, 0, 1, 0, 1, 0, 0]
        assert list(report.nan_by_col) == [1, 1, 1, 0]
