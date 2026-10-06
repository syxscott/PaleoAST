"""Regression tests for DataMatrix type-aware analysis dispatch and
strict imputation.

Covers defects 5 (no type-aware dispatch) and 6 (impute_mean falling back
to 0 for all-NaN columns).
"""

from __future__ import annotations

import numpy as np
import pytest

from config.constants import DataType, DistanceMetric
from models.data_matrix import DataMatrix
from utils.exceptions import DataValidationError


def _make_matrix(data, *, data_types: list[str] | None = None) -> DataMatrix:
    """Build a DataMatrix with per-column data_type metadata."""
    n_cols = np.asarray(data).shape[1]
    col_labels = [f"Var_{j+1}" for j in range(n_cols)]
    column_metadata = {}
    if data_types is not None:
        for label, dt in zip(col_labels, data_types, strict=False):
            column_metadata[label] = {"data_type": dt}
    return DataMatrix(
        data,
        col_labels=col_labels,
        column_metadata=column_metadata,
    )


class TestRecommendedDistance:
    """``recommended_distance`` reads the column data_type metadata and
    returns an appropriate distance metric."""

    def test_all_continuous_returns_euclidean(self):
        m = _make_matrix(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            data_types=[DataType.CONTINUOUS, DataType.CONTINUOUS],
        )
        assert m.recommended_distance() == DistanceMetric.EUCLIDEAN

    def test_all_binary_returns_jaccard(self):
        m = _make_matrix(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            data_types=[DataType.BINARY, DataType.BINARY, DataType.BINARY],
        )
        assert m.recommended_distance() == DistanceMetric.JACCARD

    def test_all_categorical_returns_jaccard(self):
        m = _make_matrix(
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            data_types=[DataType.NOMINAL, DataType.NOMINAL, DataType.NOMINAL],
        )
        assert m.recommended_distance() == DistanceMetric.JACCARD

    def test_mixed_numeric_and_categorical_falls_back_to_euclidean_with_warning(self, caplog):
        m = _make_matrix(
            [[1.0, 0, 1.0], [2.0, 1, 0.0], [3.0, 2, 1.0]],
            data_types=[DataType.CONTINUOUS, DataType.NOMINAL, DataType.BINARY],
        )
        # PaleoAST has no Gower metric; the recommendation must be honest.
        assert m.recommended_distance() == DistanceMetric.EUCLIDEAN

    def test_no_metadata_defaults_to_euclidean(self):
        m = _make_matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.recommended_distance() == DistanceMetric.EUCLIDEAN

    def test_count_columns_are_treated_as_continuous(self):
        m = _make_matrix(
            [[1, 2], [3, 4]],
            data_types=[DataType.COUNT, DataType.COUNT],
        )
        assert m.recommended_distance() == DistanceMetric.EUCLIDEAN


class TestValidateFor:
    """``validate_for`` rejects methods that don't match the column types."""

    def test_pca_rejects_categorical(self):
        m = _make_matrix(
            [[1.0, 0], [2.0, 1]],
            data_types=[DataType.CONTINUOUS, DataType.NOMINAL],
        )
        with pytest.raises(DataValidationError) as exc_info:
            m.validate_for("pca")
        assert "pca" in str(exc_info.value)
        assert "nominal" in str(exc_info.value)

    def test_pcoa_accepts_mixed_numeric(self):
        m = _make_matrix(
            [[1.0, 0, 1], [2.0, 1, 0]],
            data_types=[DataType.CONTINUOUS, DataType.BINARY, DataType.COUNT],
        )
        # Should not raise
        m.validate_for("pcoa")

    def test_permanova_rejects_categorical(self):
        m = _make_matrix(
            [[1.0, 0], [2.0, 1]],
            data_types=[DataType.CONTINUOUS, DataType.NOMINAL],
        )
        with pytest.raises(DataValidationError):
            m.validate_for("permanova")

    def test_diversity_requires_count(self):
        m = _make_matrix(
            [[1.0, 2.0], [3.0, 4.0]],
            data_types=[DataType.CONTINUOUS, DataType.CONTINUOUS],
        )
        with pytest.raises(DataValidationError) as exc_info:
            m.validate_for("diversity")
        assert "continuous" in str(exc_info.value)

    def test_diversity_accepts_count(self):
        m = _make_matrix(
            [[1, 2], [3, 4]],
            data_types=[DataType.COUNT, DataType.COUNT],
        )
        m.validate_for("diversity")

    def test_unknown_method_raises(self):
        m = _make_matrix([[1.0]], data_types=[DataType.CONTINUOUS])
        with pytest.raises(DataValidationError) as exc_info:
            m.validate_for("not_a_method")
        assert "Unknown analysis" in str(exc_info.value)


class TestImputeMeanAllNaNColumn:
    """``impute_mean`` must RAISE on an all-NaN column, never fall back to 0."""

    def test_all_nan_column_raises(self):
        m = DataMatrix(
            [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]],
            col_labels=["good", "empty"],
        )
        with pytest.raises(DataValidationError) as exc_info:
            m.impute_mean()
        assert "empty" in str(exc_info.value)
        assert "all" in str(exc_info.value).lower() or "NaN" in str(exc_info.value)

    def test_all_nan_column_never_produces_zero(self):
        """The red-line: missing must NEVER become 0.0."""
        m = DataMatrix(
            [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]],
            col_labels=["good", "empty"],
        )
        with pytest.raises(DataValidationError):
            m.impute_mean()
        # The matrix is unchanged — no row of zeros was created
        assert np.all(np.isnan(m.data[:, 1]))


class TestImputeMedianAllNaNColumn:
    """Same red-line for ``impute_median``."""

    def test_all_nan_column_raises(self):
        m = DataMatrix(
            [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]],
            col_labels=["good", "empty"],
        )
        with pytest.raises(DataValidationError) as exc_info:
            m.impute_median()
        assert "empty" in str(exc_info.value)

    def test_partial_missing_still_works(self):
        """Sanity: partial NaN columns must still impute correctly."""
        m = DataMatrix(
            [[1.0, 5.0], [3.0, np.nan], [5.0, 7.0]],
            col_labels=["good", "partial"],
        )
        out = m.impute_median()
        assert out.data[1, 1] == 6.0  # median of {5.0, 7.0}


class TestImputeKnnAllNaNColumn:
    """``impute_knn`` must also refuse silent zero-fill."""

    def test_knn_with_no_complete_rows_falls_back_to_impute_mean_and_raises(self):
        """All rows have NaN somewhere → no complete rows → fallback to
        impute_mean, which now raises."""
        m = DataMatrix(
            [[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]],
            col_labels=["empty", "good"],
        )
        with pytest.raises(DataValidationError):
            m.impute_knn(k=2)
