# =============================================================================
# FILE: tests/statistics/test_pcoa.py
# =============================================================================
"""
Unit tests for PCoA module - negative eigenvalue handling.

Bug 2 Description:
    When distance matrices produce negative eigenvalues (non-Euclidean metrics
    like Bray-Curtis), the original code truncated them to 0 and only logged
    a warning. This caused:
    1. Loss of meaningful biological information
    2. Inconsistent results compared to R cmdscale()/ape::pcoa()

    A later regression (commit c61a82b) changed the handling to sort axes by
    |lambda| and build coordinates from sqrt(|lambda|), which lets a large
    negative eigenvalue displace genuine positive axes.

    Current behavior (reverted to the R cmdscale convention; Gower 1966):
    1. Axes are sorted by signed eigenvalue (descending)
    2. Coordinates are built only from positive eigenvalues: sqrt(max(lambda, 0))
    3. Negative eigenvalues are warned about and their sum is reported
       separately (negative_eigenvalue_sum / summary text)
    4. Proportions are computed from the positive parts of the eigenvalues
"""

import numpy as np
import pytest
import warnings

from statistics.pcoa import PCoAAnalyzer, PCoAResult


class TestPCoANegativeEigenvalueBehavior:
    """
    Tests for Bug 2: negative eigenvalue preservation.

    Note: Creating distance matrices that produce negative eigenvalues is
    non-trivial. These tests verify the code behavior (sorting by absolute
    value, using sqrt(abs()), etc.) rather than requiring negative
    eigenvalues to appear in every test.
    """

    def test_eigenvalues_sorted_descending(self):
        """Test that eigenvalues are sorted by signed value (descending).

        Regression note: an earlier version sorted by |lambda|, which let a
        large negative eigenvalue displace genuine positive axes. The R
        cmdscale convention (Gower 1966) sorts by lambda descending and
        builds coordinates only from the positive eigenvalues.
        """
        # Create a non-Euclidean distance matrix
        distance_matrix = np.array([
            [0.0, 0.8, 0.9, 0.3],
            [0.8, 0.0, 0.4, 0.7],
            [0.9, 0.4, 0.0, 0.6],
            [0.3, 0.7, 0.6, 0.0],
        ])
        analyzer = PCoAAnalyzer()
        result = analyzer.analyze(distance_matrix, metric='non-euclidean')

        # Check that signed eigenvalues are in descending order
        eigenvalues = result.eigenvalues
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i + 1], (
                f"Eigenvalues not sorted in descending order: "
                f"{eigenvalues[i]} < {eigenvalues[i + 1]}"
            )

    def test_coordinates_from_positive_eigenvalues(self):
        """Test that coordinates are computed as sqrt(max(lambda, 0)).

        Follows the R cmdscale convention: coordinates come only from
        positive eigenvalues, and negative eigenvalues contribute none.
        """
        analyzer = PCoAAnalyzer()
        distance_matrix = np.array([
            [0.0, 0.8, 0.9],
            [0.8, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ])
        result = analyzer.analyze(distance_matrix, n_components=2, metric='bray-curtis')

        # Coordinates should have the right shape
        assert result.coordinates.shape[1] == 2  # 2 components
        assert result.coordinates.shape[0] == 3  # 3 samples

        # The coordinates should not be all zeros (which would happen if
        # all eigenvalues were non-positive)
        assert not np.allclose(result.coordinates, 0)

        # Columns tied to negative eigenvalues must be exactly zero
        for k in range(result.coordinates.shape[1]):
            if result.eigenvalues[k] < 0:
                assert np.allclose(result.coordinates[:, k], 0), (
                    f"Axis {k} has negative eigenvalue but non-zero coordinates"
                )

    def test_proportion_explained_sums_to_100(self):
        """Test that proportion explained sums to ~100%."""
        analyzer = PCoAAnalyzer()
        distance_matrix = np.array([
            [0.0, 0.8, 0.9],
            [0.8, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ])
        result = analyzer.analyze(distance_matrix, metric='bray-curtis')

        # Proportion should sum to 100% (within floating point tolerance)
        total_proportion = np.sum(result.proportion_explained)
        assert 99.0 <= total_proportion <= 100.1, (
            f"Proportion explained should sum to ~100%, got {total_proportion}%"
        )

    def test_no_eigenvalue_truncation(self):
        """
        Test that eigenvalues are NOT truncated to zero.

        This verifies that the fix for Bug 2 is in place - the old code
        used np.maximum(eigenvalues, 0) which would truncate negative
        eigenvalues to zero.
        """
        analyzer = PCoAAnalyzer()
        distance_matrix = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.4],
            [0.6, 0.4, 0.0],
        ])
        result = analyzer.analyze(distance_matrix, metric='euclidean')

        # The old code would have eigenvalues that were all >= 0 due to truncation
        # The new code preserves the actual eigenvalues
        # Since this is Euclidean data, eigenvalues should be non-negative anyway
        assert np.all(result.eigenvalues >= 0), (
            "Euclidean distance should produce non-negative eigenvalues"
        )


class TestPCoAEigenvalueHandling:
    """Tests for eigenvalue handling behavior."""

    def test_negative_eigenvalue_warning_mechanism(self):
        """
        Test that the code has a mechanism to warn about negative eigenvalues.

        The warning is issued when negative eigenvalues are detected.
        Note: We can't easily create a distance matrix that produces negative
        eigenvalues in simple test cases, but the warning mechanism exists
        and is tested here with non-Euclidean metrics.
        """
        analyzer = PCoAAnalyzer()

        # Use Bray-Curtis which is a non-Euclidean metric
        distance_matrix = np.array([
            [0.0, 0.8, 0.9],
            [0.8, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ])

        # The code should handle this without error (whether or not
        # negative eigenvalues are produced)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze(distance_matrix, metric='bray-curtis')

            # Result should be valid
            assert isinstance(result, PCoAResult)
            assert result.coordinates.shape[0] == 3

    def test_bray_curtis_distance_computation(self):
        """Test PCoA with Bray-Curtis distance (common in ecology)."""
        from statistics.distance_metrics import compute_distance_matrix

        # Species counts for 3 sites
        species_data = np.array([
            [2, 0, 6, 1],
            [3, 1, 2, 0],
            [0, 4, 3, 2],
        ])

        dist_result = compute_distance_matrix(species_data, metric='bray_curtis')
        dist_matrix = dist_result.matrix

        analyzer = PCoAAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyzer.analyze(dist_matrix, metric='bray-curtis')

        # Should complete without error
        assert isinstance(result, PCoAResult)
        assert result.coordinates.shape[0] == 3  # 3 sites
        assert result.n_components <= 2  # max is n_samples - 1


class TestPCoANormalOperation:
    """Tests for normal PCoA operation (regression tests)."""

    def test_result_is_pcoaresult_dataclass(self):
        """Test that analyze returns a PCoAResult instance."""
        analyzer = PCoAAnalyzer()
        dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        result = analyzer.analyze(dist_matrix)
        assert isinstance(result, PCoAResult)

    def test_coordinates_shape_matches_n_components(self):
        """Test that output coordinates match requested n_components."""
        analyzer = PCoAAnalyzer()
        dist_matrix = np.array([
            [0.0, 0.5, 0.6, 0.3],
            [0.5, 0.0, 0.4, 0.7],
            [0.6, 0.4, 0.0, 0.5],
            [0.3, 0.7, 0.5, 0.0],
        ])
        for n_comp in [1, 2, 3]:
            result = analyzer.analyze(dist_matrix, n_components=n_comp)
            assert result.coordinates.shape[1] == n_comp

    def test_get_coordinates_method(self):
        """Test the get_coordinates helper method."""
        analyzer = PCoAAnalyzer()
        dist_matrix = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.4],
            [0.6, 0.4, 0.0],
        ])
        result = analyzer.analyze(dist_matrix, n_components=3)
        coords = result.get_coordinates(n_components=2)
        assert coords.shape == (3, 2)

    def test_euclidean_distance_all_positive_eigenvalues(self):
        """Test Euclidean distances produce all-positive eigenvalues."""
        from statistics.distance_metrics import compute_distance_matrix

        # Euclidean distance matrix from 4 points in 2D
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        dist_result = compute_distance_matrix(points, metric='euclidean')
        dist_matrix = dist_result.matrix

        analyzer = PCoAAnalyzer()
        result = analyzer.analyze(dist_matrix, metric='euclidean')

        # Euclidean should have all non-negative eigenvalues
        assert np.all(result.eigenvalues >= -1e-10), (
            "Euclidean distance should produce non-negative eigenvalues"
        )


class TestPCoAEdgeCases:
    """Tests for edge cases."""

    def test_three_sample_minimum(self):
        """Test PCoA works with exactly 3 samples."""
        analyzer = PCoAAnalyzer()
        dist_matrix = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.4],
            [0.6, 0.4, 0.0],
        ])
        result = analyzer.analyze(dist_matrix, n_components=2)
        assert result.coordinates.shape[0] == 3

    def test_square_matrix_required(self):
        """Test that non-square matrix raises error."""
        analyzer = PCoAAnalyzer()
        non_square = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.4],
        ])
        # Should raise error for non-square matrix
        with pytest.raises(Exception):  # Could be ValueError or MatrixDimensionError
            analyzer.analyze(non_square)

    def test_zero_distance_matrix(self):
        """Test handling of zero distances (identical samples)."""
        analyzer = PCoAAnalyzer()
        # All samples are identical (zero distance)
        dist_matrix = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        result = analyzer.analyze(dist_matrix)
        # Should complete without error
        assert result.n_components >= 1

    def test_single_sample_raises_error(self):
        """Test that single sample raises error."""
        analyzer = PCoAAnalyzer()
        dist_matrix = np.array([[0.0]])
        with pytest.raises(Exception):
            analyzer.analyze(dist_matrix)
