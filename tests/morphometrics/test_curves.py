"""Tests for morphometrics/curves.py — triples validation and evenPts port."""

import numpy as np
import pytest

from morphometrics.curves import (
    evenly_resample_curve,
    interior_sliders,
    validate_curves,
)
from utils.exceptions import MorphometricsError


class TestValidateCurves:
    def test_accepts_valid_triples(self):
        curves = [[0, 1, 2], [2, 3, 4, 5]]
        assert validate_curves(curves, n_landmarks=6) == [[0, 1, 2], [2, 3, 4, 5]]

    def test_accepts_numpy_ints(self):
        curves = [np.array([0, 1, 2])]
        out = validate_curves(curves, n_landmarks=3)
        assert out == [[0, 1, 2]]
        assert all(isinstance(x, int) for x in out[0])

    def test_empty_returns_empty(self):
        assert validate_curves([], n_landmarks=5) == []

    def test_rejects_short_curve(self):
        with pytest.raises(MorphometricsError, match="at least 3"):
            validate_curves([[0, 1]], n_landmarks=5)

    def test_rejects_duplicate_indices(self):
        with pytest.raises(MorphometricsError, match="duplicate"):
            validate_curves([[0, 1, 0]], n_landmarks=5)

    def test_rejects_out_of_range(self):
        with pytest.raises(MorphometricsError, match="outside"):
            validate_curves([[0, 1, 7]], n_landmarks=5)

    def test_rejects_non_numeric(self):
        with pytest.raises(MorphometricsError, match="integer"):
            validate_curves([[0, "a", 2]], n_landmarks=5)


class TestInteriorSliders:
    def test_endpoints_pinned(self):
        assert interior_sliders([3, 4, 5, 6, 7]) == [4, 5, 6]

    def test_triple_has_one_slider(self):
        assert interior_sliders([0, 1, 2]) == [1]


class TestEvenlyResampleCurve:
    def test_straight_line_endpoints_and_spacing(self):
        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        out = evenly_resample_curve(coords, 5)
        assert out.shape == (5, 2)
        assert np.array_equal(out[0], coords[0])
        assert np.array_equal(out[-1], coords[1])
        assert np.allclose(np.diff(out[:, 0]), 2.5)

    def test_l_shape_equal_arc_length(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        out = evenly_resample_curve(coords, 5)  # total length 2
        steps = np.diff(out, axis=0)
        seg = np.sqrt(np.sum(steps**2, axis=1))
        assert np.allclose(seg, 0.5)
        assert np.allclose(out[2], [1.0, 0.0])  # corner hit exactly

    def test_3d_supported(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        out = evenly_resample_curve(coords, 3)
        assert np.allclose(out[1], [1.0, 2.0, 3.0])

    def test_two_point_basis_interior(self):
        coords = np.array([[0.0, 0.0], [0.0, 4.0]])
        out = evenly_resample_curve(coords, 3)
        assert np.allclose(out, [[0, 0], [0, 2], [0, 4]])

    def test_n_equals_2_returns_endpoints(self):
        coords = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]])
        out = evenly_resample_curve(coords, 2)
        assert np.array_equal(out, np.array([[0.0, 0.0], [5.0, 5.0]]))

    def test_rejects_degenerate_curve(self):
        coords = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(MorphometricsError):
            evenly_resample_curve(coords, 4)

    def test_rejects_single_point(self):
        with pytest.raises(MorphometricsError):
            evenly_resample_curve(np.array([[1.0, 2.0]]), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
