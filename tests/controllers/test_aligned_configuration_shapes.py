"""Regression tests for the aligned-configuration shape contract.

``StatisticsController._resolve_aligned_configurations`` decides what shape an
allometry or PLS analysis actually runs on. It used to hard-code ``n_dims = 2``
when a caller passed a flattened 2D array, so a 3D landmark configuration --
5 specimens x 8 landmarks x 3 dimensions, the ordinary case in morphometrics --
was reshaped to (5, 12, 2): 12 landmarks instead of 8, 2 dimensions instead of
3, no exception and no warning. Allometry then regressed that invented shape on
log centroid size and returned a number for a dataset that does not exist.

The "must have an even number of columns" guard that was there did not help:
8 * 3 = 24 is even.
"""

from __future__ import annotations

import numpy as np
import pytest

from controllers.statistics_controller import StatisticsController
from utils.exceptions import ValidationError


@pytest.fixture(scope="module")
def controller():
    return StatisticsController()


def _resolve(controller, array, **kwargs):
    """The method is private, but it is the decision point under test and the
    controller has no public entry point that exercises shape validation
    without running a full analysis."""
    with controller._lock:
        return controller._resolve_aligned_configurations(array, **kwargs)


class TestThreeDimensionalDataIsNotReadAsPlanar:
    def test_flattened_3d_needs_n_dims_three(self, controller):
        rng = np.random.default_rng(3)
        configs = rng.normal(size=(5, 8, 3))
        flat = configs.reshape(5, 8 * 3)

        # What the old code produced.
        wrong = _resolve(controller, flat)
        assert wrong.shape == (5, 12, 2), "expected the old misreading to be gone"

        # What it should produce.
        right = _resolve(controller, flat, n_dims=3)
        assert right.shape == (5, 8, 3)
        assert np.allclose(right, configs), "the coordinates themselves must survive"

    def test_planar_data_is_unaffected(self, controller):
        rng = np.random.default_rng(3)
        configs = rng.normal(size=(5, 8, 2))
        flat = configs.reshape(5, 16)
        out = _resolve(controller, flat)
        assert out.shape == (5, 8, 2)
        assert np.allclose(out, configs)

    def test_three_dimensional_array_needs_no_n_dims(self, controller):
        rng = np.random.default_rng(3)
        configs = rng.normal(size=(4, 6, 3))
        out = _resolve(controller, configs)
        assert np.array_equal(out, configs), "a 3D array is taken as-is"

    def test_landmark_count_is_preserved_not_invented(self, controller):
        """The failure mode was a landmark count that never existed."""
        rng = np.random.default_rng(11)
        for n_landmarks, n_dims in ((8, 3), (12, 3), (5, 2), (7, 2)):
            configs = rng.normal(size=(6, n_landmarks, n_dims))
            flat = configs.reshape(6, n_landmarks * n_dims)
            out = _resolve(controller, flat, n_dims=n_dims)
            assert out.shape == (6, n_landmarks, n_dims), (
                f"({n_landmarks} landmarks, {n_dims} dims) came back as {out.shape}"
            )


class TestAmbiguousInputIsRefused:
    def test_error_message_names_the_column_count(self, controller):
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 10))  # not a multiple of 3
        with pytest.raises(ValidationError) as excinfo:
            _resolve(controller, flat, n_dims=3)
        assert "10" in str(excinfo.value)

    def test_a_divisible_but_wrong_n_dims_cannot_be_detected(self, controller):
        """Documents the hazard, and why ``n_dims`` is not optional in practice.

        24 columns is a valid reshape by 2 and by 3, so nothing in the array
        says which is right -- 8 landmarks x 3 dimensions and 12 landmarks x 2
        dimensions are the same 5x24 matrix. This is precisely why the old
        "even number of columns" guard gave false comfort, and why the only
        real protection is the caller stating the dimensionality.
        """
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 24))
        # No error is possible here, by construction:
        assert _resolve(controller, flat, n_dims=2).shape == (5, 12, 2)
        assert _resolve(controller, flat, n_dims=3).shape == (5, 8, 3)

    def test_indivisible_shape_raises(self, controller):
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 10))  # not a multiple of 3
        with pytest.raises(ValidationError, match="not a multiple of n_dims"):
            _resolve(controller, flat, n_dims=3)

    def test_n_dims_below_two_is_refused(self, controller):
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 8))
        with pytest.raises(ValidationError, match="n_dims must be at least 2"):
            _resolve(controller, flat, n_dims=1)

    def test_four_dimensional_input_is_refused(self, controller):
        rng = np.random.default_rng(3)
        with pytest.raises(ValidationError, match="3D array"):
            _resolve(controller, rng.normal(size=(2, 3, 4, 5)))

    def test_error_message_does_not_assume_xy_pairs(self, controller):
        """The old message told the user to supply 'x, y pairs per landmark',
        which is wrong advice for 3D data -- the real problem is the unknown
        dimensionality, not an odd column count."""
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 10))
        with pytest.raises(ValidationError) as excinfo:
            _resolve(controller, flat, n_dims=3)
        assert "x, y pairs" not in str(excinfo.value)


class TestThePublicEntryPointThreadsTheParameter:
    def test_analyze_allometry_accepts_n_dims(self):
        import inspect

        sig = inspect.signature(StatisticsController.analyze_allometry)
        assert "n_dims" in sig.parameters
        assert sig.parameters["n_dims"].default == 2
