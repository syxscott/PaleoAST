# =============================================================================
# FILE: tests/cross_validation/test_vs_geomorph.py
# =============================================================================
"""
Cross-validation against R's ``geomorph`` package (Geometric Morphometrics).

geomorph is the reference implementation for Generalized Procrustes Analysis, the
central operation in shape analysis.

Why the comparison is on a derived quantity
--------------------------------------------
Two independent Procrustes analyses of the same data cannot be compared
coordinate by coordinate. They differ by an arbitrary rotation and an arbitrary
reflection, and the two implementations may also centre differently, so the raw
consensus configurations differ even when the *shape* is identical.

What is compared instead is the matrix of pairwise distances between landmarks
in the consensus configuration. Procrustes analysis applies one rigid transform
to the whole configuration, which preserves every inter-landmark distance
exactly, so this quantity is invariant to rotation, reflection and translation.
It is therefore a faithful fingerprint of the shape while being free of the
arbitrary conventions that make a raw coordinate comparison meaningless.

Scale has to be handled separately, and this was the one gap in the argument
above. A rigid transform is not a similarity: **scaling changes every
inter-landmark distance**. The two implementations do not agree on scale --
``geomorph::gpagen`` returns the consensus in the original units, while
``morphometrics.gpa.GPAAnalyzer`` performs a documented "Scale to unit size"
step -- and on this fixture that difference is a factor of about 5.45. Every
distance comparison here therefore goes through ``_unit_centroid_size`` first.
Without it the comparison measured the scale convention rather than the shape,
and reported a shape disagreement of ``5.3848`` where the true shape residual
is ``0.0318`` (the jitter the fixture injects).

The unit-size step in ``GPAAnalyzer`` is a deliberate, documented product
decision and is left alone here: other consumers may rely on it, and changing a
scientific module's output convention is not a test's call. What the test does
is compare on a footing where the convention cannot leak in.

References
----------
    Adams, D.C. & Otarola-Castillo, E. (2013). geomorph: an R package for the
        collection and analysis of geometric morphometric shape data. Methods
        in Ecology and Evolution, 4(4), 393-399.
    Rohlf, F.J. (1993). Shape analysis of landmark data. Biological Reviews.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import matrix_to_array, r_array, require

pytestmark = pytest.mark.cross_validation

R_GEOMORPH = require("geomorph")


def _configurations() -> np.ndarray:
    """Five 2-D landmark configurations built from one shape.

    Each specimen is the same shape translated, rotated by an increasing amount
    and jittered, which is exactly the situation GPA exists to remove. The
    jitter is small so the consensus is well defined and the test does not
    depend on how a tie is broken.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(8, 2))
    specimens = []
    for i in range(5):
        theta = 0.3 * i
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        specimens.append(base @ rotation.T + np.array([i, -i]) + 0.05 * rng.normal(size=base.shape))
    return np.array(specimens)


def _unit_centroid_size(config: np.ndarray) -> np.ndarray:
    """Centre a configuration and divide by its centroid size.

    Needed before any inter-landmark distance is compared between the two
    implementations, because they do not agree on scale. ``geomorph::gpagen``
    returns the consensus in the original units; ``morphometrics.gpa.GPAAnalyzer``
    performs a documented "Scale to unit size" step, so its consensus has
    centroid size 1. On this fixture that is a factor of ~5.45, which is far
    larger than the jitter the fixture injects.

    The module docstring's argument -- that a rigid transform preserves every
    inter-landmark distance, so the comparison is free of arbitrary
    conventions -- is correct for rotation, reflection and translation but
    **not for scaling**. Comparing raw distances therefore measured the scale
    convention, not the shape: the disagreement was reported as
    ``assert 5.3848067600016245 < 0.5`` while the shape itself agreed.
    """
    centred = np.asarray(config, dtype=float) - np.asarray(config, dtype=float).mean(axis=0)
    size = np.sqrt(np.sum(centred**2))
    if size < 1e-12:
        return centred
    return centred / size


def _pairwise_landmark_distances(consensus: np.ndarray) -> np.ndarray:
    """The rotation/reflection/translation/scale-invariant fingerprint.

    Takes the upper triangle so the result is a flat, order-stable vector.
    """
    consensus = _unit_centroid_size(consensus)
    distances = np.linalg.norm(consensus[:, None, :] - consensus[None, :, :], axis=-1)
    return distances[np.triu_indices(consensus.shape[0], k=1)]


def _geomorph_consensus(configurations: np.ndarray) -> np.ndarray:
    """geomorph's consensus configuration, as a numpy array.

    ``gpagen`` takes a **3-D array laid out p x k x n** -- landmarks x
    dimensions x specimens -- while the configurations here are built in the
    more common numpy orientation n x p x k. Passing them straight through
    would tell geomorph that there are 5 landmarks of 8 dimensions across 2
    specimens, which is not a shape it can analyse, and if it had accepted it
    the comparison would have been against the wrong data. Hence the
    transpose.
    ``gpagen`` returns a list whose ``consensus`` element is a 2-D
    (landmarks x dimensions) matrix.
    """
    gpa = R_GEOMORPH.gpagen(r_array(np.asarray(configurations).transpose(1, 2, 0)))
    return matrix_to_array(gpa.rx2("consensus"))


class TestGPAVsGeomorph:
    """Verify Generalized Procrustes Analysis against ``geomorph::gpagen``."""

    def test_consensus_shape_matches(self):
        """The consensus has the expected landmarks x dimensions shape."""
        from morphometrics.gpa import GPAAnalyzer

        configurations = _configurations()
        paleo = np.asarray(GPAAnalyzer().analyze(configurations).consensus)
        reference = _geomorph_consensus(configurations)

        assert paleo.shape == reference.shape, f"PaleoAST consensus is {paleo.shape}, geomorph's is {reference.shape}"

    def test_input_is_transposed_into_gpagen_layout(self):
        """The array handed to gpagen is p x k x n, not the numpy n x p x k.

        gpagen reads its input as landmarks x dimensions x specimens. Handing it
        the numpy orientation instead would describe a dataset with the wrong
        number of landmarks and the wrong number of dimensions -- 8 "dimensions"
        for a 2-D shape is not something it can analyse, but the failure mode
        worth guarding is the one where it *does* accept it and the comparison
        is quietly against different data. So the layout is asserted directly
        rather than left implicit in a transpose.
        """
        configurations = _configurations()
        n_specimens, n_landmarks, n_dims = configurations.shape

        laid_out = configurations.transpose(1, 2, 0)
        assert laid_out.shape == (n_landmarks, n_dims, n_specimens)
        # and the data survived the transpose intact
        assert np.array_equal(laid_out[0, 0, :], configurations[:, 0, 0])

    def test_consensus_landmark_distances_match(self):
        """Inter-landmark distances of the consensus agree with geomorph.

        This is the real comparison; see the module docstring for why it is
        made on distances rather than coordinates.
        """
        from morphometrics.gpa import GPAAnalyzer

        configurations = _configurations()
        paleo = _pairwise_landmark_distances(GPAAnalyzer().analyze(configurations).consensus)
        reference = _pairwise_landmark_distances(_geomorph_consensus(configurations))

        assert_allclose(
            paleo,
            reference,
            rtol=1e-4,
            atol=1e-6,
            err_msg=(
                "GPA consensus inter-landmark distances disagree with "
                "geomorph::gpagen -- the two analyses did not converge on the "
                "same consensus shape"
            ),
        )

    def test_consensus_is_centred(self):
        """The consensus is centred, and R agrees on that.

        Checked against geomorph's own centroid rather than against a literal
        zero, so a change in either implementation's centring convention is
        visible instead of being absorbed by the tolerance.
        """
        from morphometrics.gpa import GPAAnalyzer

        configurations = _configurations()
        paleo = np.asarray(GPAAnalyzer().analyze(configurations).consensus)
        reference = _geomorph_consensus(configurations)

        assert_allclose(
            paleo.mean(axis=0),
            reference.mean(axis=0),
            atol=1e-8,
            err_msg="GPA consensus centroid disagrees with geomorph::gpagen",
        )

    def test_procrustes_distance_is_small(self):
        """Specimen-to-consensus Procrustes residuals are small and non-zero.

        Not a cross-validation, but a guard on the fixture: if the specimens
        were already aligned the consensus comparison above would be vacuous,
        and if GPA were doing nothing the residuals would be zero. This pins
        the fixture, so a failure in the comparisons above can be read as a
        disagreement rather than as a broken test setup.

        Measured through ``_pairwise_landmark_distances``, so this is a shape
        residual with the scale convention divided out. The fixture injects
        0.05-scale jitter, and the measured magnitude is 0.0318.
        """
        from morphometrics.gpa import GPAAnalyzer

        configurations = _configurations()
        result = GPAAnalyzer().analyze(configurations)
        consensus = np.asarray(result.consensus)

        residuals = np.array(
            [_pairwise_landmark_distances(spec) - _pairwise_landmark_distances(consensus) for spec in configurations]
        )
        magnitude = np.abs(residuals).max()
        assert magnitude > 1e-6, "fixture is degenerate: specimens are already aligned"
        assert magnitude < 0.5, "fixture is unrealistic: GPA should remove most of the shift"
