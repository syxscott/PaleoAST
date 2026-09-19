"""Tests for morpho3d/sliding.py — frame-consistent closed-form sliding (W2)."""

import numpy as np
import pytest

from morpho3d.sliding import (
    SemiLandmarkSlider,
    slide_curve_semi_landmarks,
)


def _arc_configs(n_specimens=5, n_lm=8):
    """Fixed endpoints + one off-arc landmark on a circle arc; semilandmarks
    1..6 carry tangential sampling jitter plus small normal noise.

    The fixed set must span 3D (three non-collinear landmarks): with only two
    fixed points the GPA rotation is rank-deficient and the orthogonal
    completion is not unique, so specimens do not share a common frame and
    the consensus (hence sliding) becomes ill-defined — a known limitation of
    partial-GPA registration, not of the sliding code."""
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, np.pi / 2, n_lm)
    base = np.stack([np.cos(t), np.sin(t), np.zeros(n_lm)], axis=1)
    extra = np.array([[0.5, 0.5, -0.4]])
    configs = []
    for _ in range(n_specimens):
        cfg = base + rng.normal(0.0, 0.01, size=(n_lm, 3))
        cfg = np.concatenate(
            [cfg, extra + rng.normal(0.0, 0.01, size=(1, 3))], axis=0
        )
        configs.append(cfg)
    return configs


FIXED = np.array([0, 7, 8])
SEMI = np.arange(1, 7)


def _chord_residuals(pts, semi):
    """Perpendicular distance of each interior semilandmark to the chord
    through its (current) neighbours."""
    out = []
    for cfg in pts:
        p = cfg[semi]
        for i in range(1, len(semi) - 1):
            a, m, b = p[i - 1], p[i], p[i + 1]
            d = b - a
            out.append(np.linalg.norm(np.cross(m - a, d)) / np.linalg.norm(d))
    return np.array(out)


def test_slide_procrustes_reduces_perp_residuals():
    configs = _arc_configs()
    slider = SemiLandmarkSlider(criterion="procrustes", max_iterations=10)
    slider.set_landmarks(fixed_indices=FIXED, semi_indices=SEMI)
    # Baseline: residuals in the iteration-0 aligned frame (a similarity
    # transform of the raw configs; residuals themselves are not
    # similarity-invariant in magnitude, so compare in aligned frames).
    gpa0, _, _, _ = slider._gpa_with_fixed_landmarks(configs)
    pre = _chord_residuals(gpa0.aligned_configs, SEMI)

    result = slider.slide([c.copy() for c in configs])
    assert len(result.aligned_configs) == len(configs)
    post = _chord_residuals(result.aligned_configs, SEMI)

    # Sequential minPerp (Bookstein 1997 / MorphoJ) projects each interior
    # point onto the chord of its neighbours, but those neighbours move in
    # later passes, so exact collinearity of the FINAL configuration is not
    # the invariant — monotone reduction of the perpendicular residuals is.
    assert np.median(post) < 0.2 * np.median(pre)
    assert post.max() <= pre.max() + 1e-12


def test_minperp_single_interior_is_exact():
    """Closed-form check: with exactly one interior slider the minPerp
    projection lands precisely on the chord and endpoints never move."""
    slider = SemiLandmarkSlider(criterion="procrustes")
    slider.set_landmarks(fixed_indices=np.array([0, 4]), semi_indices=np.array([1, 2, 3]))
    cfg = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.0],
            [0.9, 0.4, 0.2],
            [1.5, 0.2, -0.1],
            [2.0, 0.0, 0.0],
        ]
    )
    slid = slider._slide_curve_points(cfg, cfg)
    a, m, b = slid[0], slid[1], slid[2]
    dist = np.linalg.norm(np.cross(m - a, b - a)) / np.linalg.norm(b - a)
    assert dist < 1e-12
    assert np.array_equal(slid[0], cfg[1])
    assert np.array_equal(slid[2], cfg[3])


def test_minperp_last_interior_is_exact():
    """The last interior point processed is projected onto the fixed
    block endpoint, so its final residual is exactly zero."""
    slider = SemiLandmarkSlider(criterion="procrustes")
    slider.set_landmarks(fixed_indices=FIXED, semi_indices=SEMI)
    cfg = _arc_configs(n_specimens=1)[0]
    slid = slider._slide_curve_points(cfg, cfg)
    a, m, b = slid[-3], slid[-2], slid[-1]
    dist = np.linalg.norm(np.cross(m - a, b - a)) / np.linalg.norm(b - a)
    assert dist < 1e-12


def test_slide_is_scale_invariant():
    """Sliding in the aligned frame then mapping back must not depend on the
    input coordinate scale (the old frame-mixing bug made displacement scale
    with the raw coordinates)."""
    configs = _arc_configs()

    def spread(configs):
        slider = SemiLandmarkSlider(criterion="procrustes", max_iterations=5)
        slider.set_landmarks(fixed_indices=FIXED, semi_indices=SEMI)
        res = slider.slide([c.copy() for c in configs])
        mean = res.mean_config[SEMI]
        return np.linalg.norm(res.aligned_configs[0][SEMI] - mean)

    s1 = spread(configs)
    s100 = spread([c * 100.0 for c in configs])
    # Normalised spread of the slid semilandmarks should be scale-independent
    assert np.isclose(s1, s100, rtol=1e-3), (s1, s100)


def test_be_criterion_differs_from_procrustes():
    configs = _arc_configs()

    outs = {}
    for crit in ("bending_energy", "procrustes"):
        slider = SemiLandmarkSlider(criterion=crit, max_iterations=5)
        slider.set_landmarks(fixed_indices=FIXED, semi_indices=SEMI)
        outs[crit] = slider.slide([c.copy() for c in configs]).mean_config

    # The two criteria must not collapse to identical results any more.
    assert not np.allclose(outs["bending_energy"], outs["procrustes"])


def test_wrapper_helper():
    configs = _arc_configs(n_specimens=3)
    result = slide_curve_semi_landmarks(
        configs,
        fixed_indices=FIXED,
        semi_indices=SEMI,
    )
    assert len(result.aligned_configs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
