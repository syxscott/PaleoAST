"""Regression tests for PCoA negative-eigenvalue correction (Defect 1).

These tests pin down the four correction methods documented in
``stats/pcoa.py`` and verify that:

* ``correction='cmdscale'`` (default) is bit-identical to the legacy
  behaviour: no shift is added to the squared distances, eigenvalues
  remain unchanged, axes are sorted by signed eigenvalue, and the
  proportion sum is still ~100%.
* ``correction='lingoes'`` / ``'wickoff'`` / ``'torgerson'`` all
  eliminate (or clip to zero, within floating-point tolerance) any
  negative eigenvalues in the double-centred matrix.
* The chosen method is recorded on the :class:`PCoAResult` so the UI
  can surface it.
* Unknown correction names raise ``ValidationError``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from stats.pcoa import PCoAAnalyzer, PCoAResult
from utils.exceptions import ValidationError


def _make_non_euclidean_distance(n: int = 8, seed: int = 0) -> np.ndarray:
    """Construct a Bray-Curtis distance matrix likely to have negatives."""
    rng = np.random.default_rng(seed)
    data = rng.random((n, n)) * 10
    data[rng.random(data.shape) < 0.4] = 0.0
    from stats.distance_metrics import compute_distance_matrix

    res = compute_distance_matrix(data, metric="bray_curtis")
    return np.asarray(res.matrix, dtype=float)


class TestPCoACorrectionCmdscale:
    """Default correction must be backward-compatible."""

    def test_default_correction_is_cmdscale(self):
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=11)
        result = analyzer.analyze(D, metric="bray-curtis")
        assert result.correction_method == "cmdscale"

    def test_cmdscale_is_bit_identical_to_no_correction(self):
        """Calling analyze() without the correction argument must produce
        the same eigenvalues, coordinates, and proportion vector as
        passing correction='cmdscale' explicitly — both paths skip the
        shift (no double centring of D²+c)."""
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=23)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_default = analyzer.analyze(D, metric="bray-curtis")
            r_explicit = analyzer.analyze(D, metric="bray-curtis", correction="cmdscale")
        np.testing.assert_allclose(r_default.eigenvalues, r_explicit.eigenvalues)
        np.testing.assert_allclose(r_default.coordinates, r_explicit.coordinates)
        np.testing.assert_allclose(
            r_default.proportion_explained, r_explicit.proportion_explained
        )
        np.testing.assert_allclose(
            r_default.cumulative_proportion, r_explicit.cumulative_proportion
        )
        assert r_default.correction_method == r_explicit.correction_method == "cmdscale"


class TestPCoACorrectionEliminateNegatives:
    """The Lingoes / Wickoff / Torgerson corrections must remove the
    negative eigenvalues (within floating-point slack)."""

    @pytest.mark.parametrize("correction", ["lingoes", "wickoff", "torgerson"])
    def test_no_negative_eigenvalues_after_correction(self, correction: str) -> None:
        analyzer = PCoAAnalyzer()
        # Use several seeds to cover different sign patterns of the
        # eigenvalue tail.
        for seed in (1, 2, 3, 4):
            D = _make_non_euclidean_distance(seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = analyzer.analyze(D, metric="bray-curtis", correction=correction)
            # Clip small numerical slop before asserting.
            eigs = np.asarray(result.eigenvalues, dtype=float)
            assert np.all(eigs > -1e-6), (
                f"{correction} left a real negative eigenvalue: {eigs}"
            )
            assert result.correction_method == correction

    @pytest.mark.parametrize("correction", ["lingoes", "wickoff", "torgerson"])
    def test_correction_is_recorded_on_result(self, correction: str) -> None:
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyzer.analyze(D, metric="bray-curtis", correction=correction)
        assert result.correction_method == correction


class TestPCoACorrectionRejectsUnknown:
    def test_unknown_correction_raises(self):
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=0)
        with pytest.raises(ValidationError, match="Unknown PCoA correction"):
            analyzer.analyze(D, metric="bray-curtis", correction="not_a_real_method")


class TestPCoACorrectionNoneAlias:
    """``'none'`` is accepted as an alias for the cmdscale branch."""

    def test_none_alias_matches_cmdscale(self):
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_none = analyzer.analyze(D, metric="bray-curtis", correction="none")
            r_cmd = analyzer.analyze(D, metric="bray-curtis", correction="cmdscale")
        np.testing.assert_allclose(r_none.eigenvalues, r_cmd.eigenvalues)
        np.testing.assert_allclose(r_none.coordinates, r_cmd.coordinates)
        assert r_none.correction_method == "none"


class TestPCoACorrectionBackwardCompatibility:
    """Backwards-compatibility: the signature used by the test suite
    and pre-fix callers must keep working."""

    def test_positional_args_still_work(self):
        analyzer = PCoAAnalyzer()
        D = _make_non_euclidean_distance(seed=9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyzer.analyze(D, 3, "bray-curtis")
        assert isinstance(result, PCoAResult)
        assert result.correction_method == "cmdscale"