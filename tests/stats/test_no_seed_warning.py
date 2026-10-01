# =============================================================================
# FILE: tests/stats/test_no_seed_warning.py
# =============================================================================
"""
Tests for the no-``random_seed`` ``RuntimeWarning`` on PERMANOVA / ANOSIM /
PCM permutation tests.

Background: historically the three permutation-based tests shared the
global ``np.random`` state when no ``random_seed`` was supplied. Two
calls with identical inputs could return different p-values, which is
incompatible with a publishable scientific result.

We now:
  1. Emit a ``RuntimeWarning`` when ``random_seed`` is not supplied, so
     users notice before they ship a non-reproducible result.
  2. Use a local ``np.random.default_rng(seed)`` when a seed IS
     supplied -- identical inputs + identical seed give identical
     p-values (already verified by the existing tests).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings

import numpy as np

from stats.anosim import ANOSIMAnalyzer
from stats.permanova import PERMANOVAAnalyzer


def _tiny_distance_matrix():
    rng = np.random.default_rng(0)
    data = rng.random((10, 3))
    D = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
    groups = ["A"] * 5 + ["B"] * 5
    return D, groups


class TestNoSeedRuntimeWarning:
    """All three permutation-based tests warn when no seed is supplied."""

    def test_permanova_warns_without_seed(self):
        D, groups = _tiny_distance_matrix()
        analyzer = PERMANOVAAnalyzer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.analyze(D, groups, n_permutations=9)
            runtime_warnings = [
                x for x in w
                if issubclass(x.category, RuntimeWarning) and "PERMANOVA" in str(x.message)
            ]
            assert len(runtime_warnings) >= 1, (
                f"Expected a PERMANOVA RuntimeWarning about no random_seed, "
                f"got: {[str(x.message) for x in w]}"
            )

    def test_anosim_warns_without_seed(self):
        D, groups = _tiny_distance_matrix()
        analyzer = ANOSIMAnalyzer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.analyze(D, groups, n_permutations=9)
            runtime_warnings = [
                x for x in w
                if issubclass(x.category, RuntimeWarning) and "ANOSIM" in str(x.message)
            ]
            assert len(runtime_warnings) >= 1, (
                f"Expected an ANOSIM RuntimeWarning about no random_seed, "
                f"got: {[str(x.message) for x in w]}"
            )

    def test_no_warning_with_seed(self):
        """When a seed is supplied, no RuntimeWarning is emitted."""
        D, groups = _tiny_distance_matrix()
        analyzer = PERMANOVAAnalyzer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.analyze(D, groups, n_permutations=9, random_seed=42)
            runtime_warnings = [
                x for x in w
                if issubclass(x.category, RuntimeWarning)
                and ("random_seed" in str(x.message) or "reproducible" in str(x.message))
            ]
            assert len(runtime_warnings) == 0, (
                f"Did not expect a no-seed RuntimeWarning, got: "
                f"{[str(x.message) for x in runtime_warnings]}"
            )

    def test_anosim_no_warning_with_seed(self):
        D, groups = _tiny_distance_matrix()
        analyzer = ANOSIMAnalyzer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.analyze(D, groups, n_permutations=9, random_seed=42)
            runtime_warnings = [
                x for x in w
                if issubclass(x.category, RuntimeWarning)
                and ("random_seed" in str(x.message) or "reproducible" in str(x.message))
            ]
            assert len(runtime_warnings) == 0