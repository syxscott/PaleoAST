# =============================================================================
# FILE: tests/stats/test_lda_degenerate_input.py
# =============================================================================
"""
LDA must reject degenerate input with an explanation, not a stray IndexError.

`LinearDiscriminantAnalysis(solver="svd")` selects its numerical rank from the
leading singular value, which is zero when nothing in the data varies. The rank
filter then keeps no components and sklearn indexes an empty array:

    File ".../sklearn/discriminant_analysis.py", line 628, in _solve_svd
      rank = xp.sum(xp.astype(S > self.tol * S[0], xp.int32))
    IndexError: index 0 is out of bounds for axis 0 with size 0

That traceback points into a third-party package and says nothing about what the
caller did wrong. Constant variables and classes that are constant within
themselves are ordinary things to hand a discriminant analysis, and every other
rejection in this module is a messaged `ComputationError` -- this one was the
odd exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from stats.lda import LDAAnalyzer
from utils.exceptions import ComputationError


def _two_classes(n_per: int = 12, n_vars: int = 3, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [
            rng.normal(loc=[2.0, 0.0, 0.0], scale=0.5, size=(n_per, n_vars)),
            rng.normal(loc=[-2.0, 0.0, 0.0], scale=0.5, size=(n_per, n_vars)),
        ]
    )
    y = np.array([0] * n_per + [1] * n_per)
    return x, y


class TestConstantInputIsRejected:
    def test_all_rows_identical(self):
        with pytest.raises(ComputationError, match="zero variance"):
            LDAAnalyzer().analyze(np.ones((20, 3)), np.arange(20) % 2)

    def test_one_constant_column_among_varying_ones(self):
        """A single constant variable is enough to make the fit singular."""
        x, y = _two_classes()
        x[:, 1] = 4.0
        with pytest.raises(ComputationError, match="zero variance"):
            LDAAnalyzer().analyze(x, y)

    def test_the_message_names_the_offending_variable(self):
        x, y = _two_classes()
        x[:, 2] = -1.5
        with pytest.raises(ComputationError) as excinfo:
            LDAAnalyzer().analyze(x, y, variable_names=["length", "width", "depth"])
        assert "depth" in str(excinfo.value), str(excinfo.value)

    def test_the_message_says_what_to_do(self):
        with pytest.raises(ComputationError) as excinfo:
            LDAAnalyzer().analyze(np.ones((20, 3)), np.arange(20) % 2)
        message = str(excinfo.value).lower()
        assert "remove" in message or "supply" in message, message


class TestNonDegenerateInputIsUnaffected:
    def test_well_conditioned_data_still_fits(self):
        x, y = _two_classes()
        result = LDAAnalyzer().analyze(x, y)
        assert np.all(np.isfinite(np.asarray(result.scores, dtype=float)))
        assert float(result.accuracy) > 0.9

    def test_a_near_constant_column_is_still_accepted(self):
        """The guard is for exactly-zero spread, not merely small spread."""
        x, y = _two_classes()
        x[:, 1] = 1.0 + 1e-6 * np.arange(x.shape[0], dtype=float)
        result = LDAAnalyzer().analyze(x, y)
        assert float(result.accuracy) > 0.9
