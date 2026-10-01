# =============================================================================
# FILE: tests/stats/test_tukey_q.py
# =============================================================================
"""
Tests for the Tukey HSD q-statistic (univariate.py).

The q-statistic is

    q = |mean_diff| / sqrt(MSE * (1/ni + 1/nj) / 2)

The ``/ 2`` belongs INSIDE the square root. That is the convention the
studentized-range distribution is parameterised for, and it is what
``scipy.stats.tukey_hsd`` computes internally::

    stand_err = np.sqrt(normalize * mse / 2)          # scipy/stats/_hypotests.py
    t_stat    = np.abs(mean_differences) / stand_err

A previous version of this file asserted the opposite (no ``/ 2``) on the
reasoning that the "textbook" formula omits it. That was wrong, and the
error was self-inconsistent: a q that omits the ``/ 2`` cannot reproduce the
p-value it is reported alongside. ``studentized_range.sf(q, k, df)`` is exactly
how the p-value is computed, so the two must agree. The decisive check is
``test_q_reproduces_the_reported_p_value`` below, which holds for the ``/ 2``
form and fails for the other (on scipy's own documented example, the no-``/2``
form gives p = 0.0827 / 0.9901 / 0.1037 where the true values are
0.0144 / 0.9803 / 0.0203).

R's ``TukeyHSD`` reports the same adjusted p-values as ``scipy.tukey_hsd``;
R is not required to check this, because the identity
``sf(q, k, df) == p`` is self-validating.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats as sp_stats

from stats.univariate import UnivariateAnalyzer


def _mse(groups):
    all_vals = np.concatenate(groups)
    return float(sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / (len(all_vals) - len(groups)))


def _q(mean_diff, mse, ni, nj):
    """The Tukey q-statistic, in the convention scipy/R use."""
    return abs(mean_diff) / np.sqrt(mse * (1.0 / ni + 1.0 / nj) / 2.0)


def _q_missing_halving(mean_diff, mse, ni, nj):
    """The wrong variant this file previously enshrined: no ``/ 2``."""
    return abs(mean_diff) / np.sqrt(mse * (1.0 / ni + 1.0 / nj))


# scipy's own docstring example: headache-medicine relief times. Its published
# p-values are 0.014448 / 0.980311 / 0.020331.
_HEADACHE = (
    [24.5, 23.5, 26.4, 27.1, 29.9],
    [28.4, 34.2, 29.5, 32.2, 30.1],
    [26.1, 28.3, 24.3, 26.2, 27.8],
)


class TestTukeyQStatistic:
    """PaleoAST q must match scipy.stats.tukey_hsd / R's TukeyHSD."""

    def test_q_matches_scipy_formula_equal_sizes(self):
        """Equal-size groups: q = |mean_diff| / sqrt(MSE * (1/n+1/n) / 2)."""
        np.random.seed(0)
        groups_data = [np.random.normal(loc=i * 2.0, scale=1.0, size=10) for i in range(3)]
        labels = ["G0", "G1", "G2"]
        result = UnivariateAnalyzer().one_way_anova(
            np.concatenate(groups_data),
            np.repeat(labels, [len(g) for g in groups_data]),
            tukey=True,
        )
        assert result.tukey_results is not None
        mse = _mse(groups_data)
        for pair in result.tukey_results:
            i = labels.index(pair["group_a"])
            j = labels.index(pair["group_b"])
            expected = _q(np.mean(groups_data[i]) - np.mean(groups_data[j]), mse,
                          len(groups_data[i]), len(groups_data[j]))
            assert_allclose(pair["q_stat"], expected, rtol=1e-8)

    def test_q_matches_scipy_formula_unequal_sizes(self):
        """Unequal-size groups: the same formula, with each group's own n."""
        np.random.seed(1)
        groups_data = [
            np.random.normal(loc=0.0, scale=1.0, size=5),
            np.random.normal(loc=3.0, scale=1.0, size=10),
            np.random.normal(loc=6.0, scale=1.0, size=8),
        ]
        labels = ["A", "B", "C"]
        result = UnivariateAnalyzer().one_way_anova(
            np.concatenate(groups_data),
            np.repeat(labels, [len(g) for g in groups_data]),
            tukey=True,
        )
        assert result.tukey_results is not None
        mse = _mse(groups_data)
        for pair in result.tukey_results:
            i = labels.index(pair["group_a"])
            j = labels.index(pair["group_b"])
            expected = _q(np.mean(groups_data[i]) - np.mean(groups_data[j]), mse,
                          len(groups_data[i]), len(groups_data[j]))
            assert_allclose(pair["q_stat"], expected, rtol=1e-8)

    def test_q_reproduces_the_reported_p_value(self):
        """The self-consistency identity: ``sf(q, k, df)`` IS the p-value.

        This is what makes the convention decidable without appealing to R.
        ``_tukey_hsd`` reports ``q_stat`` and ``p_adj`` side by side; a reader
        (or the fallback branch, which literally recomputes the p-value this
        way) must be able to recover one from the other. Only the ``/ 2`` form
        satisfies it.
        """
        groups_data = [np.array(g, dtype=float) for g in _HEADACHE]
        result = UnivariateAnalyzer().one_way_anova(
            np.concatenate(groups_data), [0] * 5 + [1] * 5 + [2] * 5, tukey=True
        )
        mse = _mse(groups_data)
        k, df = 3, len(np.concatenate(groups_data)) - 3
        for row_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            pair = result.tukey_results[row_idx]
            q = pair["q_stat"]
            assert_allclose(
                float(sp_stats.studentized_range.sf(q, k, df)),
                pair["p_adj"],
                rtol=1e-9,
                err_msg=f"q for pair {i}-{j} does not reproduce its own p-value",
            )
            # And it must be the /2 form, not the halved one.
            diff = float(np.mean(groups_data[i]) - np.mean(groups_data[j]))
            assert_allclose(q, _q(diff, mse, 5, 5), rtol=1e-9)
            assert not np.isclose(q, _q_missing_halving(diff, mse, 5, 5), rtol=1e-6), (
                "q matches the no-/2 variant, which cannot reproduce the p-value"
            )

    def test_p_value_matches_scipy_tukey_hsd(self):
        """The p-value (read from scipy.stats.tukey_hsd) is unchanged."""
        np.random.seed(3)
        groups_data = [np.random.normal(loc=i * 1.5, scale=1.0, size=8) for i in range(3)]
        labels = ["X", "Y", "Z"]
        result = UnivariateAnalyzer().one_way_anova(
            np.concatenate(groups_data),
            np.repeat(labels, [len(g) for g in groups_data]),
            tukey=True,
        )
        tukey_ref = sp_stats.tukey_hsd(*groups_data)
        for pair in result.tukey_results:
            i = labels.index(pair["group_a"])
            j = labels.index(pair["group_b"])
            assert_allclose(pair["p_adj"], tukey_ref.pvalue[i, j], rtol=1e-10)
            assert_allclose(pair["p_value"], pair["p_adj"], rtol=0.0)

    def test_q_on_the_scipy_fallback_path(self, monkeypatch):
        """The fallback branch must obey the same convention as the primary one.

        ``_tukey_hsd`` has two implementations: one used when
        ``scipy.stats.tukey_hsd`` succeeds, and one inside the
        ``except (AttributeError, TypeError)`` branch. The fallback does not
        borrow scipy's p-value -- it recomputes it as
        ``studentized_range.sf(q, k, df)`` -- so it is the branch where a wrong q
        propagates straight into a wrong p-value. Nothing else exercises it
        (every other test here runs on a scipy that has ``tukey_hsd``), so
        force it the way an old scipy would.
        """
        import stats.univariate as univariate_module

        np.random.seed(11)
        groups_data = [np.random.normal(loc=i * 1.5, scale=1.0, size=8) for i in range(3)]
        labels = ["X", "Y", "Z"]

        # Take the reference BEFORE patching: monkeypatch.setattr on
        # ``univariate_module.sp_stats`` patches the shared scipy.stats module,
        # so a reference computed afterwards would hit the stub too.
        reference = sp_stats.tukey_hsd(*groups_data)

        def _boom(*args, **kwargs):
            raise AttributeError(
                "module 'scipy.stats' has no attribute 'tukey_hsd' (simulated)"
            )

        monkeypatch.setattr(univariate_module.sp_stats, "tukey_hsd", _boom)

        result = UnivariateAnalyzer().one_way_anova(
            np.concatenate(groups_data),
            np.repeat(labels, [len(g) for g in groups_data]),
            tukey=True,
        )

        mse = _mse(groups_data)
        checked = 0
        for row_idx, pair in enumerate(result.tukey_results):
            i = labels.index(pair["group_a"])
            j = labels.index(pair["group_b"])
            expected = _q(np.mean(groups_data[i]) - np.mean(groups_data[j]), mse,
                          len(groups_data[i]), len(groups_data[j]))
            assert_allclose(pair["q_stat"], expected, rtol=1e-8)
            # The whole point of the fallback: its p-value is derived from q,
            # so a wrong q here means a wrong published p-value.
            assert_allclose(pair["p_adj"], float(reference.pvalue[i, j]), rtol=1e-9)
            checked += 1
        assert checked == 3, f"expected 3 pairs on the fallback path, saw {checked}"
