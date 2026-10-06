# =============================================================================
# FILE: tests/stats/test_detriding.py
# =============================================================================
"""
Tests for correspondence analysis and detrended correspondence analysis.

Two independent things are pinned down here.

CA is checked against the textbook SVD of Greenacre (1984), computed
inside the test with scipy. That is an independent implementation, not a
restatement of what the module does, and the comparison is on the
1-D subspace because a CA axis is only defined up to a sign flip.

DCA is checked on a constructed table where the answer is known before
the code runs: a dominant linear gradient plus a factor deliberately
orthogonalised against it. CA ranks the gradient first; DCA must deflate
that axis and promote the orthogonal one. Getting the multiplier the
wrong way round -- using the regression's 1/(1-R^2) instead of the
residual variance 1-R^2 -- inverts exactly this outcome, so the test is
written to fail in that case rather than merely differ from it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import linalg
from scipy import stats as sp_stats

from stats.detriding import DetrendedCAAnalyzer
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError


@pytest.fixture
def analyzer() -> DetrendedCAAnalyzer:
    """A fresh analyzer."""
    return DetrendedCAAnalyzer()


def _table(seed: int = 0, shape: tuple[int, int] = (12, 9)) -> npt.NDArray:
    """A random but strictly positive contingency table."""
    return np.random.default_rng(seed).poisson(8, size=shape).astype(float) + 1.0


def _orthogonal_block(n: int, gradient: npt.NDArray) -> npt.NDArray:
    """An alternating pattern made exactly orthogonal to the gradient."""
    raw = np.array([1.0 if i % 2 else -1.0 for i in range(n)])
    design = np.column_stack([np.ones(n), gradient])
    out = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    return out / out.std()


def _gradient_dominant_table(seed: int = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """A table whose first CA axis is a gradient and second an orthogonal factor.

    The gradient species carry 300 counts each, the orthogonal species
    40, so the gradient supplies the leading axis. The block factor is
    orthogonalised first, so the two are genuinely independent and
    "which axis is independent" has an answer that does not depend on
    the code under test.
    """
    n = 24
    gradient = np.linspace(0.0, 1.0, n)
    block = _orthogonal_block(n, gradient)
    block_scaled = (block - block.min()) / np.ptp(block)
    table = np.column_stack(
        [
            300 * (1 - gradient) + 1,
            300 * gradient + 1,
            40 * block_scaled + 1,
            40 * (1 - block_scaled) + 1,
        ]
    )
    return np.random.default_rng(seed).poisson(table).astype(float), gradient, block


# ---------------------------------------------------------------------------
# Correspondence analysis
# ---------------------------------------------------------------------------


def test_ca_first_axis_matches_the_textbook_svd(analyzer: DetrendedCAAnalyzer) -> None:
    """The first CA axis must lie in the subspace of the textbook SVD.

    Greenacre (1984) / Benzecri (1973): with
    ``S = (P - r c^T) / sqrt(r c^T)`` and ``u, sigma, v = svd(S)``, the row
    axis is ``u[:, 0] * sigma[0] / sqrt(r)``. Compared as 1-D subspaces
    via |cos|, since either sign is a valid CA axis.
    """
    F = _table()
    P = F / F.sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    expected_matrix = np.outer(r, c)
    S = (P - expected_matrix) / np.sqrt(expected_matrix)
    u, sigma, _v = linalg.svd(S)
    reference = u[:, 0] * sigma[0] / np.sqrt(r)

    result = analyzer.correspondence(F)
    mine = result.row_scores[:, 0]
    cosine = abs(
        float(np.dot(mine, reference) / (np.linalg.norm(mine) * np.linalg.norm(reference)))
    )
    assert cosine == pytest.approx(1.0, abs=1e-9)


def test_ca_eigenvalues_are_descending(analyzer: DetrendedCAAnalyzer) -> None:
    """SVD returns singular values in descending order, so the CA
    eigenvalues must be too."""
    result = analyzer.correspondence(_table(seed=1))
    assert np.all(np.diff(result.eigenvalues) <= 1e-18)


def test_ca_inertia_is_the_sum_of_eigenvalues(analyzer: DetrendedCAAnalyzer) -> None:
    """Inertia is a total, not a per-axis figure."""
    result = analyzer.correspondence(_table(seed=2))
    assert result.inertia == pytest.approx(float(np.sum(result.eigenvalues)), rel=1e-9)
    assert np.isclose(result.explained_ratio.sum(), 1.0, atol=1e-9)


def test_ca_rejects_negative_counts(analyzer: DetrendedCAAnalyzer) -> None:
    """A contingency table holds counts."""
    F = _table()
    F[0, 0] = -5.0
    with pytest.raises(ValidationError, match="negative"):
        analyzer.correspondence(F)


def test_ca_rejects_a_zero_sum_table(analyzer: DetrendedCAAnalyzer) -> None:
    """All-zero means the chi-square distances are undefined."""
    with pytest.raises(ComputationError):
        analyzer.correspondence(np.zeros((5, 4)))


def test_ca_rejects_a_one_dimensional_table(analyzer: DetrendedCAAnalyzer) -> None:
    """Needs at least 2 rows and 2 columns to have any inertia."""
    with pytest.raises(MatrixDimensionError):
        analyzer.correspondence(np.ones((1, 5)))
    with pytest.raises(MatrixDimensionError):
        analyzer.correspondence(np.ones((5, 1)))


def test_ca_drops_all_zero_rows(analyzer: DetrendedCAAnalyzer) -> None:
    """An empty row carries no information and must not poison the SVD."""
    F = _table(seed=3)
    F_with_empty = np.vstack([F, np.zeros((1, F.shape[1]))])
    with_empty = analyzer.correspondence(F_with_empty)
    without = analyzer.correspondence(F)
    assert with_empty.n_rows == without.n_rows
    np.testing.assert_allclose(with_empty.eigenvalues, without.eigenvalues, rtol=1e-12)


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------


def test_ca_ranks_the_gradient_first(analyzer: DetrendedCAAnalyzer) -> None:
    """The constructed table must actually have a gradient-dominated first
    axis, or the DCA test below would prove nothing."""
    F, gradient, _block = _gradient_dominant_table()
    ca = analyzer.correspondence(F)
    assert sp_stats.pearsonr(ca.row_scores[:, 0], gradient).statistic > 0.95


def test_dca_demotes_the_axis_the_gradient_explains(
    analyzer: DetrendedCAAnalyzer,
) -> None:
    """The method's whole purpose: push the gradient axis down.

    A multiplier of 1/(1 - R^2) instead of 1 - R^2 would ENLARGE this
    axis and leave the ranking untouched, so the assertion is on the
    ranking, not just on the numbers being finite.
    """
    F, gradient, _block = _gradient_dominant_table()
    ca = analyzer.correspondence(F)
    dca = analyzer.analyze(F, gradient=gradient)

    # CA ranks the gradient axis first; DCA must reverse that. Asserting
    # both is what makes this a test of the *change* rather than of DCA's
    # output on its own.
    assert int(np.argmax(ca.eigenvalues)) == 0
    assert dca.first_independent_axis() == 2
    # The gradient axis is deflated to nearly nothing.
    assert dca.r_squared[0] > 0.9
    assert dca.retained_variance[0] < 0.2
    assert dca.detrended_eigenvalues[0] < dca.ca_eigenvalues[0]
    # The orthogonal axis is essentially untouched. Not exactly 1.0: CA axes
    # are orthogonal under the mass-weighted inner product the SVD uses,
    # while the detrending regression is unweighted, so a little of axis 2
    # survives. 1.8e-5 of it does, against 0.997 of axis 1.
    assert dca.retained_variance[1] == pytest.approx(1.0, abs=1e-3)


def test_dca_never_inflates_an_eigenvalue(analyzer: DetrendedCAAnalyzer) -> None:
    """Deflation is a property, not a coincidence of one table.

    This is the assertion the 1/(1-R^2) version fails: it makes every
    axis the gradient explains larger than it was.
    """
    for seed in (0, 1, 2, 3):
        F, gradient, _block = _gradient_dominant_table(seed=seed)
        dca = analyzer.analyze(F, gradient=gradient)
        assert np.all(dca.detrended_eigenvalues <= dca.ca_eigenvalues + 1e-18)


def test_dca_scores_are_scaled_by_sqrt_of_retained(
    analyzer: DetrendedCAAnalyzer,
) -> None:
    """The score's variance and the eigenvalue must stay consistent."""
    F, gradient, _block = _gradient_dominant_table()
    ca = analyzer.correspondence(F)
    dca = analyzer.analyze(F, gradient=gradient)
    for k in range(min(3, ca.row_scores.shape[1])):
        var_ca = float(np.var(ca.row_scores[:, k]))
        var_dca = float(np.var(dca.row_scores[:, k]))
        assert var_dca == pytest.approx(var_ca * dca.retained_variance[k], rel=1e-9)


def test_dca_without_a_gradient_leaves_axis_one_alone(
    analyzer: DetrendedCAAnalyzer,
) -> None:
    """No gradient supplied: axis 1 is the gradient, so it is not detrended."""
    F, _gradient, _block = _gradient_dominant_table()
    ca = analyzer.correspondence(F)
    dca = analyzer.analyze(F, gradient=None)
    np.testing.assert_allclose(
        dca.row_scores[:, 0], ca.row_scores[:, 0], rtol=1e-12
    )
    assert dca.retained_variance[0] == pytest.approx(1.0)
    assert dca.gradient is None


def test_dca_axis_order_is_a_permutation_in_descending_order(
    analyzer: DetrendedCAAnalyzer,
) -> None:
    """axis_order is a re-ranking, not a re-ordering of the columns."""
    F, gradient, _block = _gradient_dominant_table()
    dca = analyzer.analyze(F, gradient=gradient)
    assert sorted(dca.axis_order.tolist()) == list(range(len(dca.detrended_eigenvalues)))
    ordered = dca.detrended_eigenvalues[dca.axis_order]
    assert np.all(np.diff(ordered) <= 1e-18)


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_constant_gradient_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """A constant gradient explains nothing; the regression is undefined."""
    F, _gradient, _block = _gradient_dominant_table()
    with pytest.raises(ValidationError, match="constant"):
        analyzer.analyze(F, gradient=np.zeros(F.shape[0]))


def test_wrong_length_gradient_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """One gradient value per row, or it does not line up."""
    F, _gradient, _block = _gradient_dominant_table()
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze(F, gradient=np.linspace(0, 1, F.shape[0] + 1))


def test_non_finite_gradient_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """NaN in the gradient is a data problem."""
    F, gradient, _block = _gradient_dominant_table()
    bad = gradient.copy()
    bad[2] = np.nan
    with pytest.raises(ValidationError):
        analyzer.analyze(F, gradient=bad)


def test_non_finite_table_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """NaN counts are not counts."""
    F = _table()
    F[1, 1] = np.nan
    with pytest.raises(ValidationError):
        analyzer.analyze(F)


def test_single_observation_table_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """One row has no inertia to decompose."""
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze(np.array([[1.0, 2.0, 3.0]]))


def test_bad_n_components_is_rejected(analyzer: DetrendedCAAnalyzer) -> None:
    """At least one axis has to be produced."""
    with pytest.raises(ValidationError):
        analyzer.correspondence(_table(), n_components=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_results_serialise(analyzer: DetrendedCAAnalyzer) -> None:
    """to_dict and summary both work for CA and DCA."""
    F, gradient, _block = _gradient_dominant_table()
    ca = analyzer.correspondence(F)
    dca = analyzer.analyze(F, gradient=gradient)

    ca_payload = ca.to_dict()
    assert set(ca_payload) >= {"eigenvalues", "inertia", "n_rows", "n_columns"}
    assert "Correspondence Analysis" in ca.summary()

    dca_payload = dca.to_dict()
    assert dca_payload["first_independent_axis"] == 2
    assert dca_payload["used_gradient"] is True
    assert "Detrended Correspondence Analysis" in dca.summary()


def test_last_results_are_retained(analyzer: DetrendedCAAnalyzer) -> None:
    """Both results are kept.

    ``analyze`` runs the CA internally, so the CA it leaves behind is
    that internal one, not an earlier external call. Checking identity
    therefore has to come after the call whose result is being asked
    about.
    """
    assert analyzer.last_ca is None
    assert analyzer.last_result is None
    F, gradient, _block = _gradient_dominant_table()
    dca = analyzer.analyze(F, gradient=gradient)
    assert analyzer.last_result is dca
    assert analyzer.last_ca is dca._ca_result
    # A subsequent explicit CA becomes the one last_ca points at.
    ca = analyzer.correspondence(F)
    assert analyzer.last_ca is ca
    assert analyzer.last_result is dca
