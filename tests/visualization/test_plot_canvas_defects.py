# =============================================================================
# Regression tests: views/ui_plot_canvas.py ordinations + PLS
# =============================================================================
"""
Regression tests for the four defects that were left in
``InteractivePlotCanvas``:

    1. PCA / PCoA / NMDS / CCA / LDA scores dropped the real sample names
       (everything became ``S1..Sn``) and the habitat groups (one colour,
       no 95% ellipses). The fix routes ``groups`` / ``labels`` /
       ``group_names`` parameters through to the canvas and uses the
       result object's attributes as the fallback.

    2. Two-Block PLS results were silently swallowed by ``ui_main_window``
       because ``InteractivePlotCanvas`` had no ``plot_pls`` method. The
       fix adds ``plot_pls`` and ``plot_pls_results`` (the alias the main
       window checks for) accepting both ``PLSResult`` dataclasses and
       the ``dict`` payload the PLSDialog already emits.

    3. PCA annotations stacked on top of each other for nearby samples
       (``S19/S20/S21``). The fix adds an ``annotate_offset=True``
       default that runs a cheap iterative repel.

    4. NMDS / CCA / LDA carried the same ``S{i+1}`` + single-group bugs.
       The fix rolls the same metadata resolution through all four.

These tests build the canvas with PyQt6 (Agg backend, no display), then
drive the plot methods directly and inspect the rendered artists.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from matplotlib.patches import Ellipse
from PyQt6.QtWidgets import QApplication

from views.ui_plot_canvas import InteractivePlotCanvas

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 20


@dataclass
class _FakePCAResult:
    """Minimal stand-in for stats.pca.PCAResult (no labels/groups attrs)."""

    scores: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    cumulative_variance: np.ndarray
    eigenvalues_raw: np.ndarray
    mean_vector: np.ndarray
    std_vector: np.ndarray | None
    n_components: int
    method: str
    singular_values: np.ndarray
    # NOTE: no `labels` and no `groups` attributes — the canvas used to
    # silently fall through to the ``S1..Sn`` placeholder.


@dataclass
class _FakePCoAResult:
    """Minimal stand-in for stats.pcoa.PCoAResult (no labels/groups attrs)."""

    coordinates: np.ndarray
    eigenvalues: np.ndarray
    proportion_explained: np.ndarray
    cumulative_proportion: np.ndarray
    distance_matrix: np.ndarray
    n_components: int
    metric: str
    negative_eigenvalue_sum: float = 0.0
    correction_method: str = "lingoes"


@dataclass
class _FakePLSResult:
    """Minimal stand-in for morphometrics.allometry.PLSResult."""

    singular_values: np.ndarray
    covariance_explained: np.ndarray
    cumulative_covariance: np.ndarray
    left_scores: np.ndarray
    right_scores: np.ndarray
    pls_loadings_left: np.ndarray
    pls_loadings_right: np.ndarray
    pls_correlations: np.ndarray
    integration_index: float
    n_components: int
    n_specimens: int
    rv_coefficient: float | None = None
    pls1_pvalue: float | None = None
    pls1_z: float | None = None
    random_correlations: np.ndarray | None = None


@dataclass
class _FakeLDAResult:
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    groups: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-viz-defects"])
    return app


@pytest.fixture
def canvas(qt_app):
    c = InteractivePlotCanvas()
    yield c
    try:
        c._figure.clear()
        c.close()
    except Exception:
        pass


@pytest.fixture
def pca_scores():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(N_SAMPLES, 5))
    # Make three visually separable clusters along PC1 / PC2.
    centres = np.array([[-3.0, -2.0], [0.0, 0.0], [3.0, 2.0]])
    for i in range(N_SAMPLES):
        base[i, :2] += centres[i % 3]
    return base


@pytest.fixture
def pca_result(pca_scores):
    rng = np.random.default_rng(1)
    return _FakePCAResult(
        scores=pca_scores[:, :3],
        loadings=rng.normal(size=(5, 3)),
        eigenvalues=np.array([5.0, 2.0, 1.0]),
        explained_variance=np.array([62.5, 25.0, 12.5]),
        cumulative_variance=np.array([62.5, 87.5, 100.0]),
        eigenvalues_raw=np.array([5.0, 2.0, 1.0, 0.5, 0.3]),
        mean_vector=np.zeros(5),
        std_vector=None,
        n_components=3,
        method="covariance",
        singular_values=np.array([np.sqrt(95), np.sqrt(38), np.sqrt(19)]),
    )


@pytest.fixture
def pcoa_result(pca_scores):
    rng = np.random.default_rng(2)
    return _FakePCoAResult(
        coordinates=pca_scores[:, :3],
        eigenvalues=np.array([0.8, 0.5, 0.2]),
        proportion_explained=np.array([0.5, 0.3, 0.2]),
        cumulative_proportion=np.array([0.5, 0.8, 1.0]),
        distance_matrix=rng.uniform(size=(N_SAMPLES, N_SAMPLES)),
        n_components=3,
        metric="braycurtis",
    )


@pytest.fixture
def site_groups():
    """5 habitat groups, 4 samples each."""
    return np.repeat(np.arange(4), N_SAMPLES // 4)


@pytest.fixture
def habitat_names():
    return ["Forest", "Grassland", "Wetland", "Coastal"]


@pytest.fixture
def site_labels():
    return [f"Site_{i + 1}" for i in range(N_SAMPLES)]


def _legend_texts(ax) -> list[str]:
    leg = ax.get_legend()
    if leg is None:
        return []
    return [str(t.get_text()) for t in leg.get_texts()]


def _annotation_texts(ax) -> list[str]:
    return [a.get_text() for a in ax.texts]


def _ellipse_patches(ax):
    return [p for p in ax.patches if isinstance(p, Ellipse)]


# ---------------------------------------------------------------------------
# Defect 1 + 4: PCA / PCoA / NMDS / LDA honour groups + labels
# ---------------------------------------------------------------------------


class TestPCAGroupsAndLabels:
    """Defect 1: PCA used to drop ``S1..Sn`` for ``Site_1..Site_20`` and
    paint every point in a single colour. The fix threads the new kwargs
    through and falls back to the result object's attributes."""

    def test_explicit_groups_and_labels_win(
        self, canvas, pca_result, site_groups, site_labels, habitat_names
    ):
        canvas.plot_pca_scores(
            pca_result,
            groups=site_groups,
            labels=site_labels,
            group_names=habitat_names,
        )
        ann = _annotation_texts(canvas._ax)
        assert "Site_1" in ann
        assert "Site_20" in ann
        # And the legacy placeholder is NOT used.
        assert "S1" not in ann
        # Group legend uses the human names, not "Group 1".
        legend = _legend_texts(canvas._ax)
        for name in habitat_names:
            assert name in legend

    def test_falls_back_to_synthetic_labels(self, canvas, pca_result):
        """Bare call (no kwargs) — back-compat path.

        No groups/labels were passed and the fake PCAResult has neither
        attribute, so the canvas must still draw something. The legacy
        ``S1..Sn`` placeholder is acceptable here; the bug only fires
        when the caller had real labels and the canvas threw them away.
        """
        canvas.plot_pca_scores(pca_result)
        # The plot should at least draw one scatter point per group slot.
        assert len(canvas._ax.collections) >= 1

    def test_ellipses_auto_on_with_two_groups(self, canvas, pca_result, site_groups):
        """≥2 groups → ellipses drawn even with the toolbar toggle off."""
        canvas._show_ellipses_check.setChecked(False)
        canvas.plot_pca_scores(pca_result, groups=site_groups)
        assert len(_ellipse_patches(canvas._ax)) >= 2

    def test_ellipses_can_be_forced_off(self, canvas, pca_result, site_groups):
        canvas.plot_pca_scores(pca_result, groups=site_groups, show_ellipses=False)
        assert len(_ellipse_patches(canvas._ax)) == 0

    def test_ellipses_can_be_forced_on_with_one_group(self, canvas, pca_result):
        """A single-group result also respects an explicit ``True``."""
        canvas.plot_pca_scores(pca_result, groups=np.zeros(N_SAMPLES, dtype=int), show_ellipses=True)
        # At least one ellipse; if the group has only 2-3 points the
        # ellipse routine may still draw it.
        assert len(_ellipse_patches(canvas._ax)) >= 1

    def test_groups_stored_for_hover(self, canvas, pca_result, site_groups, site_labels):
        canvas.plot_pca_scores(pca_result, groups=site_groups, labels=site_labels)
        np.testing.assert_array_equal(canvas._group_labels, site_groups)
        assert canvas._labels[:3] == ["Site_1", "Site_2", "Site_3"]


class TestPCoAGroupsAndLabels:
    def test_explicit_groups_and_labels(self, canvas, pcoa_result, site_groups, site_labels, habitat_names):
        canvas.plot_pcoa_scores(
            pcoa_result,
            groups=site_groups,
            labels=site_labels,
            group_names=habitat_names,
        )
        ann = _annotation_texts(canvas._ax)
        assert "Site_1" in ann
        assert "Site_20" in ann
        legend = _legend_texts(canvas._ax)
        for name in habitat_names:
            assert name in legend
        # ≥2 groups → ellipses on by default.
        assert len(_ellipse_patches(canvas._ax)) >= 2

    def test_falls_back_to_synthetic_labels(self, canvas, pcoa_result):
        canvas.plot_pcoa_scores(pcoa_result)
        assert len(canvas._ax.collections) >= 1


class TestNMDSGroupsAndLabels:
    def test_explicit_groups_and_labels(self, canvas, pcoa_result, site_groups, site_labels, habitat_names):
        canvas.plot_nmds(
            pcoa_result,
            groups=site_groups,
            labels=site_labels,
            group_names=habitat_names,
        )
        ann = _annotation_texts(canvas._ax)
        assert "Site_1" in ann
        legend = _legend_texts(canvas._ax)
        for name in habitat_names:
            assert name in legend
        assert len(_ellipse_patches(canvas._ax)) >= 2


class TestLDAPlotGroupsAndLabels:
    def test_explicit_groups_and_labels(self, canvas, site_groups, site_labels, habitat_names):
        rng = np.random.default_rng(3)
        lda_scores = rng.normal(size=(N_SAMPLES, 2))
        result = _FakeLDAResult(
            scores=lda_scores,
            explained_variance_ratio=np.array([0.7, 0.3]),
        )
        canvas.plot_lda_scores(
            result,
            groups=site_groups,
            labels=site_labels,
            group_names=habitat_names,
        )
        ann = _annotation_texts(canvas._ax)
        assert "Site_1" in ann
        legend = _legend_texts(canvas._ax)
        for name in habitat_names:
            assert name in legend
        # Ellipses on by default for ≥2 groups.
        assert len(_ellipse_patches(canvas._ax)) >= 2


# ---------------------------------------------------------------------------
# Defect 3: label repel
# ---------------------------------------------------------------------------


class TestAnnotationOffset:
    """PCA labels used to stack on top of each other (S19/S20/S21).
    The fix adds ``annotate_offset=True`` and an iterative repel."""

    def test_overlapping_labels_get_vertically_dodged(self, canvas, pca_result):
        rng = np.random.default_rng(4)
        # Stack five points on the exact same location — the historic
        # code wrote all five annotations on top of each other.
        scores = np.zeros((5, 3))
        scores[:, 0] = rng.normal(scale=0.001, size=5)
        scores[:, 1] = rng.normal(scale=0.001, size=5)
        labels = ["A", "B", "C", "D", "E"]
        pca_result.scores = scores
        canvas.plot_pca_scores(pca_result, labels=labels, annotate_offset=True)

        ann = [a for a in canvas._ax.texts if a.get_text() in labels]
        assert len(ann) == 5
        # After the pixel-space repel, the annotations sit at distinct
        # *data* y-positions because the dodge is converted back into
        # data coordinates. Round to a coarse precision to ignore float
        # jitter.
        ys = [a.get_position()[1] for a in ann]
        rounded = [round(float(y), 6) for y in ys]
        assert len(set(rounded)) >= 2, (
            f"All annotations collapsed to the same data y after dodge: {ys}"
        )

    def test_annotate_offset_false_keeps_legacy_layout(self, canvas, pca_result):
        rng = np.random.default_rng(5)
        scores = np.zeros((5, 3))
        scores[:, 0] = rng.normal(scale=0.001, size=5)
        scores[:, 1] = rng.normal(scale=0.001, size=5)
        labels = ["A", "B", "C", "D", "E"]
        pca_result.scores = scores
        canvas.plot_pca_scores(pca_result, labels=labels, annotate_offset=False)

        ann = [a for a in canvas._ax.texts if a.get_text() in labels]
        # With ``annotate_offset=False`` the canvas uses the historic
        # ``xytext=(0, 4)`` straight-above placement for every label.
        # Every annotation therefore sits at exactly the data y of its
        # point — small random jitter from the synthetic data.
        ys = sorted(a.get_position()[1] for a in ann)
        # Max-min span should be on the order of the data spread
        # (0.001), i.e. much smaller than any pixel-space dodge.
        assert max(ys) - min(ys) < 0.01


# ---------------------------------------------------------------------------
# Defect 2: PLS results were silently dropped
# ---------------------------------------------------------------------------


class TestPLS:
    """Defect 2: ``ui_main_window._on_run_pls`` checked for
    ``plot.plot_pls`` / ``plot.plot_pls_results``, neither existed on
    ``InteractivePlotCanvas``, so the ``else: return`` branch fired and
    the analysis vanished."""

    def _make_pls(self) -> _FakePLSResult:
        rng = np.random.default_rng(6)
        n = 15
        left = rng.normal(size=(n, 3))
        right = left @ rng.normal(size=(3, 3)) + 0.1 * rng.normal(size=(n, 3))
        return _FakePLSResult(
            singular_values=np.array([3.0, 1.0, 0.4]),
            covariance_explained=np.array([60.0, 30.0, 10.0]),
            cumulative_covariance=np.array([60.0, 90.0, 100.0]),
            left_scores=left,
            right_scores=right,
            pls_loadings_left=rng.normal(size=(3, 3)),
            pls_loadings_right=rng.normal(size=(3, 3)),
            pls_correlations=np.array([0.92, 0.65, 0.30]),
            integration_index=0.92,
            n_components=3,
            n_specimens=n,
        )

    def test_plot_pls_accepts_dataclass(self, canvas):
        pls = self._make_pls()
        canvas.plot_pls(pls)
        assert canvas._current_plot_type == "pls"
        # At least one scatter collection + a title that mentions the
        # integration index.
        assert len(canvas._ax.collections) >= 1
        title = canvas._ax.get_title()
        assert "0.9" in title

    def test_plot_pls_accepts_dict_payload(self, canvas):
        """``ui_main_window`` sends ``result.to_dict()`` — the canvas
        must accept that path without rewriting the controller."""
        pls = self._make_pls()
        payload = {
            "left_scores": pls.left_scores.tolist(),
            "right_scores": pls.right_scores.tolist(),
            "pls_correlations": pls.pls_correlations.tolist(),
            "integration_index": pls.integration_index,
            "rv_coefficients": pls.pls_correlations.tolist(),  # legacy key
            "n_components": pls.n_components,
            "n_specimens": pls.n_specimens,
        }
        canvas.plot_pls(payload)
        assert canvas._current_plot_type == "pls"
        assert len(canvas._ax.collections) >= 1

    def test_plot_pls_results_is_alias(self, canvas):
        """``ui_main_window._on_run_pls`` probes for ``plot_pls_results``
        — the alias must be the same callable, not a stub."""
        pls = self._make_pls()
        # Call via the alias; identical contract.
        canvas.plot_pls_results(pls)
        assert canvas._current_plot_type == "pls"
        assert len(canvas._ax.collections) >= 1

    def test_plot_pls_groups_appear_in_legend(self, canvas):
        pls = self._make_pls()
        canvas.plot_pls(pls, groups=np.array([0] * 7 + [1] * 8), group_names=["Sp1", "Sp2"])
        legend = _legend_texts(canvas._ax)
        assert "Sp1" in legend
        assert "Sp2" in legend

    def test_plot_pls_does_not_drop_result_silently(self, qt_app):
        """The main window's ``hasattr(plot, ...)`` branch must find
        the method on a freshly constructed canvas."""
        c = InteractivePlotCanvas()
        assert hasattr(c, "plot_pls")
        assert hasattr(c, "plot_pls_results")
        c._figure.clear()
        c.close()


# ---------------------------------------------------------------------------
# Back-compat smoke test: existing call sites still work
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_pca_legacy_call_signature(self, canvas, pca_result):
        """The existing ``plot.plot_pca_scores(result)`` call in
        ``ui_main_window`` must keep working — no kwargs, just the
        result object."""
        canvas.plot_pca_scores(pca_result)
        assert canvas._current_plot_type == "pca"
        assert len(canvas._ax.collections) >= 1

    def test_pcoa_legacy_call_signature(self, canvas, pcoa_result):
        canvas.plot_pcoa_scores(pcoa_result)
        assert canvas._current_plot_type == "pcoa"
        assert len(canvas._ax.collections) >= 1
