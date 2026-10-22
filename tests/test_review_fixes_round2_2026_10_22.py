# =============================================================================
# FILE: tests/test_review_fixes_round2_2026_10_22.py
# =============================================================================
"""
Regression tests for the defects found by the 2026-10-22 second-round review.

These cover the blind spots the first round did not reach: the stratigraphy
package, the phylogenetics and macroevolution packages, the plotting package,
the process/scheduler layer, the LaTeX report builder, the preset store, the
parsers, and the parts of the view layer the first round only skimmed.

Each test pins one defect that used to produce a silently wrong number, a
false success, or a process abort.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="module")
def qapp():
    """A QApplication held for the module.

    Yielded rather than created as a bare expression: a QApplication with
    no Python reference is collected the moment the statement ends, which
    tears down Qt while the widgets still exist -- a C-level abort that
    only shows up when the file runs alongside other view tests.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-round2-tests"])
    yield app


def _add(a, b):
    """A module-level function: a lambda cannot be pickled by multiprocessing."""
    return a + b


# =============================================================================
# stratigraphy: a missing value must not silence an analysis
# =============================================================================
class TestNonFiniteValues:
    def test_one_nan_does_not_suppress_every_excursion(self):
        from stratigraphy.isotope_analysis import detect_excursions_from_values

        with_nan = np.array([0.0, 0.1, np.nan, 0.05, 5.0, 0.0, 0.1, 0.0])
        clean = np.array([0.0, 0.1, 0.05, 5.0, 0.0, 0.1, 0.0])
        assert len(detect_excursions_from_values(with_nan, threshold=2.0, min_duration=1)) >= 1
        assert len(detect_excursions_from_values(clean, threshold=2.0, min_duration=1)) >= 1

    def test_one_nan_does_not_empty_remove_outliers(self):
        from stratigraphy.isotope_analysis import remove_outliers

        kept, _ = remove_outliers(np.array([1.0, 2.0, np.nan, 4.0, 100.0]), method="iqr")
        assert kept.size == 3

    def test_a_zero_variance_series_keeps_its_points(self):
        from stratigraphy.isotope_analysis import remove_outliers

        kept, _ = remove_outliers(np.array([5.0, 5.0, 5.0, 5.0]), method="zscore")
        assert kept.size == 4

    def test_lowess_keeps_the_length_of_its_input(self):
        from stratigraphy.isotope_analysis import lowess_smooth

        out = lowess_smooth(np.arange(3.0), np.array([1.0, np.nan, 3.0]), span=0.5)
        assert len(out) == 3

    def test_a_constant_series_is_not_a_significant_cycle(self):
        from stratigraphy.spectral_analysis import SpectralAnalyzer
        from utils.exceptions import ComputationError

        t = np.arange(0, 100, 0.5)
        with pytest.raises(ComputationError):
            SpectralAnalyzer().analyze(t, np.ones_like(t))

    def test_the_peak_period_never_exceeds_the_record(self):
        from stratigraphy.spectral_analysis import SpectralAnalyzer

        t = np.arange(0, 100, 0.5)
        result = SpectralAnalyzer().analyze(t, 1 + 0.05 * t, detrend=False)
        assert result.peak_period is not None
        assert result.peak_period <= (t[-1] - t[0]) * 1.001

    def test_denser_sampling_does_not_move_the_reported_period(self):
        from stratigraphy.spectral_analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer()
        periods = []
        for n in (500, 2000, 20000):
            t = np.linspace(0, 100, n)
            y = np.sin(2 * np.pi * t / 7.0)
            periods.append(analyzer.analyze(t, y, n_frequencies=1000).peak_period)
        assert max(periods) - min(periods) < 0.5

    def test_the_wavelet_axis_uses_the_time_axis_units(self):
        from stratigraphy.spectral_analysis import SpectralAnalyzer

        time = np.arange(0, 100, 0.1)
        values = np.sin(2 * np.pi * time / 5.0)
        result = SpectralAnalyzer().wavelet_transform(time, values, scales=np.arange(4, 60, 2))
        assert abs(1.0 / result.peak_frequency - 5.0) < 1.0


# =============================================================================
# stratigraphy: the statistics themselves
# =============================================================================
class TestStatistics:
    def test_the_sen_slope_interval_covers_the_slope(self):
        from stratigraphy.cycles import mann_kendall

        rng = np.random.default_rng(0)
        true_slope = 0.35
        n = 40
        t = np.arange(n, dtype=float)
        covered = 0
        trials = 100
        for _ in range(trials):
            y = 1.0 + true_slope * t + rng.normal(0, 0.4, n)
            lo, hi = mann_kendall(y, times=t).slope_ci
            covered += lo <= true_slope <= hi
        # The old formula managed 21-44% while claiming 95%.
        assert covered / trials >= 0.8

    def test_tau_is_kendalls_tau_b(self):
        from scipy import stats as sp_stats

        from stratigraphy.cycles import mann_kendall

        for data in ([1, 1, 2, 2, 3, 3], [1, 2, 2, 3], [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]):
            x = np.asarray(data, dtype=float)
            t = np.arange(x.size, dtype=float)
            assert mann_kendall(x, times=t).tau == pytest.approx(sp_stats.kendalltau(x, t, variant="b").statistic)

    def test_fitch_scores_a_polytomy_at_the_true_minimum(self):
        from phylogenetics.fitch import FitchAlgorithm
        from phylogenetics.tree import PhyloTree

        result = FitchAlgorithm().compute(
            PhyloTree.from_newick("(D,B,A,C);"),
            {"D": "0", "A": "1", "B": "1", "C": "1"},
        )
        assert result.tree_length == 1

    def test_the_kaplan_meier_upper_bound_is_one_where_the_curve_reaches_zero(self):
        from macroevolution.survival import KaplanMeierAnalyzer

        result = KaplanMeierAnalyzer().fit(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
        assert float(np.asarray(result.upper_ci)[-1]) == pytest.approx(1.0)


# =============================================================================
# stratigraphy: the time scale and the bin table
# =============================================================================
class TestTimeScale:
    def test_a_named_interval_returns_only_itself(self):
        from stratigraphy.time_bins import get_scale

        jurassic = get_scale(rank="period", interval="Jurassic")
        assert [r["interval_name"] for r in jurassic] == ["Jurassic"]
        paleocene = get_scale(rank="epoch", interval="Paleocene")
        assert [r["interval_name"] for r in paleocene] == ["Paleocene"]

    def test_a_two_name_window_keeps_what_is_between(self):
        from stratigraphy.time_bins import get_scale

        window = get_scale(rank="period", interval=("Triassic", "Cretaceous"))
        assert [r["interval_name"] for r in window] == ["Triassic", "Jurassic", "Cretaceous"]

    def test_a_numeric_window_still_keeps_what_touches_it(self):
        from stratigraphy.time_bins import get_scale

        window = get_scale(rank="period", interval=(66.0, 201.4))
        assert len(window) >= 3

    def test_an_empty_event_matrix_is_refused(self):
        from stratigraphy.biostratigraphy import UAAnalyzer
        from utils.exceptions import ComputationError

        with pytest.raises(ComputationError):
            UAAnalyzer().analyze(np.zeros((2, 0)), np.zeros((2, 0)))

    def test_a_short_event_name_list_is_refused(self):
        from stratigraphy.biostratigraphy import RASCAnalyzer
        from utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            RASCAnalyzer().analyze(
                np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                event_names=["only"],
            )

    def test_the_layers_above_a_lad_count_the_layers_above_it(self):
        from stratigraphy.extinction import ExtinctionIntervalAnalyzer

        result = ExtinctionIntervalAnalyzer().analyze(np.array([15.0, 12.0, 8.0, 5.0, 3.0]))
        assert list(np.asarray(result.n_layers_above_lad).ravel()) == [4, 3, 2, 1, 0]


# =============================================================================
# phylogenetics: the search must be able to move
# =============================================================================
class TestTreeSearch:
    def test_nni_on_the_root_edge_moves_the_unrooted_topology(self):
        from phylogenetics.heuristic_search import NNIOperation
        from phylogenetics.tree import PhyloTree

        tree = PhyloTree.from_newick("((a:1,b:1):1,(c:1,d:1):1);")
        results = set()
        for option in (1, 2):
            for target in tree.root.children:
                results.add(NNIOperation(tree.root, target, option).apply(tree).to_newick())
        assert len(results) >= 2

    def test_the_majority_rule_consensus_does_not_depend_on_the_input_order(self):
        from phylogenetics.strict_consensus import StrictConsensusTree
        from phylogenetics.tree import PhyloTree

        first = StrictConsensusTree().build_majority_rule(
            [
                PhyloTree.from_newick("((A,B),(C,D));"),
                PhyloTree.from_newick("((A,C),(B,D));"),
            ],
            threshold=0.5,
        )
        second = StrictConsensusTree().build_majority_rule(
            [
                PhyloTree.from_newick("((A,C),(B,D));"),
                PhyloTree.from_newick("((A,B),(C,D));"),
            ],
            threshold=0.5,
        )
        assert first.to_newick() == second.to_newick()


# =============================================================================
# plotting: the groups and the data must line up
# =============================================================================
class TestPlotting:
    def test_a_non_finite_point_does_not_break_the_grouping(self):
        from visualization.generic_plots import GenericPlotter

        figure = GenericPlotter().plot_xy(x=[1.0, 2.0, float("nan")], y=[1.0, 2.0, 3.0], groups=["a", "b", "c"])
        assert figure is not None

    def test_every_missing_group_label_is_one_group(self):
        from visualization.generic_plots import group_codes

        codes, names = group_codes(["a", "b", None, None], 4)
        assert len(set(codes.tolist())) == 3
        assert "missing" in [str(n) for n in names]

    def test_a_text_cell_in_a_matrix_plot_is_a_project_error(self):
        from utils.exceptions import DataValidationError
        from visualization.generic_plots import GenericPlotter

        with pytest.raises(DataValidationError):
            GenericPlotter().plot_matrix([[1.0, 2.0], ["T. rex", 3.0], [4.0, 5.0]])

    def test_the_plotting_package_leaves_no_figure_with_pyplot(self):
        import matplotlib.pyplot as plt

        from visualization.diversity_plot import DiversityPlotter

        before = len(plt.get_fignums())
        try:
            DiversityPlotter().plot_rarefaction([np.array([10.0, 8.0, 6.0])], labels=["S1"])
        except Exception:
            pass
        assert len(plt.get_fignums()) == before


# =============================================================================
# the process and scheduler layer
# =============================================================================
class TestProcessLayer:
    def test_a_shut_down_scheduler_does_not_hang(self):
        from hpc.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(n_workers=1)
        task_id = scheduler.add_task(lambda: 1)
        scheduler.shutdown()
        with pytest.raises(RuntimeError):
            scheduler.wait_all(timeout=1.0)
        with pytest.raises(RuntimeError):
            scheduler.get_result(task_id, timeout=1.0)

    def test_the_distance_metric_is_honoured(self):
        from hpc.process_pool import ProcessPool

        matrix = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        with ProcessPool(n_workers=1) as pool:
            euclidean = pool.compute_parallel_distance(matrix, metric="euclidean")
            bray_curtis = pool.compute_parallel_distance(matrix, metric="bray_curtis")
        assert euclidean[0, 1] == pytest.approx(5.0)
        assert bray_curtis[0, 1] == pytest.approx(1.0)
        assert not np.allclose(euclidean, bray_curtis)

    def test_the_task_object_can_be_handed_back_to_get_result(self):
        from hpc.process_pool import ProcessPool

        with ProcessPool(n_workers=1) as pool:
            task = pool.submit_task("t1", _add, 1, 2)
            assert pool.get_result(task, timeout=30) == 3


# =============================================================================
# the report builder
# =============================================================================
class TestReportBuilder:
    def test_the_title_and_abstract_are_escaped(self):
        from reporting.report_builder import ReportBuilder

        builder = ReportBuilder()
        builder.set_title("Body_mass & 95% CI")
        builder.add_author("Smith & Jones")
        builder.set_abstract("We measured N=5 & 95% CI")
        tex = "\n".join(builder._generate_titlepage() + builder._generate_abstract())
        # A bare "&" or "_" makes LaTeX refuse to compile the report.
        assert "Body\\_mass" in tex
        assert "Smith \\& Jones" in tex
        assert "N=5 \\& 95\\% CI" in tex


# =============================================================================
# the preset store
# =============================================================================
class TestPresetStore:
    def test_a_broken_file_does_not_hide_the_preset_you_want(self, tmp_path):
        from presets.manager import PresetManager

        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "good.json").write_text(
            '{"format": "paleoast_preset_v1", "name": "Good", "analysis": "pca", "params": {}}',
            encoding="utf-8",
        )
        manager = PresetManager(str(tmp_path))
        assert manager.load_by_name("Good") is not None


# =============================================================================
# the parsers
# =============================================================================
class TestParsers:
    def test_a_gap_inside_a_state_string_stays_a_gap(self):
        from parsers.nexus_lexer import NexusLexer, NexusTokenType

        tokens = NexusLexer().tokenize("Hsapiens 01-0")
        kinds = [t.type for t in tokens]
        assert NexusTokenType.MISSING in kinds
        assert not any(t.type is NexusTokenType.INTEGER and t.value == "-0" for t in tokens), [t.value for t in tokens]

    def test_an_empty_points_line_is_refused(self, tmp_path):
        from parsers.tps_parser import TPSParseError, TPSParser

        path = tmp_path / "curve.tps"
        path.write_text(
            "ID=spec1\nSCALE=1\nLM=2\nPOINTS=0 0\n1 0\nCO=1\nPOINTS=\n",
            encoding="utf-8",
        )
        with pytest.raises(TPSParseError):
            TPSParser().parse(str(path))


# =============================================================================
# the view layer
# =============================================================================
class TestViewLayer:
    def test_the_log_transform_refuses_a_value_at_or_below_minus_one(self, qapp):
        from utils.exceptions import DataValidationError
        from views.ui_spreadsheet import ScientificSpreadsheet

        sheet = ScientificSpreadsheet()
        sheet._data = np.array([[-1.0, 2.0], [1.0, 4.0]])
        with pytest.raises(DataValidationError):
            sheet._apply_column_transform(0, "log")

    def test_apply_preferences_accepts_a_partial_state(self, qapp, tmp_path):
        """A partial state must not raise, and must not touch the user's settings.

        The whole QSettings scope is snapshotted and put back, so running
        the suite does not quietly change the palette, language or theme
        the user chose.
        """
        from PyQt6.QtCore import QSettings

        from views.ui_main_window import MainWindow

        settings = QSettings("PaleoAST", "PaleoAST")
        snapshot = {key: settings.value(key) for key in settings.allKeys()}
        try:
            MainWindow._apply_preferences(None, {"r_palette": "viridis"})
        finally:
            for key in settings.allKeys():
                if key not in snapshot:
                    settings.remove(key)
            for key, value in snapshot.items():
                settings.setValue(key, value)
            settings.sync()

    def test_an_unknown_sample_name_does_not_analyse_row_zero(self):
        from models.data_matrix import DataMatrix
        from views.ui_main_window import MainWindow

        matrix = DataMatrix(np.array([[1.0], [2.0]]), row_labels=["Site_A", "Site_B"], col_labels=["v"])
        assert MainWindow._resolve_sample_index("Site_B", matrix) == 1
        assert MainWindow._resolve_sample_index("Site_Z", matrix) is None
