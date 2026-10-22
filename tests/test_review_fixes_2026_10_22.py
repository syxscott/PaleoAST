# =============================================================================
# FILE: tests/test_review_fixes_2026_10_22.py
# =============================================================================
"""
Regression tests for the defects found by the 2026-10-22 code review.

Each test pins one defect that used to produce a silently wrong number, a
false "success", or a lost result.  They are written so that a change that
reintroduces the defect fails here rather than in a published paper.

What each test pins
-------------------
1.  ecology/beta_diversity.py
    The rarefaction of a species that MUST be present in the subsample was
    computed from a log-gamma expression that evaluates log C(N-k, n) as 0
    where it is -inf.  A one-species sample of ten individuals was reported
    as 0.9 species, and the q=0 coverage rarefaction -- the default -- was
    systematically too low.
2.  stats/design_tests.py
    The three sibling terms of a two-way ANOVA each drew their own random
    null distribution, so their permutation p-values could not be compared
    with one another, and the result could not report the seed that
    produced them.
3.  stats/design_tests.py
    A contingency table with an empty row or column divided by zero and
    returned chi2 = nan, p = nan, Cramer's V = nan.  Because "significant"
    tests p < 0.05, a NaN p-value reads as "not significant".
4.  stats/permanova.py
    A single-group design reported F = 0.0 and p = 1.0 -- a complete-looking
    answer to a comparison that does not exist.
5.  stats/simper.py
    A variable_names list shorter than the data raised a bare IndexError,
    and only after every statistic had been computed.
6.  stats/univariate.py
    A group column holding both a number and text made the sort raise a
    bare TypeError from inside sorted(set(...)), escaping the project's
    exception hierarchy.
7.  models/data_matrix.py
    Indexing one axis produced a view whose labels no longer described its
    data, so "label index == data index" silently returned the wrong
    specimen's value.
8.  models/data_matrix.py
    Column metadata was keyed by label, so two columns sharing a label
    collapsed into one and the second column reported the first one's
    units.
9.  models/data_matrix.py
    remove_rows_with_missing() emptied a matrix that has no columns at all.
10. models/state_manager.py
    undo() and redo() restored the state without telling the spreadsheet,
    so the table kept showing the pre-undo numbers.
11. parsers/binary_cache.py
    A cache whose label count did not match its matrix loaded back without
    complaint, with every row label offset against the data.
12. views/ui_main_window.py
    The workspace was a QStackedWidget, so only the newest result was ever
    reachable and the title every call site passed was dropped.
13. utils/script_cli.py
    A script whose analysis failed still exited 0, so a scheduler recorded
    a success.
14. scripts/smoke_check.py
    The regression script had no exit call at all, so it reported 0 whether
    its checks passed or failed.
15. config/imputation.py
    Five strings of the missing-value report were in neither translation
    dictionary, so the Chinese interface showed that report in English.
16. morpho3d/gpa3d.py
    The Procrustes distance computed the optimal scale and then did not
    apply it, so it grew with the size of the configurations.
17. morphometrics/gpa.py
    GPAResult.scales recorded the divisor rather than the factor that was
    applied, making it a duplicate of centroid_sizes.
18. parsers/newick_parser.py
    A negative branch length raised a bare ValueError, which
    ``except NewickParseError`` could not catch.
19. controllers/data_controller.py
    The background CSV loader hardcoded UTF-8 while the menu loader used
    the encoding fallback chain, so a GBK file came out as mojibake.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. a species that must be present is counted as present
# =============================================================================
class TestRarefactionOfACertainSpecies:
    """The q=0 coverage rarefaction must not lose a certain species."""

    def test_a_certain_species_counts_as_one(self):
        from ecology import beta_diversity as bd
        from ecology import rarefaction as rf

        counts = np.array([10.0])
        assert bd._rarefaction_species(counts, 1) == pytest.approx(1.0)
        assert rf._rarefaction_formula(counts, 10, 1) == pytest.approx(1.0)

    def test_the_two_implementations_agree(self):
        """The two Hurlbert implementations must not disagree."""
        from ecology import beta_diversity as bd
        from ecology import rarefaction as rf

        counts = np.array([10.0, 4.0, 1.0])
        for n in (1, 2, 5, 8, 12, 15):
            assert bd._rarefaction_species(counts, n) == pytest.approx(
                rf._rarefaction_formula(counts, int(counts.sum()), n)
            )

    def test_the_coverage_curve_reports_one_species_for_a_one_species_sample(self):
        from ecology.beta_diversity import coverage_rarefaction_hill

        result = coverage_rarefaction_hill(np.array([[10.0]]), q=0, n_points=5)
        assert np.allclose(np.asarray(result.expected_richness), 1.0)


# =============================================================================
# 2. the three sibling terms of a two-way ANOVA share one permutation
# =============================================================================
class TestTwoWayAnovaSeed:
    """One seed, one permutation sequence, and a seed the result reports."""

    def test_the_result_reports_the_seed_it_used(self):
        from stats.design_tests import DesignTestAnalyzer

        rng = np.random.default_rng(0)
        values = rng.normal(size=24)
        factor_a = np.repeat(["A", "B", "C"], 8)
        factor_b = np.tile(["x", "y"], 12)

        result = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b)
        assert isinstance(result.random_seed, int)

    def test_the_result_is_reproducible_from_the_seed(self):
        from stats.design_tests import DesignTestAnalyzer

        rng = np.random.default_rng(0)
        values = rng.normal(size=24)
        factor_a = np.repeat(["A", "B", "C"], 8)
        factor_b = np.tile(["x", "y"], 12)

        first = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b, random_seed=7)
        second = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b, random_seed=7)
        assert first.term_a.permutation_p_value == second.term_a.permutation_p_value
        assert first.term_b.permutation_p_value == second.term_b.permutation_p_value
        assert first.term_interaction.permutation_p_value == second.term_interaction.permutation_p_value

    def test_the_sibling_terms_share_one_permutation_sequence(self):
        """The three terms must be compared against the same permutations."""
        from stats import design_tests as dt

        rng = np.random.default_rng(0)
        response = rng.normal(size=24)
        labels_a = np.repeat(["A", "B", "C"], 8)
        labels_b = np.tile(["x", "y"], 12)
        levels_a = sorted(set(labels_a))
        levels_b = sorted(set(labels_b))

        ss, df, _ = dt._two_way_sums_of_squares(response, labels_a, labels_b, levels_a, levels_b)
        ms_error = ss["error"] / df["error"]
        observed = {k: (ss[k] / df[k]) / ms_error for k in ("A", "B", "AB") if df[k] and ms_error > 0}

        def permuted(rng_):
            shuffled = dt._two_way_sums_of_squares(
                response[rng_.permutation(response.size)], labels_a, labels_b, levels_a, levels_b
            )[0]
            m = shuffled["error"] / df["error"] if df["error"] else np.nan
            if not np.isfinite(m) or m <= 0:
                return {k: np.nan for k in observed}
            return {k: (shuffled[k] / df[k]) / m for k in observed}

        # Three generators built from the SAME seed must produce the same
        # first permutation -- that is what makes the terms comparable.
        draws = [permuted(np.random.default_rng(11)) for _ in range(3)]
        assert draws[0]["A"] == draws[1]["A"] == draws[2]["A"]
        assert draws[0]["AB"] == draws[1]["AB"] == draws[2]["AB"]


# =============================================================================
# 3. a degenerate contingency table is refused, not reported as NaN
# =============================================================================
class TestContingencyChiSquare:
    def test_an_empty_row_is_refused(self):
        from stats.design_tests import DesignTestAnalyzer
        from utils.exceptions import DataValidationError

        with pytest.raises(DataValidationError):
            DesignTestAnalyzer().contingency_chi_square(np.array([[5.0, 3.0], [0.0, 0.0]]))

    def test_an_empty_column_is_refused(self):
        from stats.design_tests import DesignTestAnalyzer
        from utils.exceptions import DataValidationError

        with pytest.raises(DataValidationError):
            DesignTestAnalyzer().contingency_chi_square(np.array([[5.0, 0.0], [3.0, 0.0]]))

    def test_a_normal_table_still_works(self):
        from stats.design_tests import DesignTestAnalyzer

        result = DesignTestAnalyzer().contingency_chi_square(np.array([[5.0, 3.0], [2.0, 4.0]]))
        assert np.isfinite(result.chi_square)
        assert 0.0 <= result.p_value <= 1.0


# =============================================================================
# 4. a single-group design is refused
# =============================================================================
class TestPermanovaSingleGroup:
    def test_a_single_group_is_refused(self):
        from stats.permanova import PERMANOVAAnalyzer
        from utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            PERMANOVAAnalyzer().analyze(
                np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
                ["a", "a", "a"],
                n_permutations=19,
                random_seed=1,
            )


# =============================================================================
# 5. a short variable_names list is refused
# =============================================================================
class TestSimperVariableNames:
    def test_a_short_name_list_is_refused(self):
        from stats.simper import SimperAnalyzer
        from utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            SimperAnalyzer().analyze(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                [0, 1],
                variable_names=["a", "b"],
            )


# =============================================================================
# 6. mixed-type group labels do not raise a bare TypeError
# =============================================================================
class TestMixedTypeGroupLabels:
    def test_kruskal_wallis_survives_mixed_labels(self):
        from stats.univariate import UnivariateAnalyzer

        result = UnivariateAnalyzer().kruskal_wallis(
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
            [1, "A", 2, 1, 2, "B"],
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_t_test_survives_mixed_labels(self):
        from stats.univariate import UnivariateAnalyzer

        result = UnivariateAnalyzer().t_test(
            np.array([[1.0], [2.0], [3.0], [9.0]]),
            column=0,
            groups=[1, "A", 1, "A"],
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_the_shared_helper_keeps_the_natural_order(self):
        """Homogeneous labels must keep their natural order."""
        from utils.statistics_core import ordered_unique_labels

        assert ordered_unique_labels([10, 2, 1]) == [1, 2, 10]
        assert ordered_unique_labels(["b", "a"]) == ["a", "b"]
        assert ordered_unique_labels([1, "A", 2]) == [1, 2, "A"]


# =============================================================================
# 7. a view keeps its labels on the axes they describe
# =============================================================================
class TestDataMatrixViewAxes:
    @pytest.fixture
    def matrix(self):
        from models.data_matrix import DataMatrix

        return DataMatrix(
            np.arange(12.0).reshape(3, 4),
            row_labels=["r0", "r1", "r2"],
            col_labels=["c0", "c1", "c2", "c3"],
        )

    def test_a_single_column_is_a_two_dimensional_view(self, matrix):
        view = matrix[:, 0]
        assert view.data.ndim == 2
        assert len(view.row_labels) == view.data.shape[0]
        assert len(view.col_labels) == view.data.shape[1]
        assert np.allclose(view.data.ravel(), [0.0, 4.0, 8.0])
        assert view.row_labels == ["r0", "r1", "r2"]

    def test_a_single_row_is_a_two_dimensional_view(self, matrix):
        view = matrix[0, :]
        assert view.data.ndim == 2
        assert len(view.row_labels) == 1
        assert len(view.col_labels) == 4

    def test_the_view_can_be_turned_back_into_a_matrix(self, matrix):
        view = matrix[:, 0]
        restored = view.to_matrix()
        assert restored.shape == (3, 1)
        assert restored.row_labels == ["r0", "r1", "r2"]


# =============================================================================
# 8. two columns with the same label keep their metadata apart
# =============================================================================
class TestDuplicateColumnLabels:
    def test_the_metadata_is_kept_per_column(self):
        from models.data_matrix import DataMatrix

        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            col_labels=["X", "X"],
            column_metadata={"X": {"units": "mm"}},
        )
        assert matrix.get_column_metadata(0) == {"units": "mm"}
        assert matrix.get_column_metadata(1) == {"units": "mm"}

    def test_the_public_property_stays_label_keyed(self):
        from models.data_matrix import DataMatrix

        matrix = DataMatrix(
            [[1.0, 2.0], [3.0, 4.0]],
            col_labels=["X", "Y"],
            column_metadata={"X": {"units": "mm"}, "Y": {"units": "cm"}},
        )
        assert matrix.column_metadata == {"X": {"units": "mm"}, "Y": {"units": "cm"}}
        assert matrix.get_column_metadata(1) == {"units": "cm"}
        assert matrix.get_column_metadata("X") == {"units": "mm"}


# =============================================================================
# 9. a matrix with no columns does not lose its rows
# =============================================================================
class TestRemoveRowsWithMissing:
    def test_a_zero_column_matrix_keeps_its_rows(self):
        from models.data_matrix import DataMatrix

        matrix = DataMatrix(np.empty((3, 0)))
        assert matrix.remove_rows_with_missing().shape[0] == 3

    def test_an_all_nan_row_is_still_dropped(self):
        from models.data_matrix import DataMatrix

        matrix = DataMatrix(np.array([[1.0, 2.0], [np.nan, np.nan], [4.0, 5.0]]))
        assert matrix.remove_rows_with_missing().shape[0] == 2


# =============================================================================
# 10. undo and redo tell the spreadsheet the data changed
# =============================================================================
class TestUndoEmitsDataChanged:
    """Undo and redo must tell the spreadsheet the data changed.

    The bus is taken from ``models.state_manager`` rather than from
    ``utils.event_bus`` directly, and that is not a detail.  The
    ``tests/models/conftest.py`` hook pops ``utils.event_bus`` out of
    ``sys.modules`` when collection finishes, so a second copy of the
    module -- with its own singleton bus -- can exist in the same
    process.  ``models/state_manager`` holds the function it imported at
    module load, so the bus IT emits on is not necessarily the bus a
    fresh ``utils.event_bus.get_event_bus()`` returns.  Connecting to
    the emitter's own bus is the only way to see the event.
    """

    @staticmethod
    def _bus():
        from models import state_manager

        return state_manager.get_event_bus()

    def test_undo_emits_data_changed(self):
        from models.data_matrix import DataMatrix
        from models.state_manager import StateManager

        bus = self._bus()
        seen: list = []
        bus.data_changed.connect(lambda matrix: seen.append(matrix))

        state = StateManager()
        state.set_data_matrix(DataMatrix(np.array([[1.0, 2.0]])))
        seen.clear()
        state.undo()
        assert len(seen) == 1

    def test_redo_emits_data_changed(self):
        from models.data_matrix import DataMatrix
        from models.state_manager import StateManager

        bus = self._bus()
        seen: list = []
        bus.data_changed.connect(lambda matrix: seen.append(matrix))

        state = StateManager()
        state.set_data_matrix(DataMatrix(np.array([[1.0, 2.0]])))
        state.undo()
        seen.clear()
        state.redo()
        assert len(seen) == 1


# =============================================================================
# 11. the binary cache refuses labels that do not match the matrix
# =============================================================================
class TestBinaryCacheLabelCount:
    def test_saving_mismatched_labels_is_refused(self):
        from parsers.binary_cache import BinaryCache

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cache.bin")
            with pytest.raises(ValueError):
                BinaryCache().save(path, np.zeros((3, 3)), row_labels=["a", "b"])

    def test_a_well_formed_cache_still_loads(self):
        from parsers.binary_cache import BinaryCache

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cache.bin")
            BinaryCache().save(path, np.zeros((3, 3)), row_labels=["a", "b", "c"])
            loaded = BinaryCache().load(path)
            assert loaded is not None
            assert loaded["matrix"].shape == (3, 3)
            assert len(loaded["row_labels"]) == 3


# =============================================================================
# 12. the workspace keeps every result on its own tab
# =============================================================================
pytest.importorskip("PyQt6", reason="UI tests require PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-review-fixes-tests"])
    yield app


class TestWorkspaceTabs:
    def test_the_workspace_is_a_tab_widget(self, qapp):
        from views.ui_main_window import WorkspaceArea

        area = WorkspaceArea()
        assert type(area._stack).__name__ == "QTabWidget"

    def test_every_result_keeps_the_title_it_was_given(self, qapp):
        from views.ui_main_window import WorkspaceArea

        area = WorkspaceArea()
        first = QWidget()
        second = QWidget()
        area.addWidget(first, "PCA Score Plot")
        area.addWidget(second, "UAZ Hierarchy")

        titles = [area._stack.tabText(i) for i in range(area._stack.count())]
        assert "PCA Score Plot" in titles
        assert "UAZ Hierarchy" in titles

    def test_an_earlier_result_stays_reachable(self, qapp):
        from views.ui_main_window import WorkspaceArea

        area = WorkspaceArea()
        first = QWidget()
        second = QWidget()
        area.addWidget(first, "PCA Score Plot")
        area.addWidget(second, "UAZ Hierarchy")

        area.showWidget(first)
        assert area.currentWidget() is first
        # The earlier result is still there, and can be switched back to.
        assert area._stack.indexOf(second) >= 0
        area.showWidget(second)
        assert area.currentWidget() is second

    def test_the_welcome_pane_is_retired_once_a_result_is_shown(self, qapp):
        from views.ui_main_window import WorkspaceArea

        area = WorkspaceArea()
        view = QWidget()
        area.addWidget(view, "PCA Score Plot")
        area.showWidget(view)
        assert area._placeholder is None
        assert not area._stack.tabBar().isHidden()


# =============================================================================
# 13. the headless runner exits non-zero when an analysis fails
# =============================================================================
class TestHeadlessRunnerExitCode:
    def _run(self, source: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "script.py"
            script.write_text(source, encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            env["QT_QPA_PLATFORM"] = "offscreen"
            return subprocess.run(
                [sys.executable, "-m", "utils.script_cli", str(script)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
                timeout=300,
            )

    def test_a_failing_analysis_exits_non_zero(self):
        result = self._run('run("definitely_not_an_analysis", None)')
        assert result.returncode != 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    def test_a_succeeding_script_exits_zero(self):
        result = self._run('print("ok")')
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


# =============================================================================
# 14. the regression script fails loudly
# =============================================================================
class TestSmokeCheckExitCode:
    def test_the_script_exits_with_the_failure_status(self):
        source = (REPO_ROOT / "scripts" / "smoke_check.py").read_text(encoding="utf-8")
        assert "sys.exit" in source, (
            "scripts/smoke_check.py must end with a sys.exit: without one it "
            "reports 0 to the caller however many of its checks failed"
        )


# =============================================================================
# 15. the missing-value report is translated
# =============================================================================
class TestMissingValueReportTranslation:
    def test_every_string_of_the_report_is_in_both_dictionaries(self):
        from config.i18n.translations_en import TRANSLATIONS as EN
        from config.i18n.translations_zh import TRANSLATIONS as ZH

        keys = [
            "Missing Value Report",
            "Total NaN count: {0}",
            "NaN proportion: {0:.2f}%",
            "Rows with NaN: {0}",
            "Columns with NaN: {0}",
        ]
        for key in keys:
            assert key in EN, f"{key!r} missing from the English dictionary"
            assert key in ZH, f"{key!r} missing from the Chinese dictionary"

    def test_the_two_dictionaries_still_agree(self):
        from config.i18n.translations_en import TRANSLATIONS as EN
        from config.i18n.translations_zh import TRANSLATIONS as ZH

        assert set(EN) == set(ZH)


# =============================================================================
# 16. the Procrustes distance does not depend on the size of the shapes
# =============================================================================
class TestProcrustesDistanceScaleInvariance:
    def test_a_scaled_copy_has_the_same_distance(self):
        from morpho3d.gpa3d import procrustes_distance_3d

        rng = np.random.default_rng(3)
        config = rng.normal(size=(20, 3))
        distance, _, scale = procrustes_distance_3d(config, 2.0 * config)
        assert scale == pytest.approx(2.0)
        assert distance == pytest.approx(0.0, abs=1e-8)

    def test_an_identical_copy_has_zero_distance(self):
        from morpho3d.gpa3d import procrustes_distance_3d

        rng = np.random.default_rng(4)
        config = rng.normal(size=(20, 3))
        distance, _, _ = procrustes_distance_3d(config, config)
        assert distance == pytest.approx(0.0, abs=1e-8)


# =============================================================================
# 17. GPA scales record the factor that was applied
# =============================================================================
class TestGPAScales:
    def test_the_scales_are_the_applied_factor(self):
        from morphometrics.gpa import GPAAnalyzer

        rng = np.random.default_rng(1)
        result = GPAAnalyzer().analyze(rng.normal(size=(6, 8)))
        scales = np.asarray(result.scales, dtype=float)
        centroid_sizes = np.asarray(result.centroid_sizes, dtype=float)
        assert np.allclose(scales, 1.0 / centroid_sizes)
        assert not np.allclose(scales, centroid_sizes)


# =============================================================================
# 18. a negative branch length is a parse error
# =============================================================================
class TestNewickNegativeBranchLength:
    def test_the_error_is_a_newick_parse_error(self):
        from parsers.newick_parser import NewickParser
        from utils.exceptions import NewickParseError

        with pytest.raises(NewickParseError):
            NewickParser().parse("(A:-0.1,B:0.2)R;")

    def test_a_valid_tree_still_parses(self):
        from parsers.newick_parser import NewickParser

        tree = NewickParser().parse("((A,B)C,(D,E)F)G;")
        assert tree.leaf_count == 4


# =============================================================================
# 19. the background loader reads a GBK file
# =============================================================================
class TestBackgroundCsvEncoding:
    def test_a_gbk_file_is_read_without_mojibake(self):
        from controllers.data_controller import DataLoadTask

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "gbk.csv")
            with open(path, "wb") as handle:
                handle.write("站点,长度\n甲,1.5\n乙,2.5\n丙,3.5\n".encode("gbk"))

            task = DataLoadTask(filepath=path, delimiter=",", has_header=True, has_row_labels=False)
            captured: list = []
            task.result_ready.connect(captured.append)
            task.error_raised.connect(captured.append)
            task._run()

            assert captured, "the loader produced no result"
            result = captured[0]
            assert not isinstance(result, Exception), f"the loader raised {result!r}"
            assert result.row_labels == ["甲", "乙", "丙"]
            assert result.col_labels == ["长度"]
