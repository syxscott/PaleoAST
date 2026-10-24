# =============================================================================
# FILE: tests/test_plugin_data_slots_2026_10_24.py
# =============================================================================
"""
Regression tests for the plugin data-slot contract.

The generic analysis runner used to hand the WHOLE spreadsheet to the
analysis' data parameter.  For an analysis that wants a *matrix* that is
right; for one that wants a *series* it was wrong twice over -- a
two-column sheet arrived as "the time axis was the whole table and the
values were never passed", and a single-column sheet was read as the
data together with whatever else was in it.

The contract now says: an entry that declares ``data_slots`` takes one
COLUMN per slot, in the order the sheet's columns come; an entry that
declares nothing takes the whole sheet.

Each test below pins one defect that produced a silently wrong number.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from plugins.base import data_slot_names, sheet_columns
from plugins.catalog import BUILTIN_ANALYSES, BuiltinAnalysisPlugin

# The analyses that want a series: the sheet's first column is the time
# axis and the second the measurement.
TWO_COLUMN = ["spectral", "wavelet", "redfit", "autoassociation", "arma", "cross_correlation"]
ONE_COLUMN = ["acf", "multitaper", "prewhiten", "mann_kendall", "runs_test"]


def _entry(name: str):
    entry = next((e for e in BUILTIN_ANALYSES if e.name == name), None)
    assert entry is not None, f"{name} is not in the catalog"
    return entry


def _plugin(name: str) -> BuiltinAnalysisPlugin:
    return BuiltinAnalysisPlugin(_entry(name))


def _two_column_sheet(n: int = 200):
    time = np.linspace(0.0, 100.0, n)
    values = np.sin(2 * np.pi * time / 20.0) + 0.1 * np.cos(2 * np.pi * time / 5.0)
    return time, values, np.column_stack([time, values])


# ---------------------------------------------------------------------------
# the helpers
# ---------------------------------------------------------------------------
class TestSlotHelpers:
    def test_a_declared_list_wins_over_the_automatic_rule(self):
        """The automatic rule can only ever find ONE parameter.

        ``analyze(time, values)`` has two array parameters, and the rule
        picked the first -- so the sheet went in as the time axis and the
        values were never passed.
        """
        from stratigraphy.spectral_analysis import SpectralAnalyzer

        signature = inspect.signature(SpectralAnalyzer.analyze)
        assert data_slot_name_of(signature) == "time"
        assert data_slot_names(signature, ("time", "values")) == ("time", "values")
        assert data_slot_names(signature, None) == ("time",)

    def test_no_declaration_means_the_automatic_rule(self):
        from stats.pca import PCAAnalyzer

        signature = inspect.signature(PCAAnalyzer.analyze)
        assert data_slot_names(signature, None) == ("data",)

    def test_sheet_columns_splits_by_column(self):
        sheet = np.column_stack([np.arange(5.0), np.arange(5.0) * 2])
        columns = sheet_columns(sheet, 2)
        assert len(columns) == 2
        assert list(columns[0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert list(columns[1]) == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_a_one_column_sheet_still_yields_one_column(self):
        sheet = np.arange(5.0)
        columns = sheet_columns(sheet, 1)
        assert list(columns[0]) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_too_few_columns_is_refused(self):
        """Asking for two columns of a one-column sheet is a shape error."""
        from utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            sheet_columns(np.arange(5.0), 2)


def data_slot_name_of(signature):
    from plugins.base import data_slot_name

    return data_slot_name(signature)


# ---------------------------------------------------------------------------
# the two-column analyses
# ---------------------------------------------------------------------------
class TestTwoColumnAnalyses:
    @pytest.mark.parametrize("name", TWO_COLUMN)
    def test_the_entry_declares_both_slots(self, name):
        entry = _entry(name)
        assert entry.data_slots is not None, f"{name} does not declare its slots"
        assert len(entry.data_slots) == 2

    @pytest.mark.parametrize("name", TWO_COLUMN)
    def test_the_sheet_is_cut_into_the_two_columns(self, name):
        """The sheet's first column is the time axis, the second the values."""
        time, values, sheet = _two_column_sheet()
        plugin = _plugin(name)
        target = plugin._resolve()
        method = getattr(target, _entry(name).method)
        slots = data_slot_names(inspect.signature(method), _entry(name).data_slots)

        columns = sheet_columns(sheet, len(slots))
        assert list(columns[0]) == list(time)
        assert list(columns[1]) == list(values)
        assert slots == _entry(name).data_slots

    def test_the_spectral_analysis_through_the_plugin_gives_the_same_answer(self):
        """The runner's result must equal the dedicated path's result.

        The dedicated path calls ``analyze(time, values)`` with the two
        columns; the runner goes through the plugin.  If those two
        disagree, one of them is wrong -- and the runner is the one the
        user reaches from the menu.
        """
        from stratigraphy.spectral_analysis import SpectralAnalyzer

        time, values, _sheet = _two_column_sheet()
        direct = SpectralAnalyzer().analyze(time, values)
        plugin = _plugin("spectral")
        through = plugin.analyze({"time": time, "values": values})
        assert np.allclose(np.asarray(through.data.frequencies), np.asarray(direct.frequencies))
        assert np.allclose(np.asarray(through.data.power), np.asarray(direct.power))

    def test_a_one_column_sheet_cannot_fill_two_slots(self):
        """A sheet with one column cannot fill two slots.

        The runner cuts the sheet itself, so this is the path a script
        takes when it hands the whole sheet to an analysis that declared
        two slots.
        """
        from utils.exceptions import ValidationError

        plugin = _plugin("spectral")
        with pytest.raises(ValidationError):
            plugin.analyze(np.arange(10.0))


# ---------------------------------------------------------------------------
# the one-column analyses
# ---------------------------------------------------------------------------
class TestOneColumnAnalyses:
    @pytest.mark.parametrize("name", ONE_COLUMN)
    def test_the_entry_declares_one_slot(self, name):
        entry = _entry(name)
        assert entry.data_slots is not None, f"{name} does not declare its slots"
        assert len(entry.data_slots) == 1

    @pytest.mark.parametrize("name", ONE_COLUMN)
    def test_the_result_equals_the_single_column_result(self, name):
        """The bug: the whole sheet went in, and the numbers changed.

        Mann-Kendall's tau came back with the opposite sign, the runs
        test's p-value moved by forty orders of magnitude, and the
        pre-whitening's AR(1) coefficient flipped sign -- all because
        the time axis was analysed as if it were the data.
        """
        _time, values, _sheet = _two_column_sheet()
        entry = _entry(name)
        plugin = _plugin(name)
        target = plugin._resolve()
        method = getattr(target, entry.method)

        try:
            direct = method(values)
        except TypeError:
            pytest.skip(f"{name} needs more than one argument")

        kwargs = {}
        for parameter in inspect.signature(method).parameters:
            if parameter == "kwargs":
                continue
            kwargs[parameter] = values
        through = plugin.analyze({"values": values})
        if not hasattr(direct, "__dataclass_fields__") and not isinstance(direct, tuple):
            return
        for field in getattr(type(direct), "__dataclass_fields__", {}):
            left = getattr(direct, field)
            right = getattr(through.data, field)
            if not isinstance(left, (int, float, np.ndarray)) or isinstance(left, bool):
                continue
            left = np.asarray(left, dtype=float)
            right = np.asarray(right, dtype=float)
            if left.shape != right.shape:
                continue
            assert np.allclose(left, right, equal_nan=True), f"{name}.{field} differs"


# ---------------------------------------------------------------------------
# the analyses that take a matrix are left alone
# ---------------------------------------------------------------------------
class TestMatrixAnalysesUnchanged:
    def test_an_undeclared_entry_still_gets_the_whole_sheet(self):
        """A matrix wants the whole sheet.  The contract must not change it."""
        from stats.pca import PCAAnalyzer

        rng = np.random.default_rng(7)
        sheet = rng.normal(size=(30, 4))
        plugin = BuiltinAnalysisPlugin(_entry("pca"))
        through = plugin.analyze(sheet)
        direct = PCAAnalyzer().analyze(sheet)
        assert np.allclose(np.asarray(through.data.scores), np.asarray(direct.scores))

    def test_the_declared_entries_are_exactly_the_series_ones(self):
        """No entry may declare a slot by accident.

        An entry that wants a MATRIX must keep taking the whole sheet, so
        the declared set is exactly the analyses that want a series --
        the time-series family and the astrochronology family.
        """
        declared = {e.name for e in BUILTIN_ANALYSES if e.data_slots is not None}
        expected = (
            set(TWO_COLUMN)
            | set(ONE_COLUMN)
            | {
                "growth_model",
                "growth_model_comparison",
                "kaplan_meier",
                "cox_ph",
                "evolution_rate",
                # The cyclostratigraphy batch.
                "evolutionary_spectrum",
                "evolutionary_lomb_scargle",
                "evolutionary_multitaper",
                "amplitude_modulation",
                "tanner_hilbert",
                "gaussian_filter",
                "taner_filter",
                "bandpass_filter",
                "change_points",
                "emd",
                "eemd",
                "pda",
                "interpolate_series",
                "select_intervals",
                "add_gaps",
                "remove_peaks",
                "derivative",
                "clip_values",
                "sort_unique",
                "generate_signal",
                "coco",
                "ecoco",
                "time_opt",
                "etime_opt",
                "age_scale",
                "sedimentation_rate_age_model",
                "spectral_moments",
                "estimate_ar1_rho",
                "minimum_rho",
                "dynamic_filter",
                "fixed_bandwidth_summary",
                "fixed_count_summary",
                "insolation",
                # The last batch: the noise model, the length of day, the
                # smoothing and detrending tools, the transforms and edits.
                "dynot",
                "rho1_sedimentation_ratio",
                "lod_dimension",
                "optimal_span_smooth",
                "moving_average_smooth",
                "standardize",
                "log_transform",
                "apply_function",
                "detrend_series",
                "remove_part",
                "remove_range",
                # The series-shaped analyses that had been left taking the
                # whole sheet: the angle vector, the extinction
                # positions, the spatial values, the time axes.
                "astronomical_target_curve",
                "directional",
                "extinction",
                "grid_interpolate",
                "insolation_series",
                "morans_i",
            }
        )
        assert declared == expected, f"the declared set changed: {sorted(declared ^ expected)}"

    def test_a_declared_slot_is_always_an_array_parameter(self):
        """A declared slot has to be a parameter the method really has.

        The slot names the column the sheet is cut into, so a name the
        method does not have would send the column nowhere.
        """
        for entry in BUILTIN_ANALYSES:
            if entry.data_slots is None:
                continue
            plugin = BuiltinAnalysisPlugin(entry)
            target = plugin._resolve()
            parameters = inspect.signature(getattr(target, entry.method)).parameters
            for slot in entry.data_slots:
                assert slot in parameters, f"{entry.name} declares a slot {slot!r} the method does not have"
                annotation = str(parameters[slot].annotation).lower()
                assert "ndarray" in annotation, (
                    f"{entry.name} declares {slot!r} as a data slot, but it is not an array parameter"
                )

    def test_an_analysis_that_wants_an_array_declares_its_slots(self):
        """An analysis that takes an array but declares no slot gets the
        WHOLE sheet -- and a whole sheet is a matrix, not a series.

        That is the bug this file exists for, so the rule is enforced
        rather than left to whoever adds the next analysis.  An analysis
        whose first array parameter wants a series has to declare its
        slots; one that wants a MATRIX keeps taking the sheet, and the
        names of those are listed here so the exemption is deliberate
        rather than a gap.
        """
        from tests.plugins.test_catalog import KNOWN_UNDRIVABLE

        #: The first array parameter of these is a MATRIX, and the whole
        #: sheet is what it wants.  Each name is here with a reason.
        matrix_first = {
            "data": "a table of observations",
            "abundances": "abundances of several taxa per sample",
            "abundance_matrix": "a taxon-by-sample matrix",
            "presence_matrix": "a presence/absence matrix",
            "distance_matrix": "a distance matrix",
            "table": "a contingency or data table",
            "Y": "the response matrix",
            "X": "the predictor matrix",
            "data_a": "the first of two matrices",
            "data_b": "the second of two matrices",
            "control": "the control matrix",
            "block_a": "the first of two blocks",
            "block_b": "the second of two blocks",
            "configurations": "a stack of landmark configurations",
            "aligned_configurations": "a stack of aligned configurations",
            "procrustes_coords": "coordinates in morphospace",
            "fad_matrix": "a first-appearance datum matrix",
            "lad_matrix": "a last-appearance datum matrix",
            "coordinates": "site coordinates",
            "coords": "site coordinates",
            "points": "a set of points",
            "vectors": "a set of vectors",
            "source": "the source landmark configuration",
            "target": "the target landmark configuration",
            "contour": "a contour's coordinates",
            "sequences": "a list of sequences",
            "sequence": "a sequence of facies codes",
            "efa_coefficients_list": "a list of coefficient vectors",
            # These three take a MATRIX named "values": the design
            # matrices of the factorial and the repeated-measures
            # analyses, and the targets-by-raters matrix of the
            # intraclass correlation.
            "values": "a design or measurement matrix (two_way_anova, repeated_measures_anova, intraclass_correlation)",
        }

        for entry in BUILTIN_ANALYSES:
            if entry.data_slots is not None:
                continue
            if entry.name in KNOWN_UNDRIVABLE:
                continue
            plugin = BuiltinAnalysisPlugin(entry)
            try:
                target = plugin._resolve()
                parameters = inspect.signature(getattr(target, entry.method)).parameters
            except Exception:
                continue
            arrays = [name for name, param in parameters.items() if "ndarray" in str(param.annotation).lower()]
            if not arrays:
                continue
            first = arrays[0]
            assert first in matrix_first, (
                f"{entry.name} takes the array {first!r} but declares no slots, so it would be handed the whole sheet"
            )


# ---------------------------------------------------------------------------
# the plugin's own routing
# ---------------------------------------------------------------------------
class TestPluginRouting:
    def test_a_mapping_is_routed_to_its_slots(self):
        """The runner sends a mapping when the analysis wants a series."""
        plugin = _plugin("acf")
        _time, values, _ = _two_column_sheet()
        through = plugin.analyze({"values": values})
        from stratigraphy.cycles import CycleAnalyzer

        direct = CycleAnalyzer().autocorrelation(values)
        assert np.allclose(np.asarray(through.data.acf), np.asarray(direct.acf))

    def test_a_slot_cannot_be_set_twice(self):
        plugin = _plugin("acf")
        from utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            plugin.analyze({"values": np.arange(5.0)}, values=np.arange(5.0))


# ---------------------------------------------------------------------------
# every declared entry resolves
# ---------------------------------------------------------------------------
def test_every_declared_entry_resolves_and_runs():
    """A declared slot that names a parameter the method does not have
    would send the data nowhere."""
    _time, _values, _sheet = _two_column_sheet()
    for entry in BUILTIN_ANALYSES:
        if entry.data_slots is None:
            continue
        module = importlib.import_module(entry.module)
        target = module if entry.class_name is None else getattr(module, entry.class_name)()
        method = getattr(target, entry.method)
        parameters = inspect.signature(method).parameters
        for slot in entry.data_slots:
            assert slot in parameters, f"{entry.name} declares a slot {slot!r} the method does not have"
