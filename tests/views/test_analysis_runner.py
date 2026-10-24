# =============================================================================
# FILE: tests/views/test_analysis_runner.py
# =============================================================================
"""
Tests for the generated analysis-parameter form.

The form's job is to turn an analyzer's signature into widgets, so most
of these tests are about the translation layer rather than about any
one analysis. Three annotation shapes have to survive it, and getting
any of them wrong is silent -- a ``list[int]`` rendered as free text
still produces a parameter, just the wrong one:

  * a generic alias, whose ``__name__`` proxies back to the origin and
    would flatten ``list[int]`` to ``list``;
  * a plain type object, whose ``str()`` is ``"<class 'str'>"``;
  * a plain string, from ``from __future__ import annotations``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from plugins.catalog import BUILTIN_ANALYSES, register_builtin_analyses
from plugins.registry import get_plugin_registry
from views.analysis_runner import ParameterSpec, _annotation_text, _find_choices, build_specs


@pytest.fixture(scope="module")
def registry():
    """A filled registry."""
    register_builtin_analyses()
    return get_plugin_registry()


def _specs(name: str, registry) -> dict[str, ParameterSpec]:
    entry = next(e for e in BUILTIN_ANALYSES if e.name == name)
    return {s.name: s for s in build_specs(entry, registry.get(name))}


# ---------------------------------------------------------------------------
# Annotation translation
# ---------------------------------------------------------------------------


def test_generic_alias_keeps_its_subscript() -> None:
    """``list[int].__name__`` is 'list'; reading it loses the parameter."""
    assert _annotation_text(list[int]) == "list[int]"
    assert _annotation_text(list[str]) == "list[str]"


def test_plain_class_reduces_to_its_name() -> None:
    """str() of a type object is "<class 'str'>", which matches nothing."""
    assert _annotation_text(str) == "str"
    assert _annotation_text(int) == "int"
    assert _annotation_text(bool) == "bool"


def test_string_annotation_is_passed_through() -> None:
    """The PEP 563 form needs only unquoting."""
    assert _annotation_text("str") == "str"
    assert _annotation_text("int | None") == "int | None"
    assert _annotation_text("typing.Sequence[float]") == "Sequence[float]"


# ---------------------------------------------------------------------------
# Coverage across the whole catalog
# ---------------------------------------------------------------------------


def test_every_analysis_produces_a_form(registry) -> None:
    """No signature fails to introspect."""
    for entry in BUILTIN_ANALYSES:
        specs = build_specs(entry, registry.get(entry.name))
        assert specs is not None
        assert all(isinstance(s, ParameterSpec) for s in specs)


def test_data_is_never_a_rendered_parameter(registry) -> None:
    """`data` is supplied by the caller, so it is not a form field.

    A field called "data" would be a second, contradictory source of
    the data, and the runner would pass both.
    """
    for entry in BUILTIN_ANALYSES:
        names = {s.name for s in build_specs(entry, registry.get(entry.name))}
        assert "data" not in names, f"{entry.name} exposes a data field"


def test_most_parameters_are_renderable(registry) -> None:
    """The form earns its place by covering the bulk of the API.

    The remainder are callbacks, PhyloTree and dict arguments, which no
    text field can produce. The bar is set from what the catalog
    actually contains, not from a wish.
    """
    total = 0
    unsupported = 0
    for entry in BUILTIN_ANALYSES:
        for spec in build_specs(entry, registry.get(entry.name)):
            total += 1
            if spec.kind == "unsupported":
                unsupported += 1
    assert total > 250, f"expected a large parameter surface, got {total}"
    assert unsupported / total < 0.08, f"{unsupported} of {total} parameters are unrenderable; the coverage bar is 8%"


def test_unsupported_parameters_say_why(registry) -> None:
    """A blank reason leaves the user with nothing to act on."""
    for entry in BUILTIN_ANALYSES:
        for spec in build_specs(entry, registry.get(entry.name)):
            if spec.kind == "unsupported":
                assert spec.detail.strip(), f"{entry.name}.{spec.name} is unsupported with no reason"


def test_integer_parameters_get_sensible_bounds(registry) -> None:
    """A spin box left at 0-99 would turn n_permutations into 99.

    Asserted on the bounds mapping the runner consults, not on a widget
    that only exists once a dialog has built it.
    """
    import inspect

    from views.analysis_runner import _INT_BOUNDS, _widen_to_default

    for parameter in ("n_permutations", "n_clusters", "n_components", "k"):
        low, high, _step = _INT_BOUNDS[parameter]
        assert low >= 0 and high > low

    # A default outside the range was silently changed on the way to the
    # analyzer: the plot menu's "not set" sentinel of -1 became 0, which
    # turned "do not group" into "group by the first column". The range
    # the spin box is given has to contain the default.
    for entry in BUILTIN_ANALYSES:
        plugin = registry.get(entry.name)
        try:
            target = plugin._resolve()
            signature = inspect.signature(getattr(target, entry.method))
        except Exception:
            continue
        for name, param in signature.parameters.items():
            default = param.default
            if default is None or isinstance(default, bool):
                continue
            if not isinstance(default, int):
                continue
            low, high, _step = _INT_BOUNDS.get(name, (0, 100000, 1))
            low, high = _widen_to_default(low, high, default)
            assert low <= default <= high, f"{entry.name}.{name} defaults to {default}, outside the range {low}..{high}"


def test_the_data_slot_is_never_a_form_field(registry) -> None:
    """The runner hands the sheet over; the form must not also send it.

    Sending it twice is a TypeError -- "got multiple values for
    argument" -- which is what happened to orbital_forcing, whose first
    parameter is a period and not the data at all.

    An analysis that declares ``data_slots`` is the exception, and the
    exception is a column picker rather than the sheet: the sheet is cut
    into columns and each slot takes one of them, so the form asks
    WHICH column, never for the data itself.
    """
    import inspect

    from plugins.base import data_slot_name

    for entry in BUILTIN_ANALYSES:
        plugin = registry.get(entry.name)
        try:
            target = plugin._resolve()
            signature = inspect.signature(getattr(target, entry.method))
        except Exception:
            continue
        slot = data_slot_name(signature)
        if slot is None:
            continue
        specs = {s.name: s for s in build_specs(entry, plugin)}
        if slot not in specs:
            continue
        if entry.data_slots is not None and slot in entry.data_slots:
            # A declared slot is offered as a column picker. It must not
            # be a field that sends the data.
            assert specs[slot].kind == "array", (
                f"{entry.name} offers a field for its data parameter {slot!r} that is not a column picker"
            )
            continue
        assert slot not in specs, f"{entry.name} offers a field for its data parameter {slot!r}"


# ---------------------------------------------------------------------------
# Choice discovery
# ---------------------------------------------------------------------------


def test_choices_are_found_in_the_analyzers_own_module(registry) -> None:
    """correlation, scheme, form come from declared constants."""
    assert _specs("mantel", registry)["correlation"].choices == (
        "pearson",
        "spearman",
    )
    assert _specs("mantel", registry)["scheme"].choices == ("dd", "jm", "vm")
    assert _specs("intraclass_correlation", registry)["form"].choices == (
        "2,1",
        "3,1",
    )


def test_choices_come_from_the_analyzer_not_from_an_unrelated_module(registry) -> None:
    """A parameter named "method" means something different everywhere.

    The runner used to fall back on a hard-coded list of "shared
    vocabulary" modules, so stats/cca.py was offered the clustering
    module's linkage methods. Picking one of them ran RDA while the
    result reported "rda" with no error at all. The vocabulary is now
    found in the analyzer's own module, or in a module it imports --
    never in an unrelated one.
    """
    import stats.clustering as clustering

    found = _find_choices(clustering, "method")
    assert "ward" in found

    import types

    empty = types.ModuleType("empty_module_for_test")
    assert _find_choices(empty, "metric") == (), "an unrelated module's constant must not be offered"


def test_every_offered_choice_is_one_the_analyzer_accepts(registry) -> None:
    """The drop-down must not offer a value the analyzer rejects.

    This is the regression test for the silent substitution: with an
    unrelated vocabulary in the drop-down, picking "ward" for CCA ran
    RDA. The offered values are now the analyzer's own.
    """
    expected = {
        "cca": ("method", ("cca", "rda")),
        "pca": ("method", ("covariance", "correlation")),
        "nmds": ("method", ("raw_stress", "stress_1")),
        "paired_rank_test": ("method", ("wilcoxon", "sign")),
        "extinction": ("method", ("marshall", "strauss_sadler")),
        "simper": ("metric", ("bray_curtis", "euclidean")),
        "beta_diversity": ("metric", ("jaccard", "sorensen")),
    }
    for entry_name, (parameter, values) in expected.items():
        specs = _specs(entry_name, registry)
        assert parameter in specs, f"{entry_name} has no {parameter} field"
        assert specs[parameter].kind == "choice", f"{entry_name}.{parameter} is not a drop-down"
        assert tuple(specs[parameter].choices) == values, f"{entry_name}.{parameter} offers the wrong values"


def test_an_unknown_parameter_gets_no_choices(registry) -> None:
    """No token means no guessing."""
    import types

    empty = types.ModuleType("empty_module_for_test_2")
    assert _find_choices(empty, "not_a_known_parameter") == ()


# ---------------------------------------------------------------------------
# Type mapping on real signatures
# ---------------------------------------------------------------------------


def test_mantel_parameters_map_to_the_right_widgets(registry) -> None:
    """The signature a user will actually meet."""
    specs = _specs("mantel", registry)
    assert specs["n_permutations"].kind == "int"
    assert specs["compute_mc_z"].kind == "bool"
    assert specs["labels"].kind == "list_str"
    assert specs["correlation"].kind == "choice"
    # data_a is the data slot: the runner hands it the spreadsheet, so it
    # is not a form field. data_b is a second array, and an array is a
    # column of the sheet.
    assert "data_a" not in specs
    assert specs["data_b"].kind == "array"


def test_bare_list_of_labels_maps_to_a_text_field(registry) -> None:
    """anosim writes groups as list[Any] and permanova as bare list."""
    assert _specs("anosim", registry)["groups"].kind == "list_str"
    assert _specs("permanova", registry)["groups"].kind == "list_str"
    assert _specs("paired_rank_test", registry)["groups"].kind == "list_str"


def test_tree_parameters_are_reported_not_guessed(registry) -> None:
    """A PhyloTree cannot come from a text field."""
    specs = _specs("phylogenetic_signal", registry)
    assert specs["tree"].kind == "unsupported"
    assert "PhyloTree" in specs["tree"].detail


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """A QApplication held for the module.

    Yielded rather than created as a bare expression: a QApplication
    with no Python reference is collected the moment the statement ends,
    which tears down Qt while the dialogs still exist -- a C-level
    abort that only shows up when this file runs alongside another view
    test, so it never reproduced in isolation.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    """A constructed runner."""
    from controllers.statistics_controller import StatisticsController
    from views.analysis_runner import AnalysisRunnerDialog

    rng = np.random.default_rng(0)
    window = AnalysisRunnerDialog(
        controller=StatisticsController(),
        data_provider=lambda: rng.normal(size=(22, 4)),
    )
    yield window
    window.deleteLater()


def test_dialog_lists_every_analysis(dialog) -> None:
    """One row per catalogued analysis, plus a header per category."""
    from PyQt6.QtCore import Qt

    selectable = sum(1 for i in range(dialog._list.count()) if dialog._list.item(i).data(Qt.ItemDataRole.UserRole))
    assert selectable == len(BUILTIN_ANALYSES)


def test_selecting_an_analysis_builds_its_form(dialog) -> None:
    """The form follows the selection."""
    from PyQt6.QtCore import Qt

    for i in range(dialog._list.count()):
        if dialog._list.item(i).data(Qt.ItemDataRole.UserRole) == "mantel":
            dialog._list.setCurrentRow(i)
            break
    assert dialog._entry.name == "mantel"
    # Every spec got a widget. Counting QFormLayout rows instead would be
    # measuring Qt's internal bookkeeping, which does not line up with the
    # number of specs: ten addRow calls report fifteen rows.
    assert dialog._specs
    assert all(spec.widget is not None for spec in dialog._specs)
    assert {s.kind for s in dialog._specs} & {"int", "bool", "choice"}


def test_running_an_analysis_produces_output(dialog) -> None:
    """End to end, on a real analysis and real data."""
    from PyQt6.QtCore import Qt

    for i in range(dialog._list.count()):
        if dialog._list.item(i).data(Qt.ItemDataRole.UserRole) == "kmeans":
            dialog._list.setCurrentRow(i)
            break
    dialog._on_run()
    text = dialog._output.toPlainText()
    assert "K-Means" in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------
# The behaviours that were wrong
# ---------------------------------------------------------------------------


def _select(dialog, name: str) -> None:
    """Put the dialog on a named analysis."""
    from PyQt6.QtCore import Qt

    for i in range(dialog._list.count()):
        if dialog._list.item(i).data(Qt.ItemDataRole.UserRole) == name:
            dialog._list.setCurrentRow(i)
            return
    raise AssertionError(f"{name} is not in the list")


def test_a_plot_menu_figure_reaches_the_callback(qapp) -> None:
    """The figure must arrive at the canvas, not as an object repr.

    The plugin wraps whatever the analysis returned in an AnalysisResult,
    so the Figure is at .data. The runner tested the wrapper itself, the
    check was always False, and the plot menu never drew anything: the
    user got "AnalysisResult(name='xy_plot', data=<Figure ...>)" in a
    text pane, which is exactly what the figure branch exists to avoid.
    """
    from matplotlib.figure import Figure

    from views.analysis_runner import AnalysisRunnerDialog

    received = []
    window = AnalysisRunnerDialog(
        data_provider=lambda: np.arange(20, dtype=float).reshape(5, 4),
        on_figure=lambda figure, name: received.append((figure, name)),
    )
    try:
        _select(window, "xy_plot")
        window._on_run()
        assert len(received) == 1, "the figure never reached the callback"
        assert isinstance(received[0][0], Figure)
        assert received[0][1] == "xy_plot"
        assert "Figure drawn." in window._output.toPlainText()
    finally:
        window.deleteLater()


def test_a_number_keeps_the_analyzers_own_default(dialog) -> None:
    """A spin box clamped the default, and the analysis changed.

    The plot menu's "not set" sentinel of -1 was clamped to 0, so
    "groups: none" became "groups: the first column".
    """
    _select(dialog, "xy_plot")
    groups = next(s for s in dialog._specs if s.name == "groups")
    assert groups.default == -1
    assert groups.value() == -1


def test_a_number_with_no_default_is_left_to_the_analyzer(dialog) -> None:
    """None means "the analyzer decides", so the form must not send a number."""
    _select(dialog, "pca")
    components = next(s for s in dialog._specs if s.name == "n_components")
    assert components.default is None
    assert components.value() is None


def test_an_array_parameter_is_a_column_of_the_sheet(dialog) -> None:
    """An array used to be a text field that threw the typing away."""
    _select(dialog, "spectral")
    arrays = [s for s in dialog._specs if s.kind == "array"]
    assert arrays, "spectral has no array parameter"
    assert arrays[0].value() == 0, "the column picker does not start at the first column"


def test_an_analysis_the_form_cannot_drive_says_so(dialog) -> None:
    """A TypeError about a missing argument is not an explanation."""
    _select(dialog, "phylo_anova")
    dialog._on_run()
    text = dialog._output.toPlainText()
    assert "script console" in text
    assert "Traceback" not in text


def test_the_project_errors_are_shown_as_messages(dialog) -> None:
    """A PaleoASTError says what went wrong, in the project's voice."""

    _select(dialog, "kmeans")
    # A single row cannot be clustered, and the analyzer says so.
    dialog._current_data = lambda: np.zeros((1, 3))
    dialog._on_run()
    text = dialog._output.toPlainText()
    assert "Traceback" not in text
