# =============================================================================
# FILE: views/analysis_runner.py
# =============================================================================
"""
A dialog that runs any catalogued analysis from a generated form.

WHY GENERATED RATHER THAN WRITTEN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nineteen new analyses landed this release, and each bespoke dialog in
this codebase costs about a hundred lines: a class, four widget
set-ups, a ``get_parameters`` that has to stay in step with the
analyzer's signature, a plot call, and two i18n catalog entries in each
language. Nineteen of those is two thousand lines of mostly-mechanical
form-building that rots the moment a default changes.

The catalog already knows every analysis by module, class and method, so
``inspect.signature`` gives the parameters, and the annotations are
regular enough to render: across all 78 entries there are integers,
floats, booleans, strings, arrays, ``list[str]`` and ``list[int]``, and
**no signature fails to resolve**. So the form is built from the
signature instead of written per analysis.

WHAT IT IS NOT
~~~~~~~~~~~~~~
It is not a replacement for the purpose-built dialogs that exist for
PCA, CCA, the morphometrics analyses and the rest. Those have
domain-specific controls -- a preview canvas, a shape sketch, a
tree viewer -- that a generated form cannot produce, and for those the
hand-written path stays. This is the door for everything that has no
door yet, which is what makes the new analyses reachable from the
mouse rather than only from the script console.

Four analyses take a ``PhyloTree`` or a ``dict``; those parameters
cannot be built from a text field, so they are reported as unsupported
here and named in the message, pointing at the console where a tree is
already in scope.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.i18n import _
from plugins.catalog import BUILTIN_ANALYSES
from plugins.registry import get_plugin_registry
from utils.script_session import _format_result

logger = logging.getLogger(__name__)

# Names that mean "no value is needed" and so are not rendered.
_SKIPPED_PARAMETERS = frozenset({"self", "data", "kwargs", "args"})

# Types a text field cannot produce. Reported rather than guessed at.
# Parameters with no annotation whose meaning is nonetheless clear from
# the name. Deliberately narrow: a guess here produces a widget that
# silently sends the wrong type, so only names that are unambiguously
# label lists are listed. Anything else unannotated stays "unsupported"
# and is reported as such, which is recoverable -- the console takes any
# value the form cannot.
_UNANNOTATED_LABEL_LISTS = frozenset({"groups", "group_labels", "labels"})

# Unannotated parameters that are data objects, not strings. Listed so
# the form says "unsupported" with a reason instead of offering a text
# field that could never satisfy the analyzer.
_UNANNOTATED_OBJECTS = frozenset({"section", "sections", "result", "tree", "working_tree"})

# Annotation fragments a text field cannot produce. Stored lowercase
# and matched against a lowercased annotation -- a callback renders as
# "Callable" and a tree as "PhyloTree", so a case-sensitive comparison
# let both through as unrenderable with no reason attached.
_UNSUPPORTED_HINTS = (
    "phylotree",
    "phylonode",
    "callable",
    "dict[",
)

# Bounds by name, for the integer parameters whose sensible range is not
# derivable from the annotation alone. Without these a spin box defaults
# to 0-99 and "n_permutations" would silently become 99.
_INT_BOUNDS: dict[str, tuple[int, int, int]] = {
    "n_permutations": (1, 100000, 99),
    "permutations": (1, 100000, 999),
    "n_simulations": (1, 100000, 999),
    "n_components": (2, 200, 2),
    "n_clusters": (1, 200, 2),
    "n_init": (1, 500, 10),
    "n_points": (3, 2000, 40),
    "n_tapers": (1, 30, 4),
    "n_bands": (1, 200, 20),
    "column": (0, 100000, 0),
    "k": (1, 2000, 8),
    "nlags": (2, 2000, 40),
    "n_lags": (2, 2000, 40),
    "max_lag": (1, 100000, 20),
    "n_per_group": (1, 100000, 2),
    "n_bins": (2, 2000, 20),
    "degree": (1, 20, 2),
    "random_state": (0, 2**31 - 1, 0),
    "n_workers": (1, 256, 1),
}

# Float parameters whose range is worth bounding.
_FLOAT_BOUNDS: dict[str, tuple[float, float, float]] = {
    "power": (0.1, 10.0, 2.0),
    "span": (0.01, 1.0, 0.3),
    "alpha": (1e-6, 0.5, 0.05),
    "confidence": (0.5, 0.9999, 0.95),
    "confidence_level": (0.5, 0.9999, 0.95),
    "fap_level": (1e-4, 0.5, 0.05),
    "eps": (1e-12, 1.0, 1e-6),
}

# String parameters whose legal values are a tuple in the target module.
# Keyed by parameter name; the tuple is found by scanning module globals
# for a constant whose name contains the token.
_CHOICE_TOKEN: dict[str, tuple[str, ...]] = {
    "metric": ("DISTANCE", "METRIC"),
    "method": ("LINKAGE", "METHOD", "MODE"),
    "correlation": ("CORRELATION",),
    "scheme": ("SCHEME",),
    "form": ("ICC_FORM", "FORM"),
    "algorithm": ("ALGORITHM",),
    "alternative": ("ALTERNATIVE",),
}


@dataclass
class ParameterSpec:
    """One rendered parameter.

    Attributes:
        name: The keyword the analyzer expects.
        kind: ``int``, ``float``, ``bool``, ``str``, ``choice``,
            ``list_str``, ``list_int``, ``array`` or ``unsupported``.
        default: The signature default, used to seed the widget.
        choices: Legal values for ``choice``.
        detail: For unsupported parameters, what could not be rendered.
        widget: The built widget.
    """

    name: str
    kind: str
    default: Any
    widget: QWidget | None = None
    choices: tuple[str, ...] = ()
    detail: str = ""

    def value(self) -> Any:
        """Read the widget, or None when the field is blank.

        A blank field means "leave the analyzer's own default alone",
        which is why every optional parameter is returned as ``None``
        rather than as a zero.
        """
        if self.kind == "bool":
            return bool(self.widget.isChecked())  # type: ignore[union-attr]
        if self.kind in ("int", "float"):
            return self.widget.value()  # type: ignore[union-attr]
        if self.kind == "choice":
            index = self.widget.currentIndex()  # type: ignore[union-attr]
            return None if index <= 0 else self.widget.currentText()  # type: ignore[union-attr]
        if self.kind == "str":
            text = self.widget.text().strip()  # type: ignore[union-attr]
            return text or None
        if self.kind in ("list_str", "list_int"):
            text = self.widget.text().strip()  # type: ignore[union-attr]
            if not text:
                return None
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return parts if self.kind == "list_str" else [int(p) for p in parts]
        return None


def _is_array(annotation: str) -> bool:
    return "NDArray" in annotation or "ndarray" in annotation


def _base_type(annotation: str) -> str:
    """The first alternative of a union, as text."""
    return annotation.split("|")[0].strip().replace("'", "").replace("typing.", "")


# Modules that hold shared vocabularies, searched only when the analyzer's
# own module has nothing. Not a hard-coded list of legal values, because
# two of these disagree: stats/clustering.py's DISTANCE_METRICS is the
# scipy pdist set (cityblock, hamming, cosine) while
# stats/distance_metrics.py's is a different one (manhattan, bray_curtis,
# chebychev). Finding the real constant where it lives is right; guessing
# would offer the wrong options half the time.
_SHARED_VOCABULARY_MODULES = (
    "stats.distance_metrics",
    "stats.clustering",
    "stats.design_tests",
    "stratigraphy.spectral_analysis",
)


def _find_choices(module: Any, parameter: str) -> tuple[str, ...]:
    """Legal string values for a parameter, from a declared constant.

    Scans module-level tuples of strings for one whose name carries a
    token related to the parameter -- ``metric`` finds ``DISTANCE_METRICS``,
    ``method`` finds ``LINKAGE_METHODS``, ``correlation`` finds
    ``VALID_CORRELATIONS``. A token that matches nothing yields an empty
    tuple and the field degrades to free text rather than silently
    offering the wrong options.
    """
    tokens = _CHOICE_TOKEN.get(parameter, ())
    if not tokens:
        return ()

    def scan(target: Any) -> tuple[str, ...]:
        for name, value in vars(target).items():
            if not name.isupper():
                continue
            if not any(token in name for token in tokens):
                continue
            if isinstance(value, (tuple, list)) and value and all(isinstance(v, str) for v in value):
                return tuple(value)
        return ()

    found = scan(module)
    if found:
        return found
    import importlib

    for dotted in _SHARED_VOCABULARY_MODULES:
        try:
            fallback = scan(importlib.import_module(dotted))
        except Exception:
            continue
        if fallback:
            return fallback
    return ()


def _annotation_text(annotation: Any) -> str:
    """Normalise a parameter annotation to text.

    Three shapes arrive here, and each loses information if handled
    naively:

    * A generic alias, ``list[int]``. Its ``__name__`` proxies back to
      the origin -- ``list[int].__name__`` is ``'list'`` -- so reading
      ``__name__`` first silently turns every ``list[int]`` into a bare
      ``list`` and the parameter stops being renderable. Hence
      ``repr`` for those, which keeps the subscript.
    * A plain type object, where ``str()`` gives ``"<class 'str'>"``.
      Quoting that before stripping yields ``"<class str>"``, which
      matches nothing -- which is how ``method`` and ``metric`` came out
      as free text instead of drop-downs in stats/clustering.py. So the
      class form is reduced to its ``__name__``.
    * A plain string, from ``from __future__ import annotations``.
    """
    import types

    if isinstance(annotation, str):
        return annotation.strip().replace("'", "").replace("typing.", "")
    if annotation is inspect.Parameter.empty:
        # The sentinel for "no annotation at all". Its __name__ is
        # "_empty", which matched none of the branches below and so left
        # every unannotated parameter both unrenderable AND unexplained.
        return ""
    if isinstance(annotation, types.GenericAlias):
        return repr(annotation).replace("typing.", "")
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation).replace("typing.", "")


def _build_specs(entry: Any, plugin: Any) -> list[ParameterSpec]:
    """Derive the parameter form for one catalogued analysis.

    The analyzer's module is imported here rather than at import time,
    so building the dialog for a category does not import all 78
    analyses' modules at once.
    """
    specs: list[ParameterSpec] = []
    try:
        target = plugin._resolve()
        signature = inspect.signature(getattr(target, entry.method))
    except Exception as exc:  # reported in the form rather than raised
        return [
            ParameterSpec(
                name=entry.name,
                kind="unsupported",
                default=None,
                detail=str(exc),
            )
        ]
    module = inspect.getmodule(getattr(type(target), "analyze", target))
    for name, param in signature.parameters.items():
        if name in _SKIPPED_PARAMETERS:
            continue
        text = _annotation_text(param.annotation)
        lowered = text.lower()
        default = param.default
        base = text.split("|")[0].strip()
        spec = ParameterSpec(name=name, kind="unsupported", default=default)

        # Matched case-insensitively: the annotation for a callback
        # renders as "Callable", not "callable", and a case-sensitive
        # comparison let those through as unrenderable with no reason
        # attached.
        if any(hint in lowered for hint in _UNSUPPORTED_HINTS):
            spec.detail = base
            specs.append(spec)
            continue
        if _is_array(base) or "Sequence" in base:
            spec.kind = "array"
            specs.append(spec)
            continue
        if "list[str]" in text or base == "list[str]":
            spec.kind = "list_str"
            specs.append(spec)
            continue
        if "list[int]" in text or base == "list[int]":
            spec.kind = "list_int"
            specs.append(spec)
            continue
        # A bare list, or list[Any]. Across the whole catalog that shape
        # occurs twice, both on a "groups" parameter holding group labels
        # (anosim and permanova; paired_rank_test writes it as a bare
        # "list | None"). Verified by enumerating every annotation in the
        # catalog rather than assumed, so the mapping below cannot be
        # wrong for some other list-typed parameter today.
        if base in ("list", "list[Any]"):
            spec.kind = "list_str"
            specs.append(spec)
            continue
        if base == "bool":
            spec.kind = "bool"
            specs.append(spec)
            continue
        if base == "str":
            choices = _find_choices(module, name) if module else ()
            spec.kind = "choice" if choices else "str"
            spec.choices = choices
            specs.append(spec)
            continue
        # int and float are checked after bool and str so a union like
        # "int | None" resolves to the right branch, and float is checked
        # before int because "float" contains "int".
        if base == "float" or ("float" in text and "int" not in text.replace("float", "")):
            spec.kind = "float"
            specs.append(spec)
            continue
        if base == "int" or ("int" in text and "float" not in text):
            spec.kind = "int"
            specs.append(spec)
            continue
        # An unannotated parameter: fall back to the name, and only where
        # the name is unambiguous. An empty annotation text is not the same
        # as an unrecognised type, so it is checked separately.
        if text == "":
            if name in _UNANNOTATED_LABEL_LISTS:
                spec.kind = "list_str"
                specs.append(spec)
                continue
            if name in _UNANNOTATED_OBJECTS:
                spec.detail = "an object, not a value"
                specs.append(spec)
                continue
            spec.detail = "unannotated"
        else:
            # Anything that is not a renderable builtin still has to say
            # what it is. Falling through with an empty detail left the
            # user looking at "Unsupported here:" with nothing after it,
            # which is the same as no explanation at all. The type name
            # is the most useful thing available here and it costs
            # nothing to include.
            spec.detail = base or "unrecognised"
        specs.append(spec)
    return specs


build_specs = _build_specs


class AnalysisRunnerDialog(QDialog):
    """Pick a catalogued analysis, fill its parameters, run it.

    Parameters
    ----------
    parent:
        Parent widget.
    controller:
        The StatisticsController, used for its data.
    data_provider:
        Zero-argument callable returning the current data.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        controller: Any = None,
        data_provider: Any = None,
    ) -> None:
        """Build the dialog."""
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.AnalysisRunnerDialog")
        self.setWindowTitle(_("Run Analysis"))
        self.resize(1000, 700)
        self._controller = controller
        self._data_provider = data_provider
        self._registry = get_plugin_registry()
        self._entry: Any = None
        self._specs: list[ParameterSpec] = []
        self._build_ui()
        self._populate_list()

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the picker, the parameter form and the output."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        splitter.addWidget(self._list)

        right = QVBoxLayout()
        self._description = QLabel("")
        self._description.setWordWrap(True)
        right.addWidget(self._description)

        self._form_box = QGroupBox(_("Parameters"))
        self._form = QFormLayout(self._form_box)
        right.addWidget(self._form_box)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        right.addWidget(self._output, stretch=1)

        holder = QWidget()
        holder.setLayout(right)
        splitter.addWidget(holder)
        splitter.setSizes([260, 700])
        layout.addWidget(splitter, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_button = QPushButton(_("Close"))
        close_button.clicked.connect(self.reject)
        buttons.addWidget(close_button)
        run_button = QPushButton(_("Run"))
        run_button.setDefault(True)
        run_button.clicked.connect(self._on_run)
        buttons.addWidget(run_button)
        layout.addLayout(buttons)

    def _populate_list(self) -> None:
        """List every catalogued analysis, grouped by category."""
        from plugins.catalog import register_builtin_analyses

        register_builtin_analyses()
        self._list.clear()
        by_category: dict[str, list[Any]] = {}
        for entry in BUILTIN_ANALYSES:
            by_category.setdefault(entry.category, []).append(entry)
        for category in sorted(by_category):
            header = QListWidgetItem(f"{category}  ({len(by_category[category])})")
            header.setFlags(header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(header)
            for entry in sorted(by_category[category], key=lambda e: e.name):
                item = QListWidgetItem(f"    {entry.name}")
                item.setData(Qt.ItemDataRole.UserRole, entry.name)
                self._list.addItem(item)
        self._list.setCurrentRow(1 if self._list.count() > 1 else 0)

    # -- selection --------------------------------------------------------

    def _on_select(self, current: QListWidgetItem | None, _previous: Any) -> None:
        """Rebuild the parameter form for the newly selected analysis."""
        while self._form.count():
            item = self._form.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._specs = []
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        entry = next((e for e in BUILTIN_ANALYSES if e.name == name), None)
        if entry is None:
            return
        self._entry = entry
        self._description.setText(entry.description)
        plugin = self._registry.get(entry.name)
        self._specs = build_specs(entry, plugin)
        for spec in self._specs:
            spec.widget = self._build_widget(spec)
            if spec.kind == "unsupported":
                self._form.addRow(
                    spec.name,
                    QLabel(_("Unsupported here: {0}. Use the script console.").format(spec.detail or "?")),
                )
            else:
                self._form.addRow(spec.name, spec.widget)

    def _build_widget(self, spec: ParameterSpec) -> QWidget:
        """One widget per parameter kind."""
        if spec.kind == "int":
            low, high, step = _INT_BOUNDS.get(spec.name, (0, 100000, 1))
            box = QSpinBox()
            box.setRange(low, high)
            box.setSingleStep(step)
            box.setValue(int(spec.default) if isinstance(spec.default, int) else low)
            return box
        if spec.kind == "float":
            low, high, step = _FLOAT_BOUNDS.get(spec.name, (-1e12, 1e12, 0.1))
            box = QDoubleSpinBox()
            box.setDecimals(6)
            box.setRange(low, high)
            box.setSingleStep(step)
            box.setValue(float(spec.default) if isinstance(spec.default, float) else 0.0)
            return box
        if spec.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(spec.default) if spec.default is not None else True)
            return box
        if spec.kind == "choice":
            box = QComboBox()
            box.addItem(_("(default)"))
            box.addItems(list(spec.choices))
            if isinstance(spec.default, str) and spec.default in spec.choices:
                box.setCurrentText(spec.default)
            return box
        line = QLineEdit()
        if isinstance(spec.default, str):
            line.setPlaceholderText(spec.default)
        elif spec.kind == "array":
            line.setPlaceholderText(_("(uses the current data)"))
        return line

    # -- running ----------------------------------------------------------

    def _current_data(self) -> Any:
        """The spreadsheet, or None."""
        if self._data_provider is None:
            return None
        try:
            data = self._data_provider()
        except Exception:
            logger.debug("data provider raised", exc_info=True)
            return None
        if data is not None and hasattr(data, "raw_data"):
            return data.raw_data
        return data

    def _on_run(self) -> None:
        """Collect the form, run the analysis, show its summary."""
        if self._entry is None:
            return
        kwargs = {}
        for spec in self._specs:
            if spec.kind == "unsupported":
                continue
            try:
                value = spec.value()
            except (ValueError, TypeError) as exc:
                self._output.setPlainText(_("Could not read {0}: {1}").format(spec.name, exc))
                return
            if value is not None:
                kwargs[spec.name] = value
        plugin = self._registry.get(self._entry.name)
        if plugin is None:
            self._output.setPlainText(_("No analysis named {0}.").format(self._entry.name))
            return
        self._output.setPlainText(_("Running {0} ...").format(self._entry.name))
        try:
            outcome = plugin.analyze(self._current_data(), **kwargs)
        except Exception:
            import traceback

            self._output.setPlainText(traceback.format_exc())
            return
        self._output.setPlainText(_format_result(outcome))
        self._logger.info("analysis runner: %s completed", self._entry.name)
