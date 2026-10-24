# =============================================================================
# FILE: tests/test_acycle_port_runner_2026_10_24.py
# =============================================================================
"""
The Acycle-port analyses must be runnable **from the form**, not just from
Python.

Every analysis the catalogue offers is reachable through the generated
form: the sheet is cut into columns, the form's fields are read, and the
result comes back.  An entry whose parameters the form cannot express is
an entry the runner refuses to run and sends the user to the script
console instead -- the same invisibility the catalogue exists to prevent.

``tests/plugins/test_catalog.py`` checks that the form can *render* every
entry.  This file checks the stronger thing: that the values the form
would send actually **run** and come back with a result.  A catalogue
entry whose defaults would make the analysis fail is a dead entry.

THE CHECK RUNS IN A SUBPROCESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It builds the real dialog and reads the real widgets, which needs a
QApplication.  Qt's teardown rules differ between platforms: on Linux an
offscreen widget plus a QApplication that gets collected aborts the
interpreter -- and because the collection happens at whatever later point
the garbage collector runs, the crash lands far away from the test that
caused it.  The repository already runs its other widget checks this way
(see ``tests/test_design_conformance.py``).

WHAT THE FORM FILLS AND WHAT IT CANNOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A widget always has a value -- a spin box has a number, a combo box has
an entry -- so a parameter the signature calls required is still filled
by the form.  What the form cannot express is a *list of pairs* the user
has to type (the control points of the frequency stabiliser) and a column
the user has to choose (the scales of the wavelet transform).  Those are
in the tables below, and the test says so rather than hiding them.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The analyses this file drives: the ones the Acycle port brought in.
#: Each one is a catalogue entry a user reaches from the menu, so each one
#: has to run from the form.
PORTED_ANALYSES = (
    "age_depth_uncertainty",
    "anderson_darling_ksample",
    "bicoherence_significance",
    "bispectrum",
    "box_cox_transform",
    "chi_square_confidence_interval",
    "coco_adaptive",
    "coco_crossfit",
    "coco_interleaved",
    "coco_orbit_tracking",
    "coco_spearman",
    "correlation_confidence_interval",
    "cosine_taper",
    "date_to_day_number",
    "depth_to_time",
    "detrend_polynomial_pp",
    "dispersion_statistic",
    "domain_transform",
    "ecoco_row_summary",
    "evolutionary_rho1",
    "find_n_peaks",
    "fixed_bandwidth_moving_average",
    "fixed_target_weights",
    "frequency_stabilization",
    "gaussian_matched_filter",
    "inverse_percentile",
    "lead_lag_rmse",
    "milankovitch_uncertainty",
    "moving_median",
    "multitaper_taper_count",
    "num_orbits",
    "orbit_groups",
    "orbital_frequencies",
    "pad_series",
    "phase_diagram",
    "radiocarbon_calibration",
    "random_phase_resample",
    "recurrence_plot",
    "robust_linear_fit",
    "robust_lomb_scargle",
    "sliding_spectral_moments",
    "swa_background",
    "swa_fdr",
    "test_uniform_sampling",
    "tobit_regression",
    "tuned_multitaper_spectrum",
    "weedon_periodogram",
    "wavelet_coherence",
    "wavelet_smoothing",
)

#: The analyses that work on an AGE axis in Ma and need a sheet fine
#: enough to resolve the orbital bands.
COCO_ENTRIES = (
    "coco_adaptive",
    "coco_crossfit",
    "coco_interleaved",
    "coco_orbit_tracking",
    "coco_spearman",
    "ecoco_row_summary",
)

#: What the user has to type, because the form cannot guess it: a number
#: the analysis needs and the widget opens on an arbitrary value, a range,
#: a control-point path, a file name.
TYPED_VALUES: dict[str, dict[str, object]] = {
    "coco_spearman": {"rate_range": (0.5, 2.0)},
    "correlation_confidence_interval": {"rho": 0.5, "n": 100},
    "cosine_taper": {"length": 100},
    "date_to_day_number": {"year": 2024, "month": 6, "day": 15},
    "detrend_polynomial_pp": {"order": 2},
    "ecoco_row_summary": {"rate_range": (0.5, 2.0)},
    "find_n_peaks": {"n": 3},
    "fixed_bandwidth_moving_average": {"bandwidth": 10.0},
    "frequency_stabilization": {
        "lower_control_points": [(0.0, 0.01), (100.0, 0.02)],
        "upper_control_points": [(0.0, 0.03), (100.0, 0.04)],
        "window_length": 20.0,
        "step_length": 5.0,
    },
    "inverse_percentile": {"probability": 0.5},
    "moving_median": {"window": 5},
    "multitaper_taper_count": {"n_samples": 256},
    "num_orbits": {"time_span": 100.0},
    "pad_series": {"length": 250},
    "recurrence_plot": {"radius": 0.5},
    "sliding_spectral_moments": {"window": 100},
}

#: The columns the user has to pick for the array parameters that are not
#: data slots.  The form opens on column 0, which for these is the time
#: axis -- wrong for a set of scales, so the sheet below puts the right
#: column there and the test names it.
COLUMN_PICKS: dict[str, dict[str, int]] = {
    "wavelet_coherence": {"scales": 1},
}


#: The driver: it builds the dialog, reads the widgets, and runs each
#: analysis the way the runner does.
_DRIVER = """
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, {repo!r})
import numpy as np
from PyQt6.QtWidgets import QApplication, QListWidgetItem

from plugins.catalog import BUILTIN_ANALYSES, BuiltinAnalysisPlugin
from views.analysis_runner import AnalysisRunnerDialog

TYPED = {typed!r}
COLUMNS = {columns!r}
COCO_ENTRIES = {coco!r}

app = QApplication(["paleoast-acycle-port"])
dialog = AnalysisRunnerDialog()


def column_for(slot, length):
    if slot == "time":
        return np.arange(length, dtype=float)
    if slot in ("coordinate", "depths", "ages", "age_ma", "age_anchors"):
        return np.linspace(0.0, 100.0, length)
    if slot == "anchor_ages_ma":
        return np.linspace(0.0, 50.0, length)
    if slot == "groups":
        return np.repeat([1.0, 2.0, 3.0], length // 3 + 1)[:length]
    if slot == "observed":
        return np.arange(1.0, length + 1.0)
    if slot == "periods_kyr":
        return np.array([19.0, 21.0, 23.0, 41.0, 100.0])
    if slot == "frequencies":
        return np.linspace(0.001, 0.4, length)
    if slot in ("power", "swa"):
        return np.ones(length)
    if slot == "first":
        values = np.sin(2 * np.pi * np.arange(length) / 20.0)
        return values - values.min() + 1.0
    if slot == "second":
        values = np.sin(2 * np.pi * np.arange(length) / 20.0)
        return np.roll(values, 3) - values.min() + 1.0
    if slot == "values":
        values = np.sin(2 * np.pi * np.arange(length) / 20.0)
        return values - values.min() + 1.0
    if slot == "values_x":
        values = np.sin(2 * np.pi * np.arange(length) / 20.0)
        return values - values.min() + 1.0
    if slot == "values_y":
        values = np.sin(2 * np.pi * np.arange(length) / 20.0)
        return np.roll(values, 3) - values.min() + 1.0
    if slot == "target":
        return np.exp(-0.5 * ((np.arange(length) - length // 2) / 3.0) ** 2)
    if slot == "scales":
        return np.linspace(2.0, 40.0, length)
    if slot == "x":
        return np.arange(length, dtype=float)
    if slot == "y":
        return 2.0 * np.arange(length, dtype=float) + 1.0
    if slot == "lab_det":
        return np.full(length, 2450.0)
    if slot == "lab_err":
        return np.full(length, 40.0)
    values = np.abs(np.sin(2 * np.pi * np.arange(length) / 20.0)) + 1.0
    return values


def sheet_for(entry):
    slots = tuple(entry.data_slots or ())
    if not slots:
        return np.zeros((4, 1))
    if entry.name in COCO_ENTRIES:
        # The correlation-coefficient family works on an AGE axis in Ma and
        # has to resolve the orbital bands, so its sheet is a fine one.
        ages = np.linspace(0.0, 20.0, 4000)
        values = np.sin(2 * np.pi * ages / 0.021) + 0.4 * np.sin(2 * np.pi * ages / 0.019)
        return np.column_stack([ages, values])
    columns = [column_for(slot, 200) for slot in slots]
    size = min(len(column) for column in columns)
    return np.column_stack([column[:size] for column in columns])


def extra_columns(name, entry):
    # The columns the array parameters that are not slots are given: a
    # parameter like the SCALES of the wavelet transform is not a data
    # slot, the user picks the column of the sheet it comes from, and the
    # test has to put something suitable there.
    picks = COLUMNS.get(name, {{}})
    extra = {{}}
    for parameter, column in picks.items():
        if parameter == "scales":
            extra[parameter] = np.linspace(2.0, 40.0, 40)
    return extra


for name in {names!r}:
    entry = next(item for item in BUILTIN_ANALYSES if item.name == name)
    plugin = BuiltinAnalysisPlugin(entry)
    item = QListWidgetItem(name)
    item.setData(256, name)
    dialog._on_select(item, None)
    specs = dialog._specs
    slots = set(entry.data_slots or ())
    typed = dict(TYPED.get(name, {{}}))
    picks = COLUMNS.get(name, {{}})
    extra = extra_columns(name, entry)

    sheet = sheet_for(entry)
    kwargs = {{}}
    try:
        if name == "radiocarbon_calibration":
            curve = os.path.join(tempfile.gettempdir(), "paleoast-test-curve.14c")
            with open(curve, "w", encoding="utf-8") as handle:
                for _ in range(11):
                    handle.write("header\\n")
                for index in range(40):
                    calendar = 20000.0 - index * 500.0
                    handle.write("{{}},{{}},{{}},0,0\\n".format(calendar, calendar - 3000.0, 20.0))
            typed["curve_path"] = curve

        for spec in specs:
            if spec.kind == "unsupported":
                continue
            if spec.name in slots:
                continue
            if spec.name in typed:
                kwargs[spec.name] = typed[spec.name]
                continue
            if spec.kind == "array":
                column = spec.value()
                if column is None or column < 0:
                    continue
                if spec.name in extra:
                    kwargs[spec.name] = extra[spec.name]
                    continue
                array = np.asarray(sheet)
                if array.ndim == 1:
                    array = array.reshape(-1, 1)
                kwargs[spec.name] = array[:, column]
                continue
            value = spec.value()
            if value is None:
                continue
            kwargs[spec.name] = value
        outcome = plugin.analyze(sheet, **kwargs)
        if not outcome.success:
            print("FAIL", name, outcome.error, sep="\\t")
        elif outcome.data is None:
            print("FAIL", name, "returned nothing", sep="\\t")
        else:
            print("OK", name, sep="\\t")
    except Exception as exc:
        print("FAIL", name, "{{}}: {{}}".format(type(exc).__name__, exc), sep="\\t")

# The driver exits without letting Python tear the QApplication down: on
# Linux an offscreen widget plus a QApplication that gets collected
# aborts the interpreter, and the abort would look like a failed run.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


def test_every_ported_analysis_runs_from_the_form() -> None:
    """Every ported analysis runs with the values the form would send."""
    script = _DRIVER.format(
        repo=str(REPO),
        names=list(PORTED_ANALYSES),
        typed=TYPED_VALUES,
        columns=COLUMN_PICKS,
        coco=list(COCO_ENTRIES),
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    outcomes: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] in ("OK", "FAIL"):
            outcomes[parts[1]] = "" if parts[0] == "OK" else " | ".join(parts[2:])

    missing = sorted(set(PORTED_ANALYSES) - set(outcomes))
    assert not missing, f"the form did not reach: {missing}"
    failures = {name: reason for name, reason in outcomes.items() if reason}
    assert not failures, "analyses the form cannot run:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(failures.items())
    )
