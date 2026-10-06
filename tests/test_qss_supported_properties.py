# =============================================================================
# FILE: tests/test_qss_supported_properties.py
# =============================================================================
"""
Fail the build when a stylesheet uses a property Qt does not support.

Qt's style sheets implement a documented subset of CSS. A property outside
that subset is not an error: Qt prints ``Unknown property <name>`` on stderr
and drops the declaration, so the rule silently stops applying. Two such
declarations shipped for a long time before anyone noticed:

  * ``titleBarCloseButtonVisible: true`` in views/diagnostic_console.py
    (twice) -- read as if it controlled the dock's close button; it never did.
  * ``transition: all 200ms ease-out`` in config/design_system.py -- read as
    if button hover/press states eased; they have always snapped.

Neither is visible by reading the CSS, and neither is caught by the test
suite, because nothing asserts on what Qt accepted.

This test closes that gap. It installs a Qt message handler, applies the
project's real stylesheets, and fails on any ``Unknown property``.

Why capture Qt's own warnings instead of comparing against a hand-written
allow-list of property names: such a list is a guess about Qt's grammar, it
goes stale against a Qt upgrade, and it cannot know which selectors a given
stylesheet actually reaches. The earlier attempt at one produced 27 hits of
which most were false positives (words scraped out of docstrings embedded in
f-strings). Qt already knows the answer; we just have to listen.
"""

import os
import re

import pytest

pytest.importorskip("PyQt6", reason="style sheets are a Qt concept")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import qInstallMessageHandler
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabBar,
    QTreeView,
    QWidget,
)

# One concrete factory per class named by the global stylesheet.
_FACTORIES = {
    "QCheckBox": lambda p: QCheckBox("c", p),
    "QComboBox": lambda p: QComboBox(p),
    "QDialog": lambda p: QDialog(p),
    "QFrame": lambda p: QFrame(p),
    "QGraphicsDropShadowEffect": lambda p: None,
    "QGroupBox": lambda p: QGroupBox("g", p),
    "QLabel": lambda p: QLabel("l", p),
    "QLineEdit": lambda p: QLineEdit(p),
    "QListWidget": lambda p: QListWidget(p),
    "QMainWindow": lambda p: QMainWindow(p),
    "QMenu": lambda p: QMenu(p),
    "QMenuBar": lambda p: QMenuBar(p),
    "QPushButton": lambda p: QPushButton("b", p),
    "QRadioButton": lambda p: QRadioButton("r", p),
    "QScrollBar": lambda p: None,
    "QSlider": lambda p: QSlider(p),
    "QSpinBox": lambda p: QSpinBox(p),
    "QStatusBar": lambda p: QStatusBar(p),
    "QTabBar": lambda p: QTabBar(p),
    "QTreeView": lambda p: QTreeView(p),
    "QWidget": lambda p: QWidget(p),
}

BAD_MARKER = "Unknown property"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _collect_unknown_properties(app, apply_stylesheets):
    """Apply each stylesheet and return the ``Unknown property`` texts Qt emitted.

    The handler is uninstalled and replaced by a no-op afterwards, because a
    leftover Python closure handed to Qt would be called from Qt's C++ side
    long after this test module is gone.
    """
    seen = []

    def handler(mode, context, message):
        seen.append(str(message))

    previous = qInstallMessageHandler(handler)
    try:
        apply_stylesheets(app)
        app.processEvents()
    finally:
        qInstallMessageHandler(previous)
    return [m for m in seen if BAD_MARKER in m]


def test_message_handler_detects_an_unsupported_property(qapp):
    """Positive control.

    Without this, the guard below would also pass if Qt stopped reporting
    entirely (a Qt upgrade, a platform that routes messages elsewhere), and
    a test that can never fail is worse than no test at all.
    """

    def apply(app):
        probe = QPushButton("probe")
        probe.setStyleSheet("QPushButton { color: red; transition: all 200ms ease; }")
        probe.ensurePolished()
        app.processEvents()

    messages = _collect_unknown_properties(qapp, apply)
    assert any("transition" in m for m in messages), (
        "the Qt message handler did not report a deliberately unsupported "
        "property, so this guard could not detect a real one either"
    )


def test_message_handler_accepts_a_supported_property(qapp):
    """Negative control: a valid stylesheet must produce no such message."""

    def apply(app):
        probe = QPushButton("probe")
        probe.setStyleSheet("QPushButton { color: red; background: #ffffff; border-radius: 4px; padding: 4px 8px; }")
        probe.ensurePolished()
        app.processEvents()

    assert _collect_unknown_properties(qapp, apply) == []


@pytest.mark.parametrize("dark", [False, True])
def test_project_stylesheets_use_only_supported_properties(qapp, dark):
    """The real global stylesheet must contain no property Qt will drop.

    A previous version of this test applied the sheet to a bare QWidget with
    one QPushButton, and MISSED ``box-shadow: ...`` on the QMenu rule: Qt only
    reports a rejected property when it actually parses that rule, so a
    selector that was never instantiated hid the problem. It now builds a host
    containing one real widget of every class named in the stylesheet, and
    fails if a class cannot be instantiated -- otherwise a future rule on an
    untested widget type would slip through again.
    """
    from config.design_system import get_stylesheet

    stylesheet = get_stylesheet(dark)
    selectors = sorted(set(re.findall(r"(?m)^\s*(Q[A-Za-z]+)", stylesheet)))
    assert selectors, "the global stylesheet has no selectors at all"

    def apply(app):
        host = QWidget()
        host.setStyleSheet(stylesheet)
        made, skipped = [], []
        for name in selectors:
            factory = _FACTORIES.get(name)
            if factory is None:
                skipped.append(name)
                continue
            try:
                factory(host)
                made.append(name)
            except Exception as exc:  # pragma: no cover - defensive
                pytest.fail(f"could not instantiate a {name} for the guard: {exc!r}")
        host.ensurePolished()
        for name in made:
            # A menu is a top-level popup; polish it so its rules are parsed.
            pass
        app.processEvents()
        assert made, f"no widget could be instantiated; selectors={selectors}"
        assert not skipped, (
            "these stylesheet selectors are not covered by the guard, so an "
            f"unsupported property on them would go unreported: {skipped}"
        )

    bad = _collect_unknown_properties(qapp, apply)
    assert bad == [], f"Qt rejected these stylesheet properties, so the rules do not apply: {sorted(set(bad))}"


def test_diagnostic_console_stylesheet_uses_only_supported_properties(qapp):
    """The dock had its own sheet with an unsupported dock property."""
    from views.diagnostic_console import DiagnosticConsole

    console = DiagnosticConsole()

    def apply(app):
        for is_dark in (False, True):
            console.setDarkTheme(is_dark)
            app.processEvents()

    bad = _collect_unknown_properties(qapp, apply)
    assert bad == [], f"DiagnosticConsole stylesheet uses properties Qt ignores: {sorted(set(bad))}"
