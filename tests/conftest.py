# =============================================================================
# FILE: tests/conftest.py
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "property: property-based tests using Hypothesis")
    config.addinivalue_line("markers", "cross_validation: live comparison against R reference packages")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")


# The R-bridge skip deliberately does NOT live here. It used to be a
# collection hook that skipped anything marked `cross_validation` when rpy2 was
# missing, which meant the dependency was declared in two places and the skip
# was invisible at the point of use. It now lives in
# tests/cross_validation/_rbridge.py, which does `pytest.importorskip("rpy2")`
# at import time -- so a module without a working R bridge is skipped as a
# whole, with a reason, at the moment its dependency is resolved.
#
# This matters more than tidiness: the previous arrangement let the directory
# look populated (31 collected tests) while nothing in it ever touched R.


@pytest.fixture
def sample_2d_data():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def sample_abundance_matrix():
    return np.array([[10, 5, 2, 0, 0], [8, 6, 3, 1, 0], [0, 2, 4, 6, 8], [1, 1, 1, 1, 1]])


@pytest.fixture
def sample_distance_matrix():
    return np.array(
        [[0.0, 2.828, 5.657, 8.485], [2.828, 0.0, 2.828, 5.657], [5.657, 2.828, 0.0, 2.828], [8.485, 5.657, 2.828, 0.0]]
    )


@pytest.fixture
def random_seed():
    return 42


# =============================================================================
# Isolation of the process-wide singletons
# =============================================================================
# The application keeps a process-wide ``StateManager`` singleton: the loaded
# matrix, the undo stack, and the "modified" flag.  Nothing reset it between
# tests, so a matrix one test loaded stayed loaded for every test that ran
# after it -- and stayed flagged as *modified*.
#
# That leak has a sharp edge, and it cost a full debugging round to find.
# ``MainWindow.closeEvent`` asks the user to save before closing whenever the
# shared state is modified, and it asks with a MODAL ``QMessageBox``.  A
# headless test run has nobody to answer it, so the run stops dead instead of
# failing: the suite hung on the file that follows
# ``tests/test_pca_dialog_entry_point.py``, which loads a matrix and leaves it
# dirty, and every later test that closed a real main window then waited
# forever on a dialog nobody could see.
#
# The fixture below clears the shared state around every test, so each one
# starts from a known-clean slate.  ``tests/models/test_state_manager.py``
# already does the same thing by hand with ``StateManager.reset_instance()``;
# this makes it the rule for the whole suite instead of a per-file habit.
#
# The clear is deliberately ``StateManager.clear()`` and not
# ``reset_instance()``: the singleton identity has to stay stable, because
# windows and controllers capture a reference to it when they are built.


@pytest.fixture(autouse=True)
def _isolate_shared_state():
    """Give every test a clean process-wide StateManager.

    The state is cleared *before* the test as well as after it.  The "before"
    half is the one that matters: it is what stops a test from inheriting a
    dirty state, and a dirty state is what makes a later ``MainWindow.close()``
    raise a modal dialog that blocks forever.
    """
    # The palette is adjusted for the surface the figures are drawn on, and
    # that surface is process-wide state too.  It is cleared on both sides
    # of the test, so no test inherits the dark theme's palette.
    from config.colors import set_figure_background
    from models.state_manager import get_state_manager

    set_figure_background("")
    get_state_manager().clear()
    yield
    get_state_manager().clear()
    set_figure_background("")
