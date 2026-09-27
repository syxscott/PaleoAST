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
