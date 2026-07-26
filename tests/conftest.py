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
    config.addinivalue_line("markers", "cross_validation: cross-validation tests against R package gold standards")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")

def pytest_collection_modifyitems(config, items):
    try:
        import rpy2
        rpy2_available = True
    except ImportError:
        rpy2_available = False
    for item in items:
        if "cross_validation" in item.keywords and not rpy2_available:
            item.add_marker(pytest.mark.skip(reason="rpy2 not installed"))

@pytest.fixture
def sample_2d_data():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

@pytest.fixture
def sample_abundance_matrix():
    return np.array([[10, 5, 2, 0, 0], [8, 6, 3, 1, 0], [0, 2, 4, 6, 8], [1, 1, 1, 1, 1]])

@pytest.fixture
def sample_distance_matrix():
    return np.array([[0.0, 2.828, 5.657, 8.485], [2.828, 0.0, 2.828, 5.657], [5.657, 2.828, 0.0, 2.828], [8.485, 5.657, 2.828, 0.0]])

@pytest.fixture
def random_seed():
    return 42
