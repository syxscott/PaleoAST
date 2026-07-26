"""
Pytest configuration for models tests.

This conftest patches the utils module dependencies that require PyQt6,
allowing DataMatrix tests to run without a GUI environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

# Get the project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _patch_utils_modules():
    """Patch the utils module dependencies to avoid PyQt6 import."""
    if "utils" in sys.modules and isinstance(sys.modules["utils"], ModuleType):
        return  # Already patched

    # Create mock utils module
    mock_utils = ModuleType("utils")

    # Create mock validators submodule
    mock_validators = ModuleType("utils.validators")

    def validate_data_array(data, allow_nan=True, allow_inf=False, name="data_matrix"):
        """Mock validate_data_array that just converts to numpy array."""
        arr = np.asarray(data, dtype=float)
        return arr

    def check_missing_values(data, report_positions=False):
        """Mock check_missing_values."""
        nan_mask = np.isnan(data)
        return {
            "n_missing": int(np.sum(nan_mask)),
            "p_missing": float(np.sum(nan_mask) / data.size),
        }

    mock_validators.validate_data_array = validate_data_array
    mock_validators.check_missing_values = check_missing_values

    # Create mock exceptions submodule
    mock_exceptions = ModuleType("utils.exceptions")

    class MatrixDimensionError(ValueError):
        """Mock MatrixDimensionError."""
        def __init__(self, message, details=None):
            super().__init__(message)
            self.details = details or {}

    mock_exceptions.MatrixDimensionError = MatrixDimensionError

    # Create mock event_bus submodule
    mock_event_bus = ModuleType("utils.event_bus")

    class MockQObject:
        pass

    class MockSignal:
        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    mock_event_bus.QObject = MockQObject
    mock_event_bus.pyqtSignal = lambda: MockSignal

    # Create mock state_manager submodule
    mock_state_manager = ModuleType("models.state_manager")

    def mock_get_state_manager():
        return None

    mock_state_manager.get_state_manager = mock_get_state_manager

    # Register the modules
    sys.modules["utils"] = mock_utils
    sys.modules["utils.validators"] = mock_validators
    sys.modules["utils.exceptions"] = mock_exceptions
    sys.modules["utils.event_bus"] = mock_event_bus
    sys.modules["models.state_manager"] = mock_state_manager

    # Now we can import data_matrix
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "data_matrix",
        _PROJECT_ROOT / "models" / "data_matrix.py"
    )
    dm_module = importlib.util.module_from_spec(spec)
    sys.modules["data_matrix"] = dm_module
    spec.loader.exec_module(dm_module)

    # Patch DataMatrix into models.data_matrix
    sys.modules["models.data_matrix"] = dm_module


# Apply patch before any tests run
_patch_utils_modules()
