"""
PaleoAST Test Configuration

Root conftest.py for pytest. Ensures the project root is on sys.path
so that all test files can import project modules without sys.path hacks.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
