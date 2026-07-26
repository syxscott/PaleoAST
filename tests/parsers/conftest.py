"""Tests/parsers conftest - ensures project root is on sys.path."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
