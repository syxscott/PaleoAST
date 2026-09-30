# Ensure project root is on sys.path for imports.
#
# NOTE the three ``.parent`` hops. This file lives at
# ``<root>/tests/stats/conftest.py``, so:
#     .parent        -> <root>/tests/stats
#     .parent.parent -> <root>/tests        <-- WRONG: this is what it used to use
#     .parent.parent.parent -> <root>       <-- correct
#
# Inserting ``<root>/tests`` made ``tests/stats/`` importable as a top-level
# ``stats`` package, shadowing the real ``<root>/stats/``. The shadow package
# has an ``__init__.py`` but none of the modules (``univariate``, ``simper``,
# ``pcoa``, ...), so every ``from stats.X import ...`` in this directory failed
# with "No module named 'stats.X'" -- but ONLY when another test had already
# inserted the true root first and something had removed/reordered it. Run
# ``pytest tests/stats/test_pcoa.py`` alone and it passed; run it after
# ``tests/ecology`` and 38 modules failed to collect. The fix matches
# tests/models/conftest.py and tests/parsers/conftest.py, which both use the
# correct depth.
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
