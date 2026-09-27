"""The only place in the test suite that knows how to reach R.

Every test in ``tests/cross_validation/`` compares a PaleoAST result against a
value computed *live* by the reference R package -- not against a hand-written
constant. That distinction is the entire point of the directory, so nothing here
may fall back to a precomputed number if R is missing: without R the tests skip,
they do not quietly degrade into self-consistency checks.

Design notes
------------
* **No automatic numpy <-> R conversion.** ``rpy2.robjects.numpy2ri`` is
  deprecated as of rpy2 3.6 and its replacement differs between 3.5 and 3.6+.
  Matrices and vectors are therefore built and read through explicit calls
  (``FloatVector`` + ``dim=``, and index-based extraction), which behave the same
  on every rpy2 version this project supports.
* **A missing R package skips its tests, it does not fail them.** CRAN installs
  can partially fail; a skipped test is honest about that, a red test would not
  be, and neither would silently comparing against a local constant.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module at import time when there is no bridge to R. This is
# what makes `pytest tests/` work on a machine without R.
rpy2 = pytest.importorskip(
    "rpy2",
    reason="cross-validation requires rpy2 and a working R installation",
)

from rpy2 import robjects
from rpy2.robjects import FloatVector, ListVector, StrVector
from rpy2.robjects.packages import importr

__all__ = [
    "as_array",
    "as_float",
    "matrix_to_array",
    "r",
    "r_array",
    "r_matrix",
    "r_vector",
    "require",
    "vector_to_array",
]

r = robjects.r

#: Cache so a package is loaded once per session, not once per test.
_PACKAGE_CACHE: dict[str, object] = {}


def require(package: str):
    """Import an R package, or skip the calling test if it is not installed."""
    if package in _PACKAGE_CACHE:
        return _PACKAGE_CACHE[package]
    try:
        module = importr(package)
    except Exception as exc:
        pytest.skip(f"R package {package!r} is not available: {exc}")
    _PACKAGE_CACHE[package] = module
    return module


# ---------------------------------------------------------------------------
# Python -> R
# ---------------------------------------------------------------------------
def r_vector(values) -> robjects.FloatVector:
    """An R numeric vector (1-D) from any 1-D numeric input."""
    flat = np.asarray(values, dtype=float).ravel()
    return FloatVector(flat.tolist())


def r_matrix(values) -> robjects.FloatVector:
    """An R base *matrix* (2-D, column-major) from a 2-D numpy array.

    ``FloatVector`` is flat and column-major, which is exactly R's storage
    order, so flattening with ``ravel()`` and restoring the shape with ``dim=``
    is lossless.
    """
    return r_array(values)


def r_array(values) -> robjects.FloatVector:
    """An R array of any dimensionality, from a numpy array.

    geomorph takes landmark configurations as a 3-D array
    (specimens x landmarks x dimensions), so 2-D-only would not do.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError("r_array needs at least one dimension")
    vec = FloatVector(arr.ravel(order="C").tolist())
    dim = FloatVector([float(n) for n in arr.shape])
    return r["dim="](vec, dim)


def r_data_frame(columns: dict[str, object]):
    """An R data.frame from a dict of column name -> values.

    Numeric and character columns are both supported, which matters because a
    grouping factor (``adonis2(~ group)``) has to arrive in R as a factor, not
    as a numeric vector.
    """
    if not columns:
        raise ValueError("r_data_frame needs at least one column")
    names = list(columns)
    length = len(next(iter(columns.values())))
    for name in names:
        if len(columns[name]) != length:
            raise ValueError(f"column {name!r} has length {len(columns[name])}, expected {length}")

    converted = {}
    for name in names:
        values = list(columns[name])
        first = next((v for v in values if v is not None), None)
        if isinstance(first, str):
            converted[name] = StrVector([str(v) for v in values])
        else:
            converted[name] = r_vector(values)

    return r["data.frame"](
        ListVector(converted),
        stringsAsFactors=False,
        check_names=False,
    )


# ---------------------------------------------------------------------------
# R -> Python
# ---------------------------------------------------------------------------
def as_float(value) -> float:
    """Coerce an R scalar (length-1 FloatVector, or a Python float) to float.

    Done by index rather than ``float(r_obj)`` so it does not depend on which
    rpy2 version implements ``__float__`` for SexpVector.
    """
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if len(value) == 0:
        raise ValueError("cannot convert a zero-length R vector to float")
    return float(value[0])


def as_array(value) -> np.ndarray:
    """Coerce an R numeric vector to a flat numpy array."""
    return np.array([float(v) for v in value], dtype=float)


def matrix_to_array(value) -> np.ndarray:
    """Coerce an R matrix/array to a numpy array of the same shape.

    Indexing with ``[i, j]`` follows R's 1-based convention, so this returns the
    array the way it reads in R rather than in storage order. Works for
    n-dimensional arrays as long as only the first two indices are used.
    """
    dims = [int(as_float(d)) for d in r["dim"](value)]
    n_row, n_col = dims[0], dims[1]
    out = [[float(value[i, j]) for j in range(1, n_col + 1)] for i in range(1, n_row + 1)]
    arr = np.array(out, dtype=float)
    if len(dims) > 2:
        arr = arr.reshape([n_row, n_col, *dims[2:]])
    return arr


def vector_to_array(value) -> np.ndarray:
    """Coerce an R vector to a numpy array (alias kept for readability)."""
    return as_array(value)


def unnamed_list(value) -> list:
    """Convert an R list to a Python list of already-converted elements."""
    out = []
    for element in value:
        # rpy2 returns nested vectors as numpy arrays and scalars as floats.
        if isinstance(element, (int, float, np.floating, np.integer)):
            out.append(float(element))
        else:
            out.append(as_array(element))
    return out
