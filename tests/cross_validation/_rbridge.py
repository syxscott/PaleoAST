"""The only place in the test suite that knows how to reach R.

Every test in ``tests/cross_validation/`` compares a PaleoAST result against a
value computed *live* by the reference R package -- not against a hand-written
constant. That distinction is the entire point of the directory, so nothing here
may fall back to a precomputed number if R is missing: without R the tests skip,
they do not quietly degrade into self-consistency checks.

Design notes
------------
* **No automatic numpy <-> R conversion.** Arrays go across as ``FloatVector``
  plus an explicit ``dim=`` call, and come back by index. rpy2 3.6 deprecated
  ``rpy2.robjects.numpy2ri`` and its replacement differs between 3.5 and 3.6+,
  so the explicit form is what lets the same test code work on either -- which
  matters because the version is pinned in ``pyproject.toml`` for R-ABI reasons
  and may move.
* **A missing R package skips its tests, it does not fail them.** CRAN installs
  can partially fail; a skipped test is honest about that, a red test would not
  be, and neither would silently comparing against a local constant.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module at import time when there is no bridge to R. This is
# what makes `pytest tests/` work on a machine without R.
#
# The guard has to be wider than an ImportError.  rpy2 being INSTALLED is
# not the same as rpy2 being ABLE TO REACH R: with a broken R (an R_HOME
# that points somewhere R is not built, say) the import succeeds and the
# failure arrives a moment later, as a RuntimeError from R's own start-up.
# A plain importorskip would let that through as a collection ERROR --
# one broken R on one developer machine, and the whole cross-validation
# directory fails instead of skipping.  The file's own rule is "a missing
# R skips its tests, it does not fail them", and a broken R is the same
# situation.
try:
    rpy2 = pytest.importorskip(
        "rpy2",
        reason="cross-validation requires rpy2 and a working R installation",
    )
    from rpy2 import robjects
    from rpy2.robjects import FloatVector, ListVector, StrVector
    from rpy2.robjects.packages import importr
except Exception as exc:
    pytest.skip(
        f"cross-validation requires rpy2 and a working R installation: {exc}",
        allow_module_level=True,
    )

__all__ = [
    "FloatVector",
    "ListVector",
    "StrVector",
    "as_array",
    "as_float",
    "matrix_to_array",
    "r",
    "r_array",
    "r_labels",
    "r_matrix",
    "r_numbers",
    "r_program",
    "r_string_vector",
    "r_vector",
    "require",
    "vector_to_array",
]

r = robjects.r

#: Cache so a package is loaded once per session, not once per test.
_PACKAGE_CACHE: dict[str, object] = {}


def require(package: str):
    """Import an R package, or skip the calling test if it is not installed.

    The skip is raised with ``allow_module_level=True`` because this
    function is called at *module* level by the files that need a
    package: ``geomorph`` or ``iNEXT`` missing otherwise turns the whole
    cross-validation directory into a collection ERROR, which is louder
    and less honest than a skip.  pytest refuses a plain ``pytest.skip``
    outside a test for exactly that reason, and says so.
    """
    if package in _PACKAGE_CACHE:
        return _PACKAGE_CACHE[package]
    try:
        module = importr(package)
    except Exception as exc:
        pytest.skip(
            f"R package {package!r} is not available: {exc}",
            allow_module_level=True,
        )
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

    Flattening and restoring the shape is lossless **only** because
    ``r_array`` flattens in column-major (``order="F"``) order, which is R's
    storage order. Flattening row-major and letting ``array(data, dim=)`` fill
    column-major scrambles the data instead of transcribing it: for the 30x5
    matrix the cross-validation suite uses, 148 of 150 entries land in the
    wrong position, and the result is not even a transpose, so no caller-side
    reshape can recover it. R would then run prcomp/vegan/ape on different
    data than the test intended, and every comparison would be against the
    wrong matrix.
    """
    return r_array(values)


def r_array(values) -> robjects.FloatVector:
    """An R array of any dimensionality, from a numpy array.

    geomorph takes landmark configurations as a 3-D array
    (specimens x landmarks x dimensions), so 2-D-only would not do.

    The flatten is ``order="F"`` (column-major) on purpose. R's ``array()``
    fills its storage column by column, so element ``(i, j)`` of an
    ``nrow x ncol`` array is flat position ``i + nrow * j``. numpy's default
    ``ravel()`` is row-major, so pairing it with ``array(data, dim=)`` sends a
    scrambled matrix to R -- an easy mistake to make because the failure looks
    like a legitimate numerical disagreement rather than a crash.

    Uses ``array(data, dim=)`` rather than the replacement form ``dim<-``. The
    subscript spelling of a replacement function is looked up as a *variable*
    name in the global environment and raises
    ``KeyError: "'dim=' not found"``; ``r("array")`` is an ordinary function
    lookup and behaves the same on every rpy2 version.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError("r_array needs at least one dimension")
    vec = FloatVector(arr.ravel(order="F").tolist())
    return r("array")(vec, dim=FloatVector([float(n) for n in arr.shape]))


def r_string_vector(values) -> robjects.StrVector:
    """An R character vector -- for Newick strings, tip names, and the like.

    ``r_vector`` coerces to float, so anything textual has to go through here.
    """
    return StrVector([str(v) for v in values])


def r_data_frame(columns: dict[str, object], factors: tuple[str, ...] = ()):
    """An R data.frame from a dict of column name -> values.

    Numeric and character columns are both supported. Character columns arrive
    as strings, which is *not* the same thing as a factor -- and a grouping
    variable that R treats as character instead of factor is exactly the kind
    of difference that makes a model silently wrong. Name such columns in
    ``factors`` to have them converted on the R side.
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
        if name in factors:
            converted[name] = r("factor")(StrVector([str(v) for v in values]))
        elif isinstance(first, str):
            converted[name] = StrVector([str(v) for v in values])
        else:
            converted[name] = r_vector(values)

    return r("data.frame")(
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

    R arrays are stored **column-major**, and ``as.numeric()`` on an array
    returns exactly that flattened storage order with the ``dim`` attribute
    dropped. Reshaping that in numpy with ``order="F"`` therefore restores the
    array in its original shape without ever indexing it.

    The previous version built the array with ``value[i, j]`` over
    ``range(1, n + 1)``. That is wrong twice over: rpy2's ``__getitem__`` is
    **zero-based** (the R-style one-based accessor is ``rx``), so the loop
    skipped the first row and column and would have raised on the last. Going
    through the storage order instead means there is no index base to get wrong,
    and it works for any number of dimensions.

    ``dist`` objects are converted first. ``vegan::vegdist`` returns a
    ``dist``, and R's ``dim()`` on a ``dist`` is NULL (a ``dist`` carries
    ``Size``, not ``dim``), so reading the shape off it raised
    ``TypeError: 'NULLType' object is not iterable``. ``as.matrix()`` turns it
    into the full symmetric n x n matrix, which is also the shape the
    PaleoAST side produces -- so the comparison becomes like-for-like instead
    of comparing a matrix against a condensed upper triangle.
    """
    dims_obj = r("dim")(value)
    if type(dims_obj).__name__ == "NULLType":
        value = r("as.matrix")(value)
        dims_obj = r("dim")(value)
    dims = [int(as_float(d)) for d in dims_obj]
    flat = as_array(r("as.numeric")(value))
    return flat.reshape(dims, order="F")


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


# ---------------------------------------------------------------------------
# Running a reference program
# ---------------------------------------------------------------------------
def r_program(code: str, **substitutions: str) -> dict[str, float]:
    """Run an R program and read back the named numeric vector it returns.

    The program must end with the vector as its last expression; its names
    become the keys of the returned dict.

    Substitutions use ``@@name@@`` markers rather than ``str.format``'s
    ``{name}``.  R programs are full of braces -- a ``for`` loop body, a
    function definition -- and ``str.format`` reads those as placeholders,
    so a program with a loop fails with a ``KeyError`` about a line of R
    that has nothing to do with the substitution.

    Going through a *program* rather than through R's objects directly is
    deliberate.  The shapes R's statistical functions return differ from one
    another -- ``summary()`` of an ``aov`` gives a matrix for a plain model
    and a list of matrices per stratum when there is an ``Error()`` term,
    and the row names of an ``anova`` table are padded with trailing spaces
    -- and rpy2's conversion of those shapes does not always preserve the
    attributes the names live in.  Letting R do the indexing keeps every
    lookup on the R side, where the semantics are the documented ones.
    """
    text = code
    for key, value in substitutions.items():
        text = text.replace(f"@@{key}@@", str(value))
    result = r("eval")(r("parse")(text=text))
    return {str(name): as_float(result.rx2(str(name))) for name in r("names")(result)}


def r_numbers(values) -> str:
    """An R numeric vector literal.

    ``repr`` of a Python float is the shortest string that reads back as the
    same double, which is exactly the precision R's parser needs.
    """
    flat = np.asarray(values, dtype=float).ravel()
    return "c(" + ", ".join(repr(float(v)) for v in flat) + ")"


def r_labels(values) -> str:
    """An R character vector literal."""
    return "c(" + ", ".join('"' + str(v) + '"' for v in values) + ")"
