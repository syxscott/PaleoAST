# =============================================================================
# FILE: morphometrics/curves.py
# =============================================================================
"""
Curve topology utilities for semilandmark analysis.

Provides the shared representation used by ``partial_gpa`` and the TPS
parser: each curve is a list of landmark indices laid out along the curve,
whose FIRST and LAST entries are fixed endpoint landmarks and whose
interior entries are semilandmarks that may slide.  This is the geomorph
``gpagen`` "curves" convention (curves given as triples at minimum).

Reference implementations:
    - geomorph::validateConfig / slidingsemilandmarks2 (curves argument)
    - geomorph evenPts (geomorph.support.code.r:1989-2012), a simple form
      of StereoMorph's pointsAtEvenSpacing.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import MorphometricsError


def validate_curves(
    curves: list[list[int]],
    n_landmarks: int,
    name: str = "curves",
) -> list[list[int]]:
    """
    Validate curve topology definitions in geomorph's triples convention.

    Parameters:
        curves: List of curves; each curve is a list of landmark indices in
            path order.  The first and last index of every curve is a fixed
            endpoint, the interior indices are semilandmarks.
        n_landmarks: Total number of landmarks per configuration.
        name: Parameter name used in error messages.

    Returns:
        The curves as plain ``list[list[int]]`` (validated, not modified).

    Raises:
        MorphometricsError: If any curve has fewer than 3 indices, an index
            outside ``[0, n_landmarks)``, or a duplicate index within a curve.
    """
    if not curves:
        return []

    cleaned: list[list[int]] = []
    for c_i, curve in enumerate(curves):
        try:
            idx = [int(x) for x in curve]
        except (TypeError, ValueError):
            raise MorphometricsError(
                _("Curve {0} of {1} must contain integer landmark indices").format(c_i, name)
            )
        if len(idx) < 3:
            raise MorphometricsError(
                _("Curve {0} of {1} has {2} points; a curve needs at least 3 "
                  "(two fixed endpoints plus one semilandmark)").format(c_i, name, len(idx))
            )
        if len(set(idx)) != len(idx):
            raise MorphometricsError(
                _("Curve {0} of {1} contains duplicate landmark indices").format(c_i, name)
            )
        for x in idx:
            if not 0 <= x < n_landmarks:
                raise MorphometricsError(
                    _("Curve {0} of {1} references landmark {2} outside "
                      "[0, {3})").format(c_i, name, x, n_landmarks)
                )
        cleaned.append(idx)
    return cleaned


def interior_sliders(curve: list[int]) -> list[int]:
    """Sliding (interior) indices of one validated curve (endpoints pinned)."""
    return list(curve[1:-1])


def evenly_resample_curve(
    coords: npt.NDArray,
    n_points: int,
) -> npt.NDArray:
    """
    Resample an open polyline to ``n_points`` equally spaced points.

    Port of geomorph's ``evenPts`` (geomorph.support.code.r:1989): linear
    interpolation along the cumulative chord length; the first and last
    input points are preserved exactly.

    Parameters:
        coords: (m, k) array of ordered curve coordinates, m >= 2.
        n_points: Number of output points (>= 3; values below 3 collapse
            to the two endpoints, matching geomorph's fallback).

    Returns:
        (n_points, k) array of evenly spaced coordinates.
    """
    x = np.asarray(coords, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] < 2:
        raise MorphometricsError("evenly_resample_curve needs at least 2 points")
    n = int(n_points)
    if n < 2:
        raise MorphometricsError(f"n_points must be >= 2, got {n_points}")
    if n == 2:
        return x[[0, -1]].copy()

    if x.shape[0] == 2:
        # geomorph repeats the endpoint to build a 3-row basis, then cuts
        # the duplicate; equivalent to interpolating on the plain segment.
        seg = x[1] - x[0]
        ts = np.linspace(0.0, 1.0, n)
        return x[0] + np.outer(ts, seg)

    steps = np.diff(x, axis=0)
    ds = np.sqrt(np.sum(steps**2, axis=1))
    cds = np.concatenate([[0.0], np.cumsum(ds)])
    total = cds[-1]
    if total <= 0.0:
        raise MorphometricsError("Cannot resample a degenerate (zero-length) curve")

    out = np.empty((n, x.shape[1]))
    out[0] = x[0]
    out[-1] = x[-1]
    for j in range(1, n - 1):
        target = total * j / (n - 1)
        lo = int(np.searchsorted(cds, target, side="right")) - 1
        lo = max(0, min(lo, x.shape[0] - 2))
        hi = lo + 1
        span = cds[hi] - cds[lo]
        adj = 0.0 if span <= 0 else (target - cds[lo]) / span
        out[j] = x[lo] + adj * (x[hi] - x[lo])
    return out
