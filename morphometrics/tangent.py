# =============================================================================
# FILE: morphometrics/tangent.py
# =============================================================================
"""
Tangent-space projection of Procrustes configurations.

Port of geomorph's internal ``orp()`` (geomorph.support.code.r:510), the
projection gpagen applies before ordination: each aligned configuration is
projected onto the tangent hyperplane of the Procrustes sphere at the
consensus shape,

    x_j <- x_j - (x_j . y - 1) * y,   y = flatten(consensus) / ||.||

so that every projected point satisfies ``x_j . y = 1``.  This removes the
radial (size) component of Procrustes residuals while keeping translation
and rotation effects, exactly as geomorph does for ``PrinAxes=FALSE``.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import MorphometricsError


def orp(configurations: npt.NDArray | list[npt.NDArray]) -> npt.NDArray:
    """
    Project aligned configurations onto the tangent space at the consensus.

    Parameters:
        configurations: (n, p, k) array (or list) of GPA-aligned
            configurations — centroids at the origin; centroid sizes need
            not be exactly 1 but should be Procrustes-scaled.

    Returns:
        (n, p, k) array of projected configurations. Each flattened row
        ``x`` satisfies ``x . y = ||y|| = 1`` with ``y`` the unit-norm
        consensus vector, i.e. the points lie in the tangent hyperplane.

    Raises:
        MorphometricsError: on malformed input or a degenerate (zero-norm)
            consensus configuration.
    """
    X = np.asarray(configurations, dtype=float)
    if X.ndim != 3:
        raise MorphometricsError(
            _("orp expects a (n_specimens, n_landmarks, n_dims) array, got {0}D").format(X.ndim)
        )
    if X.shape[0] == 0:
        raise MorphometricsError(_("orp requires at least one configuration"))

    mean = X.mean(axis=0)
    mean = mean - mean.mean(axis=0)  # remove position (should be ~0 already)
    y = mean.ravel()
    norm = float(np.linalg.norm(y))
    if norm <= np.finfo(float).eps:
        raise MorphometricsError(_("orp cannot project onto a zero-norm consensus shape"))
    y /= norm

    flat = X.reshape(X.shape[0], -1)
    radial = flat @ y - 1.0
    projected = flat - np.outer(radial, y)
    return projected.reshape(X.shape)


def tangent_vectors(configurations: npt.NDArray | list[npt.NDArray]) -> npt.NDArray:
    """
    Orp-project, then return residuals against the (unit-norm) consensus.

    These are the classical tangent-space scores used as input to PCA /
    regression on shape (Dryden & Mardia 2016, §6.2; geomorph calls the
    pair ``orp`` + centring internally).

    Parameters:
        configurations: (n, p, k) GPA-aligned configurations.

    Returns:
        (n, p*k) matrix of tangent vectors (projected config minus the
        consensus flattened).
    """
    projected = orp(configurations)
    mean = projected.mean(axis=0)
    mean = mean - mean.mean(axis=0)
    return (projected - mean).reshape(projected.shape[0], -1)
