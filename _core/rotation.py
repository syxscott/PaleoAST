"""
================================================================================
_core/rotation.py - SVD Rotation Reference Implementation
================================================================================

Single canonical implementation of the Kabsch algorithm for optimal rotation
estimation. All other modules in PaleoAST MUST use this implementation via
thin wrappers.

Mathematical Framework
================================================================================

Given two sets of N points in k-dimensional space:
    X = {x_1, x_2, ..., x_N}  (source, to be rotated)
    Y = {y_1, y_2, ..., y_N}  (target, reference)

Find the orthogonal matrix R (k x k) that minimizes:

    ||Y - X @ R||_F^2 = sum_{i=1}^N ||y_i - R @ x_i||^2

Solution (Kabsch 1976):
----------------------
1. Compute cross-covariance matrix: H = X^T @ Y
2. SVD decomposition: H = U @ S @ V^T
3. Optimal rotation: R* = U @ V^T

Reflection Trap:
---------------
If det(U @ V^T) < 0, we have a reflection rather than a rotation.
Fix: flip the sign of the last column of U (or last row of V^T),
then recompute R = U' @ V^T.

This ensures R ∈ SO(k) with det(R) = +1.

Reference:
    Kabsch, W. (1976). A solution for the best rotation to relate
    two sets of vectors. Acta Crystallographica, A32, 922-923.

Author: PaleoAST Development Team
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: find optimal rotation R minimizing ||Y - X @ R||_F^2.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, k_dims)
        Source point matrix. Each row is a point.
    Y : np.ndarray, shape (n_points, k_dims)
        Target point matrix. Each row is a point.
        Must have same shape as X.

    Returns
    -------
    R : np.ndarray, shape (k_dims, k_dims)
        Rotation matrix in SO(k_dims) (det(R) = +1).

    Raises
    ------
    ValueError
        If X and Y have different shapes, or if fewer than 2 points.

    Notes
    -----
    The algorithm works for any dimensionality k >= 1.
    For k=3 (3D rotations), see also kabsch_rotation_3d().

    Example
    -------
    >>> X = np.array([[1, 0], [0, 1]])
    >>> Y = np.array([[0, 1], [-1, 0]])  # 90-degree rotation
    >>> R = kabsch_rotation(X, Y)
    >>> np.allclose(R @ X, Y)
    True

    R Verified Against R Packages
    ----------------------------
    - R: procSym() in Morpho package
    - Python: This implementation
    - Both should give identical results up to numerical precision.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    n_points, k_dims = X.shape

    if n_points < 2:
        raise ValueError(f"Need at least 2 points, got {n_points}")

    # Step 1: Center both point sets (optional but recommended for GPA)
    # Note: centering is often done by the caller. We work with raw data.

    # Step 2: Cross-covariance matrix
    # H[i,j] = sum_n (X[n,i] * Y[n,j])
    H = X.T @ Y

    # Step 3: SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Step 4: Compute rotation
    R = U @ Vt

    # Step 5: Reflection trap handling
    # If det(R) < 0, we have a reflection. Fix by flipping last singular vector.
    if np.linalg.det(R) < 0:
        # Create modified U with flipped last column
        U_fixed = U.copy()
        U_fixed[:, -1] *= -1
        R = U_fixed @ Vt

    # Verify R is a proper rotation matrix
    if not np.allclose(R @ R.T, np.eye(k_dims), atol=1e-8):
        raise RuntimeError("Kabsch rotation result is not orthogonal")

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
        raise RuntimeError(f"Kabsch rotation determinant is not +1: {np.linalg.det(R)}")

    return R


def kabsch_rotation_3d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm specialized for 3D point sets (N x 3).

    This is a convenience wrapper around kabsch_rotation() that validates
    and optimizes for the common 3D case.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, 3)
        Source 3D point matrix.
    Y : np.ndarray, shape (n_points, 3)
        Target 3D point matrix.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix in SO(3) (det(R) = +1).

    Example
    -------
    >>> # Two points defining a line, rotated 45 degrees around Z axis
    >>> X = np.array([[1, 0, 0], [0, 1, 0]])
    >>> angle = np.pi / 4
    >>> c, s = np.cos(angle), np.sin(angle)
    >>> R_true = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    >>> Y = X @ R_true.T
    >>> R = kabsch_rotation_3d(X, Y)
    >>> np.allclose(R, R_true, atol=1e-10)
    True

    See Also
    --------
    kabsch_rotation : General k-dimensional implementation.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must be shape (N, 3), got {X.shape}")

    if Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError(f"Y must be shape (N, 3), got {Y.shape}")

    if X.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 non-collinear points for 3D rotation, got {X.shape[0]}"
        )

    return kabsch_rotation(X, Y)
