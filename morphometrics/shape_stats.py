# =============================================================================
# FILE: morphometrics/shape_stats.py
# =============================================================================
"""
Statistical tools for Procrustes shape data.

Borrowed from Dryden & Mardia (2016, ch. 5-8) and the shapes/geomorph
ecosystem:

* :func:`kendall_preshape` — Kendall preshape sphere coordinates
  (translate to centroid, scale to unit centroid size).
* :func:`geometric_median` / :func:`procrustes_median` — rotation-refitting
  median of shapes (Small 1977), more robust than the consensus mean.
* :func:`goodall_f` / :func:`goodall_test` — Goodall's (1991) F statistic
  for group differences in shape, with a permutation test.
* :func:`hotelling_t2` — two-sample Hotelling T² on aligned (tangent)
  coordinates with the exact F-distribution p-value.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import MorphometricsError

# =============================================================================
# Preshape / distances
# =============================================================================


def kendall_preshape(configurations: npt.NDArray) -> npt.NDArray:
    """
    Map (n, p, k) landmark configurations onto the Kendall preshape sphere.

    Each configuration is centred on its centroid and scaled to unit
    Frobenius norm; configurations are returned flattened to row vectors.

    Raises:
        MorphometricsError: on a degenerate (zero-size) configuration.
    """
    X = np.asarray(configurations, dtype=float)
    if X.ndim != 3:
        raise MorphometricsError(_("kendall_preshape expects (n, p, k); got shape {0}").format(X.shape))
    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=(1, 2)))
    if np.any(norms <= np.finfo(float).eps):
        raise MorphometricsError(_("Cannot preshape a zero-size configuration"))
    return (centered / norms[:, None, None]).reshape(X.shape[0], -1)


def procrustes_distance(
    a: npt.NDArray,
    b: npt.NDArray,
    no_reflect: bool = False,
) -> float:
    """Procrustes (full) distance: min over translation/scale/similarity of
    ||A - B||.  Computed via the signed-SVD trace (Dryden & Mardia 2016,
    Prop. 2.5):

        d² = 2 - 2 * Σ_i σ_i * s_i

    where ``σ_i`` are the singular values of ``Aᵀ B`` (after centring and
    unit-norm scaling) and ``s_i ∈ {+1, -1}`` flips the sign of the *last*
    (smallest) singular value when the optimal similarity fit requires a
    reflection (``det(Vᵀ U) < 0``).  The earlier implementation multiplied
    the entire sum by ``sign(det)``, which made reflected distances
    *larger* than the true full-Procrustes distance — the opposite of the
    "full Procrustes" definition.

    Parameters:
        a, b: (p, k) landmark configurations (k = 2 or 3).
        no_reflect: if True, only proper rotations (det = +1) are
            considered and the smallest singular value is never negated
            (matches ``gpa.analyze(no_reflect=True)`` behaviour).

    Returns:
        sqrt(max(2 - 2 Σ σᵢ sᵢ, 0)).
    """
    A = np.asarray(a, dtype=float) - np.asarray(a, dtype=float).mean(axis=0)
    B = np.asarray(b, dtype=float) - np.asarray(b, dtype=float).mean(axis=0)
    na = np.sqrt(np.sum(A**2))
    nb = np.sqrt(np.sum(B**2))
    if na <= np.finfo(float).eps or nb <= np.finfo(float).eps:
        raise MorphometricsError(_("Procrustes distance needs non-degenerate configurations"))
    A /= na
    B /= nb
    U, S, Vt = np.linalg.svd(A.T @ B)
    # With N = A'B and singular values sigma_1 >= ... >= sigma_k, the trace
    # maximised over an UNCONSTRAINED orthogonal transform is sum(sigma), but
    # that maximiser has det = sign(det N).  When reflections are allowed that
    # is exactly what we want, so NO sign flip is applied.
    #
    # When only proper rotations are allowed and det N < 0, the best proper
    # transform must "sacrifice" the smallest singular value: the maximised
    # trace drops to sum(sigma) - 2*sigma_k.  That is the ONLY case where a
    # sign flip belongs here.
    #
    # The two previous implementations had this inverted. Multiplying the
    # whole sum by sign(det) inflated reflected distances; flipping sigma_k
    # when reflections were *allowed* instead of *forbidden* swapped the two
    # answers. Both were caught by brute-force search over all orthogonal
    # transforms (see tests/golden/test_procrustes_ground_truth.py).
    signs = np.ones_like(S)
    if no_reflect and np.linalg.det(Vt.T @ U.T) < 0:
        signs[-1] = -1.0
    d2 = 2.0 - 2.0 * float(np.sum(S * signs))
    # Clamp round-off at BOTH ends. For identical configurations the ideal
    # d² is exactly 0, but the singular values of a unit-norm Gram matrix sum
    # to 1 only to within rounding — on the 46-point 3D scallop fixture the
    # sum came out as 1 - 1.11e-16 (one ULP low), making d² a *positive*
    # 2.22e-16 and the distance 1.49e-8 instead of 0. Clamping only the
    # negative side (the previous ``max(d2, 0.0)``) did not catch that,
    # because a positive near-zero is exactly the case that slips through.
    #
    # The threshold must sit just above float noise and well below any real
    # displacement. Measured on that fixture: identical input leaves d² at
    # 4.4e-16, while displacing ONE landmark coordinate by 1e-4 gives d² =
    # 2.7e-13 — three orders of magnitude larger. 1e-14 separates them with
    # room to spare, and because d² is quadratic in displacement it still
    # resolves distance differences down to ~1e-7.
    if d2 <= 1e-14:
        return 0.0
    return float(np.sqrt(d2))


def _procrustes_align_to(source: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
    """Best similarity-fit of ``source`` onto ``target`` (rows=points);
    returns the rotated/scaled/centred source (in target's scale)."""
    A = source - source.mean(axis=0)
    B = target - target.mean(axis=0)
    na = np.sqrt(np.sum(A**2))
    if na <= np.finfo(float).eps:
        return A
    A = A / na * np.sqrt(np.sum(B**2))
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0] * (A.shape[1] - 1) + [d])
    R = Vt.T @ D @ U.T
    return A @ R


# =============================================================================
# Geometric medians
# =============================================================================


def geometric_median(vectors: npt.NDArray, tol: float = 1e-10, max_iter: int = 500) -> npt.NDArray:
    """Euclidean geometric median of row vectors (Weiszfeld with the
    Vardi-Zhang singularity fix).  Returns shape (d,)."""
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2 or V.shape[0] == 0:
        raise MorphometricsError(_("geometric_median expects a non-empty (n, d) array"))
    m = V.mean(axis=0)
    for _step in range(max_iter):
        diffs = V - m
        dist = np.sqrt(np.sum(diffs**2, axis=1))
        at_m = dist <= np.finfo(float).eps
        if at_m.any():
            # Vardi-Zhang: pull towards the data point(s) coinciding with m
            z = V[at_m][0]
            zeta = min(1.0, (at_m.sum() - 0.0) / V.shape[0])
            rest = ~at_m
            if rest.any():
                wr = 1.0 / dist[rest]
                m_rest = (V[rest] * wr[:, None]).sum(axis=0) / wr.sum()
            else:
                m_rest = z
            m_new = zeta * z + (1.0 - zeta) * m_rest
        else:
            w = 1.0 / dist
            m_new = (V * w[:, None]).sum(axis=0) / w.sum()
        shift = np.linalg.norm(m_new - m)
        m = m_new
        if shift <= tol * max(1.0, np.linalg.norm(m)):
            break
    return m


def procrustes_median(
    configurations: npt.NDArray,
    tol: float = 1e-10,
    max_iter: int = 250,
) -> npt.NDArray:
    """
    Geometric median of shapes under the Procrustes metric (Small 1977).

    Iterates: Procrustes-align every configuration to the current median
    estimate, weight by inverse distance, recompute the weighted mean and
    renormalise to unit centroid size.  Distances below ``tol`` make the
    corresponding specimen contribute the median itself (limit direction).

    Returns:
        (p, k) median configuration, centred, unit centroid size.
    """
    X = np.asarray(configurations, dtype=float)
    if X.ndim != 3:
        raise MorphometricsError(_("procrustes_median expects (n, p, k)"))
    n = X.shape[0]
    if n == 1:
        return X[0] - X[0].mean(axis=0)
    m = X.mean(axis=0)
    m = m - m.mean(axis=0)
    m = m / np.sqrt(np.sum(m**2))
    for _step in range(max_iter):
        aligned = np.stack([_procrustes_align_to(cfg, m) for cfg in X])
        dist = np.sqrt(np.sum((aligned - m) ** 2, axis=(1, 2)))
        zero = dist <= tol
        w = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, dist))
        if np.allclose(w, 0.0):
            break
        m_new = (aligned * w[:, None, None]).sum(axis=0) / w.sum()
        m_new = m_new - m_new.mean(axis=0)
        norm = np.sqrt(np.sum(m_new**2))
        if norm <= np.finfo(float).eps:
            raise MorphometricsError(_("procrustes_median collapsed to a point configuration"))
        m_new /= norm
        shift = np.sqrt(np.sum((m_new - m) ** 2))
        m = m_new
        if shift <= tol:
            break
    return m


# =============================================================================
# Goodall's F and permutation test
# =============================================================================


def _group_procrustes_ss(configs: npt.NDArray) -> float:
    """Within-group Procrustes SS: sum over specimens of squared residual
    distance to the group mean (configs already share a frame)."""
    mean = configs.mean(axis=0)
    return float(np.sum((configs - mean) ** 2))


@dataclass
class GoodallResult:
    """Container for Goodall's F permutation test."""

    f_statistic: float
    p_value: float
    n_permutations: int
    between_ss: float
    within_ss: float


def goodall_f(configs: npt.NDArray, group_labels: npt.NDArray) -> tuple[float, float, float]:
    """
    Goodall's (1991) F statistic on aligned configurations.

    f = (between-group Procrustes SS) / (within-group Procrustes SS),
    with group means taken in the shared (GPA-aligned) frame.
    """
    labels = np.asarray(group_labels)
    groups = np.unique(labels)
    if len(groups) < 2:
        raise MorphometricsError(_("goodall_f requires at least two groups"))
    grand = configs.mean(axis=0)
    between = 0.0
    within = 0.0
    for g in groups:
        idx = labels == g
        ng = int(idx.sum())
        gm = configs[idx].mean(axis=0)
        between += ng * float(np.sum((gm - grand) ** 2))
        within += _group_procrustes_ss(configs[idx])
    if within <= np.finfo(float).eps:
        raise MorphometricsError(_("Goodall's f is undefined when within-group SS is zero"))
    return between / within, between, within


def goodall_test(
    configurations: npt.NDArray,
    group_labels: npt.NDArray,
    n_permutations: int = 999,
    seed: int | None = None,
) -> GoodallResult:
    """
    Permutation test for group differences in shape using Goodall's f.

    Group labels are permuted (labels only — specimens stay put), so the
    null holds the within-sample shape geometry fixed.  The p-value is
    (1 + #{f_perm >= f_obs}) / (1 + n_permutations), the standard
    rejection-critical convention.
    """
    X = np.asarray(configurations, dtype=float)
    labels = np.asarray(group_labels)
    if X.ndim != 3 or labels.ndim != 1 or len(labels) != X.shape[0]:
        raise MorphometricsError(_("goodall_test expects (n, p, k) configurations and n group labels"))
    f_obs, between, within = goodall_f(X, labels)
    rng = np.random.default_rng(seed)
    count = 0
    for _i in range(n_permutations):
        perm = rng.permutation(labels)
        try:
            f_perm, _between, _within = goodall_f(X, perm)
        except MorphometricsError:
            continue
        if f_perm >= f_obs - 1e-15:
            count += 1
    p = (1.0 + count) / (1.0 + n_permutations)
    return GoodallResult(
        f_statistic=f_obs,
        p_value=p,
        n_permutations=n_permutations,
        between_ss=between,
        within_ss=within,
    )


# =============================================================================
# Hotelling T²
# =============================================================================


@dataclass
class HotellingT2Result:
    """Container for the two-sample Hotelling T² test."""

    t2: float
    f_statistic: float
    p_value: float
    df1: int
    df2: int
    mean_difference: npt.NDArray


def hotelling_t2(sample1: npt.NDArray, sample2: npt.NDArray) -> HotellingT2Result:
    """
    Two-sample Hotelling T² on flat aligned/tangent vectors (n_i, d).

    T² = (n1*n2/(n1+n2)) * d' S_pooled^{-1} d with the pooled covariance
    turned into the exact statistic
    F = (n1+n2-d-1)/(d*(n1+n2-2)) * T² ~ F_{d, n1+n2-d-1}.

    The pooled covariance is inverted through lstsq (pinv), so singular
    high-dimensional shape data degrade gracefully instead of exploding.
    """
    from scipy import stats as _stats

    X1 = np.atleast_2d(np.asarray(sample1, dtype=float))
    X2 = np.atleast_2d(np.asarray(sample2, dtype=float))
    n1, n2 = X1.shape[0], X2.shape[0]
    d = X1.shape[1]
    if X2.shape[1] != d:
        raise MorphometricsError(_("Hotelling T² samples must have equal dimensions"))
    if n1 < 2 or n2 < 2 or n1 + n2 - d - 1 <= 0:
        raise MorphometricsError(_("Hotelling T² needs n1+n2 > d+1 (got n1={0}, n2={1}, d={2})").format(n1, n2, d))
    diff = X1.mean(axis=0) - X2.mean(axis=0)
    S = (((X1 - X1.mean(axis=0)).T @ (X1 - X1.mean(axis=0))) + ((X2 - X2.mean(axis=0)).T @ (X2 - X2.mean(axis=0)))) / (
        n1 + n2 - 2
    )
    # Invert S via ``np.linalg.solve`` (fast + numerically stable in the
    # well-conditioned regime).  If S is singular, fall back to
    # ``np.linalg.lstsq`` — but emit a warning so the caller knows the
    # T² is now the minimum-norm projection, not the true quadratic form
    # ``diff' S^{-1} diff`` (which is undefined when det S = 0).
    #
    # Singularity must be detected explicitly. Relying on
    # ``np.linalg.solve`` to raise ``LinAlgError`` does not work: LAPACK's
    # LU factorisation only reports an exactly-zero pivot, and a matrix that
    # is singular in exact arithmetic (here, one column a linear combination
    # of the others) almost never produces one after rounding. Measured on a
    # 12x4 case with column 3 := 2*c0 - c1 + 0.5*c2: det(S) = 4.6e-17, rank 3
    # of 4, cond(S) = 1.9e17 — and ``solve`` returned a solution with a
    # 6.2e-16 residual and raised nothing at all. The caller got a confident
    # T² from a matrix that has no inverse, and the warning never fired.
    import warnings

    rank = int(np.linalg.matrix_rank(S))
    if rank < d:
        warnings.warn(
            f"Hotelling T²: pooled covariance is singular (rank {rank} of {d}); "
            "falling back to least-squares. The reported T² is the "
            "minimum-norm projection, not the true quadratic form.",
            stacklevel=2,
        )
        sol = np.linalg.lstsq(S, diff, rcond=None)[0]
    else:
        try:
            sol = np.linalg.solve(S, diff)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Hotelling T²: pooled covariance is singular; "
                "falling back to least-squares. The reported T² is the "
                "minimum-norm projection, not the true quadratic form.",
                stacklevel=2,
            )
            sol = np.linalg.lstsq(S, diff, rcond=None)[0]
    t2 = float(n1 * n2 / (n1 + n2) * (diff @ sol))
    df1 = d
    df2 = n1 + n2 - d - 1
    f_stat = df2 / (df1 * (n1 + n2 - 2)) * t2
    p = float(_stats.f.sf(f_stat, df1, df2))
    return HotellingT2Result(t2=t2, f_statistic=f_stat, p_value=p, df1=df1, df2=df2, mean_difference=diff)
