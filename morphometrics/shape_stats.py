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
        raise MorphometricsError(
            _("kendall_preshape expects (n, p, k); got shape {0}").format(X.shape)
        )
    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=(1, 2)))
    if np.any(norms <= np.finfo(float).eps):
        raise MorphometricsError(_("Cannot preshape a zero-size configuration"))
    return (centered / norms[:, None, None]).reshape(X.shape[0], -1)


def procrustes_distance(a: npt.NDArray, b: npt.NDArray) -> float:
    """Procrustes (full) distance: min over rotation/scale/translation of
    ||A - B|| after optimal similarity alignment (Dryden & Mardia 2016,
    Prop. 2.5): d = sqrt(1 - 2*tr((B'A(AA')^{-1}AB')^{1/2}...) ) computed
    stably via the signed SVD trace with scale normalisation."""
    A = np.asarray(a, dtype=float) - np.asarray(a, dtype=float).mean(axis=0)
    B = np.asarray(b, dtype=float) - np.asarray(b, dtype=float).mean(axis=0)
    na = np.sqrt(np.sum(A**2))
    nb = np.sqrt(np.sum(B**2))
    if na <= np.finfo(float).eps or nb <= np.finfo(float).eps:
        raise MorphometricsError(_("Procrustes distance needs non-degenerate configurations"))
    A /= na
    B /= nb
    U, S, Vt = np.linalg.svd(A.T @ B)
    d2 = 1.0 + 1.0 - 2.0 * np.sum(S) * np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(max(d2, 0.0)))


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
        raise MorphometricsError(
            _("goodall_test expects (n, p, k) configurations and n group labels")
        )
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
        raise MorphometricsError(
            _("Hotelling T² needs n1+n2 > d+1 (got n1={0}, n2={1}, d={2})").format(n1, n2, d)
        )
    diff = X1.mean(axis=0) - X2.mean(axis=0)
    S = (((X1 - X1.mean(axis=0)).T @ (X1 - X1.mean(axis=0)))
         + ((X2 - X2.mean(axis=0)).T @ (X2 - X2.mean(axis=0)))) / (n1 + n2 - 2)
    sol = np.linalg.lstsq(S, diff, rcond=None)[0]
    t2 = float(n1 * n2 / (n1 + n2) * (diff @ sol))
    df1 = d
    df2 = n1 + n2 - d - 1
    f_stat = df2 / (df1 * (n1 + n2 - 2)) * t2
    p = float(_stats.f.sf(f_stat, df1, df2))
    return HotellingT2Result(
        t2=t2, f_statistic=f_stat, p_value=p, df1=df1, df2=df2, mean_difference=diff
    )
