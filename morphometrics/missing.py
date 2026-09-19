# =============================================================================
# FILE: morphometrics/missing.py
# =============================================================================
"""
Estimation of missing landmark coordinates.

Python port of geomorph's ``estimate.missing()`` (estimate.missing.r),
following Gunz et al. (2009): landmarks missing on a specimen are encoded
as NaN rows; two estimators reconstruct them in the ORIGINAL coordinate
frame of each specimen.

* ``method="tps"``  — align the incomplete specimen to the consensus of the
  complete specimens using only its observed landmarks (partial Procrustes
  similarity fit), warp the consensus with a thin-plate spline fitted on
  the observed landmarks, and read the missing points off the warped
  consensus (Bookstein et al. 1999).
* ``method="reg"``  — same partial alignment, then predict the missing
  coordinate block from the observed block by least-squares multivariate
  regression over the complete, GPA-aligned specimens (Gunz et al. 2009;
  the ridge-penalised lstsq solve also covers p > n systems).

References:
    Bookstein, F.L. et al. (1999) Anat. Rec. 257:217-224.
    Gunz, P., Mitteroecker, P., Neubauer, S., Weber, G.W., Bookstein, F.L.
        (2009) J. Hum. Evol. 57:48-62.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import MorphometricsError

from .gpa import GPAAnalyzer
from .tps import TPSAnalyzer

logger = logging.getLogger(__name__)

METHODS = ("tps", "reg")


@dataclass
class MissingEstimateResult:
    """Container for missing-landmark estimation output."""

    filled_configurations: npt.NDArray  # (n, p, k) original frame, NaN replaced
    estimated_mask: npt.NDArray  # (n, p) bool — landmarks that were estimated
    reference: npt.NDArray  # (p, k) consensus of the complete specimens
    aligned_filled: npt.NDArray  # (n, p, k) filled configs mapped to consensus frame
    method: str
    n_complete: int

    def summary(self) -> str:
        n_est = int(self.estimated_mask.any(axis=1).sum())
        return (
            f"{_('Missing landmark estimation ({0})').format(self.method)}\n"
            f"{'=' * 50}\n"
            f"{_('Complete specimens: {0}').format(self.n_complete)}\n"
            f"{_('Specimens with estimated landmarks: {0}').format(n_est)}\n"
            f"{_('Estimated landmarks (rows): {0}').format(int(self.estimated_mask.sum()))}"
        )


def _similarity_fit(source: npt.NDArray, target: npt.NDArray):
    """
    Least-squares similarity mapping source -> target (row-vector form):
    ``target ~ (source - c) * s @ R``.

    Returns (c, s, R); inverse: ``source = (mapped @ R.T) / s + c``.
    """
    c = source.mean(axis=0)
    tc = target.mean(axis=0)
    A = source - c
    B = target - tc
    norm_a = float(np.sqrt(np.sum(A**2)))
    if norm_a <= np.finfo(float).eps:
        raise MorphometricsError(_("Cannot align a specimen whose observed landmarks coincide"))
    H = A.T @ B
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError as e:
        raise MorphometricsError(
            _("SVD failed during partial Procrustes fit: {0}").format(e)
        )
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0] * (A.shape[1] - 1) + [d])
    R = Vt.T @ D @ U.T
    s = float(np.sum(S * np.diag(D)) / np.sum(A**2))
    return c, s, R


def estimate_missing(
    configurations: npt.NDArray | list[npt.NDArray],
    method: str = "tps",
) -> MissingEstimateResult:
    """
    Estimate missing landmark coordinates (NaN rows) per specimen.

    Parameters:
        configurations: (n, p, k) array of landmark configurations with
            missing landmarks encoded as all-NaN rows (k in {2, 3}).
        method: "tps" (consensus + thin-plate spline warp) or "reg"
            (multivariate least-squares regression on the observed block).

    Returns:
        MissingEstimateResult — the NaN rows are replaced by estimates in
        the ORIGINAL frame of each specimen; ``aligned_filled`` carries the
        same estimates mapped into the consensus frame.

    Raises:
        MorphometricsError: on malformed input, no complete specimen, or an
            incomplete specimen with too few observed landmarks to fit a
            similarity (< k+1 non-collinear points required in practice).
    """
    if method not in METHODS:
        raise MorphometricsError(_("Unknown method '{0}'; expected one of {1}").format(method, METHODS))

    X = np.asarray(configurations, dtype=float)
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] not in (2, 3):
        raise MorphometricsError(
            _("estimate_missing expects (n, p>=2, k in 2|3); got {0}").format(X.shape)
        )

    n, p, k = X.shape
    lm_missing = np.isnan(X).any(axis=2)  # (n, p)
    spec_missing = lm_missing.any(axis=1)
    complete = ~spec_missing
    n_complete = int(complete.sum())
    if n_complete == 0:
        raise MorphometricsError(_("No complete specimen available as reference"))
    if not spec_missing.any():
        logger.info("estimate_missing: no NaN landmarks found; returning input unchanged")
        return MissingEstimateResult(
            filled_configurations=X.copy(),
            estimated_mask=np.zeros((n, p), dtype=bool),
            reference=X.mean(axis=0),
            aligned_filled=X.copy(),
            method=method,
            n_complete=n_complete,
        )

    # Consensus of the complete specimens, in a GPA-aligned frame.
    gpa = GPAAnalyzer().analyze(X[complete])
    aligned_complete = gpa.aligned_configurations
    ref = gpa.consensus

    # Regression design for method="reg": aligned complete specimens, split
    # per missing pattern of the incomplete specimens.
    reg_cache: dict[tuple[bool, ...], tuple[npt.NDArray, npt.NDArray]] = {}

    filled = X.copy()
    aligned_filled = X.copy()
    aligned_filled[complete] = aligned_complete

    for i in np.where(spec_missing)[0]:
        obs = ~lm_missing[i]
        miss = np.where(lm_missing[i])[0]
        if obs.sum() < k + 1:
            raise MorphometricsError(
                _("Specimen {0}: only {1} observed landmarks; need at least {2} "
                  "for a partial Procrustes fit").format(i, int(obs.sum()), k + 1)
            )
        c, s, R = _similarity_fit(X[i][obs], ref[obs])

        def to_aligned(pts, c=c, s=s, R=R):
            return (pts - c) * s @ R

        def to_raw(pts, c=c, s=s, R=R):
            return pts @ R.T / s + c

        aligned_obs = to_aligned(X[i][obs])
        if method == "tps":
            analyzer = TPSAnalyzer()
            result = analyzer.analyze(source=ref[obs], target=aligned_obs)
            est_aligned = analyzer._warp_points(ref[miss], result)
        else:  # "reg"
            key = tuple(bool(v) for v in obs)
            if key not in reg_cache:
                Ymat = aligned_complete[:, miss, :].reshape(n_complete, -1)
                Xmat = np.column_stack(
                    [aligned_complete[:, obs, :].reshape(n_complete, -1), np.ones(n_complete)]
                )
                beta, _residuals, _rank, _sv = np.linalg.lstsq(Xmat, Ymat, rcond=None)
                reg_cache[key] = (miss, beta)
            miss_cached, beta = reg_cache[key]
            xrow = np.concatenate([aligned_obs.ravel(), [1.0]])
            est_aligned = (beta.T @ xrow).reshape(len(miss_cached), k)

        # Partial-fit the observed landmarks into the reference frame as
        # well; the NaN rows are then replaced by the estimates.
        specimen_aligned = to_aligned(np.where(lm_missing[i][:, None], 0.0, X[i]))
        specimen_aligned[miss] = est_aligned
        filled[i, miss] = to_raw(est_aligned)
        aligned_filled[i] = specimen_aligned

    return MissingEstimateResult(
        filled_configurations=filled,
        estimated_mask=lm_missing,
        reference=ref,
        aligned_filled=aligned_filled,
        method=method,
        n_complete=n_complete,
    )
