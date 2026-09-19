# =============================================================================
# FILE: statistics/procD_lm.py
# =============================================================================
"""
Procrustes ANOVA linear models (geomorph ``procD.lm`` port).

Least-squares models fitted to Procrustes-aligned landmark configurations,
with sums of squares measured as *Procrustes distances* between each
observed configuration and its model fit rather than plain Euclidean
distances:

    RSS(M) = Σ_i procSS( Y_i | Ŷ_i(M) )

where procSS(Y|Ŷ) is the residual sum of squares after the optimal
(rotation+scale) Procrustes refit of the observation onto its fitted value.
Because refitting removes degrees of freedom, the residual df of the full
model is ``n - rank(design) - p + 1`` (Dryden & Mardia 2016 §8.2; Adams et
al. 2013, geomorph).

Significance of each term is assessed by permuting the error-design
units (specimen labels) — the residual permutation test of geomorph —
since Procrustes residuals are not independent Gaussian errors.

Reference:
    Adams, D.C., Collyer, M.L., Kaliontzopoulou, A., Buzi, C. (2022) —
    geomorph package documentation, procD.lm.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class ProcDLMTerm:
    """One model term in a Procrustes linear model table."""

    term: str
    ss: float
    df: int
    ms: float
    f_value: float | None
    p_value: float | None  # permutation p-value (None for intercept)


@dataclass
class ProcDLMResult:
    """Container for procD_lm output."""

    terms: list[ProcDLMTerm]
    residual_ss: float
    residual_df: int
    total_ss: float
    n_observations: int
    n_permutations: int
    r2_by_term: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = f"{'Term':<24}{'SS(Procrustes)':>16}{'Df':>6}{'MS':>14}{'F':>10}{'p(perm)':>10}\n"
        head += "-" * 80 + "\n"
        for t in self.terms:
            f_str = "NA" if t.f_value is None else f"{t.f_value:.4f}"
            p_str = "NA" if t.p_value is None else f"{t.p_value:.4f}"
            head += f"{t.term:<24}{t.ss:>16.6f}{t.df:>6}{t.ms:>14.6f}{f_str:>10}{p_str:>10}\n"
        head += (
            f"{'Residuals':<24}{self.residual_ss:>16.6f}{self.residual_df:>6}"
            f"{self.residual_ss / max(self.residual_df, 1):>14.6f}\n"
        )
        return head


def _proc_ss(y: npt.NDArray, yhat: npt.NDArray) -> npt.NDArray:
    """
    Per-observation Procrustes residual SS: ||Y - Ŷ @ R*||² minimised over
    rotations R (scale kept — geomorph scales fitted values once per model,
    not per specimen).  Vectorised closed form via SVD of Ŷ'Y per row.
    """
    n, p, k = y.shape
    out = np.zeros(n)
    for i in range(n):
        A = y[i]
        B = yhat[i]
        a2 = float(np.sum(A**2))
        b2 = float(np.sum(B**2))
        if b2 <= np.finfo(float).eps:
            out[i] = a2
            continue
        U, S, Vt = np.linalg.svd(B.T @ A)
        d = np.sign(np.linalg.det(U @ Vt))
        # max over rotations of tr(R' B'A) with det R = +1
        cross = float(np.sum(S) - (1.0 - d) * S[-1])
        ss = a2 + b2 - 2.0 * cross
        out[i] = max(ss, 0.0)
    return out


def _fit_flat(y_flat: npt.NDArray, design: npt.NDArray) -> npt.NDArray:
    fitted = np.linalg.lstsq(design, y_flat, rcond=None)[0]
    return design @ fitted


def procD_lm(
    configurations: npt.NDArray,
    design: npt.NDArray | dict[str, npt.NDArray],
    error_labels: npt.NDArray | None = None,
    n_permutations: int = 999,
    seed: int | None = None,
    names: list[str] | None = None,
) -> ProcDLMResult:
    """
    Fit a Procrustes least-squares model with sequential term SS and
    permutation tests.

    Parameters:
        configurations: (n, p, k) GPA-aligned configurations.
        design: (n, q) numeric design matrix (add an intercept column if
            wanted), or a dict {name: column} fitted in insertion order.
        error_labels: permutation units (e.g. specimen IDs for repeated
            measurements).  Defaults to one unit per observation.
        n_permutations: permutation replicates (0 skips the tests).
        seed: RNG seed for the permutations.
        names: optional term names when design is a matrix (excluding a
            leading intercept column named "Intercept" automatically).

    Returns:
        ProcDLMResult with sequential SS per term, Procrustes-corrected
        residual df, F ratios and permutation p-values.
    """
    X = np.asarray(configurations, dtype=float)
    if X.ndim != 3:
        raise DataValidationError(
            _("procD_lm expects (n, p, k) aligned configurations; got {0}D").format(X.ndim)
        )
    n, p, k = X.shape
    if isinstance(design, dict):
        term_names = list(design.keys())
        D = np.column_stack([np.asarray(design[t], dtype=float).reshape(n, -1) for t in term_names])
    else:
        D = np.asarray(design, dtype=float)
        if D.ndim != 2 or D.shape[0] != n:
            raise DataValidationError(_("design must be (n, q) with n={0}").format(n))
        if names is not None:
            term_names = list(names)
        else:
            is_intercept = D.shape[1] > 0 and np.allclose(D[:, 0], D[:, 0][0]) and np.all(D[:, 0] != 0)
            term_names = (["Intercept"] if is_intercept else []) + [
                f"X{i + 1}" for i in range(D.shape[1] - (1 if is_intercept else 0))
            ]
        if len(term_names) != D.shape[1]:
            term_names = [f"X{i + 1}" for i in range(D.shape[1])]
    if D.shape[1] >= n:
        raise DataValidationError(_("design has {0} columns for {1} observations").format(D.shape[1], n))

    rank_full = np.linalg.matrix_rank(D)
    y = X.reshape(n, p * k)

    def rss_of(sub_cols: list[int]) -> tuple[float, npt.NDArray]:
        Dsub = D[:, sub_cols]
        fit_flat = _fit_flat(y, Dsub)
        yhat = fit_flat.reshape(n, p, k)
        per_obs = _proc_ss(X, yhat)
        return float(per_obs.sum()), yhat

    # Full model residual (also needed for residual permutation)
    yhat_full = rss_of(list(range(D.shape[1])))[1]
    resid_ss_vec = _proc_ss(X, yhat_full)
    residual_ss = float(resid_ss_vec.sum())
    # Procrustes refit removes p-1 additional df per observation set
    # (Dryden & Mardia §8.2): df = n - rank - p + 1
    residual_df = n - rank_full - p + 1
    if residual_df <= 0:
        raise DataValidationError(
            _("procD_lm residual df <= 0 (n={0}, rank={1}, p={2}); need more specimens").format(
                n, rank_full, p
            )
        )
    resid_ms = residual_ss / residual_df

    # Sequential (Type I) SS over nested models
    terms: list[ProcDLMTerm] = []
    prev_cols: list[int] = []
    prev_rss = float(np.sum(_proc_ss(X, np.zeros_like(X))))  # null: fit = 0
    total_ss = prev_rss
    col_iter = 0
    for j, tname in enumerate(term_names):
        is_intercept = tname == "Intercept" and j == 0
        if is_intercept:
            # Intercept SS: null (zero) vs single common configuration.
            fit_flat = _fit_flat(y, np.ones((n, 1)))
            int_rss = float(np.sum(_proc_ss(X, fit_flat.reshape(n, p, k))))
            terms.append(
                ProcDLMTerm(
                    term=tname,
                    ss=prev_rss - int_rss,
                    df=1,
                    ms=prev_rss - int_rss,
                    f_value=None,
                    p_value=None,
                )
            )
            prev_rss = int_rss
        cols = list(range(col_iter, col_iter + 1))
        col_iter += 1
        if not is_intercept:
            full_rss = rss_of(prev_cols + cols)[0]
            term_ss = prev_rss - full_rss
            terms.append(
                ProcDLMTerm(term=tname, ss=term_ss, df=1, ms=term_ss, f_value=None, p_value=None)
            )
            prev_rss = full_rss
        prev_cols = prev_cols + cols

    # F statistics
    for t in terms:
        if t.term != "Intercept" and t.df > 0 and resid_ms > 0:
            t.f_value = (t.ss / t.df) / resid_ms

    # Permutation test: "residual" permutation in the sense of the
    # permutations package (geomorph default) — for each term the residuals
    # of the model WITHOUT that term (delete-one reduced model) are shuffled
    # among error units and added back to the reduced fit.  Permuting the
    # *full*-model residuals instead would keep the tested effect locked in
    # the fitted values and make every p-value ≈ 1.
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(seed)
        units = np.arange(n) if error_labels is None else np.asarray(error_labels)
        unit_ids = np.unique(units)
        all_cols = list(range(D.shape[1]))
        # per-term delete-one setup: reduced fit, reduced residuals, observed F
        test_stat: dict[int, tuple[npt.NDArray, npt.NDArray, float]] = {}
        for term_idx, tname in enumerate(term_names):
            if term_idx == 0 and tname == "Intercept":
                continue
            red_cols = [c for c in all_cols if c != term_idx]
            if red_cols:
                fit_red = _fit_flat(y, D[:, red_cols]).reshape(n, p, k)
            else:
                fit_red = np.zeros_like(X)
            rss_red = float(np.sum(_proc_ss(X, fit_red)))
            f_obs = ((rss_red - residual_ss) / resid_ms) if resid_ms > 0 else 0.0
            test_stat[term_idx] = (fit_red, X - fit_red, f_obs)
        counts = {term_names[j]: 0 for j in test_stat}
        for _step in range(n_permutations):
            perm_units = rng.permutation(unit_ids)
            mapping = dict(zip(unit_ids, perm_units))
            perm_idx = np.array([mapping[u] for u in units])
            for term_idx, (fit_red, resid_red, f_obs) in test_stat.items():
                y_perm = fit_red + resid_red[perm_idx]
                # refit both models on the permuted data (fixed design)
                red_cols = [c for c in all_cols if c != term_idx]
                if red_cols:
                    fit_red_p = _fit_flat(y_perm.reshape(n, p * k), D[:, red_cols]).reshape(n, p, k)
                else:
                    fit_red_p = np.zeros_like(y_perm)
                fit_full_p = _fit_flat(y_perm.reshape(n, p * k), D).reshape(n, p, k)
                rss_red_p = float(np.sum(_proc_ss(y_perm, fit_red_p)))
                rss_full_p = float(np.sum(_proc_ss(y_perm, fit_full_p)))
                f_perm = max(rss_red_p - rss_full_p, 0.0) / resid_ms if resid_ms > 0 else 0.0
                if f_perm >= f_obs - 1e-15:
                    counts[term_names[term_idx]] += 1
        for t in terms:
            if t.f_value is not None and t.term in counts:
                t.p_value = (1.0 + counts[t.term]) / (1.0 + n_permutations)

    result = ProcDLMResult(
        terms=terms,
        residual_ss=residual_ss,
        residual_df=residual_df,
        total_ss=total_ss,
        n_observations=n,
        n_permutations=int(n_permutations or 0),
    )
    result.r2_by_term = {
        t.term: (t.ss / (t.ss + residual_ss)) if (t.ss + residual_ss) > 0 else 0.0
        for t in terms
        if t.term != "Intercept"
    }
    return result
