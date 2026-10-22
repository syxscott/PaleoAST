# =============================================================================
# FILE: utils/statistics_core.py
# =============================================================================
"""
Shared statistical primitives: seeding, permutation tests, and nonlinear fits.

WHY THIS MODULE EXISTS
~~~~~~~~~~~~~~~~~~~~~~
Three pieces of machinery were re-implemented at every call site, which is
what makes "add one more permutation test" expensive:

* **Seeding.** The same "seeded -> dedicated Generator, unseeded -> global
  ``np.random`` state" block appeared in at least four modules. ``stats/anosim.py``
  and ``stats/permanova.py`` carry it line for line, warning text included.
  ``ecology/null_models.py`` already does it correctly -- an isolated
  ``default_rng`` that never touches the legacy global stream, which
  ``tests/ecology/test_null_models.py::test_seeded_runs_are_reproducible_without_touching_global_state``
  asserts. The other sites were the ones still writing to global state.

* **Permutation tests.** Seven separate Monte-Carlo loops, each with its own
  RNG handling, its own ``(1 + count) / (n + 1)`` arithmetic and its own
  log line. Mantel, partial Mantel, contingency chi-square, K-means label
  tests and every future permutation test all need exactly this.

* **Nonlinear fitting.** ``curve_fit`` appeared once in the whole codebase
  (``macroevolution/diversity.py``), as a bare call with hard-coded bounds
  that returned a bare tuple: no covariance, no confidence intervals, no
  R-squared, no AICc and no result container. Growth models are a
  five-line-each problem *once a proper harness exists*.

The dependency direction is deliberate: ``stats/``, ``ecology/`` and
``stratigraphy/`` all already import from ``utils``, and this module imports
none of them, so it can be shared without a cycle.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from utils.exceptions import ComputationError, StatisticalError, ValidationError

__all__ = [
    "MISSING_GROUP",
    "NonlinearFitResult",
    "PermutationResult",
    "aic_from_log_likelihood",
    "aicc_from_log_likelihood",
    "fit_nonlinear",
    "gaussian_log_likelihood",
    "group_indices",
    "make_rng",
    "permutation_pvalue",
]

_logger = logging.getLogger(__name__)


class _MissingGroupLabel:
    """Singleton sentinel standing in for rows with no group label.

    A NaN cannot be used as the key: ``float("nan") != float("nan")``, so a
    dict keyed by NaN can never be looked up -- ``indices[float("nan")]``
    always misses, and every lookup allocates a different NaN. A singleton
    is compared by identity, so the key is retrievable, and it cannot
    collide with a real group label.
    """

    __slots__ = ()
    _instance: _MissingGroupLabel | None = None

    def __new__(cls) -> _MissingGroupLabel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<missing group label>"


MISSING_GROUP = _MissingGroupLabel()


# =============================================================================
# Random number generation
# =============================================================================


def resolve_seed(random_seed: int | None = None, *, context: str = "analysis") -> int:
    """The seed to use, as a number.

    ``np.random.default_rng()`` draws its own entropy, which cannot be
    recovered afterwards, so a result that reports "the seed used" has
    to know it. Drawing the seed here, as a number, keeps the two
    consistent: the generator is built from a seed the caller can be
    told, so an unseeded run is still not reproducible across runs --
    it is reproducible from the seed in the result.

    Parameters
    ----------
    random_seed:
        The seed the caller asked for, or None to draw one.
    context:
        Name used in the warning and log messages.

    Returns
    -------
    int
        The seed to build the generator from.
    """
    if random_seed is not None:
        return int(random_seed)
    warnings.warn(
        f"{context}: no ``random_seed`` supplied; the result is not "
        f"reproducible across runs. Pass ``random_seed=`` for a "
        f"deterministic result.",
        RuntimeWarning,
        stacklevel=3,
    )
    _logger.warning("%s: no random_seed supplied; result is not reproducible", context)
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def make_rng(random_seed: int | None = None, *, context: str = "analysis") -> np.random.Generator:
    """Return an isolated random generator for a stochastic analysis.

    The returned generator never reads from and never writes to the legacy
    global ``np.random`` stream. That is a deliberate departure from the
    older behaviour in ``stats/anosim.py`` and ``stats/permanova.py``, which
    fell back to ``rng = np.random`` when no seed was supplied. Drawing from
    the global stream meant that merely *running* an unseeded analysis
    perturbed the state that unrelated code -- and the caller's own data
    generation -- subsequently drew from, so two analyses could not be
    compared unless both were individually seeded.

    Pass ``random_seed`` for a reproducible result. Without one the draw is
    not reproducible, which is warned about, because a published p-value
    that will not reproduce is worse than a slow one.

    Parameters
    ----------
    random_seed:
        Seed, or ``None`` to draw fresh entropy.
    context:
        Name used in the warning and log messages, e.g. ``"ANOSIM"``.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(resolve_seed(random_seed, context=context))


# =============================================================================
# Permutation tests
# =============================================================================


@dataclass
class PermutationResult:
    """Outcome of a permutation test.

    Attributes:
        observed: The statistic measured on the real data.
        null_distribution: Every permuted statistic, for diagnostics and
            for plotting a null histogram.
        p_value: One-sided p-value, ``P(null >= observed)``.
        p_value_two_sided: Two-sided p-value based on the null's distance
            from its centre.
        n_permutations: How many permutations were run.
        n_significant: How many exceeded the observed statistic.
        random_seed: The seed used, or ``None``.
    """

    observed: float
    null_distribution: npt.NDArray
    p_value: float
    p_value_two_sided: float
    n_permutations: int
    n_significant: int
    random_seed: int | None = None

    @property
    def significant(self) -> bool:
        """Whether the observed statistic rejects at the 0.05 level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """One-line human-readable result."""
        return (
            f"observed = {self.observed:.4f}, "
            f"p = {self.p_value:.4f} "
            f"({self.n_significant}/{self.n_permutations} permutations "
            f"as or more extreme)"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, without the raw null distribution."""
        return {
            "observed": float(self.observed),
            "p_value": float(self.p_value),
            "p_value_two_sided": float(self.p_value_two_sided),
            "n_permutations": int(self.n_permutations),
            "n_significant": int(self.n_significant),
            "random_seed": self.random_seed,
            "significant": self.significant,
        }


def permutation_pvalue(
    observed: float,
    statistic_fn: Callable[[np.random.Generator], float],
    *,
    n_permutations: int = 999,
    random_seed: int | None = None,
    context: str = "permutation test",
    alternative: str = "greater",
) -> PermutationResult:
    """Run a permutation test and return the p-value with its null.

    ``statistic_fn`` receives the generator and is responsible for
    re-randomising whatever the test permutes -- group labels, one of two
    distance matrices, the rows of a design matrix -- and returning the
    resulting statistic. This module owns the generator and the p-value
    arithmetic, which is the part that was being re-derived per call site.

    The p-value uses the standard ``(1 + count) / (n + 1)`` form (Phipson &
    Smyth 2010): the leading 1 keeps the p-value strictly positive, so a
    test in which *no* permutation reaches the observed value reports
    ``1 / (n + 1)`` rather than 0. Reporting 0 would claim more evidence
    than 999 permutations can supply.

    Parameters
    ----------
    observed:
        Statistic of the real data.
    statistic_fn:
        Called once per permutation with the generator; returns a float.
    n_permutations:
        Number of permutations (>= 1).
    random_seed:
        Seed for reproducibility.
    context:
        Label used in warnings and logs.
    alternative:
        ``"greater"`` for the standard "null is at least as extreme as
        observed" test, or ``"two_sided"`` to also count permutations
        falling on the opposite side.

    Returns
    -------
    PermutationResult

    Raises
    ------
    ValidationError
        If ``n_permutations`` is not positive or ``alternative`` is unknown.
    """
    if n_permutations < 1:
        raise ValidationError(f"{context}: n_permutations must be >= 1, got {n_permutations}")
    if alternative not in ("greater", "two_sided"):
        raise ValidationError(f"{context}: alternative must be 'greater' or 'two_sided', got {alternative!r}")

    seed = resolve_seed(random_seed, context=context)
    rng = make_rng(seed, context=context)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = statistic_fn(rng)

    # A permutation whose statistic is undefined -- a zero-variance group
    # pair gives 0/0 -- is not evidence against the null, and it used to
    # be counted in the denominator while never counting as "at least
    # as extreme". Worse, one NaN made the two-sided centre NaN, so
    # every comparison was False and the two-sided p-value collapsed to
    # 1/(n+1): a p-value of 0.001 for a test that found nothing.
    finite = np.isfinite(null)
    n_valid = int(np.sum(finite))
    if n_valid == 0:
        raise ComputationError(
            f"{context}: every permutation produced an undefined statistic, "
            "so there is no null distribution to compare against."
        )
    valid = null[finite]

    n_ge = int(np.sum(valid >= observed))
    p_value = (1.0 + n_ge) / (n_valid + 1.0)

    if alternative == "two_sided":
        centre = float(np.mean(valid))
        n_far = int(np.sum(np.abs(valid - centre) >= abs(observed - centre)))
        p_two = (1.0 + n_far) / (n_valid + 1.0)
    else:
        p_two = p_value

    return PermutationResult(
        observed=float(observed),
        null_distribution=null,
        p_value=float(p_value),
        p_value_two_sided=float(p_two),
        n_permutations=n_permutations,
        n_significant=n_ge,
        random_seed=seed,
    )


# =============================================================================
# Grouping helpers
# =============================================================================


def ordered_unique_labels(labels: Sequence[Any]) -> list[Any]:
    """The distinct labels, in a defined order, tolerating mixed types.

    A group column read from a spreadsheet can hold a number in one cell
    and text in the next -- the same column, one value typed as "1" and
    one typed as "A".  ``sorted(set(labels))`` then raises a bare
    ``TypeError: '<' not supported between instances of 'str' and 'int'``
    from inside the sort, which escapes the project's exception
    hierarchy and, in the GUI, the unified error handling with it.

    Homogeneous labels keep their natural order -- numbers stay numeric,
    strings stay lexicographic -- so nothing about the existing output
    changes.  Only a column that genuinely mixes types falls back to
    ordering by the string form, where the order is arbitrary anyway.

    Parameters
    ----------
    labels:
        The labels to reduce.

    Returns
    -------
    list
        The distinct labels, in order.
    """
    unique = set(labels)
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda x: str(x))


def group_indices(groups: Sequence[Any]) -> dict[Any, npt.NDArray]:
    """Map each group label to the row positions carrying it.

    Sorting and masking by group is written out by hand in at least four
    places (``stats/lda.py``, ``stats/permanova.py``, ``stats/univariate.py``,
    ``stats/simper.py``), each with its own habit for comparing a string
    array to a label. Rows whose label is missing are collected under the
    ``MISSING_GROUP`` sentinel rather than dropped, because a missing group
    label is a data problem the caller should see as its own group instead
    of quietly losing rows from every group.

    Parameters
    ----------
    groups:
        Group label per row.

    Returns
    -------
    dict
        Label -> sorted array of integer positions. Missing labels map to
        the ``MISSING_GROUP`` singleton.
    """
    array = np.asarray(groups, dtype=object)
    buckets: dict[Any, list[int]] = {}
    for i, label in enumerate(array):
        if isinstance(label, float) and np.isnan(label):
            key: Any = MISSING_GROUP
        else:
            key = label
        buckets.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in buckets.items()}


# =============================================================================
# Information criteria
# =============================================================================


def aic_from_log_likelihood(log_likelihood: float, n_params: int) -> float:
    """AIC from a log-likelihood (Burnham & Anderson 2002, eq. 2.2.1)."""
    if n_params <= 0:
        raise StatisticalError("n_params must be a positive integer")
    return float(-2.0 * log_likelihood + 2.0 * n_params)


def aicc_from_log_likelihood(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """AICc from a log-likelihood (Burnham & Anderson 2002, eq. 2.2.2).

    The small-sample correction matters here: a von Bertalanffy curve
    fitted to a handful of measurements has ``n - k - 1`` close to zero, and
    plain AIC rewards the extra parameters that its sibling models
    estimated.

    Raises
    ------
    StatisticalError
        If ``n_obs - n_params - 1 <= 0``, i.e. there is not enough data for
        the correction to be defined.
    """
    if n_params <= 0:
        raise StatisticalError("n_params must be a positive integer")
    if n_obs <= 0:
        raise StatisticalError("n_obs must be a positive integer")
    if n_obs - n_params - 1 <= 0:
        raise StatisticalError(f"AICc is undefined: n_obs ({n_obs}) must exceed n_params + 1 ({n_params + 1})")
    aic = aic_from_log_likelihood(log_likelihood, n_params)
    return float(aic + (2.0 * n_params * (n_params + 1)) / (n_obs - n_params - 1))


def gaussian_log_likelihood(residuals: npt.NDArray) -> float:
    """Gaussian log-likelihood of a residual vector, up to a constant.

    Uses the same form ``scipy.stats.linregress``-style fits imply, so AIC
    computed here is comparable with AIC computed from an explicit
    likelihood elsewhere.
    """
    rss = float(np.sum(np.asarray(residuals, dtype=float) ** 2))
    n = len(residuals)
    if n == 0 or rss <= 0:
        return float("-inf")
    return float(-0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0))


# =============================================================================
# Nonlinear least squares
# =============================================================================


@dataclass
class NonlinearFitResult:
    """A fitted nonlinear model with the diagnostics a paper needs.

    Attributes:
        name: Model name, e.g. ``"von Bertalanffy"``.
        param_names: One label per parameter.
        params: Fitted parameter values.
        covariance: Parameter covariance matrix, or ``None`` when the fit
            did not converge well enough to estimate one.
        stderr: Standard errors, the square root of the covariance
            diagonal.
        ci_lower / ci_upper: 95 % parameter confidence interval.
        r_squared: Coefficient of determination.
        adj_r_squared: R-squared adjusted for parameter count.
        rmse: Root mean squared residual.
        aic / aicc: Information criteria. ``aicc`` is ``inf`` when there is
            too little data for the correction.
        n_obs / n_params: Sample and parameter counts.
        success: Whether the optimiser reported convergence.
        message: Optimiser message.
        fitted: Model evaluated at ``x``.
        residuals: ``y - fitted``.
        converged: Alias of ``success``, kept because callers phrase this
            as "did it converge".
    """

    name: str
    param_names: tuple[str, ...]
    params: npt.NDArray
    n_obs: int
    n_params: int
    r_squared: float
    adj_r_squared: float
    rmse: float
    aic: float
    aicc: float
    success: bool
    message: str
    fitted: npt.NDArray
    residuals: npt.NDArray
    covariance: npt.NDArray | None = None
    stderr: npt.NDArray | None = None
    ci_lower: npt.NDArray | None = None
    ci_upper: npt.NDArray | None = None
    converged: bool = field(default=False)

    def __post_init__(self) -> None:
        if not self.converged:
            self.converged = self.success

    def param(self, name: str) -> float:
        """Look up one fitted parameter by name.

        Raises
        ------
        KeyError
            If the name was not a parameter of this model.
        """
        try:
            idx = self.param_names.index(name)
        except ValueError:
            raise KeyError(f"{self.name}: no parameter {name!r}; available: {', '.join(self.param_names)}") from None
        return float(self.params[idx])

    def summary(self) -> str:
        """Multi-line human-readable fit summary."""
        lines = [f"{self.name} fit:"]
        for i, pname in enumerate(self.param_names):
            line = f"  {pname} = {self.params[i]:.6g}"
            if self.stderr is not None and np.isfinite(self.stderr[i]):
                line += f" +/- {self.stderr[i]:.3g}"
            if self.ci_lower is not None and np.isfinite(self.ci_lower[i]):
                line += f"  [{self.ci_lower[i]:.4g}, {self.ci_upper[i]:.4g}]"
            lines.append(line)
        lines.append(f"  R^2 = {self.r_squared:.4f} (adj {self.adj_r_squared:.4f}), RMSE = {self.rmse:.4g}")
        lines.append(f"  AIC = {self.aic:.2f}, AICc = {self.aicc:.2f}")
        if not self.success:
            lines.append(f"  WARNING: optimiser did not converge ({self.message})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, without the arrays."""
        out: dict[str, Any] = {
            "name": self.name,
            "param_names": list(self.param_names),
            "params": [float(p) for p in self.params],
            "n_obs": int(self.n_obs),
            "n_params": int(self.n_params),
            "r_squared": float(self.r_squared),
            "adj_r_squared": float(self.adj_r_squared),
            "rmse": float(self.rmse),
            "aic": float(self.aic),
            "aicc": float(self.aicc),
            "success": bool(self.success),
            "message": self.message,
        }
        if self.stderr is not None:
            out["stderr"] = [float(s) for s in self.stderr]
        if self.ci_lower is not None:
            out["ci_lower"] = [float(c) for c in self.ci_lower]
            out["ci_upper"] = [float(c) for c in self.ci_upper]
        return out


def fit_nonlinear(
    func: Callable[..., npt.NDArray],
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    p0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] = (-np.inf, np.inf),
    param_names: Sequence[str] | None = None,
    name: str = "nonlinear model",
    sigma: npt.NDArray | None = None,
    confidence: float = 0.95,
) -> NonlinearFitResult:
    """Fit ``y = func(x, *params)`` and return parameters with diagnostics.

    This is the harness the growth models are built on. It exists because
    the single previous ``curve_fit`` call site returned a bare tuple of
    numbers, which makes model comparison impossible: you cannot rank a
    von Bertalanffy fit against a Gompertz fit without their log-likelihoods,
    and you cannot report a growth rate without its standard error.

    ``absolute_sigma`` is set only when the caller actually supplies
    ``sigma``.  With known observation standard deviations the covariance is
    already in parameter units; without them, scipy must scale the
    covariance by the reduced chi-square, and skipping that scaling is what
    made every confidence interval on this harness too narrow by a factor
    of sqrt(RSS/(M-N)).

    Parameters
    ----------
    func:
        Model, called as ``func(x, *params)``.
    x, y:
        Observations.
    p0:
        Starting parameter values.
    bounds:
        ``(lower, upper)`` arrays, as ``scipy.optimize.curve_fit`` wants.
    param_names:
        Labels; defaults to ``p0``, ``p1``, ...
    name:
        Model name carried into the result.
    sigma:
        Known observation standard deviations, if any.
    confidence:
        Level for the parameter interval, default 0.95.

    Returns
    -------
    NonlinearFitResult

    Raises
    ------
    ValidationError
        If the shapes or the starting values are wrong.
    ComputationError
        If the optimiser fails to produce a finite fit.
    """
    from scipy.optimize import curve_fit

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValidationError(f"{name}: x and y must be one-dimensional, got {x_arr.ndim}-d and {y_arr.ndim}-d")
    if len(x_arr) != len(y_arr):
        raise ValidationError(f"{name}: x and y must have equal length, got {len(x_arr)} and {len(y_arr)}")
    n_obs = len(x_arr)
    p0_arr = np.asarray(p0, dtype=float)
    n_params = len(p0_arr)
    if n_params == 0:
        raise ValidationError(f"{name}: p0 must supply at least one parameter")
    names = tuple(param_names) if param_names else tuple(f"p{i}" for i in range(n_params))
    if len(names) != n_params:
        raise ValidationError(f"{name}: got {len(names)} param_names for {n_params} parameters")

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite], y_arr[finite]
    n_obs = len(x_arr)
    if n_obs < n_params:
        raise ValidationError(f"{name}: need at least as many observations ({n_obs}) as parameters ({n_params})")

    try:
        popt, pcov = curve_fit(
            func,
            x_arr,
            y_arr,
            p0=p0_arr,
            bounds=bounds,
            sigma=sigma,
            # ``absolute_sigma`` tells scipy whether ``sigma`` is the ABSOLUTE
            # standard deviation of the observations (True) or merely their
            # RELATIVE weights (False).  With absolute_sigma=True the returned
            # covariance is used as-is; with False, scipy scales it by the
            # reduced chi-square RSS/(M-N).
            #
            # The previous code hard-coded True while ``sigma`` was None.
            # scipy then read "the measurement error of every point is exactly
            # 1.0" and returned an UNSCALED covariance, so every standard error
            # and 95% interval came out too narrow by sqrt(RSS/(M-N)) -- 3.3x
            # too narrow in the von Bertalanffy case below, i.e. the interval
            # that goes into the paper.  The old comment claimed the opposite
            # ("without it ... intervals that are too narrow"), which is what
            # made the bug so easy to miss.
            #
            # Correct rule: scale by the residual variance unless the caller
            # actually supplied an absolute sigma.
            absolute_sigma=sigma is not None,
            maxfev=20000,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ComputationError(f"{name}: nonlinear least squares failed: {exc}") from exc

    fitted = np.asarray(func(x_arr, *popt), dtype=float)
    residuals = y_arr - fitted
    rss = float(np.sum(residuals**2))
    if not np.isfinite(rss):
        raise ComputationError(f"{name}: fit produced non-finite residuals")

    tss = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params - 1) if n_obs - n_params - 1 > 0 else float("nan")
    )
    rmse = float(np.sqrt(rss / n_obs)) if n_obs else float("nan")

    log_likelihood = gaussian_log_likelihood(residuals)
    aic = aic_from_log_likelihood(log_likelihood, n_params)
    try:
        aicc = aicc_from_log_likelihood(log_likelihood, n_params, n_obs)
    except StatisticalError:
        # Too few observations for the correction. Not an error here: the
        # fit itself succeeded, and reporting inf is more honest than
        # failing a whole analysis over a model-selection statistic.
        aicc = float("inf")

    stderr: npt.NDArray | None = None
    ci_lower: npt.NDArray | None = None
    ci_upper: npt.NDArray | None = None
    if pcov is not None and np.all(np.isfinite(pcov)):
        diag = np.diag(pcov)
        if np.all(diag >= 0):
            stderr = np.sqrt(diag)
            if n_obs - n_params - 1 > 0:
                t_crit = float(sp_stats.t.ppf(0.5 + confidence / 2.0, n_obs - n_params - 1))
                ci_lower = popt - t_crit * stderr
                ci_upper = popt + t_crit * stderr

    return NonlinearFitResult(
        name=name,
        param_names=names,
        params=popt,
        n_obs=n_obs,
        n_params=n_params,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        rmse=rmse,
        aic=float(aic),
        aicc=float(aicc),
        success=True,
        message="ok",
        fitted=fitted,
        residuals=residuals,
        covariance=pcov,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
