# =============================================================================
# FILE: ecology/null_models.py
# =============================================================================
"""
Null Model Analysis for PaleoAST

Implements EcoSim-style null model analysis for testing co-occurrence patterns.

Stone, L. & Roberts, A. (1990). The checkerboard score: tests of
，刷到欠漏和物种组合随机性的统计方法.

Gotelli, N.J. & Entsminger, G.L. (2001). EcoSim: null models software for
ecology. Aquatic Sciences, 63(1), 5-11.

Mathematical Foundation:
==============================================================================

C-score (Checkerboard Score):
    C-score(i,j) = (r_i - 1) * (r_j - 1)

    where r_i = total occurrences of species i

    For a matrix, the observed C-score is the average
    over all species pairs.

Null Model Algorithms (Gotelli 2000; Gotelli & Entsminger 2001):
    1. "swap"    - fixed-fixed: random 2x2 exchanges keep *both* the row
                   (species frequency) and column (site richness) totals.
    2. "rrs"     - random rows: each species keeps its occurrences, they are
                   re-placed among the sites; row totals fixed, column
                   totals free.
    3. "rcs"     - random columns: the transpose of "rrs"; column totals
                   (site richness) fixed, row totals free.
    4. "shuffle" - unconstrained: all cells pooled and re-dealt, only the
                   grand total is preserved.

Significance:
    SES = (observed - mean_simulated) / std_simulated
    p-value = (1 + #{simulated >= observed}) / (n_simulated + 1)

    The +1 correction (Waller & Gotelli 2000) keeps a Monte-Carlo p-value
    strictly positive and counts the observed matrix as one of the
    randomizations.  ``n_permutations`` in the result is the number of
    randomizations *actually performed*, which can differ from the request
    when the parallel chunks are trimmed.  A degenerate null (zero variance
    in the simulated scores, e.g. a single-species-realistic matrix) makes
    the SES undefined: it is reported as NaN - never as 0, which would
    falsely read as "observed equals expected" - and a warning is logged.

Author: PaleoAST Development Team
version: 1.0.2
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: Randomization algorithms understood by :class:`NullModelAnalyzer`.
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("swap", "rrs", "rcs", "shuffle")


# =============================================================================
# Result Class
# =============================================================================


@dataclass
class NullModelResult:
    """
    Container for null model analysis results.

    Attributes:
        observed_score: Observed co-occurrence index value
        simulated_scores: Distribution of simulated index values
        mean_simulated: Mean of simulated values
        std_simulated: Standard deviation of simulated values
        standardized_effect_size: SES = (observed - mean) / std; NaN when the
            randomisations have zero variance (SES is then undefined)
        p_value: (1 + #{simulated >= observed}) / (n + 1)
        n_permutations: Number of permutations actually performed
        algorithm: Null model algorithm used
        metric: Co-occurrence metric used
        n_species: Number of species
        n_sites: Number of sites
    """

    observed_score: float
    simulated_scores: npt.NDArray[np.float64]
    mean_simulated: float
    std_simulated: float
    standardized_effect_size: float
    p_value: float
    n_permutations: int
    algorithm: str
    metric: str
    n_species: int
    n_sites: int

    def summary(self) -> str:
        """Generate summary text."""
        if self.p_value < 0.001:
            sig = "***"
        elif self.p_value < 0.01:
            sig = "**"
        elif self.p_value < 0.05:
            sig = "*"
        else:
            sig = ""

        if not np.isfinite(self.standardized_effect_size):
            # Zero-variance null: SES carries no information, and printing
            # "0.0000" would wrongly suggest observed == expected.
            ses_text = _("SES: n/a (zero-variance null)")
            interpretation = _("Not defined")
        else:
            ses_text = _("SES: {0:.4f}").format(self.standardized_effect_size)
            if self.standardized_effect_size > 2:
                interpretation = _("Aggregation")
            elif self.standardized_effect_size < -2:
                interpretation = _("Segregation")
            else:
                interpretation = _("Random")

        return (
            f"{_('Null Model Analysis')}\n"
            f"{'=' * 50}\n"
            f"{_('Metric: {0}').format(self.metric.upper())}\n"
            f"{_('Algorithm: {0}').format(self.algorithm.upper())}\n"
            f"{_('Species: {0}, Sites: {1}').format(self.n_species, self.n_sites)}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"\n"
            f"{_('Observed score: {0:.4f}').format(self.observed_score)}\n"
            f"{_('Mean simulated: {0:.4f}').format(self.mean_simulated)}\n"
            f"{_('Std simulated: {0:.4f}').format(self.std_simulated)}\n"
            f"{ses_text}\n"
            f"{_('P-value: {0:.4f} {1}').format(self.p_value, sig)}\n"
            f"\n"
            f"{_('Interpretation: {0}').format(interpretation)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "observed_score": self.observed_score,
            "simulated_scores": self.simulated_scores.tolist(),
            "mean_simulated": self.mean_simulated,
            "std_simulated": self.std_simulated,
            "standardized_effect_size": self.standardized_effect_size,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "n_species": self.n_species,
            "n_sites": self.n_sites,
            "summary": self.summary(),
        }


# =============================================================================
# Main Analyzer Class
# =============================================================================


class NullModelAnalyzer:
    """
    EcoSim-style null model analysis for co-occurrence patterns.

    Tests whether observed co-occurrence patterns differ significantly from
    random expectation using Monte Carlo simulations.

    Example:
        >>> analyzer = NullModelAnalyzer()
        >>> # Binary presence/absence matrix (species x sites)
        >>> presence = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]])
        >>> result = analyzer.analyze(presence, n_permutations=9999)
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize null model analyzer."""
        self._logger = logging.getLogger(f"{__name__}.NullModelAnalyzer")
        self._lock = threading.RLock()
        self._last_result: NullModelResult | None = None

    @property
    def last_result(self) -> NullModelResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze(
        self,
        presence_matrix: npt.NDArray,
        metric: str = "c_score",
        n_permutations: int = 9999,
        algorithm: str = "swap",
        n_workers: int | None = None,
        random_seed: int | None = None,
        progress_callback: Any | None = None,
    ) -> NullModelResult:
        """
        Perform null model analysis.

        Parameters:
            presence_matrix: Binary presence/absence matrix (n_species, n_sites)
            metric: Co-occurrence metric ("c_score", "checkerboard", "combo")
            n_permutations: Number of randomizations
            algorithm: Randomization algorithm ("swap", "rrs", "rcs",
                "shuffle"); anything else is rejected instead of silently
                falling back to another algorithm
            n_workers: Number of parallel workers (None = sequential)
            random_seed: Seed for a private ``numpy.random.Generator``; the
                global ``numpy.random`` state is deliberately left untouched
                so that a null-model run cannot perturb other analyses

        Returns:
            NullModelResult with observed and simulated statistics

        Raises:
            ValidationError: If input data, metric or algorithm is invalid
        """
        with self._lock:
            self._logger.info(f"Null model analysis: metric={metric}, n_perm={n_permutations}")

            # Validate input
            presence_matrix = np.asarray(presence_matrix, dtype=np.int32)
            if presence_matrix.ndim != 2:
                raise ValidationError(_("Presence matrix must be 2D (species x sites)"))

            n_species, n_sites = presence_matrix.shape

            if n_species < 2:
                raise ValidationError(_("Need at least 2 species for null model analysis"))

            if n_sites < 2:
                raise ValidationError(_("Need at least 2 sites for null model analysis"))

            if algorithm not in SUPPORTED_ALGORITHMS:
                raise ValidationError(
                    _("Unknown algorithm: {0}. Use 'swap', 'rrs', 'rcs', or 'shuffle'").format(algorithm)
                )

            if int(n_permutations) < 1:
                raise ValidationError(_("Number of permutations must be at least 1"))
            n_permutations = int(n_permutations)

            # Compute observed score
            if metric == "c_score":
                observed = self._compute_c_score(presence_matrix)
            elif metric == "checkerboard":
                observed = self._compute_checkerboard(presence_matrix)
            elif metric == "combo":
                observed = self._compute_combo_score(presence_matrix)
            else:
                raise ValidationError(
                    _("Unknown metric: {0}. Use 'c_score', 'checkerboard', or 'combo'").format(metric)
                )

            rng = np.random.default_rng(random_seed)

            if n_workers is not None and n_workers > 1:
                simulated = self._run_parallel(
                    presence_matrix, n_permutations, algorithm, n_workers, metric, rng
                )
            else:
                simulated = self._run_sequential(
                    presence_matrix, n_permutations, algorithm, metric, progress_callback, rng
                )

            # Statistics over the randomisations that were actually run:
            # the parallel path trims its chunks to n_permutations, and a
            # multiprocessing failure may fall back to fewer replicates.
            simulated = np.asarray(simulated, dtype=np.float64).reshape(-1)
            n_simulated = int(simulated.size)
            if n_simulated == 0:
                raise ValidationError(_("No randomization could be performed"))

            mean_sim = float(np.mean(simulated))
            std_sim = float(np.std(simulated))
            if std_sim > 0:
                ses: float = (observed - mean_sim) / std_sim
            else:
                # SES is undefined for a degenerate null; 0.0 would read as
                # "observed equals expected", which is a different claim.
                ses = float("nan")
                self._logger.warning(
                    "Zero variance in the %d simulated %s scores under '%s': "
                    "standardized effect size is undefined.",
                    n_simulated, metric, algorithm,
                )
            p_value = float((1 + np.sum(simulated >= observed)) / (n_simulated + 1))

            result = NullModelResult(
                observed_score=observed,
                simulated_scores=simulated,
                mean_simulated=mean_sim,
                std_simulated=std_sim,
                standardized_effect_size=ses,
                p_value=p_value,
                n_permutations=n_simulated,
                algorithm=algorithm,
                metric=metric,
                n_species=n_species,
                n_sites=n_sites,
            )

            self._last_result = result
            self._logger.info(f"Null model: observed={observed:.4f}, SES={ses:.4f}, p={p_value:.4f}")
            return result

    def _compute_score(self, matrix: npt.NDArray, metric: str) -> float:
        """Dispatch to the correct co-occurrence metric function."""
        if metric == "checkerboard":
            return self._compute_checkerboard(matrix)
        elif metric == "combo":
            return self._compute_combo_score(matrix)
        else:
            return self._compute_c_score(matrix)

    def _run_sequential(
        self,
        matrix: npt.NDArray,
        n_permutations: int,
        algorithm: str,
        metric: str = "c_score",
        progress_callback: Any | None = None,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Run permutations sequentially with optional progress callback.

        Each replicate is randomised from the observed matrix, so the
        simulated scores are independent draws rather than a correlated
        Markov chain of the previous replicate.
        """
        simulated = np.zeros(n_permutations)
        report_interval = max(1, n_permutations // 100)

        for i in range(n_permutations):
            permuted = self._permute_matrix(matrix, algorithm, rng)
            if len(permuted) > 0:
                simulated[i] = self._compute_score(permuted, metric)
            if progress_callback and (i + 1) % report_interval == 0:
                progress_callback((i + 1) / n_permutations)

        return simulated

    def _run_parallel(
        self,
        matrix: npt.NDArray,
        n_permutations: int,
        algorithm: str,
        n_workers: int,
        metric: str = "c_score",
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Run permutations in parallel using multiprocessing.

        The replicates are split into ``ceil(n_permutations / n_workers)``
        chunks per worker (ceiling, so that ``n_workers`` chunks always
        cover the request instead of silently dropping the remainder), and
        the concatenated result is trimmed to exactly ``n_permutations``.
        Worker streams come from :class:`numpy.random.SeedSequence`
        children of one parent seed, which is the documented way to get
        statistically independent streams (in contrast to ``base + i``,
        a set of near-identical seeds).
        """
        n_workers = max(1, int(n_workers))
        chunk_size = max(1, -(-int(n_permutations) // n_workers))
        n_chunks = min(n_workers, -(-int(n_permutations) // chunk_size))

        parent_seed = (
            int(rng.integers(0, 2**31, dtype=np.uint64))
            if rng is not None
            else int(np.random.randint(0, 2**31))
        )
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in np.random.SeedSequence(parent_seed).spawn(n_chunks)]

        counts = []
        remaining = int(n_permutations)
        for _ in range(n_chunks):
            count = min(chunk_size, remaining)
            counts.append(count)
            remaining -= count

        try:
            from multiprocessing import Pool

            args_list = [
                (np.ascontiguousarray(matrix), counts[i], algorithm, metric, seeds[i])
                for i in range(n_chunks)
            ]

            with Pool(processes=n_chunks) as pool:
                results = pool.starmap(_worker_permute, args_list)

            simulated = np.concatenate(results)
            return simulated[: n_permutations]

        except ImportError:
            self._logger.warning("Multiprocessing not available, running sequentially")
            return self._run_sequential(matrix, n_permutations, algorithm, metric, None, rng)
        except Exception as e:  # pragma: no cover - platform dependent
            self._logger.warning("Parallel randomization failed (%s), running sequentially", e)
            return self._run_sequential(matrix, n_permutations, algorithm, metric, None, rng)

    def _permute_matrix(
        self,
        matrix: npt.NDArray,
        algorithm: str,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        """Apply permutation algorithm to matrix.

        Raises:
            ValidationError: for an unknown algorithm. The previous
                implementation silently substituted a ``shuffle`` (only the
                grand total fixed), so asking for a constrained null model
                quietly produced an unconstrained one.
        """
        return _permute_with_algorithm(matrix, algorithm, rng)

    def _shuffle_matrix(
        self, matrix: npt.NDArray, rng: np.random.Generator | None = None
    ) -> npt.NDArray:
        """Simple random shuffle of matrix elements (grand total fixed)."""
        return _shuffle_matrix_impl(matrix, rng)

    def _swap_matrix(
        self, matrix: npt.NDArray, rng: np.random.Generator | None = None
    ) -> npt.NDArray:
        """
        Swap algorithm - preserves row and column sums.

        Randomly select two rows and two columns. A valid fixed-fixed
        swap only toggles checkerboard submatrices:
        [[1, 0], [0, 1]] <-> [[0, 1], [1, 0]].

        Delegates to :func:`_swap_matrix_impl` so that the sequential and
        the multiprocessing path cannot drift apart (they used to differ in
        the number of swap attempts, which zeroed out small matrices).

        ``rng=None`` keeps the legacy global ``numpy.random`` stream.
        """
        return _swap_matrix_impl(matrix, rng)

    def _randomize_rows(
        self, matrix: npt.NDArray, rng: np.random.Generator | None = None
    ) -> npt.NDArray:
        """Gotelli "random rows": permute the cells within each row."""
        return _randomize_rows_impl(matrix, rng)

    def _randomize_columns(
        self, matrix: npt.NDArray, rng: np.random.Generator | None = None
    ) -> npt.NDArray:
        """Gotelli "random columns": permute the cells within each column."""
        return _randomize_columns_impl(matrix, rng)

    def _compute_c_score(self, matrix: npt.NDArray) -> float:
        """
        Compute C-score for presence/absence matrix.

        The Stone & Roberts (1990) C-score for a species pair (i, j) is:

            C_ij = (r_i - S_ij) * (r_j - S_ij)

        where ``r_i`` and ``r_j`` are the total number of sites occupied
        by species i and j, and ``S_ij`` is the number of sites where the
        two species *co-occur*. The score therefore measures the degree
        of checkerboard structure between the two species — it is
        maximised when the species never co-occur and small when they
        share many sites.

        The previous implementation used ``(r_i - 1)(r_j - 1)`` and
        ignored ``S_ij`` entirely, so the C-score reduced to a function
        of marginal species richness only and carried no information
        about co-occurrence patterns. Use the canonical formula here.
        """
        n_species, _n_sites = matrix.shape
        # Binary presence/absence row sums = number of occupied sites.
        pa = (matrix > 0).astype(np.int64)
        row_sums = pa.sum(axis=1)

        c_scores = []
        for i in range(n_species):
            for j in range(i + 1, n_species):
                # Number of sites where BOTH species i and j occur.
                s_ij = int(np.sum(pa[i] & pa[j]))
                c_ij = float((row_sums[i] - s_ij) * (row_sums[j] - s_ij))
                c_scores.append(c_ij)

        return float(np.mean(c_scores)) if c_scores else 0.0

    def _compute_checkerboard(self, matrix: npt.NDArray) -> float:
        """
        Compute checkerboard count.

        Counts the number of perfect checkerboard patterns:
        [[1, 0], [0, 1]] or [[0, 1], [1, 0]]
        """
        n_species, n_sites = matrix.shape
        checkerboards = 0

        for i in range(n_species):
            for j in range(i + 1, n_species):
                for k in range(n_sites):
                    for l in range(k + 1, n_sites):
                        # Check for checkerboard pattern
                        if (matrix[i, k] == 1 and matrix[i, l] == 0 and matrix[j, k] == 0 and matrix[j, l] == 1) or (
                            matrix[i, k] == 0 and matrix[i, l] == 1 and matrix[j, k] == 1 and matrix[j, l] == 0
                        ):
                            checkerboards += 1

        return float(checkerboards)

    def _compute_combo_score(self, matrix: npt.NDArray) -> float:
        """
        Compute combined score = normalized C-score + checkerboard.

        Combines both metrics for a more robust test.
        """
        c_score = self._compute_c_score(matrix)
        checkerboard = self._compute_checkerboard(matrix)

        # Normalize checkerboard to similar scale as C-score
        n_species, n_sites = matrix.shape
        max_checkerboard = n_species * (n_species - 1) / 2 * n_sites * (n_sites - 1) / 2
        norm_checkerboard = checkerboard / max_checkerboard if max_checkerboard > 0 else 0

        return (c_score + norm_checkerboard) / 2


# =============================================================================
# Randomization primitives (shared by the sequential and the worker path)
# =============================================================================


def _swap_matrix_impl(matrix: npt.NDArray, rng: np.random.Generator | None = None) -> npt.NDArray:
    """Fixed-fixed swap: random 2x2 exchanges that keep both margins.

    Only checkerboard submatrices are toggled, and the transfer amount is
    ``min`` of the two occupied diagonal cells so that quantitative
    (non-binary) matrices also stay margin-consistent.  At least one swap
    is attempted even for tiny matrices - the worker copy used to compute
    ``int(n_species * n_sites * 0.1)`` without the floor, which for a 2x2
    or 3x3 matrix meant *zero* swaps and therefore a simulated null made of
    exact copies of the observed matrix (SES = 0, p = 1).

    ``rng=None`` falls back to the global ``numpy.random`` stream.
    """
    result = np.array(matrix, copy=True)
    n_species, n_sites = result.shape
    if n_species < 2 or n_sites < 2:
        return result

    n_swaps = max(1, int(n_species * n_sites * 0.1))

    for _ in range(n_swaps):
        if rng is not None:
            rows = rng.choice(n_species, 2, replace=False)
            cols = rng.choice(n_sites, 2, replace=False)
        else:
            rows = np.random.choice(n_species, 2, replace=False)
            cols = np.random.choice(n_sites, 2, replace=False)

        r1, r2 = int(rows[0]), int(rows[1])
        c1, c2 = int(cols[0]), int(cols[1])

        a = result[r1, c1]
        b = result[r1, c2]
        c = result[r2, c1]
        d = result[r2, c2]

        if a > 0 and d > 0 and b == 0 and c == 0:
            delta = min(a, d)
            result[r1, c1] = a - delta
            result[r1, c2] = b + delta
            result[r2, c1] = c + delta
            result[r2, c2] = d - delta
        elif b > 0 and c > 0 and a == 0 and d == 0:
            delta = min(b, c)
            result[r1, c1] = a + delta
            result[r1, c2] = b - delta
            result[r2, c1] = c - delta
            result[r2, c2] = d + delta

    return result


def _shuffle_matrix_impl(matrix: npt.NDArray, rng: np.random.Generator | None = None) -> npt.NDArray:
    """Unconstrained shuffle: all cells pooled, only the grand total fixed."""
    flat = np.array(matrix.ravel(), copy=True)
    if rng is not None:
        rng.shuffle(flat)
    else:
        np.random.shuffle(flat)
    return flat.reshape(matrix.shape)


def _randomize_rows_impl(matrix: npt.NDArray, rng: np.random.Generator | None = None) -> npt.NDArray:
    """Gotelli "random rows" (rrs): permute within each row.

    Row totals (species frequencies) are preserved exactly; column totals
    (site richness) are free.  For a binary row every arrangement of its
    occurrences is equally likely, which is the intended null.
    """
    result = np.array(matrix, copy=True)
    for i in range(result.shape[0]):
        result[i] = rng.permutation(matrix[i]) if rng is not None else np.random.permutation(matrix[i])
    return result


def _randomize_columns_impl(matrix: npt.NDArray, rng: np.random.Generator | None = None) -> npt.NDArray:
    """Gotelli "random columns" (rcs): permute within each column.

    The transpose of :func:`_randomize_rows_impl` - column totals (site
    richness) are preserved, row totals are free.
    """
    result = np.array(matrix, copy=True)
    for j in range(result.shape[1]):
        col = matrix[:, j]
        result[:, j] = rng.permutation(col) if rng is not None else np.random.permutation(col)
    return result


def _permute_with_algorithm(
    matrix: npt.NDArray, algorithm: str, rng: np.random.Generator | None = None
) -> npt.NDArray:
    """Dispatch a randomization to its implementation.

    Raises:
        ValidationError: unknown algorithm names are rejected; they used to
            be replaced silently by a ``shuffle``.
    """
    if algorithm == "swap":
        return _swap_matrix_impl(matrix, rng)
    if algorithm == "rrs":
        return _randomize_rows_impl(matrix, rng)
    if algorithm == "rcs":
        return _randomize_columns_impl(matrix, rng)
    if algorithm == "shuffle":
        return _shuffle_matrix_impl(matrix, rng)
    raise ValidationError(
        _("Unknown algorithm: {0}. Use 'swap', 'rrs', 'rcs', or 'shuffle'").format(algorithm)
    )


def _worker_permute(
    matrix: npt.NDArray,
    n_perms: int,
    algorithm: str,
    metric: str = "c_score",
    seed: int | None = None,
) -> npt.NDArray:
    """
    Worker function for parallel permutation.

    This is a module-level function to allow pickling for multiprocessing.
    Each worker gets its own :class:`numpy.random.Generator` seeded from a
    :class:`numpy.random.SeedSequence` child of the parent seed, instead of
    reseeding the global legacy stream (which would leak into anything else
    the child process computes and would correlate the streams).
    """
    rng = np.random.default_rng(seed) if seed is not None else None

    results = np.zeros(n_perms)

    for i in range(n_perms):
        result = _permute_with_algorithm(matrix, algorithm, rng)
        results[i] = _compute_score_worker(result, metric)

    return results


def _swap_matrix_worker(
    matrix: npt.NDArray, rng: np.random.Generator | None = None
) -> npt.NDArray:
    """Swap algorithm in worker function (same code as the sequential path)."""
    return _swap_matrix_impl(matrix, rng)


def _compute_c_score_worker(matrix: npt.NDArray) -> float:
    """C-score computation in worker function (Stone & Roberts 1990).

    C_ij = (r_i - S_ij) * (r_j - S_ij)
    where S_ij is the number of sites where both species co-occur.
    """
    n_species, _ = matrix.shape
    pa = (matrix > 0).astype(np.int64)
    row_sums = pa.sum(axis=1)

    c_scores = []
    for i in range(n_species):
        for j in range(i + 1, n_species):
            s_ij = int(np.sum(pa[i] & pa[j]))
            c_ij = float((row_sums[i] - s_ij) * (row_sums[j] - s_ij))
            c_scores.append(c_ij)

    return float(np.mean(c_scores)) if c_scores else 0.0


def _compute_checkerboard_worker(matrix: npt.NDArray) -> float:
    """Checkerboard count computation in worker function."""
    n_species, n_sites = matrix.shape
    checkerboards = 0
    for i in range(n_species):
        for j in range(i + 1, n_species):
            for k in range(n_sites):
                for l in range(k + 1, n_sites):
                    if (matrix[i, k] == 1 and matrix[i, l] == 0 and matrix[j, k] == 0 and matrix[j, l] == 1) or (
                        matrix[i, k] == 0 and matrix[i, l] == 1 and matrix[j, k] == 1 and matrix[j, l] == 0
                    ):
                        checkerboards += 1
    return float(checkerboards)


def _compute_score_worker(matrix: npt.NDArray, metric: str) -> float:
    """Dispatch to the correct co-occurrence metric in the worker."""
    if metric == "checkerboard":
        return _compute_checkerboard_worker(matrix)
    elif metric == "combo":
        c_score = _compute_c_score_worker(matrix)
        n_species, n_sites = matrix.shape
        max_checkerboard = n_species * (n_species - 1) / 2 * n_sites * (n_sites - 1) / 2
        checkerboard = _compute_checkerboard_worker(matrix)
        norm_checkerboard = checkerboard / max_checkerboard if max_checkerboard > 0 else 0
        return (c_score + norm_checkerboard) / 2
    else:
        return _compute_c_score_worker(matrix)
