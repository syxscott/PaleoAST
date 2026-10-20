# =============================================================================
# FILE: ecology/diversity_menu.py
# =============================================================================
"""
The diversity sampling and comparison tools, as the analysis runner can reach
them.

WHY THIS EXISTS
---------------
``ecology/diversity_extra.py`` holds six analyses the application could not
run at all: shareholder-quorum subsampling, sample coverage, rarefaction,
the diversity profile, and the t-test and permutation test that compare
diversity between groups. The analysis runner's contract is::

    plugin.analyze(data, **kwargs)

where ``data`` is the whole spreadsheet and ``**kwargs`` comes from a form
generated from the method's signature. ``subsample(abundances: NDArray,
quorum: int)`` handed that contract would receive the entire sheet as
``abundances`` -- a matrix where one sample's per-taxon counts are wanted.

So this module is a thin facade, in the same shape as
``visualization/plot_menu.py``: every method takes the sheet first, pulls
the rows and columns out, and delegates. No statistic is recomputed here; a
facade that recomputes anything is a second implementation, and two
implementations drift.

WHAT THE SHEET MEANS
--------------------
* The sheet is a **samples x taxa** table: one row per sample, one column
  per taxon.
* The subsample and the profile work on **one** sample, so they take the
  number of the row holding it.
* The coverage, the rarefaction and the two group tests work on the whole
  table.
* The group tests need one group label per sample, which the form cannot
  pick out of the table, so the labels are typed in, comma separated -- the
  same convention ``stats/permanova.py`` and ``stats/anosim.py`` use.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from utils.exceptions import DataValidationError

from .diversity_extra import (
    DiversityExtraAnalyzer,
    DiversityProfileResult,
    DiversityTestResult,
    SampleCoverageResult,
    SampleRarefactionResult,
    SQSResult,
)

logger = logging.getLogger(__name__)

__all__ = ["DiversityExtraMenu"]

#: The diversity indices the group tests can compare. The generated form
#: finds this constant by scanning this module for a name carrying "METRIC",
#: and offers it as a drop-down; leaving it unset tests every index.
DIVERSITY_METRICS = ("shannon", "simpson", "pielou", "margalef", "fisher_alpha", "chao1")


def _matrix(data: object) -> npt.NDArray:
    """The sheet as a 2-D float matrix.

    A 1-D input is promoted to a single row rather than rejected, because
    "one sample" is a legitimate thing to analyse and the user did not do
    anything wrong.
    """
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "these analyses need numeric columns; the sheet has a column that is not numeric"
        ) from exc
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise DataValidationError(f"these analyses need a 1-D or 2-D data set; got shape {arr.shape}")
    if arr.size == 0:
        raise DataValidationError("the data set is empty")
    return arr


def _row(matrix: npt.NDArray, index: int, name: str) -> npt.NDArray:
    """One row by number, with the out-of-range case named.

    Falling back to the last row would silently analyse the wrong sample,
    which is the failure this whole module exists to make impossible.
    """
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise DataValidationError(f"{name} must be a row number, got {index!r}")
    if index < 0 or index >= matrix.shape[0]:
        raise DataValidationError(f"{name} is row {index + 1}, but the data has {matrix.shape[0]} rows")
    return matrix[index, :]


def _group_labels(groups: list[str] | None, n_samples: int) -> npt.NDArray:
    """The group labels of each sample, one label per row.

    Parameters
    ----------
    groups:
        The labels, as typed in the form: comma separated, one per sample.
    n_samples:
        How many samples the table has.

    Returns
    -------
    npt.NDArray
        One label per sample.
    """
    if groups is None or len(groups) == 0:
        raise DataValidationError("groups needs one label per sample, comma separated, e.g. 1,1,2,2 for four samples")
    if len(groups) != n_samples:
        raise DataValidationError(
            f"groups has {len(groups)} label(s) but the table has {n_samples} sample(s); "
            "they must line up one label per sample"
        )
    return np.asarray([str(label) for label in groups])


def _metric_indices(metric: str | None) -> list[str] | None:
    """The index keys to test, or None for every index."""
    if metric is None or not str(metric).strip():
        return None
    key = str(metric).strip()
    if key not in DIVERSITY_METRICS:
        raise DataValidationError(f"metric must be one of {DIVERSITY_METRICS}, got {key!r}")
    return [key]


class DiversityExtraMenu:
    """The diversity sampling and comparison tools, reachable from the form.

    Every method takes the sheet as its first argument, which is what the
    analysis runner's generated form can express.
    """

    #: The analyzers this facade wraps. The catalog's staleness test uses
    #: this to know that the analyzers are reachable: a facade that is
    #: catalogued makes its analyzers reachable too.
    wrapped_analyzers = (DiversityExtraAnalyzer,)

    def __init__(self) -> None:
        """Initialise the facade and the analyzer behind it."""
        self._logger = logging.getLogger(f"{__name__}.DiversityExtraMenu")
        self._inner = DiversityExtraAnalyzer()

    # -- sampling -----------------------------------------------------------

    def subsample(
        self,
        data: object,
        sample_row: int = 0,
        quorum: int = 10,
        sample_name: str = "Sample",
    ) -> SQSResult:
        """Reduce one sample to a quorum of individuals (SQS subsampling)."""
        m = _matrix(data)
        return self._inner.subsample(
            _row(m, sample_row, "sample_row"),
            quorum=quorum,
            sample_name=sample_name,
        )

    def coverage(self, data: object) -> SampleCoverageResult:
        """Estimate how completely a table of samples has been sampled."""
        return self._inner.coverage(_matrix(data))

    def rarefaction(
        self,
        data: object,
        sample_sizes: list[int] | None = None,
    ) -> SampleRarefactionResult:
        """Rarefy a table of samples to a common size (Tao rarefaction)."""
        sizes = None if sample_sizes is None else np.asarray([int(v) for v in sample_sizes])
        return self._inner.rarefaction(_matrix(data), sample_sizes=sizes)

    def profile(
        self,
        data: object,
        sample_row: int = 0,
        quorums: list[int] | None = None,
        sample_name: str = "Sample",
        n_points: int = 20,
    ) -> DiversityProfileResult:
        """Compute every diversity index of one sample across sampling depths."""
        m = _matrix(data)
        depths = None if quorums is None else np.asarray([int(v) for v in quorums])
        return self._inner.profile(
            _row(m, sample_row, "sample_row"),
            quorums=depths,
            sample_name=sample_name,
            n_points=n_points,
        )

    # -- comparison ---------------------------------------------------------

    def t_test(
        self,
        data: object,
        groups: list[str] | None = None,
        metric: str = "",
        equal_var: bool = True,
    ) -> DiversityTestResult:
        """Compare diversity between groups of samples (t-test on the indices)."""
        m = _matrix(data)
        return self._inner.t_test(
            m,
            _group_labels(groups, m.shape[0]),
            indices=_metric_indices(metric),
            equal_var=equal_var,
        )

    def permutation_test(
        self,
        data: object,
        groups: list[str] | None = None,
        metric: str = "",
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> DiversityTestResult:
        """Compare diversity between groups of samples (permutation test on the indices)."""
        m = _matrix(data)
        return self._inner.permutation_test(
            m,
            _group_labels(groups, m.shape[0]),
            indices=_metric_indices(metric),
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
