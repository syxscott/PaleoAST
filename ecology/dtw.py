# =============================================================================
# FILE: ecology/dtw.py
# =============================================================================
"""
Dynamic Time Warping Module for PaleoAST

Dynamic Time Warping (DTW) for comparing unevenly-sampled
paleontological and ecological sequences.

Mathematical Foundation:

DTW finds the optimal alignment between two sequences by
minimizing the cumulative distance along a warping path.

Given sequences X = (x₁, ..., xₙ) and Y = (y₁, ..., yₘ):

DTW(X, Y) = min_π Σ d(x_{π(i)}, y_{π(j)})

where π is a monotonic warping path subject to:
    - Boundary conditions: π(1) = (1,1), π(K) = (n,m)
    - Step size: π(k+1) - π(k) ∈ {(1,0), (0,1), (1,1)}
    - Monotonicity: i₁ ≤ i₂ and j₁ ≤ j₂

Sakoe-Chiba Band Constraint:
    |i - j| ≤ r  (window radius r)

LB_Keogh Lower Bound:
    LB_Keogh(Q, C) = √(Σ max_{j∈[i-r,i+r]} (c_j - U_j)² + ...)
    where U = upper envelope, L = lower envelope

Reference:
    Sakoe, H. and Chiba, S. (1978). Dynamic programming
    algorithm optimization for spoken word recognition.
    IEEE Transactions on Acoustics, Speech, and Signal
    Processing, 26(1), 43-49.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class DTWResult:
    """
    Container for DTW alignment results.

    Attributes:
        distance: DTW distance between sequences
        path: Warping path as list of (i, j) indices
        warped_seq1: Warped version of sequence 1
        warped_seq2: Warped version of sequence 2
        cumulative_matrix: Accumulated distance matrix
        n1: Length of first sequence
        n2: Length of second sequence
    """

    distance: float
    path: list[tuple[int, int]]
    warped_seq1: npt.NDArray
    warped_seq2: npt.NDArray
    cumulative_matrix: npt.NDArray
    n1: int
    n2: int

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Dynamic Time Warping (DTW) Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Sequence lengths: {0}, {1}').format(self.n1, self.n2)}\n"
            f"{_('DTW distance: {0:.4f}').format(self.distance)}\n"
            f"{_('Path length: {0}').format(len(self.path))}"
        )


class DTWAnalyzer:
    """
    Dynamic Time Warping analyzer for sequence comparison.

    Computes optimal alignment between unevenly-sampled
    paleontological and ecological time series.
    """

    def __init__(self) -> None:
        """Initialize the DTW analyzer."""
        self._logger = logging.getLogger(f"{__name__}.DTWAnalyzer")
        self._lock = threading.RLock()
        self._last_result: DTWResult | None = None
        self._logger.info("DTWAnalyzer initialized")

    def compute(
        self,
        seq1: npt.NDArray,
        seq2: npt.NDArray,
        metric: str = "euclidean",
        window: int | None = None,
    ) -> DTWResult:
        """
        Compute DTW alignment between two sequences.

        Parameters:
            seq1: First sequence (n1, ...) - can be multivariate
            seq2: Second sequence (n2, ...) - can be multivariate
            metric: Distance metric ('euclidean', 'cityblock', 'cosine')
            window: Sakoe-Chiba band radius (None = no constraint)

        Returns:
            DTWResult with alignment distance and path

        Note:
            Sequences can be 1D (n,) or 2D (n, n_features).
            The window parameter constrains the warping to prevent
            pathological alignments.
        """
        with self._lock:
            s1 = validate_data_array(seq1, allow_nan=False, name="seq1")
            s2 = validate_data_array(seq2, allow_nan=False, name="seq2")

            # Handle 1D sequences
            if s1.ndim == 1:
                s1 = s1.reshape(-1, 1)
            if s2.ndim == 1:
                s2 = s2.reshape(-1, 1)

            n1, n_features = s1.shape
            n2 = s2.shape[0]

            self._logger.info(
                f"Computing DTW alignment: seq1({n1}), seq2({n2}), "
                f"metric={metric}, window={window}"
            )

            # Compute pairwise distance matrix
            dist_mat = cdist(s1, s2, metric=metric)

            # Initialize cumulative distance matrix
            cum_dist = np.full((n1, n2), np.inf)
            cum_dist[0, 0] = dist_mat[0, 0]

            # Fill first row and column
            for i in range(1, n1):
                if window is None or i <= window:
                    cum_dist[i, 0] = cum_dist[i - 1, 0] + dist_mat[i, 0]

            for j in range(1, n2):
                if window is None or j <= window:
                    cum_dist[0, j] = cum_dist[0, j - 1] + dist_mat[0, j]

            # Fill rest of matrix with Sakoe-Chiba constraint
            for i in range(1, n1):
                for j in range(1, n2):
                    if window is not None and abs(i - j) > window:
                        continue

                    cum_dist[i, j] = dist_mat[i, j] + min(
                        cum_dist[i - 1, j],  # insertion
                        cum_dist[i, j - 1],  # deletion
                        cum_dist[i - 1, j - 1],  # match
                    )

            # Backtrack to find optimal path
            path = []
            i, j = n1 - 1, n2 - 1
            path.append((i, j))

            while i > 0 or j > 0:
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    candidates = [
                        (cum_dist[i - 1, j - 1], i - 1, j - 1),
                        (cum_dist[i - 1, j], i - 1, j),
                        (cum_dist[i, j - 1], i, j - 1),
                    ]
                    _, i, j = min(candidates, key=lambda x: x[0])

                path.append((i, j))

            path.reverse()

            # Warp sequences along path
            warped_seq1 = s1[[p[0] for p in path]]
            warped_seq2 = s2[[p[1] for p in path]]

            dtw_distance = cum_dist[n1 - 1, n2 - 1]

            result = DTWResult(
                distance=float(dtw_distance),
                path=path,
                warped_seq1=warped_seq1,
                warped_seq2=warped_seq2,
                cumulative_matrix=cum_dist,
                n1=n1,
                n2=n2,
            )

            self._last_result = result
            self._logger.info(f"DTW distance: {dtw_distance:.4f}")
            return result

    def distance_matrix(
        self,
        sequences: list[npt.NDArray],
        metric: str = "euclidean",
        window: int | None = None,
    ) -> tuple[npt.NDArray, list[npt.NDArray]]:
        """
        Compute pairwise DTW distance matrix for multiple sequences.

        Parameters:
            sequences: List of sequences
            metric: Distance metric
            window: Sakoe-Chiba band radius

        Returns:
            Tuple of (distance_matrix, warped_sequences)
        """
        n = len(sequences)

        if n == 0:
            return np.array([]), []

        self._logger.info(f"Computing DTW distance matrix for {n} sequences")

        dist_mat = np.zeros((n, n))
        warped = []

        for i in range(n):
            for j in range(i + 1, n):
                result = self.compute(sequences[i], sequences[j], metric, window)
                dist_mat[i, j] = result.distance
                dist_mat[j, i] = result.distance

                if i == 0:
                    warped.append(result.warped_seq1)

            if i == 0:
                warped.append(result.warped_seq2 if len(warped) == 0 else warped[0])

        return dist_mat, warped

    def lb_keogh(
        self,
        query: npt.NDArray,
        reference: npt.NDArray,
        window: int = 5,
    ) -> float:
        """
        Compute LB_Keogh lower bound for DTW distance.

        The LB_Keogh bound can be used to quickly eliminate
        sequence pairs that cannot have a small DTW distance
        without actually computing DTW.

        Parameters:
            query: Query sequence
            reference: Reference sequence
            window: Window radius for envelope

        Returns:
            Lower bound distance
        """
        if query.ndim > 1:
            query = query.reshape(-1)
        if reference.ndim > 1:
            reference = reference.reshape(-1)

        n_q = len(query)
        n_r = len(reference)

        # Compute envelope of reference
        U = np.full(n_r, np.inf)
        L = np.full(n_r, -np.inf)

        for j in range(n_r):
            lower = max(0, j - window)
            upper = min(n_r - 1, j + window)
            U[j] = np.max(reference[lower : upper + 1])
            L[j] = np.min(reference[lower : upper + 1])

        # Compute lower bound
        lb = 0.0
        for i in range(n_q):
            j = min(i, n_r - 1)
            if query[i] > U[j]:
                lb += (query[i] - U[j]) ** 2
            elif query[i] < L[j]:
                lb += (L[j] - query[i]) ** 2

        return float(np.sqrt(lb))

    @property
    def last_result(self) -> DTWResult | None:
        """Get the last DTW result."""
        with self._lock:
            return self._last_result
