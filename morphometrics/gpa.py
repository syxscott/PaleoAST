# =============================================================================
# FILE: morphometrics/gpa.py
# =============================================================================
"""
Generalized Procrustes Analysis (GPA) Module for PaleoAST

GPA is the fundamental preprocessing step in geometric morphometrics that
removes non-shape variation from landmark configurations.

Mathematical Foundation:

Given N landmark configurations X₁, X₂, ..., Xₙ ∈ ℝ^(k×m):
    k = number of landmarks
    m = number of dimensions (2 for 2D, 3 for 3D)

The GPA algorithm minimizes Procrustes distance by iteratively:

1. Translation: Remove centroid from each configuration
   X_centered = X - centroid(X)

2. Scaling: Normalize to unit centroid size
   X_scaled = X_centered / size(X_centered)

   where size = sqrt(trace(X_centered' * X_centered))

3. Rotation: Optimal rotation via SVD
   X_rotated = R * X_scaled

   where R maximizes trace(Y' * R * X_scaled)
   and R = U * V' from SVD of Y' * X_scaled = U * Σ * V'

Procrustes Distance:
    d_P(X, Y) = sqrt(||X - Y||² / size(X))

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from config.constants import GPA_CONVERGENCE_TOLERANCE, GPA_MAX_ITERATIONS
from config.i18n import _
from utils.exceptions import ComputationError, MorphometricsError

logger = logging.getLogger(__name__)


@dataclass
class GPAResult:
    """
    Container for GPA analysis results.
    """

    aligned_configurations: npt.NDArray  # (n_specimens, k, m)
    consensus: npt.NDArray  # (k, m) - mean configuration
    procrustes_distances: npt.NDArray  # Distance to consensus
    rotations: list[npt.NDArray]  # Rotation matrices
    scales: npt.NDArray  # Scale factors
    centroids: npt.NDArray  # Original centroids
    n_iterations: int
    converged: bool
    final_sse: float  # Sum of squared errors

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Generalized Procrustes Analysis Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Number of specimens: {0}').format(self.aligned_configurations.shape[0])}\n"
            f"{_('Number of landmarks: {0}').format(self.aligned_configurations.shape[1])}\n"
            f"{_('Dimensions: {0}').format(self.aligned_configurations.shape[2])}\n"
            f"{_('Iterations: {0}').format(self.n_iterations)}\n"
            f"{_('Converged: {0}').format(self.converged)}\n"
            f"{_('Final SSE: {0}').format(f'{self.final_sse:.6f}')}"
        )


@dataclass
class PartialGPAResult:
    """
    Container for Partial GPA (semilandmark sliding) results.

    Semilandmark sliding follows Bookstein (1997) and Gunz et al. (2005):
    - Fixed landmarks are aligned via standard GPA
    - Semilandmarks slide along their curve/surface to minimize bending energy
    - Iterative refinement until convergence

    Bending Energy Function (Bookstein 1997):
        B(f) = ∫∫ (∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)² dx dy

    The sliding algorithm finds positions for semilandmarks that minimize:
        objective = Procrustes_distance + λ * bending_energy

    where λ is a weighting factor (typically 1.0 for geometric morphometrics).
    """

    aligned_configurations: npt.NDArray  # (n_specimens, k, m)
    consensus: npt.NDArray  # (k, m) - mean configuration
    procrustes_distances: npt.NDArray  # Distance to consensus
    bending_energies: npt.NDArray  # Bending energy for each specimen
    sliding_iterations: int
    converged: bool
    final_sse: float  # Sum of squared errors

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Partial GPA (Semilandmark Sliding) Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Number of specimens: {0}').format(self.aligned_configurations.shape[0])}\n"
            f"{_('Number of landmarks: {0}').format(self.aligned_configurations.shape[1])}\n"
            f"{_('Dimensions: {0}').format(self.aligned_configurations.shape[2])}\n"
            f"{_('Sliding iterations: {0}').format(self.sliding_iterations)}\n"
            f"{_('Converged: {0}').format(self.converged)}\n"
            f"{_('Final SSE: {0}').format(f'{self.final_sse:.6f}')}"
        )


class GPAAnalyzer:
    """
    Generalized Procrustes Analysis engine for landmark data.

    GPA removes non-shape variation (translation, scaling, rotation)
    from landmark configurations to obtain shape coordinates.
    """

    def __init__(self) -> None:
        """Initialize the GPA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.GPAAnalyzer")
        self._logger.info("GPAAnalyzer initialized")
        self._lock = threading.RLock()
        self._last_result: GPAResult | None = None
        self._tolerance = GPA_CONVERGENCE_TOLERANCE
        self._max_iterations = GPA_MAX_ITERATIONS

    def analyze(
        self,
        configurations: npt.NDArray,
        n_iterations: int | None = None,
        tolerance: float | None = None,
        n_landmarks: int | None = None,
        n_dims: int | None = None,
    ) -> GPAResult:
        """
        Perform Generalized Procrustes Analysis.

        Parameters:
            configurations: 3D array of shape (n_specimens, n_landmarks, n_dims)
                          or 2D array (n_specimens, n_landmarks*n_dims) for flat format.
                          For ambiguous flat dimensions (e.g., 12 which could be 6*2 or 4*3),
                          provide either n_landmarks or n_dims to disambiguate.
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for Procrustes distance
            n_landmarks: Number of landmarks per specimen (resolves ambiguous flat dimensions)
            n_dims: Number of dimensions per landmark (2 for 2D, 3 for 3D).
                   If None, inferred from shape heuristics.

        Returns:
            GPAResult: GPA analysis results

        Raises:
            MorphometricsError: If configurations are invalid or dimensions ambiguous
        """
        with self._lock:
            # Validate and prepare configurations
            X = self._prepare_configurations(configurations, n_landmarks=n_landmarks, n_dims=n_dims)

            n_specimens, n_landmarks, n_dims = X.shape
            self._logger.info(
                f"GPA alignment started: n_specimens={n_specimens}, n_landmarks={n_landmarks}, n_dimensions={n_dims}"
            )

            if n_iterations is None:
                n_iterations = self._max_iterations
            if tolerance is None:
                tolerance = self._tolerance

            # Initialize aligned configurations
            aligned = X.copy()
            rotations = []
            scales = np.ones(n_specimens)
            original_centroids = np.zeros((n_specimens, n_dims))

            # Iterative Procrustes superimposition
            prev_sse = float("inf")

            for iteration in range(n_iterations):
                # Step 1: Compute current centroids and translate to common origin
                for i in range(n_specimens):
                    centroid = np.mean(aligned[i], axis=0)
                    aligned[i] = aligned[i] - centroid
                    if iteration == 0:
                        original_centroids[i] = centroid

                # Step 2: Scale to unit size
                sizes = self._compute_sizes(aligned)
                for i in range(n_specimens):
                    aligned[i] = aligned[i] / sizes[i]
                    scales[i] *= sizes[i]

                # Step 3: Compute consensus (mean)
                consensus = np.mean(aligned, axis=0)

                # Step 4: Rotate each specimen to consensus
                new_aligned = np.zeros_like(aligned)
                iter_rotations = []

                for i in range(n_specimens):
                    rotation = self._find_rotation(consensus, aligned[i])
                    iter_rotations.append(rotation)
                    # _find_rotation(consensus, target) returns the Kabsch
                    # rotation R such that ``R @ target ≈ consensus`` (the
                    # standard left-multiplication convention). Because the
                    # rotation is applied on the *left* but our specimens
                    # are stored with landmarks on the rows (k x m), the
                    # equivalent operation is ``target @ R.T``. Verified
                    # empirically with a 2D rotation round-trip.
                    new_aligned[i] = aligned[i] @ rotation.T

                aligned = new_aligned
                rotations = iter_rotations

                # Compute sum of squared Procrustes distances
                sse = 0.0
                for i in range(n_specimens):
                    diff = aligned[i] - consensus
                    sse += np.sum(diff**2)

                self._logger.debug(f"GPA iteration {iteration + 1}: SSE={sse:.6f}, delta={abs(prev_sse - sse):.6e}")

                # Check convergence
                if abs(prev_sse - sse) < tolerance:
                    self._logger.info(f"GPA converged after {iteration + 1} iterations with final SSE={sse:.6f}")
                    result = GPAResult(
                        aligned_configurations=aligned,
                        consensus=consensus,
                        procrustes_distances=self._compute_distances_to_consensus(aligned, consensus),
                        rotations=rotations,
                        scales=scales,
                        centroids=original_centroids,
                        n_iterations=iteration + 1,
                        converged=True,
                        final_sse=sse,
                    )
                    self._last_result = result
                    return result

                prev_sse = sse

            # Did not converge within max iterations
            self._logger.warning(f"GPA did not converge after {n_iterations} iterations, final SSE={prev_sse:.6f}")
            result = GPAResult(
                aligned_configurations=aligned,
                consensus=consensus,
                procrustes_distances=self._compute_distances_to_consensus(aligned, consensus),
                rotations=rotations,
                scales=scales,
                centroids=original_centroids,
                n_iterations=n_iterations,
                converged=False,
                final_sse=prev_sse,
            )

            self._last_result = result
            return result

    def _prepare_configurations(
        self,
        configurations: npt.NDArray,
        n_landmarks: int | None = None,
        n_dims: int | None = None,
    ) -> npt.NDArray:
        """
        Prepare configurations array to standard 3D format.

        Parameters:
            configurations: Input array (n, k, m) or (n, k*m).
                          Can be a list of configurations that will be converted to array.
            n_landmarks: Number of landmarks per specimen (required for ambiguous cases)
            n_dims: Number of dimensions per landmark (2 for 2D, 3 for 3D).
                   If None, inferred from shape heuristics.

        Returns:
            npt.NDArray: Standardized 3D array (n, k, m)

        Raises:
            MorphometricsError: If dimensions cannot be determined unambiguously

        Notes:
            Dimension inference heuristic:
            - If data.shape[1] in [2, 3] and data.shape[0] > 3: treat as (data.shape[0], 1, dim)
              which means data.shape[0] landmarks with dim dimensions for a SINGLE specimen
            - Otherwise: first dimension is n_specimens, infer dim from divisibility
            - Ambiguous cases (e.g., 12 which could be 6*2 or 4*3) require explicit parameters
        """
        # Convert list to numpy array if needed
        if isinstance(configurations, list):
            configurations = np.array(configurations)

        if configurations.ndim == 2:
            n_specimens = configurations.shape[0]
            flat_dim = configurations.shape[1]

            # Special case: if shape[1] in [2, 3] and shape[0] > 3, this is likely
            # a single specimen with shape[0] landmarks and shape[1] dimensions
            # e.g., (10, 2) = 10 landmarks, 2D for 1 specimen
            if flat_dim in [2, 3] and n_specimens > 3 and n_landmarks is None and n_dims is None:
                n_dims = flat_dim
                n_landmarks = n_specimens
                n_specimens = 1
                X = configurations.reshape(n_specimens, n_landmarks, n_dims)
            # Case 1: Explicit parameters provided - use them
            elif n_landmarks is not None and n_dims is not None:
                if n_landmarks * n_dims != flat_dim:
                    raise MorphometricsError(
                        f"Flat dimension {flat_dim} does not match "
                        f"n_landmarks ({n_landmarks}) * n_dims ({n_dims})"
                    )
                X = configurations.reshape(n_specimens, n_landmarks, n_dims)

            # Case 2: Explicit n_dims provided, infer n_landmarks
            elif n_dims is not None:
                if flat_dim % n_dims != 0:
                    raise MorphometricsError(
                        f"Flat dimension {flat_dim} not divisible by n_dims={n_dims}"
                    )
                n_landmarks = flat_dim // n_dims
                X = configurations.reshape(n_specimens, n_landmarks, n_dims)

            # Case 3: Explicit n_landmarks provided, infer n_dims
            elif n_landmarks is not None:
                if flat_dim % n_landmarks == 0:
                    n_dims = flat_dim // n_landmarks
                    if n_dims not in [2, 3]:
                        raise MorphometricsError(
                            f"Cannot determine dimensions: inferred n_dims={n_dims} "
                            f"but must be 2 or 3. Provide explicit n_dims parameter."
                        )
                    X = configurations.reshape(n_specimens, n_landmarks, n_dims)
                else:
                    raise MorphometricsError(
                        f"Flat dimension {flat_dim} not divisible by n_landmarks={n_landmarks}"
                    )

            # Case 4: Heuristic inference (ambiguous cases require explicit params)
            else:
                # Special case: if shape[1] is 2 or 3, treat as single specimen
                if flat_dim in [2, 3]:
                    # Single specimen with 2 or 3 landmarks in 1D... unlikely but handle
                    raise MorphometricsError(
                        f"Ambiguous case: flat_dim={flat_dim}. "
                        f"Provide explicit n_landmarks and n_dims parameters."
                    )

                # Check divisibility
                dim_candidates = []
                for candidate_dim in [2, 3]:
                    if flat_dim % candidate_dim == 0:
                        candidate_landmarks = flat_dim // candidate_dim
                        dim_candidates.append((candidate_dim, candidate_landmarks))

                if len(dim_candidates) == 1:
                    # Unambiguous case
                    n_dims, n_landmarks = dim_candidates[0]
                    X = configurations.reshape(n_specimens, n_landmarks, n_dims)
                elif len(dim_candidates) == 0:
                    raise MorphometricsError(
                        f"Cannot determine dimensions from flat_dim={flat_dim}. "
                        f"Provide explicit n_landmarks or n_dims parameter."
                    )
                else:
                    # Ambiguous case: e.g., flat_dim=12 could be 6*2 or 4*3
                    raise MorphometricsError(
                        f"Ambiguous case: flat_dim={flat_dim} is divisible by both "
                        f"{dim_candidates[0][0]} and {dim_candidates[1][0]}. "
                        f"Provide explicit n_landmarks or n_dims parameter to disambiguate. "
                        f"Candidates: {[(d, l) for d, l in dim_candidates]} landmarks * dims."
                    )

        elif configurations.ndim == 3:
            if n_dims is not None and configurations.shape[2] != n_dims:
                raise MorphometricsError(
                    f"Explicit n_dims={n_dims} does not match "
                    f"configuration shape {configurations.shape}"
                )
            X = configurations
        else:
            raise MorphometricsError(
                "Configurations must be 2D (n_specimens, k*m) or 3D (n_specimens, k, m)"
            )

        return X.astype(float)

    def _compute_sizes(self, configurations: npt.NDArray) -> npt.NDArray:
        """
        Compute centroid size for each configuration (vectorized).

        Centroid size = sqrt(trace(X' * X))

        where X is the centered configuration matrix.
        """
        # Vectorized: compute mean for all configurations at once
        means = np.mean(configurations, axis=1, keepdims=True)  # (n, 1, m)
        centered = configurations - means  # Broadcasting

        # Compute centroid sizes for all at once
        sizes = np.sqrt(np.sum(centered**2, axis=(1, 2)))

        return sizes

    def _find_rotation(self, reference: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
        """
        Find optimal rotation to align target to reference using SVD.

        Given reference Y and target X, find R that minimizes:
            ||Y - X @ R||²

        Solution: R = V * U' where U * Σ * V' = X' * Y

        When det(R) < 0, a reflection is detected (improper rotation).
        The standard fix (following Bookstein 1989, Dryden & Mardia 2016) is to
        flip the sign of the last singular vector in Vt only, then recompute R.
        This ensures det(R) = +1 while preserving the optimal least-squares fit.

        Returns:
            npt.NDArray: Rotation matrix R with det(R) = +1

        References:
            - Bookstein, F.L. (1989). Principal warps: thin-plate splines and
              the decomposition of deformations. IEEE TPAMI.
            - Dryden, I.L. & Mardia, K.V. (2016). Statistical shape analysis.
        """
        # Compute cross-covariance
        H = target.T @ reference

        # SVD
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError as e:
            raise ComputationError("SVD failed during GPA rotation", original_exception=e)

        # Compute rotation
        R = Vt.T @ U.T

        # Ensure proper rotation (determinant = +1)
        # Only flip Vt to avoid in-place modification issues; this follows
        # the standard approach (Morpho::procSym in R, procrustes.py)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
        R = Vt.T @ U.T

        return R

    def _compute_distances_to_consensus(self, aligned: npt.NDArray, consensus: npt.NDArray) -> npt.NDArray:
        """
        Compute Procrustes distances to consensus configuration (vectorized).
        """
        # Vectorized: diff[i] = aligned[i] - consensus for all i
        diff = aligned - consensus  # Broadcasting: (n, k, m) - (k, m) -> (n, k, m)
        distances = np.sqrt(np.sum(diff**2, axis=(1, 2)))

        return distances

    @property
    def last_result(self) -> GPAResult | None:
        """Get the last GPA result."""
        with self._lock:
            return self._last_result


# =============================================================================
# Partial GPA (Semilandmark Sliding)
# =============================================================================

def partial_gpa(
    configurations: npt.NDArray,
    fixed_landmarks: list[int] | npt.NDArray,
    curve_indices: list[list[int]] | None = None,
    surface_indices: list[list[int]] | None = None,
    n_dims: Literal[2, 3] = 2,
    sliding_weight: float = 1.0,
    n_iterations: int = 20,
    tolerance: float = 1e-6,
    n_landmarks: int | None = None,
) -> PartialGPAResult:
    """
    Perform Partial GPA with semilandmark sliding.

    This implements the Bookstein (1997) and Gunz et al. (2005) algorithm for
    sliding semilandmarks along curves (2D) or surfaces (3D) to minimize
    bending energy while maintaining Procrustes alignment with fixed landmarks.

    Parameters:
        configurations: Landmark configurations of shape (n_specimens, n_landmarks, n_dims)
                      or (n_specimens, n_landmarks * n_dims) for flat format.
        fixed_landmarks: Indices of fixed (non-sliding) landmarks.
        curve_indices: For 2D, list of lists where each sublist contains indices
                      of semilandmarks that slide along the same curve.
                      e.g., [[3, 4, 5, 6], [7, 8, 9]] for two separate curves.
        surface_indices: For 3D, analogous to curve_indices but for surfaces.
        n_dims: Number of dimensions (2 for 2D, 3 for 3D).
        sliding_weight: Weight for bending energy in the objective function.
                       Higher values = more smoothing (default: 1.0).
        n_iterations: Maximum number of sliding iterations.
        tolerance: Convergence tolerance for sliding.
        n_landmarks: Number of landmarks (required if ambiguous from shape).

    Returns:
        PartialGPAResult: Results containing aligned configurations and statistics.

    Raises:
        MorphometricsError: If configurations are invalid or parameters inconsistent.

    Notes:
        The sliding algorithm works as follows:
        1. Perform standard GPA on all landmarks
        2. For each semilandmark curve/surface:
           - Hold fixed landmarks stationary
           - Slide semilandmarks along tangent directions
           - Find positions that minimize: Procrustes_distance + λ * bending_energy
        3. Iterate until convergence (typically 5-10 iterations)

        Bending energy is computed using the TPS (Thin Plate Spline) formulation:
            B(f) = Σ Σ w_i * K_ij * w_j

        where K_ij = ||x_i - x_j||² * log(||x_i - x_j||) is the TPS kernel.

    References:
        - Bookstein, F.L. (1997). Morphometric tools for landmark data.
          Cambridge University Press. Chapter 8.
        - Gunz, P., Mitteroecker, P., & Bookstein, F.L. (2005).
          Semilandmarks in three dimensions. Anatomical Record.
        - Rohlf, F.J. (1999). Shape statistics: Procrustes superimposition
          and tangent spaces. Taxon.
    """
    # Validate inputs
    if isinstance(fixed_landmarks, list):
        fixed_landmarks = np.array(fixed_landmarks)
    fixed_landmarks = fixed_landmarks.astype(int)

    # Get sliding indices (semilandmarks)
    all_sliding_indices: list[int] = []
    if curve_indices is not None:
        for curve in curve_indices:
            all_sliding_indices.extend(curve)
    if surface_indices is not None:
        for surface in surface_indices:
            all_sliding_indices.extend(surface)
    sliding_indices = np.array(all_sliding_indices) if all_sliding_indices else np.array([], dtype=int)

    # Prepare configurations
    if configurations.ndim == 2:
        n_specimens = configurations.shape[0]
        flat_dim = configurations.shape[1]
        if n_landmarks is not None:
            inferred_dims = flat_dim // n_landmarks
            if inferred_dims in [2, 3]:
                n_dims = inferred_dims
        elif flat_dim % 3 == 0 and flat_dim % 2 == 0:
            raise MorphometricsError(
                f"Ambiguous flat_dim={flat_dim}. Provide n_landmarks parameter."
            )
        elif flat_dim % n_dims != 0:
            raise MorphometricsError(
                f"flat_dim={flat_dim} not divisible by n_dims={n_dims}"
            )

    # Initialize GPA analyzer
    gpa = GPAAnalyzer()

    # Iterative partial GPA with sliding
    current_configs = configurations.copy()
    prev_sse = float("inf")

    for iteration in range(n_iterations):
        # Step 1: Standard GPA alignment
        gpa_result = gpa.analyze(current_configs)

        aligned = gpa_result.aligned_configurations
        consensus = gpa_result.consensus
        sse = gpa_result.final_sse

        logger.debug(f"Partial GPA iteration {iteration + 1}: SSE={sse:.6f}")

        # Step 2: Slide semilandmarks
        if len(sliding_indices) > 0 and (curve_indices is not None or surface_indices is not None):
            aligned = _slide_semilandmarks(
                aligned=aligned,
                consensus=consensus,
                fixed_landmarks=fixed_landmarks,
                curve_indices=curve_indices,
                surface_indices=surface_indices,
                n_dims=n_dims,
                sliding_weight=sliding_weight,
            )

        # Step 3: Check convergence
        if abs(prev_sse - sse) < tolerance:
            logger.info(f"Partial GPA converged after {iteration + 1} iterations")
            break
        prev_sse = sse
        current_configs = aligned

    # Final GPA alignment
    final_result = gpa.analyze(aligned)

    # Compute bending energies
    bending_energies = np.array([
        _compute_bending_energy(spec, consensus, fixed_landmarks, n_dims)
        for spec in final_result.aligned_configurations
    ])

    return PartialGPAResult(
        aligned_configurations=final_result.aligned_configurations,
        consensus=final_result.consensus,
        procrustes_distances=final_result.procrustes_distances,
        bending_energies=bending_energies,
        sliding_iterations=iteration + 1,
        converged=abs(prev_sse - final_result.final_sse) < tolerance,
        final_sse=final_result.final_sse,
    )


def _slide_semilandmarks(
    aligned: npt.NDArray,
    consensus: npt.NDArray,
    fixed_landmarks: npt.NDArray,
    curve_indices: list[list[int]] | None,
    surface_indices: list[list[int]] | None,
    n_dims: Literal[2, 3],
    sliding_weight: float,
) -> npt.NDArray:
    """
    Slide semilandmarks along their tangent directions to minimize bending energy.

    For each semilandmark, the algorithm finds the position along the curve/surface
    that minimizes the combined Procrustes distance and bending energy objective.
    """
    n_specimens, n_landmarks, _ = aligned.shape
    result = aligned.copy()

    # Process each curve/surface
    if n_dims == 2 and curve_indices is not None:
        for curve in curve_indices:
            result = _slide_2d_curve(
                configs=result,
                consensus=consensus,
                curve=curve,
                fixed_landmarks=fixed_landmarks,
                sliding_weight=sliding_weight,
            )
    elif n_dims == 3 and surface_indices is not None:
        for surface in surface_indices:
            result = _slide_3d_surface(
                configs=result,
                consensus=consensus,
                surface=surface,
                fixed_landmarks=fixed_landmarks,
                sliding_weight=sliding_weight,
            )

    return result


def _slide_2d_curve(
    configs: npt.NDArray,
    consensus: npt.NDArray,
    curve: list[int],
    fixed_landmarks: npt.NDArray,
    sliding_weight: float,
) -> npt.NDArray:
    """
    Slide semilandmarks along a 2D curve.

    For each semilandmark on the curve, computes the tangent direction from
    adjacent landmarks and slides along this tangent to minimize the
    objective: Procrustes_distance + λ * bending_energy.
    """
    n_specimens = configs.shape[0]
    result = configs.copy()

    # Build tangent directions for each curve point using consensus
    tangents = _compute_curve_tangents(consensus, curve)

    for i, lm_idx in enumerate(curve):
        # Tangent direction at this point
        tangent = tangents[i]
        if np.linalg.norm(tangent) < 1e-10:
            continue

        tangent = tangent / np.linalg.norm(tangent)

        for spec_idx in range(n_specimens):
            # Current position
            current_pos = result[spec_idx, lm_idx].copy()

            # Search for optimal position along tangent
            # Use golden section search for efficiency
            best_pos = current_pos.copy()
            best_score = float("inf")

            # Try multiple positions along tangent
            for delta in np.linspace(-0.1, 0.1, 21):
                candidate_pos = current_pos + delta * tangent
                test_config = result[spec_idx].copy()
                test_config[lm_idx] = candidate_pos

                # Compute objective: Procrustes + λ * bending
                proc_dist = np.linalg.norm(test_config - consensus)
                bend_energy = _compute_local_bending_energy(
                    test_config, consensus, lm_idx, fixed_landmarks
                )
                score = proc_dist + sliding_weight * bend_energy

                if score < best_score:
                    best_score = score
                    best_pos = candidate_pos

            result[spec_idx, lm_idx] = best_pos

    return result


def _slide_3d_surface(
    configs: npt.NDArray,
    consensus: npt.NDArray,
    surface: list[int],
    fixed_landmarks: npt.NDArray,
    sliding_weight: float,
) -> npt.NDArray:
    """
    Slide semilandmarks along a 3D surface.

    Similar to 2D curve sliding but uses surface normals and tangent plane
    for sliding directions.
    """
    n_specimens = configs.shape[0]
    result = configs.copy()

    # Compute surface normals and tangent plane basis
    normals, tangent_basis = _compute_surface_tangents_and_normals(consensus, surface)

    for i, lm_idx in enumerate(surface):
        normal = normals[i]
        basis = tangent_basis[i]

        if np.linalg.norm(normal) < 1e-10:
            continue

        normal = normal / np.linalg.norm(normal)

        for spec_idx in range(n_specimens):
            current_pos = result[spec_idx, lm_idx].copy()

            best_pos = current_pos.copy()
            best_score = float("inf")

            # Search in tangent plane (2D grid)
            for du in np.linspace(-0.1, 0.1, 11):
                for dv in np.linspace(-0.1, 0.1, 11):
                    candidate_pos = current_pos + du * basis[0] + dv * basis[1]
                    test_config = result[spec_idx].copy()
                    test_config[lm_idx] = candidate_pos

                    proc_dist = np.linalg.norm(test_config - consensus)
                    bend_energy = _compute_local_bending_energy(
                        test_config, consensus, lm_idx, fixed_landmarks
                    )
                    score = proc_dist + sliding_weight * bend_energy

                    if score < best_score:
                        best_score = score
                        best_pos = candidate_pos

            result[spec_idx, lm_idx] = best_pos

    return result


def _compute_curve_tangents(consensus: npt.NDArray, curve: list[int]) -> npt.NDArray:
    """
    Compute tangent directions for points on a curve.

    For interior points, tangent is the normalized difference between
    adjacent points. For endpoints, uses the direction to the nearest point.
    """
    n_points = len(curve)
    tangents = np.zeros((n_points, consensus.shape[1]))

    for i, idx in enumerate(curve):
        if i == 0:
            # First point: tangent toward next
            if n_points > 1:
                next_idx = curve[1]
                tangents[i] = consensus[next_idx] - consensus[idx]
        elif i == n_points - 1:
            # Last point: tangent from previous
            prev_idx = curve[i - 1]
            tangents[i] = consensus[idx] - consensus[prev_idx]
        else:
            # Interior: average of both directions
            prev_idx = curve[i - 1]
            next_idx = curve[i + 1]
            tangents[i] = (consensus[next_idx] - consensus[prev_idx]) / 2

    return tangents


def _compute_surface_tangents_and_normals(
    consensus: npt.NDArray, surface: list[int]
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute surface normals and tangent basis for surface semilandmarks.

    For a triangulated surface, uses the normal to the tangent plane.
    Returns normals and two tangent vectors spanning the plane.
    """
    n_points = len(surface)
    normals = np.zeros((n_points, 3))
    tangent_basis = np.zeros((n_points, 2, 3))

    for i, idx in enumerate(surface):
        # Find neighbors in the surface (simplified: use all other surface points)
        neighbors = [j for j in surface if j != idx]

        if len(neighbors) < 2:
            normals[i] = np.array([0, 0, 1])
            tangent_basis[i, 0] = np.array([1, 0, 0])
            tangent_basis[i, 1] = np.array([0, 1, 0])
            continue

        # Compute normal as average cross product of neighbor vectors
        normal = np.zeros(3)
        for j in neighbors[:3]:  # Use first 3 neighbors
            v1 = consensus[j] - consensus[idx]
            v2 = consensus[surface[0]] - consensus[idx] if surface[0] != idx else consensus[surface[1]] - consensus[idx]
            normal += np.cross(v1, v2)

        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normals[i] = normal / norm
        else:
            normals[i] = np.array([0, 0, 1])

        # Compute orthonormal basis in tangent plane
        if abs(normals[i, 2]) < 0.9:
            tangent_basis[i, 0] = np.cross(normals[i], np.array([0, 0, 1]))
        else:
            tangent_basis[i, 0] = np.cross(normals[i], np.array([0, 1, 0]))
        tangent_basis[i, 0] = tangent_basis[i, 0] / np.linalg.norm(tangent_basis[i, 0])
        tangent_basis[i, 1] = np.cross(normals[i], tangent_basis[i, 0])

    return normals, tangent_basis


def _compute_bending_energy(
    specimen: npt.NDArray,
    consensus: npt.NDArray,
    fixed_landmarks: npt.NDArray,
    n_dims: Literal[2, 3],
) -> float:
    """
    Compute bending energy of a specimen relative to consensus.

    Implements the Bookstein (1989) TPS bending energy formula:
        L(f) = trace(W^T K W) = ||L W||²

    where W is the non-affine weight matrix from TPS decomposition.
    The bending energy measures the non-affine (local) deformation component.

    This function solves the TPS system for the non-affine weights w
    and computes w^T K w as the bending energy.

    Parameters:
        specimen: Aligned specimen configuration (n_landmarks, n_dims)
        consensus: Consensus (mean) configuration (n_landmarks, n_dims)
        fixed_landmarks: Indices of fixed landmarks for TPS kernel
        n_dims: Number of dimensions (2 or 3)

    Returns:
        Bending energy as float (w^T K w)

    References:
        - Bookstein, F.L. (1989). Principal warps: thin-plate splines and
          the decomposition of deformations. IEEE TPAMI.
        - Dryden, I.L. & Mardia, K.V. (2016). Statistical shape analysis.
        - Gunz et al. (2005). Geometerrie morphometrics.
    """
    # Build TPS kernel matrix for fixed landmarks
    fixed = fixed_landmarks if isinstance(fixed_landmarks, np.ndarray) else np.array(fixed_landmarks)
    n_fixed = len(fixed)

    if n_fixed < 3:
        return 0.0

    # Extract fixed landmarks from consensus
    fixed_coords = consensus[fixed]  # (n_fixed, n_dims)

    # Build kernel matrix K_ij = r_ij² * log(r_ij) where r_ij = ||landmark_i - landmark_j||
    K = np.zeros((n_fixed, n_fixed))
    for i in range(n_fixed):
        for j in range(n_fixed):
            if i != j:
                r = np.linalg.norm(fixed_coords[i] - fixed_coords[j])
                if r > 1e-10:
                    K[i, j] = r**2 * np.log(r)

    # Build P matrix for affine component: [1, x, y] for 2D or [1, x, y, z] for 3D
    if n_dims == 2:
        P = np.column_stack([np.ones(n_fixed), fixed_coords])
    else:
        P = np.column_stack([np.ones(n_fixed), fixed_coords])

    n_affine = P.shape[1]  # 3 for 2D, 4 for 3D

    # Build block TPS system:
    # [K  P] [w]   [target]
    # [P^T 0] [a] = [0]
    #
    # where target = specimen - consensus (for the fixed landmarks)
    top = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((n_affine, n_affine))])
    L = np.vstack([top, bottom])

    # Right-hand side: difference between specimen and consensus at fixed landmarks
    # (the TPS interpolates this deformation)
    target_diff = specimen[fixed] - consensus[fixed]  # (n_fixed, n_dims)

    # Solve TPS system for each dimension
    n_total = n_fixed + n_affine
    w_all = np.zeros((n_total, n_dims))

    for d in range(n_dims):
        rhs = np.zeros(n_total)
        rhs[:n_fixed] = target_diff[:, d]
        try:
            solution = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            solution, _, _, _ = np.linalg.lstsq(L, rhs, rcond=None)
        w_all[:, d] = solution

    # Extract non-affine weights w (first n_fixed entries)
    w = w_all[:n_fixed]  # (n_fixed, n_dims)

    # Compute bending energy: w^T K w (per dimension, then sum)
    # This is the trace(W^T K W) formula from Bookstein 1989
    bending_energy = float(np.sum(w * (K @ w)))

    return bending_energy


def _compute_local_bending_energy(
    config: npt.NDArray,
    consensus: npt.NDArray,
    lm_idx: int,
    fixed_landmarks: npt.NDArray,
) -> float:
    """
    Compute local bending energy contribution from a single semilandmark.

    Uses the TPS bending energy formulation measuring how much the
    deformation is non-affine (purely local deformation vs. global stretch).
    """
    fixed = fixed_landmarks if isinstance(fixed_landmarks, np.ndarray) else np.array(fixed_landmarks)

    # Compute bending energy contribution
    # Simplified: measures local distortion based on deviation from consensus
    deviation = np.linalg.norm(config[lm_idx] - consensus[lm_idx])
    return float(deviation**2)
