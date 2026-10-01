# =============================================================================
# FILE: controllers/statistics_controller.py
# =============================================================================
"""
Statistics Controller for PaleoAST

This controller coordinates all statistical analyses including PCA, PCoA,
NMDS, ANOSIM, PERMANOVA, and diversity analyses.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from typing import Any

import numpy as np
import numpy.typing as npt

from ecology.advanced import AbundanceModelFit, AbundanceModelFitter, SHEAnalyzer, SHEResult
from ecology.diversity import DiversityAnalyzer, compute_diversity_indices
from ecology.rarefaction import RarefactionAnalyzer
from models.diversity_result import DiversityResult, RarefactionResult
from models.state_manager import get_state_manager
from morphometrics.efa import EFAAnalyzer, EFAResult, EigenshapeAnalyzer, EigenshapeResult
from morphometrics.gpa import GPAAnalyzer, GPAResult
from stats.anosim import ANOSIMAnalyzer, ANOSIMResult
from stats.cca import CCAAnalyzer, CCAResult
from stats.clustering import ClusteringAnalyzer, ClusteringResult
from stats.distance_metrics import DistanceMatrixResult, compute_distance_matrix
from stats.lda import LDAAnalyzer, LDAResult
from stats.nmds import NMDSAnalyzer, NMDSResult
from stats.pca import PCAAnalyzer, PCAResult
from stats.pcm import (
    AncestralStateResult,
    ContrastResult,
    PCMAnalyzer,
    PhyloANOVAResult,
    PhylogeneticSignalResult,
)
from stats.pcoa import PCoAAnalyzer, PCoAResult
from stats.permanova import PERMANOVAAnalyzer, PERMANOVAResult
from stats.simper import SimperAnalyzer, SimperResult
from stats.spatial import RipleyKAnalyzer, SpatialResult
from stats.univariate import (
    ANOVAResult,
    KruskalResult,
    NormalityResult,
    SummaryResult,
    TTestResult,
    UnivariateAnalyzer,
)
from stratigraphy.coniss import CONISSAnalyzer, CONISSResult
from stratigraphy.directional import DirectionalAnalyzer, DirectionalResult
from stratigraphy.markov import MarkovAnalyzer, MarkovResult
from stratigraphy.spectral_analysis import SpectralAnalyzer, SpectralResult
from utils.exceptions import ValidationError
from utils.transformations import (
    boxcox_transform,
    hellinger_transform,
    impute_knn,
    log_transform,
    sqrt_transform,
    zscore_standardize,
)

logger = logging.getLogger(__name__)


class StatisticsController:
    """
    Controller for statistical analysis operations.

    This controller acts as a facade, providing a unified interface to
    all statistical analysis engines while managing state and caching.
    """

    def __init__(self) -> None:
        """Initialize the statistics controller."""
        self._logger = logging.getLogger(f"{__name__}.StatisticsController")
        self._lock = threading.RLock()

        # Initialize analyzers
        self._pca_analyzer = PCAAnalyzer()
        self._pcoa_analyzer = PCoAAnalyzer()
        self._nmds_analyzer = NMDSAnalyzer()
        self._anosim_analyzer = ANOSIMAnalyzer()
        self._permanova_analyzer = PERMANOVAAnalyzer()
        self._cca_analyzer = CCAAnalyzer()
        self._diversity_analyzer = DiversityAnalyzer()
        self._rarefaction_analyzer = RarefactionAnalyzer()
        self._spectral_analyzer = SpectralAnalyzer()
        self._simper_analyzer = SimperAnalyzer()
        self._univariate_analyzer = UnivariateAnalyzer()
        self._lda_analyzer = LDAAnalyzer()
        self._clustering_analyzer = ClusteringAnalyzer()
        self._abundance_fitter = AbundanceModelFitter()
        self._she_analyzer = SHEAnalyzer()
        self._coniss_analyzer = CONISSAnalyzer()
        self._markov_analyzer = MarkovAnalyzer()
        self._directional_analyzer = DirectionalAnalyzer()
        self._efa_analyzer = EFAAnalyzer()
        self._eigenshape_analyzer = EigenshapeAnalyzer()
        self._gpa_analyzer = GPAAnalyzer()
        self._spatial_analyzer = RipleyKAnalyzer()
        self._pcm_analyzer = PCMAnalyzer()

        # State manager
        self._state = get_state_manager()

        self._logger.info("StatisticsController initialized")

    # =========================================================================
    # Shared Analysis Helpers
    # =========================================================================

    def _ensure_data(self, data: npt.NDArray | None) -> npt.NDArray:
        """Get data from state if not provided. Must be called with lock held."""
        if data is None:
            if not self._state.has_data:
                raise ValidationError("No data available.")
            return self._state.data_matrix.data
        return data

    def _column_names_or_fallback(self, data: npt.NDArray) -> list[str]:
        """State column labels when they fit ``data``, else generic names.

        Directly reading ``self._state.data_matrix.col_labels`` crashed with
        AttributeError whenever the caller passed explicit ``data`` while no
        dataset was loaded (or gave a shape mismatch).
        """
        n_vars = data.shape[1] if data.ndim == 2 else 1
        matrix = self._state.data_matrix
        if matrix is not None and len(matrix.col_labels) == n_vars:
            return list(matrix.col_labels)
        return [f"Var_{i + 1}" for i in range(n_vars)]

    def _ensure_distance_matrix(
        self, distance_matrix: npt.NDArray | None, data: npt.NDArray | None, metric: str
    ) -> npt.NDArray:
        """Compute distance matrix if not provided. Must be called with lock held."""
        if distance_matrix is None:
            data = self._ensure_data(data)
            dm = compute_distance_matrix(data, metric=metric)
            return dm.matrix
        return distance_matrix

    # =========================================================================
    # PCA Operations
    # =========================================================================

    def run_pca(
        self, data: npt.NDArray | None = None, n_components: int | None = None, method: str = "covariance"
    ) -> PCAResult:
        """
        Run Principal Component Analysis.

        Parameters:
            data: Input data. If None, uses current state data.
            n_components: Number of components to extract.
            method: 'covariance' or 'correlation'

        Returns:
            PCAResult: PCA analysis results
        """
        with self._lock:
            data = self._ensure_data(data)
            self._logger.info(
                f"run_pca called with data shape={data.shape}, n_components={n_components}, method='{method}'"
            )
            result = self._pca_analyzer.analyze(data, n_components, method)
            self._state.cache_result("pca_result", result)
            self._logger.info(f"PCA completed: variance explained={result.explained_variance}")
            return result

    # =========================================================================
    # PCoA Operations
    # =========================================================================

    def run_pcoa(
        self,
        distance_matrix: npt.NDArray | None = None,
        n_components: int = 10,
        metric: str = "bray_curtis",
        correction: str = "cmdscale",
    ) -> PCoAResult:
        """
        Run Principal Coordinate Analysis.

        Parameters:
            distance_matrix: Distance matrix. If None, computes from state data.
            n_components: Number of coordinates.
            metric: Distance metric for computation.
            correction: How to handle negative eigenvalues — one of
                ``"cmdscale"`` (default; matches R ``cmdscale`` and the
                historical behavior), ``"lingoes"``, ``"wickoff"``,
                ``"torgerson"``. See :mod:`stats.pcoa` for formulas and
                references.

        Returns:
            PCoAResult: PCoA analysis results
        """
        with self._lock:
            self._logger.info(
                f"run_pcoa called with distance matrix shape={distance_matrix.shape if distance_matrix is not None else None}, metric='{metric}', correction='{correction}'"
            )
            distance_matrix = self._ensure_distance_matrix(distance_matrix, None, metric)
            result = self._pcoa_analyzer.analyze(
                distance_matrix, n_components, metric=metric, correction=correction
            )
            self._state.cache_result("pcoa_result", result)
            self._state.cache_result("pcoa_metric", metric)
            self._logger.info(
                f"PCoA completed: {n_components} coordinates extracted (correction={result.correction_method})"
            )
            return result

    # =========================================================================
    # NMDS Operations
    # =========================================================================

    def run_nmds(
        self,
        distance_matrix: npt.NDArray | None = None,
        n_dimensions: int = 2,
        metric: str = "bray_curtis",
        n_restarts: int = 10,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        random_seed: int | None = None,
        progress_callback=None,
    ) -> NMDSResult:
        """
        Run Non-metric Multidimensional Scaling.

        Parameters:
            distance_matrix: Distance matrix. If None, computes from state data.
            n_dimensions: Number of dimensions for ordination.
            metric: Distance metric for computation.
            n_restarts: Number of random restarts.
            random_seed: Random seed for reproducibility.
            progress_callback: Optional callable(restart_index, total_restarts,
                stress) forwarded to the analyzer; invoked from the worker
                thread, so GUI callers must marshal via a signal.

        Returns:
            NMDSResult: NMDS analysis results
        """
        with self._lock:
            self._logger.info(
                f"run_nmds called with distance matrix shape={distance_matrix.shape if distance_matrix is not None else None}, n_dimensions={n_dimensions}, n_restarts={n_restarts}"
            )
            distance_matrix = self._ensure_distance_matrix(distance_matrix, None, metric)
            result = self._nmds_analyzer.analyze(
                distance_matrix,
                n_dimensions=n_dimensions,
                metric=metric,
                n_restarts=n_restarts,
                max_iterations=max_iterations,
                random_seed=random_seed,
                tolerance=tolerance,
                progress_callback=progress_callback,
            )
            self._state.cache_result("nmds_result", result)
            self._state.cache_result("nmds_metric", metric)
            self._logger.info(f"NMDS completed with stress={result.stress:.6f}")
            return result

    # =========================================================================
    # Group Comparison Tests
    # =========================================================================

    def run_anosim(
        self,
        groups: list[int],
        distance_matrix: npt.NDArray | None = None,
        metric: str = "bray_curtis",
        n_permutations: int = 9999,
    ) -> ANOSIMResult:
        """
        Run Analysis of Similarities (ANOSIM).

        Parameters:
            groups: Group assignments for each sample.
            distance_matrix: Distance matrix. If None, computes from state data.
            metric: Distance metric.
            n_permutations: Number of permutations for p-value.

        Returns:
            ANOSIMResult: ANOSIM analysis results
        """
        with self._lock:
            distance_matrix = self._ensure_distance_matrix(distance_matrix, None, metric)
            result = self._anosim_analyzer.analyze(
                distance_matrix, groups, n_permutations=n_permutations, metric=metric
            )
            self._state.cache_result("anosim_result", result)
            return result

    def run_permanova(
        self,
        groups: list[int],
        distance_matrix: npt.NDArray | None = None,
        metric: str = "bray_curtis",
        n_permutations: int = 9999,
    ) -> PERMANOVAResult:
        """
        Run Permutational MANOVA (PERMANOVA).

        Parameters:
            groups: Group assignments for each sample.
            distance_matrix: Distance matrix. If None, computes from state data.
            metric: Distance metric.
            n_permutations: Number of permutations for p-value.

        Returns:
            PERMANOVAResult: PERMANOVA analysis results
        """
        with self._lock:
            distance_matrix = self._ensure_distance_matrix(distance_matrix, None, metric)
            result = self._permanova_analyzer.analyze(
                distance_matrix, groups, n_permutations=n_permutations, metric=metric
            )
            self._state.cache_result("permanova_result", result)
            return result

    # =========================================================================
    # Diversity Analysis
    # =========================================================================

    def analyze_diversity(self, abundances: npt.NDArray | None = None, sample_name: str = "Sample") -> DiversityResult:
        """
        Analyze diversity indices for a sample.

        Parameters:
            abundances: Abundance array. If None, uses first row of state data.
            sample_name: Name for the sample.

        Returns:
            DiversityResult: Diversity analysis results
        """
        with self._lock:
            if abundances is None:
                if not self._state.has_data:
                    self._logger.error("analyze_diversity called with no data available")
                    raise ValidationError("No data available.")
                abundances = self._state.data_matrix.data[0]
            self._logger.info(
                f"analyze_diversity called for sample '{sample_name}' with {len(abundances)} abundance values"
            )
            result = compute_diversity_indices(abundances, sample_name)
            self._state.cache_result(f"diversity_{sample_name}", result)
            self._logger.info(f"Diversity analysis completed for sample '{sample_name}'")
            return result

    def analyze_rarefaction(
        self, abundances: npt.NDArray | None = None, sample_name: str = "Sample", n_points: int = 50
    ) -> RarefactionResult:
        """
        Compute rarefaction curve.

        Parameters:
            abundances: Abundance array. If None, uses first row of state data.
            sample_name: Name for the sample.
            n_points: Number of points on the curve.

        Returns:
            RarefactionResult: Rarefaction analysis results
        """
        with self._lock:
            if abundances is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                abundances = self._state.data_matrix.data[0]
            result = self._rarefaction_analyzer.analyze(abundances, sample_name, n_points=n_points)
            self._state.cache_result(f"rarefaction_{sample_name}", result)
            return result

    # =========================================================================
    # Distance Matrix Operations
    # =========================================================================

    def compute_distances(self, data: npt.NDArray | None = None, metric: str = "bray_curtis") -> DistanceMatrixResult:
        """
        Compute distance matrix.

        Parameters:
            data: Input data. If None, uses state data.
            metric: Distance metric.

        Returns:
            DistanceMatrixResult: Computed distance matrix
        """
        with self._lock:
            data = self._ensure_data(data)
            result = compute_distance_matrix(data, metric=metric)
            self._state.cache_result("distance_matrix", result)
            self._state.cache_result("distance_metric", metric)
            return result

    # =========================================================================
    # Spectral Analysis
    # =========================================================================

    def analyze_spectral(
        self,
        data: npt.NDArray | None = None,
        frequency_range: tuple[float, float] | None = None,
        n_frequencies: int = 1000,
    ) -> SpectralResult:
        """
        Perform spectral analysis using Lomb-Scargle periodogram.

        Parameters:
            data: Input data. If None, uses state data. First column is treated
                  as time, remaining columns as signal values.
            frequency_range: Tuple of (min_freq, max_freq). If None, auto-calculated.
            n_frequencies: Number of frequencies to evaluate.

        Returns:
            SpectralResult: Spectral analysis results
        """
        with self._lock:
            data = self._ensure_data(data)
            self._logger.info(f"analyze_spectral called with data shape={data.shape}")
            if data.ndim == 2:
                time = data[:, 0]
                values = data[:, 1] if data.shape[1] > 1 else data[:, 0]
            else:
                time = np.arange(len(data), dtype=float)
                values = data
            result = self._spectral_analyzer.analyze(
                time, values, frequency_range=frequency_range, n_frequencies=n_frequencies
            )
            self._state.cache_result("spectral_result", result)
            self._logger.info(f"Spectral analysis completed: peak frequency={result.peak_frequency}")
            return result

    # =========================================================================
    # Group Comparison (data-based wrappers)
    # =========================================================================

    def analyze_anosim(
        self,
        data: npt.NDArray | None = None,
        groups: list[int] | None = None,
        metric: str = "bray_curtis",
        n_permutations: int = 9999,
    ) -> ANOSIMResult:
        """
        Run ANOSIM analysis from raw data.

        Parameters:
            data: Input data matrix. If None, uses state data.
            groups: Group assignments. If None, assigns all samples to one group.
            metric: Distance metric.
            n_permutations: Number of permutations.

        Returns:
            ANOSIMResult: ANOSIM analysis results
        """
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            distance_matrix = self._ensure_distance_matrix(None, data, metric)
            return self.run_anosim(
                groups=groups, distance_matrix=distance_matrix, metric=metric, n_permutations=n_permutations
            )

    def analyze_permanova(
        self,
        data: npt.NDArray | None = None,
        groups: list[int] | None = None,
        metric: str = "bray_curtis",
        n_permutations: int = 9999,
    ) -> PERMANOVAResult:
        """
        Run PERMANOVA analysis from raw data.

        Parameters:
            data: Input data matrix. If None, uses state data.
            groups: Group assignments. If None, assigns all samples to one group.
            metric: Distance metric.
            n_permutations: Number of permutations.

        Returns:
            PERMANOVAResult: PERMANOVA analysis results
        """
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            distance_matrix = self._ensure_distance_matrix(None, data, metric)
            return self.run_permanova(
                groups=groups, distance_matrix=distance_matrix, metric=metric, n_permutations=n_permutations
            )

    # =========================================================================
    # SIMPER Analysis
    # =========================================================================

    def analyze_simper(
        self,
        data: npt.NDArray | None = None,
        groups: list[int] | None = None,
        metric: str = "bray_curtis",
    ) -> SimperResult:
        """Run SIMPER analysis to identify discriminating variables."""
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            result = self._simper_analyzer.analyze(data, groups, metric=metric)
            self._state.cache_result("simper_result", result)
            return result

    # =========================================================================
    # Univariate Statistics
    # =========================================================================

    def analyze_univariate_summary(
        self, data: npt.NDArray | None = None, column_names: list[str] | None = None
    ) -> SummaryResult:
        """Compute summary statistics for each variable."""
        with self._lock:
            data = self._ensure_data(data)
            if column_names is None:
                column_names = self._column_names_or_fallback(data)
            result = self._univariate_analyzer.summary_statistics(data, column_names)
            self._state.cache_result("univariate_summary", result)
            return result

    def analyze_normality(
        self, data: npt.NDArray | None = None, column_names: list[str] | None = None
    ) -> list[NormalityResult]:
        """Run normality tests on each variable."""
        with self._lock:
            data = self._ensure_data(data)
            if column_names is None:
                column_names = self._column_names_or_fallback(data)
            n_vars = data.shape[1] if data.ndim == 2 else 1
            results = [self._univariate_analyzer.normality_test(data, column=i) for i in range(n_vars)]
            self._state.cache_result("normality_results", results)
            return results

    def analyze_t_test(
        self, data: npt.NDArray | None = None, groups: list[int] | None = None, paired: bool = False
    ) -> list[TTestResult]:
        """Run t-tests for each variable between two groups."""
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            n_vars = data.shape[1] if data.ndim == 2 else 1
            results = [
                self._univariate_analyzer.t_test(data, column=i, groups=groups, paired=paired) for i in range(n_vars)
            ]
            self._state.cache_result("t_test_results", results)
            return results

    def analyze_anova(self, data: npt.NDArray | None = None, groups: list[int] | None = None) -> list[ANOVAResult]:
        """Run one-way ANOVA for each variable."""
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            n_vars = data.shape[1] if data.ndim == 2 else 1
            results = [self._univariate_analyzer.one_way_anova(data, groups, column=i) for i in range(n_vars)]
            self._state.cache_result("anova_results", results)
            return results

    def analyze_kruskal_wallis(
        self, data: npt.NDArray | None = None, groups: list[int] | None = None
    ) -> list[KruskalResult]:
        """Run Kruskal-Wallis test for each variable."""
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            n_vars = data.shape[1] if data.ndim == 2 else 1
            results = [self._univariate_analyzer.kruskal_wallis(data, groups, column=i) for i in range(n_vars)]
            self._state.cache_result("kruskal_results", results)
            return results

    # =========================================================================
    # LDA / CVA
    # =========================================================================

    def analyze_lda(
        self,
        data: npt.NDArray | None = None,
        groups: list[int] | None = None,
        n_components: int | None = None,
        cross_validate: bool = True,
        cv_folds: int = 5,
    ) -> LDAResult:
        """Run Linear Discriminant Analysis.

        Parameters:
            cross_validate: When False, the leave-group-out accuracy check is
                skipped. The engine skips cross-validation whenever the
                effective fold count drops below 2, so this maps to
                ``cv_folds=1`` rather than to a special flag.
            cv_folds: Requested fold count; the engine clamps it to the size
                of the smallest group.
        """
        with self._lock:
            data = self._ensure_data(data)
            if groups is None:
                groups = [0] * data.shape[0]
            result = self._lda_analyzer.analyze(
                data,
                groups,
                n_components=n_components,
                cv_folds=cv_folds if cross_validate else 1,
            )
            self._state.cache_result("lda_result", result)
            return result

    # =========================================================================
    # CCA / RDA
    # =========================================================================

    def run_cca(
        self,
        Y: npt.NDArray | None = None,
        X: npt.NDArray | None = None,
        n_components: int | None = None,
        method: str = "cca",
        species_names: list[str] | None = None,
        env_names: list[str] | None = None,
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> CCAResult:
        """
        Run Canonical Correspondence Analysis (CCA) or Redundancy Analysis (RDA).

        Parameters:
            Y: Species abundance matrix (n_samples, n_species). If None, uses state data.
            X: Environmental variable matrix (n_samples, n_env). If None, uses columns
               from state data not used for Y.
            n_components: Number of constrained axes to extract.
            method: 'cca' or 'rda'
            species_names: Names of species/variables.
            env_names: Names of environmental variables.
            n_permutations: Permutation count for the significance test. The
                engine permutes the RESPONSE (Y), never the predictors (X).
            random_seed: Seed for the permutation test. When None the engine
                warns and the p-value is not reproducible.

        Returns:
            CCAResult: CCA/RDA analysis results, including ``f_statistic``,
            ``p_value``, per-axis F/p and Wilks' lambda.
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available. Please load data first.")

            if Y is None:
                data = self._state.data_matrix.data
                # Default: first half as species, second half as env
                mid = max(1, data.shape[1] // 2)
                Y = data[:, :mid]
                X = data[:, mid : mid + min(mid, data.shape[1] - mid)]

            if X is None:
                raise ValidationError("Environmental variables required for CCA/RDA")

            self._logger.info(
                f"run_cca called with Y.shape={Y.shape}, X.shape={X.shape}, method={method}, "
                f"n_permutations={n_permutations}, random_seed={random_seed}"
            )

            result = self._cca_analyzer.analyze(
                Y,
                X,
                n_components=n_components,
                method=method,
                species_names=species_names,
                env_names=env_names,
                n_permutations=n_permutations,
                random_seed=random_seed,
            )

            self._state.cache_result("cca_result", result)
            self._state.cache_result("cca_method", method)

            self._logger.info(
                f"CCA/RDA completed: constrained variance={result.constrained_variance:.2f}%, "
                f"p={getattr(result, 'p_value', float('nan')):.4f}, "
                f"Wilks lambda={getattr(result, 'wilks_lambda', float('nan')):.4f}"
            )
            return result

    def analyze_cca(
        self,
        species_data: npt.NDArray | None = None,
        env_data: npt.NDArray | None = None,
        n_components: int | None = None,
        method: str = "cca",
    ) -> CCAResult:
        """
        Run CCA/RDA analysis from raw data matrices.

        This is a wrapper around run_cca that accepts separate species
        and environmental data.
        """
        with self._lock:
            if species_data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data
                mid = max(1, data.shape[1] // 2)
                species_data = data[:, :mid]
                # Only auto-split the env half when the caller did not
                # supply env_data — previously this clobbered a caller-
                # provided env_data whenever species_data was omitted.
                if env_data is None:
                    env_data = data[:, mid : mid + min(mid, data.shape[1] - mid)]

            return self.run_cca(Y=species_data, X=env_data, n_components=n_components, method=method)

    # =========================================================================
    # Spatial Point Pattern Analysis (Ripley's K)
    # =========================================================================

    def analyze_spatial_ripley_k(
        self,
        coords: npt.NDArray | None = None,
        r_max: float | None = None,
        n_r_values: int = 50,
        n_simulations: int = 99,
    ) -> SpatialResult:
        """
        Run Ripley's K spatial point pattern analysis.

        Parameters:
            coords: Point coordinates (n_points, 2). If None, uses first 2 columns
                   of state data.
            r_max: Maximum distance for analysis.
            n_r_values: Number of distance values.
            n_simulations: Monte Carlo simulations for envelope.

        Returns:
            SpatialResult: Ripley's K analysis results
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available. Please load data first.")

            if coords is None:
                data = self._state.data_matrix.data
                coords = data[:, :2]  # Use first 2 columns as x, y

            self._logger.info(f"analyze_spatial_ripley_k called: n_points={coords.shape[0]}, r_max={r_max}")

            result = self._spatial_analyzer.analyze(
                coords,
                r_max=r_max,
                n_r_values=n_r_values,
                n_simulations=n_simulations,
            )

            self._state.cache_result("spatial_result", result)

            self._logger.info(f"RipleyK completed: {result.interpretation[:50]}")
            return result

    # =========================================================================
    # Data Transformations
    # =========================================================================

    def transform_data(
        self,
        data: npt.NDArray | None = None,
        method: str = "hellinger",
        **kwargs,
    ) -> npt.NDArray:
        """Apply data transformation."""
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data.copy()
            transforms = {
                "log": lambda d: log_transform(d, base=kwargs.get("base", 10)),
                "sqrt": sqrt_transform,
                "zscore": zscore_standardize,
                "hellinger": hellinger_transform,
                "boxcox": lambda d: (
                    np.column_stack([boxcox_transform(d[:, c])[0] for c in range(d.shape[1])])
                    if d.ndim == 2
                    else boxcox_transform(d)[0]
                ),
            }
            fn = transforms.get(method)
            if fn is None:
                raise ValidationError(f"Unknown transformation: {method}")
            result = fn(data)
            self._state.cache_result("transformed_data", result)
            return result

    def impute_missing(self, data: npt.NDArray | None = None, method: str = "knn", **kwargs) -> npt.NDArray:
        """Impute missing values."""
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data.copy()
            if method == "knn":
                result = impute_knn(data, k=kwargs.get("k", 5))
            else:
                from utils.transformations import impute_column_mean

                result = impute_column_mean(data)
            self._state.cache_result("imputed_data", result)
            return result

    # =========================================================================
    # Hierarchical Clustering
    # =========================================================================

    def analyze_clustering(
        self,
        data: npt.NDArray | None = None,
        n_clusters: int = 3,
        method: str = "ward",
        metric: str = "euclidean",
    ) -> ClusteringResult:
        """Run hierarchical clustering."""
        with self._lock:
            data = self._ensure_data(data)
            result = self._clustering_analyzer.analyze(data, n_clusters=n_clusters, method=method, metric=metric)
            self._state.cache_result("clustering_result", result)
            return result

    # =========================================================================
    # Abundance Distribution Models
    # =========================================================================

    def analyze_abundance_models(self, abundances: npt.NDArray | None = None) -> dict[str, AbundanceModelFit]:
        """Fit species-abundance distribution models."""
        with self._lock:
            if abundances is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                abundances = np.sum(self._state.data_matrix.data, axis=0)
            result = self._abundance_fitter.fit_all(abundances)
            self._state.cache_result("abundance_models", result)
            return result

    # =========================================================================
    # SHE Analysis
    # =========================================================================

    def analyze_she(self, data: npt.NDArray | None = None) -> SHEResult:
        """Run SHE analysis."""
        with self._lock:
            data = self._ensure_data(data)
            result = self._she_analyzer.analyze(data)
            self._state.cache_result("she_result", result)
            return result

    # =========================================================================
    # Stratigraphy: CONISS
    # =========================================================================

    def analyze_coniss(
        self,
        data: npt.NDArray | None = None,
        n_zones: int = 4,
        depths: npt.NDArray | None = None,
    ) -> CONISSResult:
        """Run CONISS constrained clustering.

        ``CONISSAnalyzer.analyze`` returns the legacy 2-tuple
        ``(CONISSResult, broken_stick_dict | None)`` — that shape is locked in
        by ``tests/stratigraphy/test_coniss_broken_stick.py::test_analyze_return_type``,
        so it is unpacked here rather than changed at the source.

        This controller previously passed the tuple straight through, so
        ``views/ui_main_window.py`` calling ``result.linkage_matrix`` raised
        ``AttributeError: 'tuple' object has no attribute 'linkage_matrix'``
        every time CONISS was run from the ribbon. The broken-stick payload is
        cached separately so it stays reachable.
        """
        with self._lock:
            data = self._ensure_data(data)
            outcome = self._coniss_analyzer.analyze(data, n_zones=n_zones, depths=depths)

            if isinstance(outcome, tuple):
                result, broken_stick = outcome[0], outcome[1] if len(outcome) > 1 else None
            else:
                result, broken_stick = outcome, None

            self._state.cache_result("coniss_result", result)
            if broken_stick is not None:
                self._state.cache_result("coniss_broken_stick", broken_stick)
            return result

    # =========================================================================
    # Stratigraphy: Markov Chain
    # =========================================================================

    def analyze_markov(
        self,
        sequence: list[int] | npt.NDArray | None = None,
        facies_names: list[str] | None = None,
    ) -> MarkovResult:
        """Run Markov chain analysis on a facies sequence."""
        with self._lock:
            if sequence is None:
                data = self._ensure_data(None)
                sequence = np.argmax(data, axis=1)
            result = self._markov_analyzer.analyze(sequence, facies_names=facies_names)
            self._state.cache_result("markov_result", result)
            return result

    # =========================================================================
    # Stratigraphy: Directional Statistics
    # =========================================================================

    def analyze_directional(
        self,
        angles_deg: npt.NDArray | None = None,
        column_index: int = 0,
    ) -> DirectionalResult:
        """Run directional (circular) statistics.

        Parameters:
            angles_deg: Optional pre-extracted angle array. If None,
                the column at ``column_index`` of the state data is
                used.
            column_index: Zero-based column of the state data matrix
                to read angles from when ``angles_deg`` is None.
        """
        with self._lock:
            if angles_deg is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data
                if column_index < 0 or column_index >= data.shape[1]:
                    raise ValidationError(
                        f"column_index {column_index} out of range for data with {data.shape[1]} columns."
                    )
                angles_deg = data[:, column_index]
            result = self._directional_analyzer.analyze(angles_deg)
            self._state.cache_result("directional_result", result)
            return result

    def bin_rose_diagram(
        self,
        angles_deg: npt.NDArray | None = None,
        n_bins: int = 12,
        column_index: int = 0,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Bin angles for rose diagram.

        Parameters:
            angles_deg: Optional pre-extracted angle array. If None,
                the column at ``column_index`` of the state data is
                used.
            column_index: Zero-based column of the state data matrix
                to read angles from when ``angles_deg`` is None.
        """
        with self._lock:
            if angles_deg is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data
                if column_index < 0 or column_index >= data.shape[1]:
                    raise ValidationError(
                        f"column_index {column_index} out of range for data with {data.shape[1]} columns."
                    )
                angles_deg = data[:, column_index]
            return self._directional_analyzer.bin_for_rose(angles_deg, n_bins=n_bins)

    # =========================================================================
    # Morphometrics: EFA
    # =========================================================================

    def analyze_efa(
        self,
        contour: npt.NDArray | None = None,
        n_harmonics: int = 10,
        n_points: int = 200,
    ) -> EFAResult:
        """Run Elliptic Fourier Analysis."""
        with self._lock:
            if contour is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                contour = self._state.data_matrix.data[:, :2]
            result = self._efa_analyzer.analyze(contour, n_harmonics=n_harmonics, n_points=n_points)
            self._state.cache_result("efa_result", result)
            return result

    def analyze_eigenshape(
        self,
        efa_coefficients_list: list[npt.NDArray] | None = None,
        n_components: int | None = None,
    ) -> EigenshapeResult:
        """Run Eigenshape analysis on EFA coefficients."""
        with self._lock:
            if efa_coefficients_list is None:
                # EFA results are cached under "efa_result" (see analyze_efa);
                # "efa_coefficients_list" was never written anywhere.
                cached = self._state.get_cached_result("efa_result")
                if cached is None:
                    raise ValidationError("No EFA coefficients available. Run EFA first.")
                efa_coefficients_list = [cached.coefficients]
            result = self._eigenshape_analyzer.analyze(efa_coefficients_list, n_components=n_components)
            self._state.cache_result("eigenshape_result", result)
            return result

    # =========================================================================
    # Morphometrics: GPA
    # =========================================================================

    def analyze_gpa(
        self,
        data: npt.NDArray | None = None,
        n_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> GPAResult:
        """
        Run Generalized Procrustes Analysis.

        Parameters:
            data: Input data. If None, uses state data.
                  Expected shape: (n_specimens, n_landmarks, n_dims) for 3D,
                  or will attempt to reshape 2D data.
            n_iterations: Maximum iterations.
            tolerance: Convergence tolerance.

        Returns:
            GPAResult: GPA analysis results
        """
        with self._lock:
            data = self._ensure_data(data)
            if data.ndim == 2:
                n_rows, n_cols = data.shape
                if n_cols % 2 == 0:
                    n_landmarks = n_cols // 2
                    data = data.reshape(n_rows, n_landmarks, 2)
                else:
                    raise ValidationError(
                        f"Cannot reshape 2D data with {n_cols} columns into landmark configurations. Expected even number of columns (x, y pairs)."
                    )
            result = self._gpa_analyzer.analyze(data, n_iterations=n_iterations, tolerance=tolerance)
            self._state.cache_result("gpa_result", result)
            return result

    # =========================================================================
    # Phylogenetic Comparative Methods (PCM)
    # =========================================================================

    def analyze_pic(
        self,
        tree_newick: str,
        trait_values: dict[str, float],
    ) -> ContrastResult:
        """
        Compute Phylogenetic Independent Contrasts.

        Parameters:
            tree_newick: Newick-format phylogenetic tree string
            trait_values: {taxon_name: trait_value} dictionary

        Returns:
            ContrastResult with contrasts and standard errors
        """
        from phylogenetics.tree import PhyloTree

        with self._lock:
            tree = PhyloTree.from_newick(tree_newick)
            result = self._pcm_analyzer.compute_contrasts(tree, trait_values)
            self._logger.info(f"PIC computed: {result.n_contrasts} contrasts")
            return result

    def analyze_ancestral_states(
        self,
        tree_newick: str,
        trait_values: dict[str, float],
        model: str = "bm",
    ) -> AncestralStateResult:
        """
        Reconstruct ancestral states via weighted squared-change parsimony.

        Parameters:
            tree_newick: Newick-format phylogenetic tree string
            trait_values: {taxon_name: trait_value} dictionary
            model: Evolution model ('bm' or 'ou')

        Returns:
            AncestralStateResult with reconstructed states
        """
        from phylogenetics.tree import PhyloTree

        with self._lock:
            tree = PhyloTree.from_newick(tree_newick)
            result = self._pcm_analyzer.reconstruct_ancestral_states(tree, trait_values, model=model)
            self._logger.info(f"ASR completed: {len(result.node_states)} internal nodes")
            return result

    def analyze_phylogenetic_signal(
        self,
        tree_newick: str,
        trait_values: dict[str, float],
        n_randomizations: int = 999,
    ) -> PhylogeneticSignalResult:
        """
        Measure phylogenetic signal using Blomberg's K.

        Parameters:
            tree_newick: Newick-format phylogenetic tree string
            trait_values: {taxon_name: trait_value} dictionary
            n_randomizations: Permutations for significance test

        Returns:
            PhylogeneticSignalResult with K, Z-score, p-value
        """
        from phylogenetics.tree import PhyloTree

        with self._lock:
            tree = PhyloTree.from_newick(tree_newick)
            result = self._pcm_analyzer.compute_phylogenetic_signal(
                tree, trait_values, n_randomizations=n_randomizations
            )
            self._logger.info(f"Blomberg's K = {result.k:.4f}")
            return result

    def analyze_phylo_anova(
        self,
        tree_newick: str,
        trait_values: dict[str, float],
        group_labels: dict[str, str],
        n_permutations: int = 999,
    ) -> PhyloANOVAResult:
        """
        Phylogenetic ANOVA: test for trait differences between groups.

        Parameters:
            tree_newick: Newick-format phylogenetic tree string
            trait_values: {taxon_name: trait_value} dictionary
            group_labels: {taxon_name: group_name} dictionary
            n_permutations: Permutations for significance test

        Returns:
            PhyloANOVAResult with F-statistic and p-value
        """
        from phylogenetics.tree import PhyloTree

        with self._lock:
            tree = PhyloTree.from_newick(tree_newick)
            result = self._pcm_analyzer.phylogenetic_anova(
                tree, trait_values, group_labels, n_permutations=n_permutations
            )
            self._logger.info(f"Phylo-ANOVA: F={result.f_statistic:.4f}, p={result.p_value:.4f}")
            return result

    # =========================================================================
    # Macroevolution & 3-D morphometrics
    #
    # These engines (macroevolution/, morpho3d/) were implemented and tested
    # but had no entry point, so the README's "FBD Process", "Cohort
    # Survivorship", "Diversity Dynamics" and "2D/3D GPA" were unreachable
    # from the running application. The methods below are thin dispatchers;
    # the engines stay the single source of truth.
    # =========================================================================

    def _records_from_matrix(self, fad_col: int, lad_col: int) -> list[tuple[float, float]]:
        """Read (FAD, LAD) ranges for each row from the loaded matrix.

        FAD = first appearance date, LAD = last appearance date, both in Ma
        (older = larger), matching the convention used by
        stratigraphy.time_bins and macroevolution.cohort.
        """
        data = self._ensure_data(None)
        n_vars = data.shape[1]
        for col, name in ((fad_col, "FAD"), (lad_col, "LAD")):
            if not 0 <= col < n_vars:
                raise ValidationError(f"Column {col} ({name}) is out of range: the data has {n_vars} column(s)")
        return [(float(r[fad_col]), float(r[lad_col])) for r in data]

    def analyze_cohort_survivorship(
        self,
        data: npt.NDArray | None = None,
        fad_column: int = 0,
        lad_column: int = 1,
        n_intervals: int = 4,
        confidence_level: float = 0.95,
    ) -> Any:
        """Foote (1997, 2000) cohort survivorship.

        The taxon ranges are read from two columns of the loaded matrix
        (FAD, LAD); time intervals are derived from the observed age span so
        the caller does not have to invent bin edges.
        """
        from macroevolution.cohort import analyze_cohort_survivorship

        with self._lock:
            if data is not None:
                self._state.set_data_matrix(data)
            records = self._records_from_matrix(fad_column, lad_column)
            if not records:
                raise ValidationError("No taxon ranges available for cohort analysis")

            ages = np.array([r for rec in records for r in rec], dtype=float)
            lo, hi = float(np.nanmin(ages)), float(np.nanmax(ages))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValidationError("Taxon ranges have no usable age span; check the FAD/LAD columns")
            edges = np.linspace(hi, lo, int(n_intervals) + 1)  # oldest first
            intervals = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

            result = analyze_cohort_survivorship(records, intervals, confidence_level=confidence_level)
            self._state.cache_result("cohort_survivorship_result", result)
            return result

    def analyze_diversity_dynamics(
        self,
        data: npt.NDArray | None = None,
        fad_column: int = 0,
        lad_column: int = 1,
        n_intervals: int = 8,
    ) -> Any:
        """Foote per-capita diversity / origination / extinction over time bins."""
        from macroevolution.diversity import DiversityDynamics

        with self._lock:
            if data is not None:
                self._state.set_data_matrix(data)
            records = self._records_from_matrix(fad_column, lad_column)
            if not records:
                raise ValidationError("No taxon ranges available for diversity dynamics")

            ages = np.array([r for rec in records for r in rec], dtype=float)
            lo, hi = float(np.nanmin(ages)), float(np.nanmax(ages))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValidationError("Taxon ranges have no usable age span")
            edges = np.linspace(hi, lo, int(n_intervals) + 1)
            intervals = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

            result = DiversityDynamics().estimate_diversity(records, intervals)
            self._state.cache_result("diversity_dynamics_result", result)
            return result

    def analyze_survival(
        self,
        data: npt.NDArray | None = None,
        time_column: int = 0,
        event_column: int = 1,
    ) -> Any:
        """Kaplan-Meier survival curve from a duration + event-indicator pair."""
        from macroevolution.survival import KaplanMeierAnalyzer

        with self._lock:
            if data is not None:
                self._state.set_data_matrix(data)
            matrix = self._ensure_data(None)
            n_vars = matrix.shape[1]
            for col, name in ((time_column, "duration"), (event_column, "event")):
                if not 0 <= col < n_vars:
                    raise ValidationError(f"Column {col} ({name}) is out of range: the data has {n_vars} column(s)")
            times = np.asarray(matrix[:, time_column], dtype=float)
            events = np.asarray(matrix[:, event_column], dtype=float)
            keep = np.isfinite(times) & np.isfinite(events)
            times, events = times[keep], events[keep]
            if times.size == 0:
                raise ValidationError("No finite duration/event rows for survival analysis")

            result = KaplanMeierAnalyzer().fit(times, (events > 0).astype(int))
            self._state.cache_result("survival_result", result)
            return result

    def compare_survival_groups(
        self,
        data: npt.NDArray | None = None,
        time_a: int = 0,
        event_a: int = 1,
        time_b: int = 2,
        event_b: int = 3,
    ) -> Any:
        """Log-rank test between two duration/event column pairs."""
        from macroevolution.survival import log_rank_test

        with self._lock:
            if data is not None:
                self._state.set_data_matrix(data)
            matrix = self._ensure_data(None)
            n_vars = matrix.shape[1]
            for col, name in ((time_a, "A duration"), (event_a, "A event"), (time_b, "B duration"), (event_b, "B event")):
                if not 0 <= col < n_vars:
                    raise ValidationError(f"Column {col} ({name}) is out of range: the data has {n_vars} column(s)")

            def _pair(t_col: int, e_col: int) -> tuple[np.ndarray, np.ndarray]:
                t = np.asarray(matrix[:, t_col], dtype=float)
                e = np.asarray(matrix[:, e_col], dtype=float)
                keep = np.isfinite(t) & np.isfinite(e)
                return t[keep], (e[keep] > 0).astype(int)

            t1, e1 = _pair(time_a, event_a)
            t2, e2 = _pair(time_b, event_b)
            if t1.size == 0 or t2.size == 0:
                raise ValidationError("One of the two groups has no finite duration/event rows")

            result = log_rank_test(t1, e1, t2, e2)
            self._state.cache_result("survival_logrank_result", result)
            return result

    def simulate_fbd(
        self,
        speciation_rate: float = 0.5,
        extinction_rate: float = 0.2,
        fossilization_rate: float = 0.1,
        duration: float = 10.0,
        n_replicates: int = 1,
        random_seed: int | None = 42,
    ) -> Any:
        """Gillespie simulation of the fossilised birth-death process."""
        from macroevolution.fbd import simulate_fbd_process

        with self._lock:
            results = simulate_fbd_process(
                lambda_=speciation_rate,
                mu=extinction_rate,
                psi=fossilization_rate,
                duration=duration,
                n_replicates=int(n_replicates),
                random_seed=random_seed,
            )
            self._state.cache_result("fbd_result", results)
            return results

    def analyze_gpa3d(self, configurations: Any, **kwargs: Any) -> Any:
        """Generalized Procrustes analysis on 3-D landmark configurations."""
        from morpho3d.gpa3d import GPA3D

        with self._lock:
            result = GPA3D(**kwargs).analyze(configurations)
            self._state.cache_result("gpa3d_result", result)
            return result

    def run_tps3d(self, source: Any, target: Any, **kwargs: Any) -> Any:
        """Thin-plate spline in 3-D with a Jacobian and deformation grid."""
        from morpho3d.tps3d import TPS3D

        with self._lock:
            tps = TPS3D(**kwargs)
            tps.fit(source, target)
            result = tps.create_deformation_grid()
            self._state.cache_result("tps3d_result", result)
            return result

    # =========================================================================
    # Morphometrics: Allometry & Two-Block PLS (Integration)
    # =========================================================================

    def _resolve_aligned_configurations(
        self,
        aligned: npt.NDArray | None,
        n_dims: int = 2,
    ) -> npt.NDArray:
        """Return a (n, k, m) Procrustes-aligned array.

        Falls back to the most recent ``gpa_result`` cached in state
        when ``aligned`` is ``None``. A 3D array is taken as-is; a 2D
        array is split into landmarks, and ``n_dims`` says how many
        coordinates each landmark has. Raises a clear
        :class:`ValidationError` when no input is available. The caller
        must hold ``self._lock``.

        ``n_dims`` exists because a flattened (n, k*m) array is ambiguous,
        and guessing wrong is silent. This used to hard-code 2, so a
        flattened 3D configuration -- 5 specimens x 8 landmarks x 3
        dimensions, the ordinary case in morphometrics -- was reshaped to
        (5, 12, 2): 12 landmarks instead of 8, 2 dimensions instead of 3,
        no exception and no warning. Allometry then regressed that
        invented shape on log centroid size and returned a number for a
        dataset that does not exist. The old "must have an even number of
        columns" guard did not help either: 8*3 = 24 is even.
        """
        if aligned is not None:
            arr = np.asarray(aligned, dtype=float)
            if arr.ndim == 2:
                n_rows, n_cols = arr.shape
                if n_dims < 2:
                    raise ValidationError(f"n_dims must be at least 2, got {n_dims}")
                if n_cols % n_dims != 0:
                    raise ValidationError(
                        f"2D aligned data has {n_cols} columns, which is not a "
                        f"multiple of n_dims={n_dims}. A flattened "
                        f"(n_specimens, n_landmarks x n_dims) array needs "
                        f"n_cols = n_landmarks * {n_dims}; pass the 3D array "
                        f"directly, or set n_dims to match the data."
                    )
                arr = arr.reshape(n_rows, n_cols // n_dims, n_dims)
            if arr.ndim != 3:
                raise ValidationError(
                    "Aligned configurations must be a 3D array "
                    "(n_specimens, n_landmarks, n_dims)."
                )
            return arr

        cached = self._state.get_cached_result("gpa_result")
        if cached is None:
            raise ValidationError(
                "No aligned configurations available. "
                "Run Generalized Procrustes Analysis (GPA) first, or pass "
                "explicit configurations to analyze_allometry."
            )
        aligned_attr = getattr(cached, "aligned_configurations", None)
        if aligned_attr is None:
            raise ValidationError(
                "The cached GPA result has no aligned_configurations attribute."
            )
        return np.asarray(aligned_attr, dtype=float)

    def analyze_allometry(
        self,
        aligned_configurations: npt.NDArray | None = None,
        n_components: int | None = None,
        n_dims: int = 2,
    ) -> Any:
        """Multivariate regression of Procrustes shape on log centroid size.

        Parameters:
            aligned_configurations: Optional ``(n, k, m)`` array from GPA.
                When ``None``, the controller falls back to the most
                recent ``gpa_result`` cached in state.
            n_components: Optional PCA-reduced dimensionality for the
                regression. ``None`` keeps the full Procrustes shape
                vector (default).
            n_dims: Coordinates per landmark, used **only** when
                ``aligned_configurations`` is a flattened 2D array. Pass the
                3D array and this is ignored. The default of 2 is right for
                planar landmarks; for 3D landmarks pass 3, because the
                flattened shape alone does not say which it is.

        Returns:
            :class:`morphometrics.allometry.AllometryResult`.
        """
        from morphometrics.allometry import AllometryAnalyzer

        with self._lock:
            configurations = self._resolve_aligned_configurations(aligned_configurations, n_dims=n_dims)
            self._logger.info(
                f"analyze_allometry called with shape {configurations.shape}, "
                f"n_components={n_components}, n_dims={n_dims}"
            )
            analyzer = AllometryAnalyzer()
            result = analyzer.analyze_allometry(configurations, n_components=n_components)
            self._state.cache_result("allometry_result", result)
            return result

    def analyze_pls(
        self,
        block_a: npt.NDArray | None = None,
        block_b: npt.NDArray | None = None,
        division: str = "anterior_posterior",
        n_components: int | None = None,
        permutations: int = 0,
        seed: int | None = None,
    ) -> Any:
        """Two-Block PLS (morphological integration).

        When ``block_a``/``block_b`` are ``None``, the controller splits
        the cached GPA-aligned configurations by ``division`` (see
        :meth:`morphometrics.allometry.IntegrationAnalyzer.divide_configuration_into_blocks`).

        Parameters:
            block_a: Optional explicit ``(n, p)`` first block.
            block_b: Optional explicit ``(n, q)`` second block.
            division: Landmark-split rule used only when both blocks
                are ``None``. One of ``"anterior_posterior"``,
                ``"size_matched"`` or ``"random"``.
            n_components: Number of PLS components to retain.
            permutations: Permutations for the r₁ significance test
                (``0`` to skip).
            seed: RNG seed for the permutation test.

        Returns:
            :class:`morphometrics.allometry.PLSResult`.
        """
        from morphometrics.allometry import IntegrationAnalyzer

        with self._lock:
            analyzer = IntegrationAnalyzer()
            if block_a is None or block_b is None:
                configurations = self._resolve_aligned_configurations(None)
                a, b = analyzer.divide_configuration_into_blocks(
                    configurations, division=division, random_seed=seed
                )
            else:
                a = np.asarray(block_a, dtype=float)
                b = np.asarray(block_b, dtype=float)
            self._logger.info(
                f"analyze_pls called with block_a {a.shape}, block_b {b.shape}, "
                f"n_components={n_components}, permutations={permutations}"
            )
            result = analyzer.analyze_pls(
                a, b, n_components=n_components, permutations=permutations, seed=seed
            )
            self._state.cache_result("pls_result", result)
            return result

    # =========================================================================
    # Cached Results Access
    # =========================================================================

    def get_cached_result(self, key: str) -> Any | None:
        """Get cached analysis result."""
        return self._state.get_cached_result(key)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._state.clear_cache()

    # =========================================================================
    # Plugin System Integration
    # =========================================================================

    def list_available_analyses(self) -> list[str]:
        """
        List all available analyses (built-in + registered plugins).

        Returns:
            List of analysis names
        """
        # Built-in analyses (methods on this controller). Look for both
        # ``run_<name>`` and ``analyze_<name>`` patterns; deduplicate by
        # the bare analysis name. The previous implementation crammed a
        # ternary inside ``getattr`` which is technically correct but
        # fragile to read — split the lookup onto its own line.
        builtin: list[str] = []
        seen: set[str] = set()
        for method_name in dir(self):
            if method_name.startswith("_"):
                continue
            if method_name.startswith("run_"):
                base = method_name[len("run_") :]
            elif method_name.startswith("analyze_"):
                base = method_name[len("analyze_") :]
            else:
                continue
            if not base or base in seen:
                continue
            if callable(getattr(self, method_name, None)):
                seen.add(base)
                builtin.append(base)

        # Registered plugins
        try:
            from plugins import get_plugin_registry

            plugins = get_plugin_registry().list_plugins()
        except ImportError:
            plugins = []

        return sorted(set(builtin)) + sorted(plugins)

    def run_plugin(self, plugin_name: str, data: npt.NDArray | None = None, **kwargs: Any) -> Any:
        """
        Run a registered analysis plugin.

        Parameters:
            plugin_name: Name of the plugin to run
            data: Input data (uses state data if None)
            **kwargs: Additional parameters for the plugin

        Returns:
            Plugin-specific result

        Raises:
            ValidationError: If no data available or plugin not found
        """
        with self._lock:
            data = self._ensure_data(data)

        try:
            from plugins import get_plugin_registry

            registry = get_plugin_registry()
        except ImportError:
            raise ValidationError("Plugin system not available")

        plugin = registry.get(plugin_name)
        if plugin is None:
            raise ValidationError(f"Plugin '{plugin_name}' not found. Available: {registry.list_plugins()}")

        result = plugin.analyze(data, **kwargs)

        # Cache if plugin provides a cache key
        cache_key = plugin.cache_key
        if cache_key and hasattr(result, "data"):
            self._state.cache_result(cache_key, result)

        return result

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        try:
            from plugins import get_plugin_registry

            return get_plugin_registry().list_plugins()
        except ImportError:
            return []

    def list_plugin_categories(self) -> list[str]:
        """List all unique plugin categories."""
        try:
            from plugins import get_plugin_registry

            return get_plugin_registry().list_categories()
        except ImportError:
            return []
