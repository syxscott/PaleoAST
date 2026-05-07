# =============================================================================
# FILE: controllers/statistics_controller.py
# =============================================================================
"""
Statistics Controller for PaleoAST

This controller coordinates all statistical analyses including PCA, PCoA,
NMDS, ANOSIM, PERMANOVA, and diversity analyses.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import threading

from statistics.pca import PCAAnalyzer, PCAResult
from statistics.pcoa import PCoAAnalyzer, PCoAResult
from statistics.nmds import NMDSAnalyzer, NMDSResult
from statistics.anosim import ANOSIMAnalyzer, ANOSIMResult
from statistics.permanova import PERMANOVAAnalyzer, PERMANOVAResult
from statistics.distance_metrics import compute_distance_matrix, DistanceMatrix
from ecology.diversity import DiversityAnalyzer, compute_diversity_indices
from ecology.rarefaction import RarefactionAnalyzer
from models.diversity_result import DiversityResult, RarefactionResult
from models.state_manager import get_state_manager
from utils.exceptions import ComputationError, ValidationError


class StatisticsController:
    """
    Controller for statistical analysis operations.
    
    This controller acts as a facade, providing a unified interface to
    all statistical analysis engines while managing state and caching.
    """
    
    def __init__(self) -> None:
        """Initialize the statistics controller."""
        self._lock = threading.RLock()
        
        # Initialize analyzers
        self._pca_analyzer = PCAAnalyzer()
        self._pcoa_analyzer = PCoAAnalyzer()
        self._nmds_analyzer = NMDSAnalyzer()
        self._anosim_analyzer = ANOSIMAnalyzer()
        self._permanova_analyzer = PERMANOVAAnalyzer()
        self._diversity_analyzer = DiversityAnalyzer()
        self._rarefaction_analyzer = RarefactionAnalyzer()
        
        # State manager
        self._state = get_state_manager()
    
    # =========================================================================
    # PCA Operations
    # =========================================================================
    
    def run_pca(
        self,
        data: Optional[npt.NDArray] = None,
        n_components: Optional[int] = None,
        method: str = 'covariance'
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
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available. Please load data first.")
                data = self._state.data_matrix.data
            
            result = self._pca_analyzer.analyze(data, n_components, method)
            
            # Cache result
            self._state.cache_result('pca_result', result)
            
            return result
    
    # =========================================================================
    # PCoA Operations
    # =========================================================================
    
    def run_pcoa(
        self,
        distance_matrix: Optional[npt.NDArray] = None,
        n_components: int = 10,
        metric: str = 'bray_curtis'
    ) -> PCoAResult:
        """
        Run Principal Coordinate Analysis.
        
        Parameters:
            distance_matrix: Distance matrix. If None, computes from state data.
            n_components: Number of coordinates.
            metric: Distance metric for computation.
        
        Returns:
            PCoAResult: PCoA analysis results
        """
        with self._lock:
            if distance_matrix is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                
                data = self._state.data_matrix.data
                dm = compute_distance_matrix(data, metric=metric)
                distance_matrix = dm.matrix
            
            result = self._pcoa_analyzer.analyze(distance_matrix, n_components)
            
            self._state.cache_result('pcoa_result', result)
            self._state.cache_result('pcoa_metric', metric)
            
            return result
    
    # =========================================================================
    # NMDS Operations
    # =========================================================================
    
    def run_nmds(
        self,
        distance_matrix: Optional[npt.NDArray] = None,
        n_dimensions: int = 2,
        metric: str = 'bray_curtis',
        n_restarts: int = 10,
        random_seed: Optional[int] = None
    ) -> NMDSResult:
        """
        Run Non-metric Multidimensional Scaling.
        
        Parameters:
            distance_matrix: Distance matrix. If None, computes from state data.
            n_dimensions: Number of dimensions for ordination.
            metric: Distance metric for computation.
            n_restarts: Number of random restarts.
            random_seed: Random seed for reproducibility.
        
        Returns:
            NMDSResult: NMDS analysis results
        """
        with self._lock:
            if distance_matrix is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                
                data = self._state.data_matrix.data
                dm = compute_distance_matrix(data, metric=metric)
                distance_matrix = dm.matrix
            
            result = self._nmds_analyzer.analyze(
                distance_matrix,
                n_dimensions=n_dimensions,
                metric=metric,
                n_restarts=n_restarts,
                random_seed=random_seed
            )
            
            self._state.cache_result('nmds_result', result)
            self._state.cache_result('nmds_metric', metric)
            
            return result
    
    # =========================================================================
    # Group Comparison Tests
    # =========================================================================
    
    def run_anosim(
        self,
        groups: List[int],
        distance_matrix: Optional[npt.NDArray] = None,
        metric: str = 'bray_curtis',
        n_permutations: int = 9999
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
            if distance_matrix is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                
                data = self._state.data_matrix.data
                dm = compute_distance_matrix(data, metric=metric)
                distance_matrix = dm.matrix
            
            result = self._anosim_analyzer.analyze(
                distance_matrix,
                groups,
                n_permutations=n_permutations,
                metric=metric
            )
            
            self._state.cache_result('anosim_result', result)
            
            return result
    
    def run_permanova(
        self,
        groups: List[int],
        distance_matrix: Optional[npt.NDArray] = None,
        metric: str = 'bray_curtis',
        n_permutations: int = 9999
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
            if distance_matrix is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                
                data = self._state.data_matrix.data
                dm = compute_distance_matrix(data, metric=metric)
                distance_matrix = dm.matrix
            
            result = self._permanova_analyzer.analyze(
                distance_matrix,
                groups,
                n_permutations=n_permutations,
                metric=metric
            )
            
            self._state.cache_result('permanova_result', result)
            
            return result
    
    # =========================================================================
    # Diversity Analysis
    # =========================================================================
    
    def analyze_diversity(
        self,
        abundances: Optional[npt.NDArray] = None,
        sample_name: str = "Sample"
    ) -> DiversityResult:
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
                    raise ValidationError("No data available.")
                
                abundances = self._state.data_matrix.data[0]
            
            result = compute_diversity_indices(abundances, sample_name)
            
            self._state.cache_result(f'diversity_{sample_name}', result)
            
            return result
    
    def analyze_rarefaction(
        self,
        abundances: Optional[npt.NDArray] = None,
        sample_name: str = "Sample",
        n_points: int = 50
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
            
            result = self._rarefaction_analyzer.analyze(
                abundances, sample_name, n_points=n_points
            )
            
            self._state.cache_result(f'rarefaction_{sample_name}', result)
            
            return result
    
    # =========================================================================
    # Distance Matrix Operations
    # =========================================================================
    
    def compute_distances(
        self,
        data: Optional[npt.NDArray] = None,
        metric: str = 'bray_curtis'
    ) -> DistanceMatrix:
        """
        Compute distance matrix.
        
        Parameters:
            data: Input data. If None, uses state data.
            metric: Distance metric.
        
        Returns:
            DistanceMatrix: Computed distance matrix
        """
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available.")
                data = self._state.data_matrix.data
            
            result = compute_distance_matrix(data, metric=metric)
            
            self._state.cache_result('distance_matrix', result)
            self._state.cache_result('distance_metric', metric)
            
            return result
    
    # =========================================================================
    # Cached Results Access
    # =========================================================================
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached analysis result."""
        return self._state.get_cached_result(key)
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._state.clear_cache()
