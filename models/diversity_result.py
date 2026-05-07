# =============================================================================
# FILE: models/diversity_result.py
# =============================================================================
"""
Diversity Analysis Result Classes for PaleoAST

This module defines result classes for biodiversity and paleoecological
diversity analyses, providing structured output for various diversity indices.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DiversityIndexResult:
    """
    Result container for a single diversity index calculation.
    
    Attributes:
        index_name: Name of the diversity index
        value: Calculated index value
        standard_error: Standard error (if available)
        confidence_interval: Tuple of (lower, upper) bounds
        formula: LaTeX formula used for calculation
        interpretation: Text interpretation of the result
    """
    index_name: str
    value: float
    standard_error: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    formula: Optional[str] = None
    interpretation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'index_name': self.index_name,
            'value': self.value,
            'standard_error': self.standard_error,
            'confidence_interval': self.confidence_interval,
            'formula': self.formula,
            'interpretation': self.interpretation,
        }


@dataclass
class DiversityResult:
    """
    Comprehensive result container for diversity analysis.
    
    Contains results for all calculated diversity indices along with
    metadata about the analysis and sample properties.
    
    Attributes:
        sample_name: Name/identifier of the sample
        taxa_count: Number of unique taxa in the sample
        individuals: Total number of individuals in the sample
        indices: Dictionary of DiversityIndexResult objects
        metadata: Additional analysis metadata
    
    Mathematical Context:
        Diversity analysis quantifies two key aspects of biological communities:
        1. Richness: The number of different species/taxa present
        2. Evenness: How evenly individuals are distributed among taxa
        
        Common indices include:
        - Shannon Index (H'): -Σ p_i ln(p_i)
        - Simpson Index (D): 1 - Σ p_i²
        - Fisher's Alpha: Solves S = α ln(1 + n/α)
    
    Example:
        >>> result = DiversityResult(
        ...     sample_name="Site_A",
        ...     taxa_count=15,
        ...     individuals=150,
        ...     indices={
        ...         'shannon': DiversityIndexResult('Shannon', 2.45),
        ...         'simpson': DiversityIndexResult('Simpson', 0.89)
        ...     }
        ... )
        >>> result['shannon']
        DiversityIndexResult(index_name='Shannon', value=2.45, ...)
    """
    sample_name: str
    taxa_count: int
    individuals: int
    indices: Dict[str, DiversityIndexResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> DiversityIndexResult:
        """Access index result by name."""
        return self.indices[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if index exists."""
        return key in self.indices
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get index value or default."""
        if key in self.indices:
            return self.indices[key].value
        return default
    
    @property
    def shannon_index(self) -> Optional[float]:
        """Get Shannon diversity index value."""
        if 'shannon' in self.indices:
            return self.indices['shannon'].value
        return None
    
    @property
    def shannon_base_e(self) -> Optional[float]:
        """Get Shannon index with natural log."""
        return self.shannon_index
    
    @property
    def simpson_index(self) -> Optional[float]:
        """Get Simpson diversity index value."""
        if 'simpson' in self.indices:
            return self.indices['simpson'].value
        return None
    
    @property
    def species_richness(self) -> int:
        """Get number of taxa."""
        return self.taxa_count
    
    @property
    def evenness(self) -> Optional[float]:
        """Get Pielou's evenness index (J = H / ln(S))."""
        if 'shannon' in self.indices and self.taxa_count > 0:
            h = self.indices['shannon'].value
            max_h = np.log(self.taxa_count)
            if max_h > 0:
                return h / max_h
        return None
    
    def summary(self) -> str:
        """Generate text summary of diversity results."""
        lines = [
            f"Diversity Analysis: {self.sample_name}",
            "=" * 50,
            f"Taxa Richness (S): {self.taxa_count}",
            f"Total Individuals (N): {self.individuals}",
            ""
        ]
        
        if 'shannon' in self.indices:
            lines.append(f"Shannon Index (H'): {self.indices['shannon'].value:.4f}")
        
        if 'simpson' in self.indices:
            lines.append(f"Simpson Index (1-D): {self.indices['simpson'].value:.4f}")
        
        if self.evenness is not None:
            lines.append(f"Pielou's Evenness (J): {self.evenness:.4f}")
        
        if 'margalef' in self.indices:
            lines.append(f"Margalef Index: {self.indices['margalef'].value:.4f}")
        
        if 'fisher_alpha' in self.indices:
            lines.append(f"Fisher's Alpha: {self.indices['fisher_alpha'].value:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'sample_name': self.sample_name,
            'taxa_count': self.taxa_count,
            'individuals': self.individuals,
            'indices': {
                name: index.to_dict()
                for name, index in self.indices.items()
            },
            'metadata': self.metadata,
            'summary': self.summary(),
        }
    
    def __repr__(self) -> str:
        return (
            f"DiversityResult(sample='{self.sample_name}', "
            f"S={self.taxa_count}, N={self.individuals}, "
            f"n_indices={len(self.indices)})"
        )


@dataclass
class RarefactionResult:
    """
    Result container for rarefaction analysis.
    
    Attributes:
        sample_name: Name of the original sample
        expected_taxa: Array of expected taxa counts
        confidence_interval_lower: Lower CI bounds
        confidence_interval_upper: Upper CI bounds
        sample_sizes: Array of sample sizes used
        standard_error: Array of standard errors
        method: Rarefaction method used ('individual' or 'sample')
    """
    sample_name: str
    expected_taxa: npt.NDArray
    confidence_interval_lower: Optional[npt.NDArray] = None
    confidence_interval_upper: Optional[npt.NDArray] = None
    sample_sizes: Optional[npt.NDArray] = None
    standard_error: Optional[npt.NDArray] = None
    method: str = "individual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'sample_name': self.sample_name,
            'expected_taxa': self.expected_taxa.tolist(),
            'confidence_interval_lower': (
                self.confidence_interval_lower.tolist() 
                if self.confidence_interval_lower is not None else None
            ),
            'confidence_interval_upper': (
                self.confidence_interval_upper.tolist()
                if self.confidence_interval_upper is not None else None
            ),
            'sample_sizes': (
                self.sample_sizes.tolist()
                if self.sample_sizes is not None else None
            ),
            'method': self.method,
        }
    
    def __repr__(self) -> str:
        return (
            f"RarefactionResult(sample='{self.sample_name}', "
            f"method='{self.method}', "
            f"n_points={len(self.expected_taxa)})"
        )
