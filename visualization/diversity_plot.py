# =============================================================================
# FILE: visualization/diversity_plot.py
# =============================================================================
"""
Diversity Visualization Module for PaleoAST

This module implements publication-quality diversity plots including:
    - Diversity indices comparison plots
    - Rarefaction curves
    - Species accumulation curves

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from models.diversity_result import DiversityResult, RarefactionResult
from config.colors import get_color_scheme

logger = logging.getLogger(__name__)


class DiversityPlotter:
    """
    Publication-quality diversity visualization engine.
    """
    
    def __init__(self) -> None:
        """Initialize the diversity plotter."""
        self._logger = logging.getLogger(f"{__name__}.DiversityPlotter")
        self._logger.info("DiversityPlotter initialized")
        self._style = 'seaborn-v0_8-paper'
        self._figure_size = (8, 6)
        self._dpi = 300
        self._font_size = 10

    def _apply_style(self) -> None:
        """Apply matplotlib style with fallback for older versions."""
        try:
            plt.style.use(self._style)
        except OSError:
            try:
                plt.style.use(self._style.replace('v0_8-', ''))
            except OSError:
                pass
    
    def plot_rarefaction(
        self,
        result: RarefactionResult,
        show_ci: bool = False,
        title: Optional[str] = None
    ) -> Figure:
        """
        Create rarefaction curve plot.
        
        Parameters:
            result: Rarefaction analysis result
            show_ci: Whether to show confidence interval
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        self._apply_style()
        self._logger.info(
            f"plot_rarefaction called: sample_name='{result.sample_name}'"
        )

        fig, ax = plt.subplots(figsize=self._figure_size)

        ax.plot(
            result.sample_sizes,
            result.expected_taxa,
            'o-',
            color='#3498DB',
            linewidth=2,
            markersize=5,
            markerfacecolor='white',
            markeredgewidth=1.5
        )
        
        if show_ci and result.confidence_interval_lower is not None:
            ax.fill_between(
                result.sample_sizes,
                result.confidence_interval_lower,
                result.confidence_interval_upper,
                alpha=0.2,
                color='#3498DB'
            )
        
        if title is None:
            title = f"Rarefaction Curve: {result.sample_name}"
        
        ax.set_xlabel('Number of Individuals', fontsize=self._font_size)
        ax.set_ylabel('Expected Species Richness', fontsize=self._font_size)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_rarefaction(
        self,
        results: List[RarefactionResult],
        title: str = "Rarefaction Curves Comparison"
    ) -> Figure:
        """
        Plot multiple rarefaction curves for comparison.
        
        Parameters:
            results: List of RarefactionResult objects
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        self._apply_style()
        
        fig, ax = plt.subplots(figsize=self._figure_size)
        
        colors = get_color_scheme(len(results))
        
        for i, result in enumerate(results):
            ax.plot(
                result.sample_sizes,
                result.expected_taxa,
                'o-',
                color=colors[i],
                linewidth=2,
                markersize=5,
                label=result.sample_name,
                markerfacecolor='white',
                markeredgewidth=1.5
            )
        
        ax.set_xlabel('Number of Individuals', fontsize=self._font_size)
        ax.set_ylabel('Expected Species Richness', fontsize=self._font_size)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fancybox=True)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_diversity_comparison(
        self,
        results: List[DiversityResult],
        index: str = 'shannon',
        title: Optional[str] = None
    ) -> Figure:
        """
        Create bar plot comparing diversity indices across samples.
        
        Parameters:
            results: List of DiversityResult objects
            index: Which index to compare ('shannon', 'simpson', 'margalef')
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        self._apply_style()
        
        fig, ax = plt.subplots(figsize=self._figure_size)
        
        samples = [r.sample_name for r in results]
        values = [r.get(index, 0) for r in results]
        
        colors = get_color_scheme(len(results))
        
        bars = ax.bar(
            range(len(results)),
            values,
            color=colors,
            edgecolor='white',
            linewidth=1.5
        )
        
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(samples, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        index_labels = {
            'shannon': "Shannon Index (H')",
            'simpson': 'Simpson Index (1-D)',
            'margalef': 'Margalef Index',
            'pielou': "Pielou's Evenness (J)"
        }
        
        ylabel = index_labels.get(index, index)
        
        if title is None:
            title = f"{ylabel} Comparison"
        
        ax.set_ylabel(ylabel, fontsize=self._font_size)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_diversity_summary(
        self,
        result: DiversityResult
    ) -> Figure:
        """
        Create comprehensive diversity summary plot.
        
        Parameters:
            result: DiversityResult object
        
        Returns:
            matplotlib Figure object
        """
        self._apply_style()
        self._logger.info(
            f"plot_diversity_summary called: sample_name='{result.sample_name}', "
            f"taxa_count={result.taxa_count}"
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Shannon vs Simpson comparison
        ax1 = axes[0, 0]
        
        # 2. Diversity indices radar chart (simplified as bar)
        ax2 = axes[0, 1]
        
        indices_to_show = []
        values_to_show = []
        
        for key in ['shannon', 'simpson', 'pielou', 'margalef']:
            if key in result.indices:
                indices_to_show.append(key)
                # Normalize for display
                value = result.indices[key].value
                if key == 'simpson':
                    values_to_show.append(value * 100)  # Scale to percentage
                else:
                    values_to_show.append(value)
        
        colors = get_color_scheme(len(indices_to_show))
        bars = ax2.bar(indices_to_show, values_to_show, color=colors, edgecolor='white')
        
        ax2.set_ylabel('Value', fontsize=9)
        ax2.set_title('Diversity Indices', fontsize=11, fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # 3. Sample info
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        info_text = (
            f"Sample: {result.sample_name}\n"
            f"{'='*30}\n"
            f"Taxa Richness (S): {result.taxa_count}\n"
            f"Total Individuals (N): {result.individuals}\n"
            f"{'='*30}\n"
        )
        
        for key, index_result in result.indices.items():
            info_text += f"{index_result.index_name}: {index_result.value:.4f}\n"
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        
        # 4. Evenness pie
        ax4 = axes[1, 1]
        if result.evenness is not None:
            labels = ['Even', 'Uneven']
            sizes = [result.evenness, 1 - result.evenness]
            colors_pie = ['#3498DB', '#E74C3C']
            ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                    startangle=90, explode=(0.05, 0))
            ax4.set_title("Pielou's Evenness", fontsize=11, fontweight='bold')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, "Evenness not available", transform=ax4.transAxes,
                     ha='center', va='center', fontsize=10)
        
        plt.suptitle(f"Diversity Analysis: {result.sample_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
