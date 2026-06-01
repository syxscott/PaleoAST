# =============================================================================
# FILE: visualization/stratigraphy_plot.py
# =============================================================================
"""
Stratigraphy Visualization Module for PaleoAST

This module implements publication-quality stratigraphic plots including:
    - Extinction range charts with confidence intervals
    - Stratigraphic columns
    - CONISS clustering diagrams
    - Markov chain transition diagrams

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


class StratigraphyPlotter:
    """
    Publication-quality stratigraphic visualization engine.

    Produces range charts and extinction interval plots showing:
        - Observed stratigraphic ranges (solid lines)
        - 95% confidence intervals for true extinction (whiskers)
        - Taxonomic diversity through time
    """

    def __init__(self) -> None:
        """Initialize the stratigraphy plotter."""
        self._logger = logging.getLogger(f"{__name__}.StratigraphyPlotter")
        self._logger.info("StratigraphyPlotter initialized")
        self._style = "seaborn-v0_8-paper"
        self._figure_size = (10, 8)
        self._dpi = 300
        self._font_size = 10
        self._title_font_size = 12

    def set_style(self, style: str) -> None:
        """Set matplotlib style."""
        try:
            plt.style.use(style)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not apply matplotlib style '{style}': {e}")

    def plot_extinction_ranges(
        self,
        lad_positions: npt.NDArray[np.float64],
        ci_lower: npt.NDArray[np.float64],
        ci_upper: npt.NDArray[np.float64],
        true_extinction_layer: npt.NDArray[np.float64] | None = None,
        taxon_names: list[str] | None = None,
        sampling_interval: float = 1.0,
        confidence_level: float = 0.95,
        method: str = "marshall",
        title: str = "Extinction Confidence Intervals",
        ylabel: str = "Stratigraphic Height (layers from top)",
    ) -> Figure:
        """
        Create stratigraphic range chart with extinction confidence intervals.

        Parameters:
            lad_positions: LAD positions (layer numbers from top)
            ci_lower: Lower 95% CI bound
            ci_upper: Upper 95% CI bound
            true_extinction_layer: Estimated true extinction layer
            taxon_names: Names for each taxon
            sampling_interval: Spacing between layers in meters
            confidence_level: Confidence level (default 0.95)
            method: Method used ("marshall" or "strauss_sadler")
            title: Plot title
            ylabel: Y-axis label

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        lad_positions = np.asarray(lad_positions, dtype=np.float64)
        ci_lower = np.asarray(ci_lower, dtype=np.float64)
        ci_upper = np.asarray(ci_upper, dtype=np.float64)

        n_taxa = len(lad_positions)
        if taxon_names is None:
            taxon_names = [f"Taxon {i + 1}" for i in range(n_taxa)]

        self._logger.info(f"plot_extinction_ranges: n_taxa={n_taxa}, method={method}")

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        # Sort taxa by LAD position (oldest at top)
        sorted_indices = np.argsort(lad_positions)[::-1]
        lad_sorted = lad_positions[sorted_indices]
        ci_lower_sorted = ci_lower[sorted_indices]
        ci_upper_sorted = ci_upper[sorted_indices]
        names_sorted = [taxon_names[i] for i in sorted_indices]
        if true_extinction_layer is not None:
            true_ext_sorted = true_extinction_layer[sorted_indices]
        else:
            true_ext_sorted = None

        # Plot each taxon's range
        for i, (lad, ci_l, ci_u, name) in enumerate(
            zip(lad_sorted, ci_lower_sorted, ci_upper_sorted, names_sorted)
        ):
            # Observed range: from 0 (top) to LAD
            # Draw vertical line for observed range
            ax.plot(
                [0.3, 0.7],
                [0, lad],
                "b-",
                linewidth=3,
                solid_capstyle="butt",
            )

            # Add horizontal bars at LAD for visibility
            ax.plot(
                [0.2, 0.8],
                [lad, lad],
                "b-",
                linewidth=2,
            )

            # Confidence interval whiskers (extending upward from LAD)
            # CI_lower is typically older (larger number), CI_upper is younger (smaller)
            whisker_lower = lad  # LAD is the observed position
            whisker_upper = ci_u  # Upper CI extends above LAD (smaller value)

            # Draw whiskers
            ax.plot(
                [0.5, 0.5],
                [whisker_upper, whisker_lower],
                "r--",
                linewidth=1.5,
            )

            # Add error bar caps
            ax.plot(
                [0.35, 0.65],
                [whisker_upper, whisker_upper],
                "r-",
                linewidth=2,
            )
            ax.plot(
                [0.35, 0.65],
                [whisker_lower, whisker_lower],
                "b-",
                linewidth=2,
            )

            # Add confidence interval box
            ci_box = Rectangle(
                (0.25, whisker_upper),
                0.5,
                whisker_lower - whisker_upper,
                linewidth=1,
                edgecolor="red",
                facecolor="red",
                alpha=0.2,
                linestyle="--",
            )
            ax.add_patch(ci_box)

            # Mark true extinction estimate if available
            if true_ext_sorted is not None:
                ax.plot(
                    [0.5],
                    [true_ext_sorted[i]],
                    "g^",
                    markersize=10,
                    markeredgecolor="black",
                )

            # Taxon label on right
            ax.text(
                0.85,
                lad,
                name,
                fontsize=9,
                va="center",
                ha="left",
            )

        # Customize axes
        ax.set_xlim(0, 1)
        max_lad = max(lad_sorted) if len(lad_sorted) > 0 else 10
        ax.set_ylim(-1, max_lad + 2)
        ax.invert_yaxis()  # Older layers at top

        ax.set_xlabel("Taxonomic Range", fontsize=self._font_size)
        ax.set_ylabel(ylabel, fontsize=self._font_size)
        ax.set_title(
            f"{title}\n({method.upper()} method, {int(confidence_level * 100)}% CI)",
            fontsize=self._title_font_size,
            fontweight="bold",
        )

        # Remove x ticks
        ax.set_xticks([])

        # Custom legend
        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=3, label="Observed LAD"),
            Line2D([0], [0], color="red", linewidth=1.5, linestyle="--", label=f"{int(confidence_level * 100)}% CI"),
            Line2D([0], [0], marker="^", color="green", linestyle="none", markersize=10, label="True extinction estimate"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)

        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_stratigraphic_column(
        self,
        layer_thicknesses: npt.NDArray[np.float64],
        layer_names: list[str] | None = None,
        fossil_occurrences: dict[str, list[int]] | None = None,
        lad_positions: dict[str, int] | None = None,
        title: str = "Stratigraphic Column",
    ) -> Figure:
        """
        Create stratigraphic column diagram.

        Parameters:
            layer_thicknesses: Thickness of each layer
            layer_names: Names/labels for each layer
            fossil_occurrences: Dict mapping fossil name to list of layer indices
            lad_positions: Last appearance data (fossil -> layer index)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        layer_thicknesses = np.asarray(layer_thicknesses, dtype=np.float64)
        n_layers = len(layer_thicknesses)

        if layer_names is None:
            layer_names = [f"L{i + 1}" for i in range(n_layers)]

        fig = Figure(figsize=(8, 10))
        ax = fig.add_subplot(111)

        # Cumulative depth
        cumulative = np.concatenate([[0], np.cumsum(layer_thicknesses)])

        # Draw column
        for i in range(n_layers):
            rect = Rectangle(
                (0, cumulative[i]),
                1,
                layer_thicknesses[i],
                linewidth=1,
                edgecolor="black",
                facecolor="lightgray",
                alpha=0.5,
            )
            ax.add_patch(rect)

            # Layer label
            ax.text(
                0.5,
                cumulative[i] + layer_thicknesses[i] / 2,
                layer_names[i],
                fontsize=8,
                ha="center",
                va="center",
            )

            # Plot fossil occurrences if provided
            if fossil_occurrences:
                for fossil, layers in fossil_occurrences.items():
                    if i in layers:
                        ax.plot(
                            1.2,
                            cumulative[i] + layer_thicknesses[i] / 2,
                            "bo",
                            markersize=6,
                        )

        # Plot LADs if provided
        if lad_positions:
            for fossil, layer_idx in lad_positions.items():
                if layer_idx < n_layers:
                    # Draw range bar
                    ax.plot(
                        [1.4, 1.6],
                        [cumulative[0], cumulative[layer_idx]],
                        "r-",
                        linewidth=2,
                    )
                    # LAD marker
                    ax.plot(
                        [1.5],
                        [cumulative[layer_idx]],
                        "rv",
                        markersize=12,
                    )
                    ax.text(1.7, cumulative[layer_idx], fossil, fontsize=8, va="center")

        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, cumulative[-1] + layer_thicknesses[-1])
        ax.invert_yaxis()

        ax.set_xlabel("")
        ax.set_ylabel("Depth (meters)", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.set_xticks([])
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_diversity_curve(
        self,
        sample_positions: npt.NDArray[np.float64],
        diversity_values: npt.NDArray[np.float64],
        sample_names: list[str] | None = None,
        confidence_lower: npt.NDArray[np.float64] | None = None,
        confidence_upper: npt.NDArray[np.float64] | None = None,
        title: str = "Taxonomic Diversity Through Time",
    ) -> Figure:
        """
        Create diversity curve with confidence intervals.

        Parameters:
            sample_positions: Stratigraphic positions of samples
            diversity_values: Diversity index at each position
            sample_names: Labels for samples
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        sample_positions = np.asarray(sample_positions, dtype=np.float64)
        diversity_values = np.asarray(diversity_values, dtype=np.float64)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        # Plot diversity curve
        ax.plot(sample_positions, diversity_values, "b-o", linewidth=2, markersize=8, label="Diversity")

        # Confidence band
        if confidence_lower is not None and confidence_upper is not None:
            ax.fill_between(
                sample_positions,
                confidence_lower,
                confidence_upper,
                alpha=0.2,
                color="blue",
                label="95% CI",
            )

        # Annotate samples
        if sample_names:
            for i, (pos, div, name) in enumerate(zip(sample_positions, diversity_values, sample_names)):
                ax.annotate(name, (pos, div), fontsize=7, xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Stratigraphic Position", fontsize=self._font_size)
        ax.set_ylabel("Diversity Index", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.invert_xaxis()

        fig.tight_layout()
        return fig
