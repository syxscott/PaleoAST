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
    - Multi-section stratigraphic correlation with DTW warping paths

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
            zip(lad_sorted, ci_lower_sorted, ci_upper_sorted, names_sorted, strict=False)
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
            Line2D(
                [0], [0], marker="^", color="green", linestyle="none", markersize=10, label="True extinction estimate"
            ),
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
            for i, (pos, div, name) in enumerate(zip(sample_positions, diversity_values, sample_names, strict=False)):
                ax.annotate(name, (pos, div), fontsize=7, xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Stratigraphic Position", fontsize=self._font_size)
        ax.set_ylabel("Diversity Index", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.invert_xaxis()

        fig.tight_layout()
        return fig

    # =====================================================================
    # Multi-section stratigraphic correlation plot
    # =====================================================================

    def plot_stratigraphic_correlation(
        self,
        correlation_result,
        title: str = "Stratigraphic Correlation",
        cmap_name: str = "viridis",
        max_pairs: int = 3,
    ) -> Figure:
        """
        Render a multi-section stratigraphic correlation diagram with
        similarity-coded warping paths.

        Parameters:
            correlation_result: A
                :class:`stratigraphy.correlation.StratigraphicCorrelationResult`
                object containing the sections and their pairwise best
                matches.
            title: Plot title.
            cmap_name: Name of a matplotlib colormap used to encode the
                similarity score of each warping path. Default
                ``"viridis"`` follows the design-system conventions.
            max_pairs: Maximum number of section pairs (ranked by
                similarity) to display in the correlation panel. Use
                ``-1`` to render all pairs.

        Returns:
            matplotlib Figure object with a left column for the
            stratigraphic columns and a right column for the
            warping-path correlation panel.

        Note:
            The function reads the design-system palette from
            ``config.design_system`` or falls back to the legacy
            ``config.colors`` constants. Each warping path is drawn
            between the per-section height arrays, with colour, alpha
            and line-width controlled by the per-pair DTW similarity
            score.
        """
        self.set_style(self._style)

        # Defer heavy imports to keep module import cost low
        from stratigraphy.correlation import StratigraphicCorrelationResult

        if not isinstance(correlation_result, StratigraphicCorrelationResult):
            raise TypeError(
                "correlation_result must be a StratigraphicCorrelationResult instance, "
                f"got {type(correlation_result).__name__}"
            )

        sections = correlation_result.sections
        n_sections = len(sections)
        if n_sections < 2:
            raise ValueError("At least 2 sections are required for correlation plotting")

        # Resolve colour palette (prefer design system, fall back to config.colors)
        try:
            from config.design_system import colors as ds_colors
            primary_color = ds_colors.primary
            text_color = ds_colors.text_primary
            border_color = ds_colors.border_medium
        except Exception:
            from config.colors import PRIMARY_COLOR, CELL_HEADER_TEXT, DEFAULT_EDGE_COLOR
            primary_color = PRIMARY_COLOR
            text_color = CELL_HEADER_TEXT
            border_color = DEFAULT_EDGE_COLOR

        # Select the best-matching section pairs to render as warping paths
        best_matches = list(correlation_result.best_matches or [])
        if max_pairs == -1:
            render_matches = best_matches
        else:
            render_matches = best_matches[:max_pairs]
        if not render_matches:
            self._logger.warning("No best_matches available; rendering columns only")
            render_matches = []

        # Figure layout: each section gets a column, plus extra axes for warping
        n_cols = n_sections + len(render_matches) if render_matches else n_sections
        if render_matches:
            # n_sections section columns + len(render_matches) warping panels
            col_widths: list[float] = [3.0] * n_sections + [1.2] * len(render_matches)
        else:
            col_widths = [3.0] * n_sections

        fig = Figure(figsize=(max(10.0, 1.5 * n_cols + 4.0), 10))
        gs = fig.add_gridspec(
            1,
            n_cols,
            width_ratios=col_widths,
            wspace=0.15,
        )

        # Determine global y-range across all sections
        all_heights = np.concatenate([np.asarray(s.heights, dtype=np.float64) for s in sections])
        y_min = float(np.nanmin(all_heights))
        y_max = float(np.nanmax(all_heights))
        # Add a 5% pad
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

        # Per-section axes.
        # NOTE: ``_render_section_column`` no longer flips the y-axis;
        # we own the final orientation here so that the section columns
        # and the warping panels stay in lockstep ("older at top"
        # convention). ``set_ylim(low, high)`` first, then a single
        # ``invert_yaxis()`` for both column and warp axes.
        section_axes: list = []
        for i, section in enumerate(sections):
            ax = fig.add_subplot(gs[0, i])
            section_axes.append(ax)
            self._render_section_column(ax, section, primary_color, text_color, border_color)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.invert_yaxis()

        # Warping-path axes - one per pair
        warp_axes: list = []
        for j, (idx_a, idx_b, similarity) in enumerate(render_matches):
            ax = fig.add_subplot(gs[0, n_sections + j])
            warp_axes.append((ax, idx_a, idx_b, similarity))
            self._render_warping_panel(
                ax=ax,
                section_a=sections[idx_a],
                section_b=sections[idx_b],
                similarity=float(similarity),
                cmap_name=cmap_name,
                primary_color=primary_color,
                text_color=text_color,
                border_color=border_color,
                y_min=y_min - y_pad,
                y_max=y_max + y_pad,
            )

        # Suppress redundant y-axis labels for warping panels
        for ax, *_ in warp_axes:
            ax.set_yticklabels([])

        fig.suptitle(title, fontsize=self._title_font_size + 1, fontweight="bold")
        # Colorbar for similarity scores
        if render_matches:
            sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap(cmap_name))
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Similarity score", fontsize=self._font_size)

        fig.subplots_adjust(left=0.04, right=0.9, top=0.92, bottom=0.06)
        return fig

    def _render_section_column(
        self,
        ax,
        section,
        primary_color: str,
        text_color: str,
        border_color: str,
    ) -> None:
        """Render a single stratigraphic column on the given axis.

        Each layer is drawn as a rectangle *centred on* its actual
        stratigraphic height, with the rectangle's vertical extent set
        to the layer's thickness. This way the column respects the
        real height array instead of stacking everything from zero.
        The y-axis is inverted so that older layers (smaller height)
        appear at the top, matching the standard geological convention.
        """
        heights = np.asarray(section.heights, dtype=np.float64)
        thicknesses = (
            np.asarray(section.thicknesses, dtype=np.float64)
            if section.thicknesses is not None
            and len(section.thicknesses) == len(heights)
            else np.ones_like(heights)
        )
        lithologies = section.lithologies or [f"L{i + 1}" for i in range(len(heights))]

        for k in range(len(heights)):
            h_center = float(heights[k])
            t = float(thicknesses[k])
            y_bottom = h_center - t / 2.0
            rect = Rectangle(
                (0.0, y_bottom),
                1.0,
                t,
                linewidth=0.6,
                edgecolor=border_color,
                facecolor=primary_color,
                alpha=0.18 + 0.04 * (k % 4),
            )
            ax.add_patch(rect)
            ax.text(
                0.5,
                h_center,
                str(lithologies[k]) if k < len(lithologies) else "",
                ha="center",
                va="center",
                fontsize=max(6, self._font_size - 3),
                color=text_color,
            )

        # Add a representative "core" line through the actual height range
        h_min = float(np.min(heights))
        h_max = float(np.max(heights))
        ax.plot([0.5, 0.5], [h_min, h_max], color=primary_color, linewidth=2.0)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([])
        ax.set_title(getattr(section, "name", "Section"), fontsize=self._font_size)
        ax.set_ylabel("Stratigraphic height (m)", fontsize=self._font_size)
        # NOTE: do NOT invert the y-axis here. The orientation is
        # owned by ``plot_stratigraphic_correlation`` so the section
        # columns and the warping panels stay synchronised. Inverting
        # twice (once here, once again outside) would flip the column
        # back to ascending order and visually de-sync it from the
        # warping panels.

    def _render_warping_panel(
        self,
        ax,
        section_a,
        section_b,
        similarity: float,
        cmap_name: str,
        primary_color: str,
        text_color: str,
        border_color: str,
        y_min: float,
        y_max: float,
    ) -> None:
        """Render the warping-path correlation panel for a single pair."""
        h_a = np.asarray(section_a.heights, dtype=np.float64)
        h_b = np.asarray(section_b.heights, dtype=np.float64)
        n_a, n_b = len(h_a), len(h_b)
        cmap = plt.get_cmap(cmap_name)

        if n_a == 0 or n_b == 0:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", color=text_color)
            return

        # Compute DTW warping path
        path = self._dtw_warping_path(h_a, h_b)
        if not path:
            return

        # Style parameters driven by the similarity score
        similarity = float(np.clip(similarity, 0.0, 1.0))
        # Higher similarity => more saturated colour, thicker line, less transparent
        line_width = 0.4 + 2.6 * similarity
        alpha = 0.25 + 0.6 * similarity
        is_solid = similarity >= 0.5
        linestyle = "-" if is_solid else "--"

        # Draw all path segments in a single LineCollection for speed.
        # Building one ``ax.plot`` per segment is O(N) Matplotlib draws
        # and is prohibitively slow for long warping paths (e.g. N>500).
        # ``LineCollection`` submits everything in one shot.
        color = cmap(similarity)
        segments = np.empty((len(path), 2, 2), dtype=np.float64)
        for k, (i_a, i_b) in enumerate(path):
            segments[k, 0, 0] = 0.0
            segments[k, 0, 1] = h_a[i_a]
            segments[k, 1, 0] = 1.0
            segments[k, 1, 1] = h_b[i_b]
        lc = LineCollection(
            segments,
            colors=[color] * len(segments),
            linewidths=line_width,
            alpha=alpha,
            linestyles=linestyle,
            capstyle="round",
        )
        ax.add_collection(lc)

        # Add a faint background grid for orientation
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.grid(True, axis="y", linestyle=":", alpha=0.3, color=border_color)
        ax.set_title(
            f"{getattr(section_a, 'name', 'A')} - {getattr(section_b, 'name', 'B')}\n"
            f"sim={similarity:.2f}",
            fontsize=max(6, self._font_size - 1),
        )

    @staticmethod
    def _dtw_warping_path(
        h_a: npt.NDArray[np.float64],
        h_b: npt.NDArray[np.float64],
    ) -> list[tuple[int, int]]:
        """Compute the optimal DTW warping path between two height arrays.

        Parameters:
            h_a: Heights of section A (length ``n_a``).
            h_b: Heights of section B (length ``n_b``).

        Returns:
            List of ``(i_a, i_b)`` pairs along the optimal warping path
            (in forward order).
        """
        n_a = len(h_a)
        n_b = len(h_b)
        if n_a == 0 or n_b == 0:
            return []
        # Guard against OOM: the DP table is (n_a+1)*(n_b+1) float64
        # values. Two 10k-sample sections would need ~800 MiB just for
        # the cost matrix. Refuse and let the caller decide whether to
        # subsample or use a banded variant.
        max_dim = 5000  # 5000*5000*8 bytes ≈ 200 MiB
        if n_a > max_dim or n_b > max_dim:
            raise ValueError(
                f"DTW input too large ({n_a}x{n_b}); "
                f"max supported dimension is {max_dim}. "
                "Subsample the input or use a banded DTW variant."
            )
        # Vectorise the local cost matrix: |h_a[i] - h_b[j]|.
        local = np.abs(h_a[:, None] - h_b[None, :])
        cost = np.full((n_a + 1, n_b + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0
        # Row-wise DP. Inner loop is in NumPy via cumulative ``minimum``.
        for i in range(1, n_a + 1):
            for j in range(1, n_b + 1):
                cost[i, j] = local[i - 1, j - 1] + min(
                    cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
                )

        # Backtrack the optimal path
        path: list[tuple[int, int]] = []
        i, j = n_a, n_b
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            step = int(np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]))
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.reverse()
        return path
