# =============================================================================
# FILE: visualization/evo_rate_plot.py
# =============================================================================
"""
Evolutionary Rate Visualization Module for PaleoAST

This module implements publication-quality evolutionary rate plots including:
    - Phenogram: trait evolution trajectory over geological time
    - Rate comparison bar charts
    - Model selection visualizations
    - Trajectory confidence intervals

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class EvolutionRatePlotter:
    """
    Publication-quality evolutionary rate visualization engine.

    Produces phenograms and rate comparison plots showing:
        - Trait evolution over geological time
        - Model fits (Random Walk, Directional, Stasis)
        - Rate estimates with confidence intervals
    """

    def __init__(self) -> None:
        """Initialize the evolution rate plotter."""
        self._logger = logging.getLogger(f"{__name__}.EvolutionRatePlotter")
        self._logger.info("EvolutionRatePlotter initialized")
        self._style = "seaborn-v0_8-paper"
        self._figure_size = (10, 6)
        self._dpi = 300
        self._font_size = 10
        self._title_font_size = 12

    def set_style(self, style: str) -> None:
        """Set matplotlib style."""
        try:
            plt.style.use(style)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not apply matplotlib style '{style}': {e}")

    def plot_phenogram(
        self,
        trait_series: npt.NDArray[np.float64],
        time_intervals: npt.NDArray[np.float64] | None,
        time_unit: str = "Depth/Height",
        best_model: str = "random_walk",
        rate_estimate: float = 0.0,
        trend_estimate: float | None = None,
        aic_weights: dict[str, float] | None = None,
        specimen_labels: list[str] | None = None,
        title: str = "Phenogram: Morphological Evolution Over Time",
    ) -> Figure:
        """
        Create phenogram showing trait evolution trajectory.

        Parameters:
            trait_series: Trait values in stratigraphic/time order
            time_intervals: Time/depth intervals between measurements
            time_unit: Unit label for x-axis (e.g., "Ma", "m", "cm")
            best_model: Best-fit evolutionary model
            rate_estimate: Estimated evolution rate
            trend_estimate: Directional trend (if applicable)
            aic_weights: AIC weights for model comparison
            specimen_labels: Optional labels for each measurement point
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        trait_series = np.asarray(trait_series, dtype=np.float64)
        n = len(trait_series)

        # Compute cumulative time/depth
        if time_intervals is None:
            time_intervals = np.ones(n - 1)
        time_intervals = np.asarray(time_intervals, dtype=np.float64)
        time_points = np.zeros(n)
        time_points[1:] = np.cumsum(time_intervals)

        fig = Figure(figsize=self._figure_size)

        # Main phenogram subplot
        ax = fig.add_subplot(211)

        # Plot trajectory
        ax.plot(time_points, trait_series, "b-o", linewidth=2, markersize=8, label="Observed trait")

        # Add confidence band based on rate
        if rate_estimate > 0:
            # Compute variance envelope (random walk variance = rate * time)
            cumulative_time = np.concatenate([[0], np.cumsum(time_intervals)])
            variance = rate_estimate * cumulative_time
            std = np.sqrt(variance)
            ax.fill_between(
                time_points,
                trait_series - 1.96 * std,
                trait_series + 1.96 * std,
                alpha=0.2,
                color="blue",
                label="95% CI",
            )

        # Annotate points
        if specimen_labels is not None:
            for i, label in enumerate(specimen_labels):
                ax.annotate(
                    label,
                    (time_points[i], trait_series[i]),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Model interpretation text
        model_label = f"Best model: {best_model.upper()}"
        if best_model == "directional" and trend_estimate is not None:
            model_label += f"\nTrend: {trend_estimate:.4f} per unit"
        model_label += f"\nRate: {rate_estimate:.6f}"

        ax.text(
            0.02,
            0.98,
            model_label,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel(f"Stratigraphic {time_unit}", fontsize=self._font_size)
        ax.set_ylabel("Trait Value", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.invert_xaxis()  # Older time/depth on right

        # Model comparison subplot (if AIC weights provided)
        if aic_weights:
            ax2 = fig.add_subplot(212)
            models = list(aic_weights.keys())
            weights = list(aic_weights.values())

            # Color bars by model
            colors = []
            for m in models:
                if m == best_model:
                    colors.append("#E74C3C")  # Red for best
                else:
                    colors.append("#3498DB")  # Blue for others

            bars = ax2.barh(models, weights, color=colors, alpha=0.7)
            ax2.set_xlabel("AIC Weight", fontsize=self._font_size)
            ax2.set_title("Model Comparison", fontsize=self._font_size)
            ax2.set_xlim(0, 1)

            # Add value labels
            for bar, w in zip(bars, weights, strict=False):
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center")

        fig.tight_layout()
        return fig

    def plot_rate_comparison(
        self,
        rate_estimates: dict[str, float],
        rate_ci_lower: dict[str, float] | None = None,
        rate_ci_upper: dict[str, float] | None = None,
        title: str = "Evolution Rate Comparison Across Models",
    ) -> Figure:
        """
        Create bar chart comparing rates across models.

        Parameters:
            rate_estimates: Rate estimate for each model
            rate_ci_lower: Lower CI for each model (optional)
            rate_ci_upper: Upper CI for each model (optional)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        models = list(rate_estimates.keys())
        rates = list(rate_estimates.values())
        n_models = len(models)

        # Bar positions
        positions = np.arange(n_models)
        bars = ax.bar(positions, rates, color="#3498DB", alpha=0.7, edgecolor="black")

        # Error bars for CI
        if rate_ci_lower is not None and rate_ci_upper is not None:
            errors_lower = [rates[i] - rate_ci_lower[m] for i, m in enumerate(models)]
            errors_upper = [rate_ci_upper[m] - rates[i] for i, m in enumerate(models)]
            ax.errorbar(
                positions,
                rates,
                yerr=[errors_lower, errors_upper],
                fmt="none",
                color="black",
                capsize=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylabel("Evolution Rate", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars, rates, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{rate:.4f}", ha="center", fontsize=9)

        fig.tight_layout()
        return fig

    def plot_trait_distribution(
        self,
        trait_series: npt.NDArray[np.float64],
        groups: list[int] | None = None,
        title: str = "Trait Value Distribution",
    ) -> Figure:
        """
        Create boxplot/violin plot of trait distribution.

        Parameters:
            trait_series: Trait values
            groups: Optional group assignments
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        trait_series = np.asarray(trait_series, dtype=np.float64)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if groups is not None:
            unique_groups = sorted(set(groups))
            data_by_group = [trait_series[np.array(groups) == g] for g in unique_groups]

            parts = ax.violinplot(data_by_group, positions=range(len(unique_groups)), showmeans=True, showmedians=True)

            # Color the violin bodies
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
            for idx, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[idx])
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(unique_groups)))
            ax.set_xticklabels([f"Group {g}" for g in unique_groups])
            ax.legend(["Mean", "Median"], loc="best")
        else:
            ax.hist(trait_series, bins=20, color="#3498DB", alpha=0.7, edgecolor="black")

        ax.set_xlabel("Trait Value", fontsize=self._font_size)
        ax.set_ylabel("Frequency", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig
