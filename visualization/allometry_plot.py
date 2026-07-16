# =============================================================================
# FILE: visualization/allometry_plot.py
# =============================================================================
"""
Allometry Visualization Module for PaleoAST

This module implements publication-quality allometry plots including:
    - Scatter plot of Log Centroid Size vs shape scores
    - Linear regression with confidence bands
    - Residual plots
    - Group comparison plots

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)


class AllometryPlotter:
    """
    Publication-quality allometry visualization engine.

    Produces scatter plots showing size-shape relationships with:
        - Log Centroid Size on X-axis
        - Shape scores (PC1 or regression predictions) on Y-axis
        - Linear regression line
        - 95% confidence and prediction bands
    """

    def __init__(self) -> None:
        """Initialize the allometry plotter."""
        self._logger = logging.getLogger(f"{__name__}.AllometryPlotter")
        self._logger.info("AllometryPlotter initialized")
        self._style = "seaborn-v0_8-paper"
        self._figure_size = (8, 6)
        self._dpi = 300
        self._font_size = 10
        self._title_font_size = 12

    def set_style(self, style: str) -> None:
        """Set matplotlib style."""
        try:
            plt.style.use(style)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not apply matplotlib style '{style}': {e}")

    def plot_allometry_scatter(
        self,
        centroid_sizes: npt.NDArray[np.float64],
        shape_scores: npt.NDArray[np.float64],
        regression_coefficients: npt.NDArray[np.float64],
        regression_intercept: npt.NDArray[np.float64],
        r_squared: float,
        groups: list[int] | None = None,
        specimen_labels: list[str] | None = None,
        title: str = "Allometry: Size-Shape Relationship",
        xlabel: str = "Log Centroid Size",
        ylabel: str = "Shape Score (PC1)",
        show_confidence_band: bool = True,
        show_prediction_band: bool = True,
    ) -> Figure:
        """
        Create allometry scatter plot with regression line and confidence bands.

        Parameters:
            centroid_sizes: Centroid sizes for each specimen (n_specimens,)
            shape_scores: Shape scores (e.g., PC1 scores or predicted values)
            regression_coefficients: Regression slope coefficient
            regression_intercept: Regression intercept
            r_squared: R-squared of the regression
            groups: Optional group assignments for coloring
            specimen_labels: Optional specimen labels for annotation
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_confidence_band: Whether to show 95% confidence band
            show_prediction_band: Whether to show 95% prediction band

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        centroid_sizes = np.asarray(centroid_sizes, dtype=np.float64)
        shape_scores = np.asarray(shape_scores, dtype=np.float64)
        log_cs = np.log(centroid_sizes)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        self._logger.info(
            f"plot_allometry_scatter: n_points={len(log_cs)}, "
            f"r_squared={r_squared:.4f}, n_groups={len(set(groups)) if groups else 0}"
        )

        # Plot points by group
        if groups is not None:
            unique_groups = sorted(set(groups))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

            for idx, group in enumerate(unique_groups):
                mask = np.array(groups) == group
                ax.scatter(
                    log_cs[mask],
                    shape_scores[mask],
                    c=[colors[idx]],
                    label=f"Group {group}",
                    s=80,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.5,
                )
            ax.legend(loc="best", frameon=True, fancybox=True)
        else:
            ax.scatter(log_cs, shape_scores, c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Annotate specimens
        if specimen_labels is not None:
            for i, label in enumerate(specimen_labels):
                ax.annotate(
                    label,
                    (log_cs[i], shape_scores[i]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                    alpha=0.7,
                )

        # Regression line
        x_range = np.linspace(log_cs.min(), log_cs.max(), 100)
        coef = float(regression_coefficients[0]) if regression_coefficients.ndim > 0 else float(regression_coefficients)
        intercept = float(regression_intercept[0]) if regression_intercept.ndim > 0 else float(regression_intercept)
        y_pred = coef * x_range + intercept

        ax.plot(x_range, y_pred, "r-", linewidth=2, label="Regression line")

        # Confidence and prediction bands
        n = len(log_cs)
        x_mean = np.mean(log_cs)
        ss_x = np.sum((log_cs - x_mean) ** 2)

        # Standard error of regression
        residuals = shape_scores - (coef * log_cs + intercept)
        mse = np.sum(residuals**2) / (n - 2) if n > 2 else 1.0
        se = np.sqrt(mse)

        # Guard against zero variance in x (would cause division by zero)
        has_x_variance = ss_x > 0

        if show_confidence_band and n > 2 and has_x_variance:
            # 95% confidence band for the regression line
            t_val = stats.t.ppf(0.975, n - 2)
            se_line = se * np.sqrt(1 / n + (x_range - x_mean) ** 2 / ss_x)
            ci_lower = y_pred - t_val * se_line
            ci_upper = y_pred + t_val * se_line
            ax.fill_between(x_range, ci_lower, ci_upper, alpha=0.2, color="red", label="95% CI")

        if show_prediction_band and n > 2 and has_x_variance:
            # 95% prediction band for individual points
            t_val = stats.t.ppf(0.975, n - 2)
            se_pred = se * np.sqrt(1 + 1 / n + (x_range - x_mean) ** 2 / ss_x)
            pi_lower = y_pred - t_val * se_pred
            pi_upper = y_pred + t_val * se_pred
            ax.fill_between(x_range, pi_lower, pi_upper, alpha=0.1, color="blue", label="95% PI")

        # Labels and title
        ax.set_xlabel(xlabel, fontsize=self._font_size)
        ax.set_ylabel(ylabel, fontsize=self._font_size)
        ax.set_title(
            f"{title}\n$R^2$ = {r_squared:.4f}",
            fontsize=self._title_font_size,
            fontweight="bold",
        )

        # Grid and styling
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_allometry_residuals(
        self,
        centroid_sizes: npt.NDArray[np.float64],
        residuals: npt.NDArray[np.float64],
        groups: list[int] | None = None,
        title: str = "Allometry: Residual Plot",
    ) -> Figure:
        """
        Create residual plot for allometry regression.

        Parameters:
            centroid_sizes: Centroid sizes for each specimen
            residuals: Residual values (observed - predicted)
            groups: Optional group assignments
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        centroid_sizes = np.asarray(centroid_sizes, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)
        log_cs = np.log(centroid_sizes)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        if groups is not None:
            unique_groups = sorted(set(groups))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

            for idx, group in enumerate(unique_groups):
                mask = np.array(groups) == group
                ax.scatter(
                    log_cs[mask],
                    residuals[mask],
                    c=[colors[idx]],
                    label=f"Group {group}",
                    s=80,
                    alpha=0.7,
                    edgecolors="white",
                )
            ax.legend(loc="best", frameon=True, fancybox=True)
        else:
            ax.scatter(log_cs, residuals, c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Zero reference line
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

        ax.set_xlabel("Log Centroid Size", fontsize=self._font_size)
        ax.set_ylabel("Residuals", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_pls_scores(
        self,
        left_scores: npt.NDArray[np.float64],
        right_scores: npt.NDArray[np.float64],
        rv_coefficients: npt.NDArray[np.float64],
        integration_index: float,
        groups: list[int] | None = None,
        title: str = "2B-PLS: Morphological Integration",
    ) -> Figure:
        """
        Create PLS scores plot showing morphological integration.

        Parameters:
            left_scores: PLS scores for block A (n_specimens, n_components)
            right_scores: PLS scores for block B (n_specimens, n_components)
            rv_coefficients: RV coefficients per component
            integration_index: Mean absolute RV coefficient
            groups: Optional group assignments
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        left_scores = np.asarray(left_scores, dtype=np.float64)
        right_scores = np.asarray(right_scores, dtype=np.float64)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        # Plot first PLS component for both blocks
        x = left_scores[:, 0]
        y = right_scores[:, 0]

        if groups is not None:
            unique_groups = sorted(set(groups))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

            for idx, group in enumerate(unique_groups):
                mask = np.array(groups) == group
                ax.scatter(
                    x[mask],
                    y[mask],
                    c=[colors[idx]],
                    label=f"Group {group}",
                    s=80,
                    alpha=0.7,
                    edgecolors="white",
                )
            ax.legend(loc="best", frameon=True, fancybox=True)
        else:
            ax.scatter(x, y, c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=1.5, alpha=0.7)

        # Labels
        ax.set_xlabel("Block A PLS Score (Comp 1)", fontsize=self._font_size)
        ax.set_ylabel("Block B PLS Score (Comp 1)", fontsize=self._font_size)

        # Title with integration info
        rv1 = rv_coefficients[0] if len(rv_coefficients) > 0 else 0
        ax.set_title(
            f"{title}\nRV coefficient = {rv1:.4f}, Integration index = {integration_index:.4f}",
            fontsize=self._title_font_size,
            fontweight="bold",
        )

        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig
