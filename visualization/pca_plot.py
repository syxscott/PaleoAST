# =============================================================================
# FILE: visualization/pca_plot.py
# =============================================================================
"""
PCA Visualization Module for PaleoAST

This module implements publication-quality PCA plots including:
    - Score plots (scatter plots of PC scores)
    - Loading plots (variable contributions)
    - Biplots (combined score and loading plots)
    - Scree plots (variance explained)

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from config.colors import get_color_scheme
from statistics.pca import PCAResult

logger = logging.getLogger(__name__)


class PCAPlotter:
    """
    Publication-quality PCA visualization engine.
    """

    def __init__(self) -> None:
        """Initialize the PCA plotter."""
        self._logger = logging.getLogger(f"{__name__}.PCAPlotter")
        self._logger.info("PCAPlotter initialized")
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

    def plot_scores(
        self,
        result: PCAResult,
        pc1: int = 0,
        pc2: int = 1,
        groups: list[int] | None = None,
        labels: list[str] | None = None,
        title: str = "PCA Score Plot",
        show_loadings: bool = False,
        loading_scale: float = 1.0,
        annotate_samples: bool = False,
    ) -> Figure:
        """
        Create PCA score plot.

        Parameters:
            result: PCA analysis result
            pc1: First principal component (0-indexed)
            pc2: Second principal component (0-indexed)
            groups: Optional group assignments for coloring
            labels: Optional sample labels
            title: Plot title
            show_loadings: Whether to overlay loading vectors
            loading_scale: Scale factor for loading arrows
            annotate_samples: Whether to annotate sample points

        Returns:
            matplotlib Figure object
        """
        # Set style
        self.set_style(self._style)

        # Create figure
        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        # Get scores
        scores = result.get_scores(n_components=max(pc1, pc2) + 1)
        x = scores[:, pc1]
        y = scores[:, pc2]
        self._logger.info(
            f"plot_scores called: n_points={len(x)}, "
            f"n_groups={len(set(groups)) if groups else 0}, "
            f"PC{pc1 + 1} vs PC{pc2 + 1}"
        )

        # Plot points
        if groups is not None:
            unique_groups = sorted(set(groups))
            colors = get_color_scheme("default")

            for i, group in enumerate(unique_groups):
                mask = np.array(groups) == group
                ax.scatter(
                    x[mask],
                    y[mask],
                    c=[colors[i % len(colors)]],
                    label=f"Group {group}",
                    s=80,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.5,
                )

            ax.legend(loc="best", frameon=True, fancybox=True)
        else:
            ax.scatter(x, y, c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Annotate samples
        if annotate_samples and labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")

        # Add loadings vectors
        if show_loadings and result.loadings is not None:
            self._add_loadings(ax, result.loadings, loading_scale, result.mean_vector)

        # Labels and title
        var1 = result.explained_variance[pc1]
        var2 = result.explained_variance[pc2]

        ax.set_xlabel(f"PC{pc1 + 1} ({var1:.1f}% variance)", fontsize=self._font_size)
        ax.set_ylabel(f"PC{pc2 + 1} ({var2:.1f}% variance)", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")

        # Grid and spine styling
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

        # Add 95% confidence ellipse if groups provided
        if groups is not None and len(set(groups)) >= 2:
            self._add_confidence_ellipses(ax, scores, groups, pc1, pc2)

        fig.tight_layout()
        return fig

    def plot_scree(
        self, result: PCAResult, n_components: int = 10, show_cumulative: bool = True, title: str = "Scree Plot"
    ) -> Figure:
        """
        Create scree plot showing variance explained.

        Parameters:
            result: PCA analysis result
            n_components: Number of components to show
            show_cumulative: Whether to show cumulative variance
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        n_show = min(n_components, len(result.eigenvalues))
        components = np.arange(1, n_show + 1)

        # Individual variance
        variance = result.explained_variance[:n_show]

        ax.bar(components, variance, color="#3498DB", alpha=0.8, label="Individual")

        # Cumulative variance
        if show_cumulative:
            cumulative = result.cumulative_variance[:n_show]
            ax2 = ax.twinx()
            ax2.plot(components, cumulative, "o-", color="#E74C3C", linewidth=2, markersize=6, label="Cumulative")
            ax2.set_ylabel("Cumulative Variance (%)", fontsize=self._font_size, color="#E74C3C")
            ax2.tick_params(axis="y", labelcolor="#E74C3C")
            ax2.set_ylim(0, 105)

            # Add legend
            ax2.legend(loc="center right")

        ax.set_xlabel("Principal Component", fontsize=self._font_size)
        ax.set_ylabel("Variance Explained (%)", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.set_xticks(components)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_biplot(
        self,
        result: PCAResult,
        pc1: int = 0,
        pc2: int = 1,
        labels: list[str] | None = None,
        loading_labels: list[str] | None = None,
        title: str = "PCA Biplot",
    ) -> Figure:
        """
        Create biplot combining scores and loadings.

        Parameters:
            result: PCA analysis result
            pc1: First principal component
            pc2: Second principal component
            labels: Sample labels
            loading_labels: Variable labels for loadings
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        self.set_style(self._style)

        fig = Figure(figsize=self._figure_size)
        ax = fig.add_subplot(111)

        # Get scores
        scores = result.get_scores(n_components=max(pc1, pc2) + 1)
        x = scores[:, pc1]
        y = scores[:, pc2]

        # Plot samples
        ax.scatter(x, y, c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Annotate samples
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")

        # Plot loadings
        if result.loadings is not None:
            loadings = result.loadings[:, [pc1, pc2]]

            # Scale factor for visibility
            scale = np.max(np.abs(scores[:, [pc1, pc2]])) / np.max(np.abs(loadings)) * 0.8

            origin = np.array([0, 0])

            for i in range(len(loadings)):
                dx, dy = loadings[i] * scale

                ax.arrow(
                    origin[0],
                    origin[1],
                    dx,
                    dy,
                    head_width=0.1,
                    head_length=0.05,
                    fc="#E74C3C",
                    ec="#E74C3C",
                    alpha=0.8,
                )

                if loading_labels and i < len(loading_labels):
                    ax.annotate(loading_labels[i], (dx * 1.1, dy * 1.1), fontsize=9, color="#E74C3C", fontweight="bold")

        # Labels
        var1 = result.explained_variance[pc1]
        var2 = result.explained_variance[pc2]

        ax.set_xlabel(f"PC{pc1 + 1} ({var1:.1f}%)", fontsize=self._font_size)
        ax.set_ylabel(f"PC{pc2 + 1} ({var2:.1f}%)", fontsize=self._font_size)
        ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

        fig.tight_layout()
        return fig

    def _add_loadings(self, ax, loadings: npt.NDArray, scale: float, mean_vector: npt.NDArray | None = None) -> None:
        """Add loading vectors to plot."""
        origin = np.array([0, 0])

        # loadings shape: (n_variables, n_components)
        # Draw arrow for each variable using its loadings on PC1 (col 0) and PC2 (col 1)
        n_vars = min(loadings.shape[0], 10)  # Limit to 10 arrows for readability
        for i in range(n_vars):
            dx = loadings[i, 0] * scale
            dy = loadings[i, 1] * scale

            ax.arrow(
                origin[0], origin[1], dx, dy, head_width=0.05, head_length=0.03, fc="#E74C3C", ec="#E74C3C", alpha=0.8
            )

    def _add_confidence_ellipses(self, ax, scores: npt.NDArray, groups: list[int], pc1: int, pc2: int) -> None:
        """Add 95% confidence ellipses for each group."""
        from matplotlib.patches import Ellipse

        unique_groups = sorted(set(groups))
        colors = get_color_scheme("default")

        for idx, group in enumerate(unique_groups):
            mask = np.array(groups) == group
            x = scores[mask, pc1]
            y = scores[mask, pc2]

            if len(x) < 3:
                continue

            # Compute covariance and mean
            mean_x = np.mean(x)
            mean_y = np.mean(y)

            cov = np.cov(x, y)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # 95% confidence (chi-squared with 2 df)
            chi2 = 2.447  # sqrt(-2*ln(0.05)) approximately
            width = 2 * np.sqrt(eigenvalues[0]) * chi2
            height = 2 * np.sqrt(eigenvalues[1]) * chi2

            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            ellipse = Ellipse(
                (mean_x, mean_y),
                width,
                height,
                angle=angle,
                fill=False,
                edgecolor=colors[idx % len(colors)],
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(ellipse)
