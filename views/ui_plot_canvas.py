# =============================================================================
# FILE: views/ui_plot_canvas.py
# =============================================================================
"""
Interactive Plot Canvas for PaleoAST

This module implements a Matplotlib-based interactive canvas with:
    - Hover tooltips showing data point information
    - Lasso/rectangle selection for point highlighting
    - Bidirectional sync with spreadsheet selection
    - High-resolution export capabilities
    - Publication-quality styling

Design Patterns:
    - Observer Pattern: Canvas observes StateManager
    - Strategy Pattern: Different selection strategies
    - Factory Pattern: Plot type factory

Mathematical Context:
    The canvas displays ordination results and statistical plots:
        - PCA/PCoA scores: PC_j = X @ v_j
        - NMDS coordinates: Minimize stress function
        - Confidence ellipses: Mahalanobis distance

Author: PaleoAST Development Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.widgets import LassoSelector, RectangleSelector
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config.design_system import get_palette
from config.i18n import _

# Publication quality style settings
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#FAFBFC",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E4E7EB",
        "axes.labelcolor": "#2C3E50",
        "xtick.color": "#2C3E50",
        "ytick.color": "#2C3E50",
        "text.color": "#2C3E50",
        "grid.color": "#E4E7EB",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
    }
)


@dataclass
class PlotPoint:
    """Data point for interactive plotting."""

    x: float
    y: float
    label: str
    group: int | None = None
    row_index: int = 0
    metadata: dict | None = None


class InteractivePlotCanvas(QWidget):
    """
    Interactive plot canvas with hover and selection capabilities.

    Features:
        - Hover tooltips showing point information
        - Lasso and rectangle selection
        - Bidirectional sync with spreadsheet
        - High-resolution export
        - Multiple plot types (scatter, bar, line)

    Signals:
        pointsSelected: Emitted when points are selected (List[int])
        plotChanged: Emitted when plot type changes
    """

    pointsSelected = pyqtSignal(list)  # List of selected row indices
    plotChanged = pyqtSignal(str)  # Plot type

    # Colorblind-friendly palette
    COLORS = [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
        "#BBBBBB",  # Gray
        "#000000",  # Black
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._is_dark_theme = False

        # Data
        self._points: list[PlotPoint] = []
        self._selected_indices: list[int] = []
        self._groups: dict[int, list[int]] = {}

        # Plot data
        self._scores: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._labels: list[str] = []
        self._group_labels: list[int] | None = None
        self._hover_annotation = None
        self._current_plot_type: str | None = None
        self._current_dim1: int = 0
        self._current_dim2: int = 1
        self._stress: float = 0.0
        self._last_plot_call: tuple | None = None  # (method_name, args, kwargs)

        # UI
        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(QSize(20, 20))
        self._setup_toolbar()
        layout.addWidget(self._toolbar)

        # Matplotlib canvas
        self._figure = Figure(figsize=(8, 6), facecolor="#FFFFFF")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Setup axes
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#FAFBFC")

        # Enable interaction
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        layout.addWidget(self._canvas)

        # Apply dark theme
        self._apply_stylesheet()

    def _setup_toolbar(self) -> None:
        """Setup toolbar buttons."""
        # Selection mode
        self._selection_label = QLabel(_("Selection:"))
        self._toolbar.addWidget(self._selection_label)

        self._selection_combo = QComboBox()
        self._selection_combo.addItems([_("None"), _("Rectangle"), _("Lasso")])
        self._toolbar.addWidget(self._selection_combo)

        self._toolbar.addSeparator()

        # Zoom controls
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setMaximumWidth(30)
        zoom_in_btn.clicked.connect(self._zoom_in)
        self._toolbar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setMaximumWidth(30)
        zoom_out_btn.clicked.connect(self._zoom_out)
        self._toolbar.addWidget(zoom_out_btn)

        reset_btn = QPushButton(_("Reset"))
        reset_btn.clicked.connect(self._reset_view)
        self._toolbar.addWidget(reset_btn)

        self._toolbar.addSeparator()

        # Export
        export_btn = QPushButton(_("Export"))
        export_btn.clicked.connect(self._export_plot)
        self._toolbar.addWidget(export_btn)

        # Show labels toggle
        self._toolbar.addSeparator()
        self._show_labels_check = QCheckBox(_("Show Labels"))
        self._show_labels_check.setChecked(True)
        self._show_labels_check.toggled.connect(self._toggle_labels)
        self._toolbar.addWidget(self._show_labels_check)

        # Show ellipses toggle
        self._show_ellipses_check = QCheckBox(_("Show 95% Ellipses"))
        self._show_ellipses_check.setChecked(False)
        self._show_ellipses_check.toggled.connect(self._toggle_ellipses)
        self._toolbar.addWidget(self._show_ellipses_check)

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {c.bg_primary};
            }}
            QToolBar {{
                background-color: {c.bg_secondary};
                border: 1px solid {c.border_light};
                spacing: 8px;
                padding: 6px;
            }}
            QPushButton {{
                background-color: {c.bg_tertiary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 50px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {c.bg_hover};
                border: 1px solid {c.primary};
            }}
            QPushButton:pressed {{
                background-color: {c.primary};
                color: white;
            }}
            QComboBox {{
                background-color: {c.bg_primary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 6px;
                padding: 6px 8px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border: 1px solid {c.primary};
            }}
            QComboBox:focus {{
                border: 1px solid {c.primary};
            }}
            QComboBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {c.primary};
            }}
            QComboBox QAbstractItemView {{
                background-color: {c.bg_primary};
                color: {c.text_primary};
                selection-background-color: {c.primary};
                selection-color: white;
            }}
            QCheckBox {{
                color: {c.text_primary};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {c.border_medium};
                border-radius: 3px;
                background-color: {c.bg_primary};
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {c.primary};
            }}
            QCheckBox::indicator:checked {{
                background-color: {c.primary};
                border-color: {c.primary_dark};
            }}
            QLabel {{
                color: {c.text_primary};
                padding: 0 4px;
            }}
        """)

        self._figure.patch.set_facecolor(c.bg_primary)

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and sync with matplotlib."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

        # Sync matplotlib theme
        if is_dark:
            # Dark theme colors for matplotlib
            plt.rcParams.update({
                "axes.facecolor": "#1E1E1E",
                "figure.facecolor": "#2D2D2D",
                "axes.edgecolor": "#555555",
                "axes.labelcolor": "#E0E0E0",
                "xtick.color": "#E0E0E0",
                "ytick.color": "#E0E0E0",
                "text.color": "#E0E0E0",
                "grid.color": "#444444",
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
        else:
            # Light theme colors for matplotlib
            plt.rcParams.update({
                "axes.facecolor": "#FAFBFC",
                "figure.facecolor": "#FFFFFF",
                "axes.edgecolor": "#E4E7EB",
                "axes.labelcolor": "#2C3E50",
                "xtick.color": "#2C3E50",
                "ytick.color": "#2C3E50",
                "text.color": "#2C3E50",
                "grid.color": "#E4E7EB",
                "axes.edgecolor": "#E4E7EB",
                "axes.spines.top": False,
                "axes.spines.right": False,
            })

        # Update figure background
        c = get_palette(is_dark)
        self._figure.patch.set_facecolor(c.bg_primary)

        # Redraw if there's an active plot
        if self._current_plot_type and self._ax is not None:
            self._canvas.draw_idle()

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self._selection_combo.currentIndexChanged.connect(self._on_selection_mode_changed)

    # =========================================================================
    # PCA Plotting Methods
    # =========================================================================

    def plot_pca_scores(self, result: Any, pc1: int = 0, pc2: int = 1) -> None:
        """
        Plot PCA scores.

        Mathematical Context:
            PC scores: PC_j = X_centered @ v_j

            where:
                X_centered = X - μ (centered data)
                v_j = j-th eigenvector of covariance matrix

        The plot shows:
            - X-axis: PC1 scores
            - Y-axis: PC2 scores
            - Points colored by group
            - Eigenvalue percentages on axes
        """
        self._record_plot_call("plot_pca_scores", result, pc1=pc1, pc2=pc2)
        self._ax.clear()
        self._current_plot_type = "pca"

        # Extract data from result
        if hasattr(result, "scores"):
            scores = result.scores
        else:
            scores = result

        if hasattr(result, "explained_variance"):
            eigenvalues = result.explained_variance
        else:
            eigenvalues = np.ones(scores.shape[1])

        if hasattr(result, "labels"):
            labels = result.labels
        else:
            labels = [f"S{i + 1}" for i in range(scores.shape[0])]

        if hasattr(result, "groups"):
            groups = result.groups
        else:
            groups = np.zeros(scores.shape[0], dtype=int)

        # Store data
        self._scores = scores
        self._eigenvalues = eigenvalues
        self._labels = labels
        self._group_labels = groups
        self._current_dim1 = pc1
        self._current_dim2 = pc2

        # Calculate variance explained
        total_var = np.sum(eigenvalues)
        var_pc1 = eigenvalues[pc1] / total_var * 100
        var_pc2 = eigenvalues[pc2] / total_var * 100

        # Set up groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot points by group
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            self._ax.scatter(
                scores[idx, pc1],
                scores[idx, pc2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=f"Group {group + 1}" if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Add labels if enabled
        if self._show_labels_check.isChecked():
            for i, (x, y) in enumerate(zip(scores[:, pc1], scores[:, pc2])):
                self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8, ha="center", va="bottom")

        # Add ellipses if enabled
        if self._show_ellipses_check.isChecked():
            self._add_confidence_ellipses(scores, groups, pc1, pc2)

        # Labels and title
        self._ax.set_xlabel(_("PC{0} ({1:.1f}% variance)").format(pc1 + 1, var_pc1))
        self._ax.set_ylabel(_("PC{0} ({1:.1f}% variance)").format(pc2 + 1, var_pc2))
        self._ax.set_title(_("PCA Scores Plot"))

        # Style
        self._ax.set_facecolor("#FAFBFC")
        self._ax.tick_params(colors="#2C3E50")
        self._ax.xaxis.label.set_color("#2C3E50")
        self._ax.yaxis.label.set_color("#2C3E50")
        self._ax.title.set_color("#2C3E50")

        for spine in self._ax.spines.values():
            spine.set_color("#E4E7EB")

        if len(unique_groups) > 1:
            self._ax.legend(
                loc="upper right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E4E7EB", labelcolor="#2C3E50"
            )

        # Add grid
        self._ax.grid(True, alpha=0.3, color="#E4E7EB")

        # Draw reference lines
        self._ax.axhline(y=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)
        self._ax.axvline(x=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)

        self._canvas.draw()

    # =========================================================================
    # PCoA Plotting Methods
    # =========================================================================

    def plot_pcoa_scores(self, result: Any, coord1: int = 0, coord2: int = 1) -> None:
        """
        Plot PCoA scores.

        Mathematical Context:
            PCoA coordinates: X = U Λ^(1/2)

            where:
                U = eigenvector matrix
                Λ = diagonal eigenvalue matrix
        """
        self._record_plot_call("plot_pcoa_scores", result, coord1=coord1, coord2=coord2)
        self._ax.clear()
        self._current_plot_type = "pcoa"

        # Extract data
        if hasattr(result, "coordinates"):
            coords = result.coordinates
        else:
            coords = result

        if hasattr(result, "eigenvalues"):
            eigenvalues = result.eigenvalues
        else:
            eigenvalues = np.ones(coords.shape[1])

        if hasattr(result, "labels"):
            labels = result.labels
        else:
            labels = [f"S{i + 1}" for i in range(coords.shape[0])]

        if hasattr(result, "groups"):
            groups = result.groups
        else:
            groups = np.zeros(coords.shape[0], dtype=int)

        # Store data
        self._scores = coords
        self._eigenvalues = eigenvalues
        self._labels = labels
        self._group_labels = groups
        self._current_dim1 = coord1
        self._current_dim2 = coord2

        # Calculate variance explained
        total_var = np.sum(np.abs(eigenvalues))
        var_1 = np.abs(eigenvalues[coord1]) / total_var * 100
        var_2 = np.abs(eigenvalues[coord2]) / total_var * 100

        # Setup groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            self._ax.scatter(
                coords[idx, coord1],
                coords[idx, coord2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=f"Group {group + 1}" if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Labels
        if self._show_labels_check.isChecked():
            for i, (x, y) in enumerate(zip(coords[:, coord1], coords[:, coord2])):
                self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Ellipses
        if self._show_ellipses_check.isChecked():
            self._add_confidence_ellipses(coords, groups, coord1, coord2)

        # Axis labels
        self._ax.set_xlabel(_("PCo{0} ({1:.1f}% variance)").format(coord1 + 1, var_1))
        self._ax.set_ylabel(_("PCo{0} ({1:.1f}% variance)").format(coord2 + 1, var_2))
        self._ax.set_title(_("PCoA Scores Plot"))

        # Style
        self._apply_axis_style()

        self._canvas.draw()

    # =========================================================================
    # CCA/RDA Plotting Methods (Triplot)
    # =========================================================================

    def plot_cca_triplot(self, result: Any, ax1: int = 0, ax2: int = 1) -> None:
        """
        Plot CCA/RDA triplot showing samples, species, and environmental vectors.

        Triplot Components:
            - Sample scores: points (colored by group if available)
            - Species scores: points (different marker)
            - Environmental vectors: arrows (biplot scores)

        Mathematical Context:
            Site scores: Y_centered @ U
            Species scores: U * sqrt(Λ)
            Biplot scores: X_centered' @ site_scores
        """
        self._record_plot_call("plot_cca_triplot", result, ax1=ax1, ax2=ax2)
        self._ax.clear()
        self._current_plot_type = "cca"

        # Extract data from result
        if hasattr(result, "site_scores"):
            site_scores = result.site_scores
        else:
            site_scores = result

        if hasattr(result, "species_scores"):
            species_scores = result.species_scores
        else:
            species_scores = None

        if hasattr(result, "biplot_scores"):
            biplot_scores = result.biplot_scores
        else:
            biplot_scores = None

        if hasattr(result, "eigenvalues"):
            eigenvalues = result.eigenvalues
        else:
            eigenvalues = np.ones(site_scores.shape[1])

        if hasattr(result, "proportion_explained"):
            proportion = result.proportion_explained
        else:
            proportion = np.ones(len(eigenvalues)) * 100 / len(eigenvalues)

        if hasattr(result, "method"):
            method = result.method.upper()
        else:
            method = "CCA"

        if hasattr(result, "groups"):
            groups = result.groups
        else:
            groups = np.zeros(site_scores.shape[0], dtype=int)

        if hasattr(result, "labels"):
            labels = result.labels
        else:
            labels = [f"S{i + 1}" for i in range(site_scores.shape[0])]

        # Store data for selection
        self._scores = site_scores
        self._eigenvalues = eigenvalues
        self._labels = labels
        self._group_labels = groups
        self._current_dim1 = ax1
        self._current_dim2 = ax2

        # Calculate variance explained
        total_var = np.sum(eigenvalues)
        var_ax1 = eigenvalues[ax1] / total_var * 100 if total_var > 0 else 0
        var_ax2 = eigenvalues[ax2] / total_var * 100 if total_var > 0 else 0

        # Set up groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot sample scores (sites) by group
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            self._ax.scatter(
                site_scores[idx, ax1],
                site_scores[idx, ax2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                marker="o",
                label=f"Group {group + 1}" if len(unique_groups) > 1 else _("Samples"),
                picker=True,
            )

        # Plot species scores
        if species_scores is not None:
            self._ax.scatter(
                species_scores[:, ax1],
                species_scores[:, ax2],
                c="#E74C3C",
                s=60,
                alpha=0.6,
                marker="^",
                edgecolors="white",
                linewidths=0.5,
                label=_("Species"),
            )

        # Plot environmental vectors (biplot arrows)
        if biplot_scores is not None:
            n_env = biplot_scores.shape[0]
            for i in range(n_env):
                x_end = biplot_scores[i, ax1]
                y_end = biplot_scores[i, ax2]
                length = np.sqrt(x_end**2 + y_end**2)

                # Scale arrow for visibility
                scale = 1.0
                if length > 0:
                    # Determine env name
                    env_name = result.env_names[i] if hasattr(result, "env_names") else f"Env_{i + 1}"

                    self._ax.annotate(
                        "",
                        xy=(x_end * scale, y_end * scale),
                        xytext=(0, 0),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=self.COLORS[2],
                            lw=2,
                        ),
                    )

                    # Label position offset
                    label_offset = 1.1
                    self._ax.text(
                        x_end * scale * label_offset,
                        y_end * scale * label_offset,
                        env_name,
                        fontsize=9,
                        color=self.COLORS[2],
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

        # Add labels if enabled
        if self._show_labels_check.isChecked():
            for i, (x, y) in enumerate(zip(site_scores[:, ax1], site_scores[:, ax2])):
                self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Axis labels
        self._ax.set_xlabel(_("{0}1 ({1:.1f}% variance)").format(method, var_ax1))
        self._ax.set_ylabel(_("{0}2 ({1:.1f}% variance)").format(method, var_ax2))
        self._ax.set_title(_("{0} Triplot").format(method))

        # Style
        self._apply_axis_style()

        # Legend
        if len(unique_groups) > 1 or species_scores is not None or biplot_scores is not None:
            self._ax.legend(
                loc="upper right",
                framealpha=0.95,
                facecolor="#FFFFFF",
                edgecolor="#E4E7EB",
                labelcolor="#2C3E50",
            )

        # Reference lines
        self._ax.axhline(y=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)
        self._ax.axvline(x=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)

        # Grid
        self._ax.grid(True, alpha=0.3, color="#E4E7EB")

        self._canvas.draw()

    # =========================================================================
    # TPS Deformation Grid
    # =========================================================================

    def plot_tps_deformation_grid(
        self,
        tps_result: Any,
        grid_shape: tuple[int, int] = (15, 15),
        show_vectors: bool = True
    ) -> None:
        """
        Plot TPS deformation grid visualization.

        Shows how a grid is warped from source to target configuration
        using Thin-Plate Spline interpolation.

        Parameters:
            tps_result: TPSResult containing source, target, warped configurations
            grid_shape: Shape of the deformation grid (rows, cols)
            show_vectors: Whether to show displacement vectors
        """
        self._record_plot_call("plot_tps_deformation_grid", tps_result, grid_shape=grid_shape, show_vectors=show_vectors)
        self._ax.clear()
        self._current_plot_type = "tps_grid"

        # Extract data from TPS result
        if hasattr(tps_result, "source"):
            source = tps_result.source
        else:
            source = tps_result

        if hasattr(tps_result, "target"):
            target = tps_result.target
        else:
            target = source

        if hasattr(tps_result, "warped"):
            warped = tps_result.warped
        else:
            warped = target

        if hasattr(tps_result, "landmarks"):
            landmarks = tps_result.landmarks
        else:
            landmarks = None

        # Generate original grid points
        x_min, x_max = source[:, 0].min(), source[:, 0].max()
        y_min, y_max = source[:, 1].min(), source[:, 1].max()

        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Create regular grid
        x_grid = np.linspace(x_min, x_max, grid_shape[1])
        y_grid = np.linspace(y_min, y_max, grid_shape[0])
        xx_orig, yy_orig = np.meshgrid(x_grid, y_grid)

        # Warp grid using TPS result's control point mapping
        # We need to compute the warp transformation
        from morphometrics.tps import TPSAnalyzer

        tps_analyzer = TPSAnalyzer()
        try:
            warped_grid = tps_analyzer.warp_grid(tps_result, grid_shape=grid_shape)
        except Exception:
            # Fallback: just use target shape
            warped_grid = np.zeros((grid_shape[0], grid_shape[1], 2))
            tx_min, tx_max = target[:, 0].min(), target[:, 0].max()
            ty_min, ty_max = target[:, 1].min(), target[:, 1].max()
            wx_grid = np.linspace(tx_min, tx_max, grid_shape[1])
            wy_grid = np.linspace(ty_min, ty_max, grid_shape[0])
            warped_grid[:, :, 0], warped_grid[:, :, 1] = np.meshgrid(wx_grid, wy_grid)

        # Plot original grid (reference) - light gray dashed
        for i in range(grid_shape[0]):
            self._ax.plot(xx_orig[i, :], yy_orig[i, :], color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)
        for j in range(grid_shape[1]):
            self._ax.plot(xx_orig[:, j], yy_orig[:, j], color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)

        # Plot warped grid - colored lines
        for i in range(grid_shape[0]):
            self._ax.plot(warped_grid[i, :, 0], warped_grid[i, :, 1], color="#3498DB", linewidth=1.5, alpha=0.8)
        for j in range(grid_shape[1]):
            self._ax.plot(warped_grid[:, j, 0], warped_grid[:, j, 1], color="#3498DB", linewidth=1.5, alpha=0.8)

        # Plot source landmarks
        self._ax.scatter(source[:, 0], source[:, 1], c="#E74C3C", s=80, marker="o",
                        edgecolors="white", linewidths=1, label=_("Source"), zorder=5)

        # Plot target landmarks
        self._ax.scatter(target[:, 0], target[:, 1], c="#27AE60", s=80, marker="s",
                        edgecolors="white", linewidths=1, label=_("Target"), zorder=5)

        # Show displacement vectors if requested
        if show_vectors and len(source) == len(target):
            # Draw arrows from source to target for each landmark
            for i in range(len(source)):
                self._ax.annotate(
                    "",
                    xy=(target[i, 0], target[i, 1]),
                    xytext=(source[i, 0], source[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.5, alpha=0.6),
                )

        # Labels and title
        self._ax.set_xlabel(_("X"))
        self._ax.set_ylabel(_("Y"))
        self._ax.set_title(_("TPS Deformation Grid"))

        # Style
        self._apply_axis_style()

        # Legend
        self._ax.legend(loc="upper right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E4E7EB")

        # Aspect ratio
        self._ax.set_aspect("equal", adjustable="box")

        self._canvas.draw()

    # =========================================================================
    # NMDS Plotting Methods
    # =========================================================================

    def plot_nmds(self, result: Any) -> None:
        """
        Plot NMDS ordination.

        Mathematical Context:
            NMDS minimizes the stress function:

            Stress = √(Σ(d_ij - d̂_ij)² / Σd_ij²)

            where:
                d_ij = original dissimilarity
                d̂_ij = ordination distance
        """
        self._record_plot_call("plot_nmds", result)
        self._ax.clear()
        self._current_plot_type = "nmds"

        # Extract data
        if hasattr(result, "coordinates"):
            coords = result.coordinates
        else:
            coords = result

        stress = getattr(result, "stress", 0)

        if hasattr(result, "labels"):
            labels = result.labels
        else:
            labels = [f"S{i + 1}" for i in range(coords.shape[0])]

        if hasattr(result, "groups"):
            groups = result.groups
        else:
            groups = np.zeros(coords.shape[0], dtype=int)

        # Store data
        self._scores = coords
        self._labels = labels
        self._group_labels = groups
        self._stress = stress

        # Setup groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            self._ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c=color,
                s=100,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=f"Group {group + 1}" if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Labels
        if self._show_labels_check.isChecked():
            for i, (x, y) in enumerate(zip(coords[:, 0], coords[:, 1])):
                self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Ellipses
        if self._show_ellipses_check.isChecked():
            self._add_confidence_ellipses(coords, groups, 0, 1)

        # Axis labels and title
        self._ax.set_xlabel(_("NMDS1"))
        self._ax.set_ylabel(_("NMDS2"))
        self._ax.set_title(_("NMDS Ordination (Stress = {0:.4f})").format(stress))

        # Style
        self._apply_axis_style()

        self._canvas.draw()

    # =========================================================================
    # Diversity Plotting Methods
    # =========================================================================

    def plot_diversity_summary(self, result: Any) -> None:
        """
        Plot biodiversity summary.

        Mathematical Context:
            Displays multiple diversity indices:
                - Shannon: H' = -Σ p_i ln(p_i)
                - Simpson: D = 1 - Σ p_i²
                - Fisher's α: N = α ln(1 + N/α)
        """
        self._record_plot_call("plot_diversity_summary", result)
        self._ax.clear()
        self._current_plot_type = "diversity"

        # Extract data
        if hasattr(result, "indices"):
            indices = result.indices
        else:
            indices = ["Shannon", "Simpson", "Fisher"]

        if hasattr(result, "values"):
            values = result.values
        else:
            values = [1.5, 0.8, 2.5]

        if hasattr(result, "labels"):
            labels = result.labels
        else:
            labels = indices

        # Bar chart
        x_pos = np.arange(len(indices))
        bars = self._ax.bar(x_pos, values, color=self.COLORS[: len(indices)], alpha=0.7, edgecolor="white", linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            self._ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#2C3E50",
                fontweight="bold",
            )

        # Labels
        self._ax.set_xticks(x_pos)
        self._ax.set_xticklabels(labels, color="#2C3E50")
        self._ax.set_ylabel(_("Index Value"), color="#2C3E50")
        self._ax.set_title(_("Biodiversity Indices"), color="#2C3E50")

        # Style
        self._apply_axis_style()

        self._canvas.draw()

    def plot_rarefaction(self, result: Any) -> None:
        """
        Plot rarefaction curves.

        Mathematical Context:
            E(S_n) = Σ[1 - C(N-n_i, n) / C(N, n)]
        """
        self._record_plot_call("plot_rarefaction", result)
        self._ax.clear()
        self._current_plot_type = "rarefaction"

        # Extract data from CoverageRarefactionResult
        if hasattr(result, "coverage_levels") and hasattr(result, "expected_richness"):
            coverage_levels = result.coverage_levels
            expected_richness = result.expected_richness
            sample_names = getattr(result, "sample_names", [f"Sample {i+1}" for i in range(len(coverage_levels))])
            ci_lower = getattr(result, "confidence_lower", None)
            ci_upper = getattr(result, "confidence_upper", None)

            # Build curve_data dict for consistent plotting
            curve_data = {}
            for i, name in enumerate(sample_names):
                curve_data[name] = (coverage_levels, expected_richness)
        elif hasattr(result, "curve_data"):
            # Legacy format
            curve_data = result.curve_data
            ci_lower = None
            ci_upper = None
        else:
            # No data available
            self._ax.text(0.5, 0.5, _("No rarefaction data available"),
                         ha="center", va="center", fontsize=12)
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_title(_("Rarefaction Curves: No Data"), fontsize=12, fontweight="bold")
            self._canvas.draw()
            return

        # Plot curves
        for i, (sample, (x, y)) in enumerate(curve_data.items()):
            color = self.COLORS[i % len(self.COLORS)]
            self._ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, alpha=0.7, label=sample)

        # Labels
        self._ax.set_xlabel(_("Number of Individuals"), color="#2C3E50")
        self._ax.set_ylabel(_("Expected Species Richness"), color="#2C3E50")
        self._ax.set_title(_("Rarefaction Curves"), color="#2C3E50")

        # Legend
        self._ax.legend(
            loc="lower right", framealpha=0.9, facecolor="#2C3E50", edgecolor="#34495E", labelcolor="#ECF0F1"
        )

        # Style
        self._apply_axis_style()

        self._canvas.draw()

    # =========================================================================
    # Spectral / ANOSIM / PERMANOVA Plotting
    # =========================================================================

    def plot_spectral(self, result: Any) -> None:
        """
        Plot spectral analysis results (Lomb-Scargle periodogram).

        Shows the power spectrum with frequency on x-axis and power on y-axis.
        """
        self._record_plot_call("plot_spectral", result)
        self._ax.clear()
        self._current_plot_type = "spectral"

        # Extract data
        if hasattr(result, "frequencies") and hasattr(result, "power"):
            frequencies = result.frequencies
            power = result.power
        else:
            # No spectral data available
            self._ax.text(0.5, 0.5, _("No spectral data available"),
                         ha="center", va="center", fontsize=12)
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_title(_("Spectral Analysis: No Data"), fontsize=12, fontweight="bold")
            self._canvas.draw()
            return

        peak_frequency = getattr(result, "peak_frequency", None)
        peak_period = getattr(result, "peak_period", None)

        # Plot power spectrum
        self._ax.plot(frequencies, power, color=self.COLORS[0], linewidth=1.5, alpha=0.8)

        # Highlight peak
        if peak_frequency is not None:
            peak_idx = np.argmin(np.abs(frequencies - peak_frequency))
            self._ax.axvline(
                x=peak_frequency,
                color=self.COLORS[3],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"{_('Peak')}: f={peak_frequency:.4f}, T={peak_period:.2f}" if peak_period else f"{_('Peak')}: f={peak_frequency:.4f}",
            )

        # Fill under curve
        self._ax.fill_between(frequencies, power, alpha=0.15, color=self.COLORS[0])

        # Labels
        self._ax.set_xlabel(_("Frequency"))
        self._ax.set_ylabel(_("Power"))
        self._ax.set_title(_("Lomb-Scargle Periodogram"))

        if peak_frequency is not None:
            self._ax.legend(loc="upper right", framealpha=0.9)

        # Style
        self._apply_axis_style()
        self._canvas.draw()

    def plot_wavelet_scalogram(self, result: Any) -> None:
        """
        Plot wavelet CWT scalogram.

        Shows time-frequency power distribution as a heatmap.
        The Cone of Influence is shown as a hatched region.
        """
        self._record_plot_call("plot_wavelet_scalogram", result)
        self._ax.clear()
        self._current_plot_type = "wavelet"

        # Remove old colorbar if exists to prevent memory leak
        if hasattr(self, "_colorbar") and self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        # Extract data
        time = getattr(result, "time", None)
        frequencies = getattr(result, "frequencies", None)
        power = getattr(result, "power", None)
        coi = getattr(result, "coi", None)
        wavelet = getattr(result, "wavelet", "Morlet")

        # Check if required data is available
        if time is None or frequencies is None or power is None:
            self._ax.text(0.5, 0.5, _("No wavelet data available"),
                         ha="center", va="center", fontsize=12)
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_title(_("Wavelet Scalogram: No Data"), fontsize=12, fontweight="bold")
            self._canvas.draw()
            return

        # Use pcolormesh for the scalogram
        time_grid, freq_grid = np.meshgrid(time, frequencies)

        # Plot power as colormap
        self._pcolormesh = self._ax.pcolormesh(
            time_grid, freq_grid, power,
            shading="gouraud",
            cmap="hot",
        )

        # Add colorbar
        self._colorbar = self._figure.colorbar(self._pcolormesh, ax=self._ax, label=_("Power"))

        # Mark COI if available
        if coi is not None and len(coi) > 0:
            # COI is a mask - show edge effects as hatched region
            # Use fill_between to show regions where edge effects are significant
            coi_region = np.where(coi, frequencies.max(), np.nan)
            self._ax.fill_between(
                time,
                coi_region,
                frequencies.max(),
                alpha=0.3,
                color="white",
                hatch="///",
                label=_("Cone of Influence"),
            )

        # Labels
        self._ax.set_xlabel(_("Time"))
        self._ax.set_ylabel(_("Frequency"))
        self._ax.set_title(_("Wavelet CWT Scalogram ({0})").format(wavelet))

        # Invert y-axis so low frequencies are at bottom
        self._ax.invert_yaxis()

        # Style
        self._ax.set_facecolor("#1A1A2E")
        self._canvas.draw()

    def plot_anosim_results(self, result: Any) -> None:
        """
        Plot ANOSIM results.

        Shows a bar chart of the R statistic with significance indicator.
        """
        self._record_plot_call("plot_anosim_results", result)
        self._ax.clear()
        self._current_plot_type = "anosim"

        # Extract data
        R = getattr(result, "statistic", 0.0)
        p_value = getattr(result, "p_value", 1.0)
        n_groups = getattr(result, "n_groups", 1)
        n_samples = getattr(result, "n_samples", 0)

        # Bar chart for R statistic
        color = self.COLORS[3] if p_value < 0.05 else self.COLORS[0]
        bar = self._ax.bar([0], [R], color=color, alpha=0.7, edgecolor="white", width=0.5)

        # Add value label
        self._ax.annotate(
            f"R = {R:.4f}",
            xy=(0, R),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#2C3E50",
        )

        # Significance annotation
        sig_text = f"p = {p_value:.4f}"
        if p_value < 0.01:
            sig_text += " **"
        elif p_value < 0.05:
            sig_text += " *"
        self._ax.annotate(
            sig_text,
            xy=(0, R / 2),
            ha="center",
            va="center",
            fontsize=11,
            color="#FFFFFF" if R > 0.3 else "#2C3E50",
            fontweight="bold",
        )

        # Info text
        info = f"{_('Groups')}: {n_groups}  |  {_('Samples')}: {n_samples}"
        self._ax.text(
            0, -0.1 * max(abs(R), 0.1), info, ha="center", va="top", fontsize=9, color="#7F8C8D",
            transform=self._ax.get_xaxis_transform(),
        )

        # Labels
        self._ax.set_xticks([0])
        self._ax.set_xticklabels(["ANOSIM R"])
        self._ax.set_ylabel(_("R Statistic"))
        self._ax.set_title(_("ANOSIM Results"))
        self._ax.set_ylim(-0.1, max(1.0, R * 1.3))

        # Style
        self._apply_axis_style()
        self._canvas.draw()

    def plot_permanova_results(self, result: Any) -> None:
        """
        Plot PERMANOVA results.

        Shows a summary panel with F statistic, p-value, and variance decomposition.
        """
        self._record_plot_call("plot_permanova_results", result)
        self._ax.clear()
        self._current_plot_type = "permanova"

        # Extract data
        F = getattr(result, "f_statistic", 0.0)
        p_value = getattr(result, "p_value", 1.0)
        ss_between = getattr(result, "ss_between", 0.0)
        ss_within = getattr(result, "ss_within", 0.0)
        df_between = getattr(result, "df_between", 1)
        df_within = getattr(result, "df_within", 1)
        ms_between = getattr(result, "ms_between", 0.0)
        ms_within = getattr(result, "ms_within", 0.0)
        n_groups = getattr(result, "n_groups", 1)
        n_samples = getattr(result, "n_samples", 0)

        # Variance decomposition pie chart
        ss_between = max(0.0, float(ss_between) if np.isfinite(ss_between) else 0.0)
        ss_within = max(0.0, float(ss_within) if np.isfinite(ss_within) else 0.0)
        total_ss = ss_between + ss_within

        colors = [self.COLORS[0], self.COLORS[2]]

        if total_ss > 0 and ss_between > 0:
            sizes = [ss_between / total_ss, ss_within / total_ss]
            labels_pie = [
                f"{_('Between')}\nSS={ss_between:.2f}\n({sizes[0] * 100:.1f}%)",
                f"{_('Within')}\nSS={ss_within:.2f}\n({sizes[1] * 100:.1f}%)",
            ]
            wedges, texts = self._ax.pie(sizes, labels=labels_pie, colors=colors, startangle=90, textprops={"fontsize": 9})

            # Add center text with F and p (pie axes centered at origin)
            sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            center_text = f"F = {F:.3f}{sig}\np = {p_value:.4f}"
            self._ax.text(
                0, 0, center_text, ha="center", va="center", fontsize=10, fontweight="bold", color="#2C3E50",
            )
        else:
            # Single group or zero between-group SS: show text instead of pie
            sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            info_text = (
                f"F = {F:.3f}{sig}\n"
                f"p = {p_value:.4f}\n\n"
                f"{_('Within-group SS')}: {ss_within:.2f}\n"
                f"({_('No between-group variance')})"
            )
            self._ax.text(
                0.5, 0.5, info_text,
                ha="center", va="center", fontsize=11, color="#2C3E50",
                transform=self._ax.transAxes,
            )

        # Title
        self._ax.set_title(
            f"{_('PERMANOVA')} ({_('Groups')}: {n_groups}, {_('Samples')}: {n_samples})",
        )

        self._canvas.draw()

    # =========================================================================
    # New Feature Plot Methods
    # =========================================================================

    def plot_simper_results(self, result: Any) -> None:
        """Plot SIMPER contribution bar chart with cumulative line."""
        self._record_plot_call("plot_simper_results", result)
        self._current_plot_type = "simper"
        self._simper_result = result  # Store for replotting
        self._figure.clear()

        # Use twiny() for cumulative contribution line on top
        self._ax = self._figure.add_subplot(111)
        ax_top = self._ax.twiny()

        # Get top contributors (flat list from SimperResult)
        all_contribs: dict[str, float] = {}
        for vc in result.contributions:
            all_contribs[vc.name] = max(all_contribs.get(vc.name, 0), vc.average)

        # Sort by contribution
        sorted_items = sorted(all_contribs.items(), key=lambda x: x[1], reverse=True)[:15]
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Calculate cumulative contributions
        total = sum(values)
        cumulative = np.cumsum([v / total * 100 for v in values])

        y_pos = np.arange(len(names))
        bars = self._ax.barh(y_pos, values, color=self.COLORS[0], alpha=0.8)
        self._ax.set_yticks(y_pos)
        self._ax.set_yticklabels(names, fontsize=8)
        self._ax.invert_yaxis()
        self._ax.set_xlabel(_("Average Contribution (%)"), color=self.COLORS[0])
        self._ax.set_title(_("SIMPER: Top Contributing Variables"))

        for bar, val in zip(bars, values):
            self._ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=8, color="#2C3E50")

        # Cumulative contribution line on top axis
        ax_top.plot(cumulative, y_pos, color=self.COLORS[3], linewidth=2, marker="o", markersize=4, alpha=0.8)
        ax_top.set_xlabel(_("Cumulative Contribution (%)"), color=self.COLORS[3])
        ax_top.tick_params(axis="x", colors=self.COLORS[3])
        ax_top.set_xlim([0, 100])

        # Style
        self._ax.set_facecolor("#FAFBFC")
        self._ax.spines["top"].set_color(self.COLORS[0])
        ax_top.spines["top"].set_color(self.COLORS[3])

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_anova_boxplot(self, data: np.ndarray, groups: list[int],
                           variable_name: str = "", group_names: list[str] | None = None) -> None:
        """Plot boxplot comparing groups for a variable."""
        self._record_plot_call("plot_anova_boxplot", data, groups, variable_name=variable_name, group_names=group_names)
        self._current_plot_type = "anova_boxplot"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        unique_groups = sorted(set(groups))
        if group_names is None:
            group_names = [f"Group {g+1}" for g in unique_groups]

        plot_data = []
        for g in unique_groups:
            mask = np.array(groups) == g
            plot_data.append(data[mask])

        bp = self._ax.boxplot(plot_data, labels=group_names[:len(unique_groups)], patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(self.COLORS[i % len(self.COLORS)])
            patch.set_alpha(0.7)

        self._ax.set_ylabel(variable_name)
        self._ax.set_title(f"{_('Group Comparison')}: {variable_name}")
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_lda_scores(self, result: Any) -> None:
        """Plot LDA scatter plot with confidence ellipses."""
        self._record_plot_call("plot_lda_scores", result)
        self._current_plot_type = "lda"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        scores = result.scores
        n_dims = scores.shape[1]

        # Store data for replotting
        self._scores = scores
        self._group_labels = result.groups if hasattr(result, "groups") else np.zeros(len(scores), dtype=int)
        self._labels = getattr(result, "labels", [f"S{i}" for i in range(len(scores))])

        if n_dims >= 2:
            x_data = scores[:, 0]
            y_data = scores[:, 1]
            x_label = "LD1"
            y_label = "LD2"
            pc1, pc2 = 0, 1
        else:
            x_data = scores[:, 0]
            y_data = np.zeros(len(x_data))
            x_label = "LD1"
            y_label = ""
            pc1, pc2 = 0, 0

        groups = self._group_labels
        unique_groups = np.unique(groups)

        for i, g in enumerate(unique_groups):
            mask = groups == g
            color = self.COLORS[i % len(self.COLORS)]
            self._ax.scatter(x_data[mask], y_data[mask], c=color, s=50, alpha=0.7,
                           edgecolors="white", linewidth=0.5, label=f"Group {g+1}")

        # Add labels if enabled
        if self._show_labels_check.isChecked():
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                self._ax.annotate(self._labels[i], (x, y), fontsize=8, alpha=0.8, ha="center", va="bottom")

        # Add ellipses if enabled
        if self._show_ellipses_check.isChecked() and n_dims >= 2:
            self._add_confidence_ellipses(scores, groups, pc1, pc2)

        if hasattr(result, "explained_variance_ratio") and len(result.explained_variance_ratio) >= 2:
            x_label = f"LD1 ({result.explained_variance_ratio[0]:.1%})"
            y_label = f"LD2 ({result.explained_variance_ratio[1]:.1%})"

        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        self._ax.set_title(_("Linear Discriminant Analysis"))

        # Style
        self._ax.set_facecolor("#FAFBFC")
        self._ax.tick_params(colors="#2C3E50")
        self._ax.xaxis.label.set_color("#2C3E50")
        self._ax.yaxis.label.set_color("#2C3E50")
        self._ax.title.set_color("#2C3E50")

        for spine in self._ax.spines.values():
            spine.set_color("#E4E7EB")

        if len(unique_groups) > 1:
            self._ax.legend(
                loc="upper right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E4E7EB", labelcolor="#2C3E50"
            )

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_dendrogram(self, result: Any, labels: list[str] | None = None) -> None:
        """Plot hierarchical clustering dendrogram."""
        self._record_plot_call("plot_dendrogram", result, labels=labels)
        self._current_plot_type = "dendrogram"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        scipy_dendrogram(result.linkage_matrix, labels=labels, ax=self._ax,
                        leaf_rotation=90, leaf_font_size=8,
                        color_threshold=result.linkage_matrix[-(result.n_clusters - 1), 2]
                        if result.n_clusters > 1 else 0)

        self._ax.set_title(f"{_('Hierarchical Clustering')} (cophenetic r={result.cophenetic_corr:.3f})")
        self._ax.set_ylabel(_("Distance"))
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_rose_diagram(self, bin_centers: np.ndarray, counts: np.ndarray,
                          mean_direction_deg: float = 0.0) -> None:
        """Plot rose diagram for directional data."""
        self._record_plot_call("plot_rose_diagram", bin_centers, counts, mean_direction_deg=mean_direction_deg)
        self._current_plot_type = "rose"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111, projection="polar")

        n_bins = len(counts)
        bin_width = 2 * np.pi / n_bins
        # Convert degree centers to radians if values > 2*pi
        centers = np.deg2rad(bin_centers) if np.max(bin_centers) > 2 * np.pi else bin_centers

        bars = self._ax.bar(centers, counts, width=bin_width * 0.8,
                           color=self.COLORS[0], alpha=0.7, edgecolor="white")

        # Mean direction arrow
        max_count = max(counts) if len(counts) > 0 else 1
        mean_rad = np.deg2rad(mean_direction_deg)
        self._ax.annotate("", xy=(mean_rad, max_count * 0.9), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=self.COLORS[3], lw=2))

        self._ax.set_title(_("Rose Diagram"), pad=20)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_efa_contours(self, original: np.ndarray, reconstructed: np.ndarray,
                          title: str = "") -> None:
        """Plot original vs reconstructed EFA contours."""
        self._record_plot_call("plot_efa_contours", original, reconstructed, title=title)
        self._current_plot_type = "efa"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        self._ax.plot(original[:, 0], original[:, 1], "o-",
                     color=self.COLORS[0], markersize=2, linewidth=1, label=_("Original"))
        self._ax.plot(reconstructed[:, 0], reconstructed[:, 1], "-",
                     color=self.COLORS[3], linewidth=1.5, label=_("Reconstructed"))

        self._ax.set_aspect("equal")
        self._ax.legend(fontsize=8)
        self._ax.set_title(title or _("Elliptic Fourier Analysis"))
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_she_curve(self, result: Any) -> None:
        """Plot SHE analysis curves (S, H, E vs sample size)."""
        self._record_plot_call("plot_she_curve", result)
        self._current_plot_type = "she"
        self._figure.clear()

        ax1 = self._figure.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(result.sample_sizes, result.s_values, "o-",
                color=self.COLORS[0], markersize=3, label="S (Richness)")
        ax2.plot(result.sample_sizes, result.e_values, "s-",
                color=self.COLORS[2], markersize=3, label="E (Evenness)")

        ax1.set_xlabel(_("Sample Size"))
        ax1.set_ylabel("S", color=self.COLORS[0])
        ax2.set_ylabel("E", color=self.COLORS[2])
        ax1.set_title(_("SHE Analysis"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        self._figure.tight_layout()
        self._canvas.draw()

    # =========================================================================
    # Ripley's K Plotting
    # =========================================================================

    def plot_ripley_k(
        self,
        result: Any,
        show_points: bool = True
    ) -> None:
        """
        Plot Ripley's K-function results.

        Shows L(r) - r curve with confidence envelope.
        L(r) > 0 indicates clustering, L(r) < 0 indicates regularity.

        Parameters:
            result: SpatialResult from Ripley's K analysis
            show_points: Whether to show point locations inset
        """
        self._record_plot_call("plot_ripley_k", result, show_points=show_points)
        self._ax.clear()
        self._current_plot_type = "ripley_k"

        # Extract data
        r_values = result.r_values
        l_values = result.l_values
        envelope_upper = result.envelope_upper
        envelope_lower = result.envelope_lower

        # Plot envelope as shaded region
        self._ax.fill_between(
            r_values, envelope_lower, envelope_upper,
            color="#3498DB", alpha=0.2, label=_("95% Envelope")
        )

        # Plot L(r) curve
        self._ax.plot(
            r_values, l_values,
            color="#E74C3C", linewidth=2.5,
            label=_("L(r) - r")
        )

        # Plot zero reference line
        self._ax.axhline(y=0, color="#2C3E50", linestyle="--", linewidth=1, alpha=0.7)

        # Labels
        self._ax.set_xlabel(_("Distance (r)"))
        self._ax.set_ylabel(_("L(r) - r"))
        self._ax.set_title(_("Ripley's K Spatial Point Pattern Analysis"))

        # Legend
        self._ax.legend(loc="upper left", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E4E7EB")

        # Style
        self._apply_axis_style()
        self._ax.grid(True, alpha=0.3, color="#E4E7EB")

        # Interpretation text
        interp = result.interpretation
        self._ax.text(
            0.98, 0.02, interp,
            transform=self._ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#F8F9F9", edgecolor="#E4E7EB", alpha=0.9)
        )

        self._canvas.draw()

    def plot_abundance_models(self, results: dict) -> None:
        """Plot rank-abundance curves with fitted models."""
        self._record_plot_call("plot_abundance_models", results)
        self._current_plot_type = "abundance_models"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        for i, (name, fit) in enumerate(results.items()):
            obs = np.sort(fit.observed)[::-1]
            pred = np.sort(fit.predicted)[::-1]
            n = min(len(obs), len(pred))
            ranks = np.arange(1, n + 1)

            if i == 0:
                self._ax.scatter(ranks, obs, s=20, color="#2C3E50", alpha=0.6, label=_("Observed"), zorder=5)

            self._ax.plot(ranks, pred, "-", color=self.COLORS[i % len(self.COLORS)],
                         linewidth=1.5, label=f"{fit.model_name} (R²={fit.r_squared:.3f})")

        self._ax.set_yscale("log")
        self._ax.set_xlabel(_("Rank"))
        self._ax.set_ylabel(_("Abundance (log)"))
        self._ax.set_title(_("Species-Abundance Models"))
        self._ax.legend(fontsize=7, loc="upper right")
        self._figure.tight_layout()
        self._canvas.draw()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _add_confidence_ellipses(self, data: np.ndarray, groups: np.ndarray, x_col: int, y_col: int) -> None:
        """
        Add 95% confidence ellipses for each group.

        Mathematical Context:
            Confidence ellipses based on:
                - Mean vector μ = (μ_x, μ_y)
                - Covariance matrix Σ
                - Mahalanobis distance for 95%: χ²(2, 0.95) ≈ 5.991

            Ellipse equation: (x - μ)ᵀ Σ⁻¹ (x - μ) = χ²
        """
        unique_groups = np.unique(groups)

        for group in unique_groups:
            idx = np.where(groups == group)[0]
            group_data = data[np.ix_(idx, [x_col, y_col])]

            # Calculate mean and covariance
            mean = np.mean(group_data, axis=0)
            cov = np.cov(group_data[:, 0], group_data[:, 1])

            # Eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Chi-squared value for 95% confidence
            chi2 = 5.991

            # Calculate ellipse parameters
            if eigenvalues[0] > 0:
                width = 2 * np.sqrt(eigenvalues[0] * chi2)
                height = 2 * np.sqrt(eigenvalues[1] * chi2)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                # Draw ellipse
                ellipse = Ellipse(
                    mean,
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=self.COLORS[group % len(self.COLORS)],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                )
                self._ax.add_patch(ellipse)

    def _apply_axis_style(self) -> None:
        """Apply consistent axis styling."""
        self._ax.set_facecolor("#FAFBFC")
        self._ax.tick_params(colors="#2C3E50")
        self._ax.xaxis.label.set_color("#2C3E50")
        self._ax.yaxis.label.set_color("#2C3E50")
        self._ax.title.set_color("#2C3E50")

        for spine in self._ax.spines.values():
            spine.set_color("#E4E7EB")

        self._ax.grid(True, alpha=0.3, color="#E4E7EB")
        self._ax.axhline(y=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)
        self._ax.axvline(x=0, color="#BDC3C7", linestyle="--", linewidth=0.5, alpha=0.5)

    # =========================================================================
    # Interaction Methods
    # =========================================================================

    def _on_motion(self, event) -> None:
        """
        Handle mouse motion for hover tooltips.

        Mathematical Context:
            Point proximity calculation using Euclidean distance:
                d = √((x - x_i)² + (y - y_i)²)

            Hover threshold: d < ε (epsilon)
        """
        if event.inaxes != self._ax:
            self._canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            if self._hover_annotation is not None:
                self._hover_annotation.remove()
                self._hover_annotation = None
                self._canvas.draw_idle()
            return

        if self._scores is None:
            return

        # Find nearest point
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        d1, d2 = self._current_dim1, self._current_dim2
        distances = np.sqrt((self._scores[:, d1] - x) ** 2 + (self._scores[:, d2] - y) ** 2)
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]

        # Check threshold (relative to axis range)
        x_range = self._ax.get_xlim()[1] - self._ax.get_xlim()[0]
        y_range = self._ax.get_ylim()[1] - self._ax.get_ylim()[0]
        threshold = max(x_range, y_range) * 0.02
        if min_dist < threshold:
            self._canvas.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

            # Remove previous annotation
            if self._hover_annotation is not None:
                self._hover_annotation.remove()
                self._hover_annotation = None

            # Show tooltip
            label = self._labels[nearest_idx] if nearest_idx < len(self._labels) else f"Point {nearest_idx}"
            group = self._group_labels[nearest_idx] if self._group_labels is not None else 0

            tooltip = f"{label}\n"
            tooltip += f"X: {self._scores[nearest_idx, d1]:.4f}\n"
            tooltip += f"Y: {self._scores[nearest_idx, d2]:.4f}\n"
            if self._group_labels is not None:
                tooltip += f"Group: {group + 1}"

            # Draw annotation
            self._hover_annotation = self._ax.annotate(
                tooltip,
                xy=(self._scores[nearest_idx, d1], self._scores[nearest_idx, d2]),
                xytext=(20, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="white", lw=1),
                bbox=dict(boxstyle="round", facecolor="#2C3E50", alpha=0.9),
                color="#ECF0F1",
                fontsize=9,
            )

            self._canvas.draw_idle()
        else:
            # Not near any point — remove annotation if present
            if self._hover_annotation is not None:
                self._hover_annotation.remove()
                self._hover_annotation = None
                self._canvas.draw_idle()
            self._canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _on_press(self, event) -> None:
        """Handle mouse press for selection."""
        if event.dblclick:
            # Double-click: reset view
            self._reset_view()

    def _on_scroll(self, event) -> None:
        """Handle scroll for zoom."""
        if event.inaxes != self._ax:
            return

        # Zoom factor
        scale_factor = 1.1 if event.step > 0 else 1 / 1.1

        # Get current limits
        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()

        # Calculate new limits centered on mouse position
        x_center = event.xdata if event.xdata is not None else (x_lim[0] + x_lim[1]) / 2
        y_center = event.ydata if event.ydata is not None else (y_lim[0] + y_lim[1]) / 2

        x_range = (x_lim[1] - x_lim[0]) / scale_factor
        y_range = (y_lim[1] - y_lim[0]) / scale_factor

        self._ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self._ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])

        self._canvas.draw()

    def _on_selection_mode_changed(self, index: int) -> None:
        """Handle selection mode change."""
        if index == 1:  # Rectangle
            self._selector = RectangleSelector(
                self._ax,
                self._on_rectangle_select,
                useblit=True,
                button=[1],
                minspanx=0.01,
                minspany=0.01,
                spancoords="data",
            )
        elif index == 2:  # Lasso
            self._selector = LassoSelector(self._ax, self._on_lasso_select, useblit=True)
        else:
            if hasattr(self, "_selector"):
                self._selector.disconnect()
                del self._selector

    def _on_rectangle_select(self, eclick, erelease) -> None:
        """Handle rectangle selection."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Find points in rectangle
        d1, d2 = self._current_dim1, self._current_dim2
        selected = []
        for i, point in enumerate(self._scores):
            if min(x1, x2) <= point[d1] <= max(x1, x2) and min(y1, y2) <= point[d2] <= max(y1, y2):
                selected.append(i)

        self._highlight_selection(selected)

    def _on_lasso_select(self, verts) -> None:
        """Handle lasso selection."""
        from matplotlib.path import Path

        path = Path(verts)

        # Find points inside path
        d1, d2 = self._current_dim1, self._current_dim2
        selected = []
        for i, point in enumerate(self._scores):
            if path.contains_point((point[d1], point[d2])):
                selected.append(i)

        self._highlight_selection(selected)

    def _highlight_selection(self, indices: list[int]) -> None:
        """Highlight selected points and emit signal."""
        self._selected_indices = indices

        # Visual feedback: update edge color and line width on scatter collections
        import matplotlib.collections as mcoll

        selected_set = set(indices)
        offset = 0
        for collection in self._ax.collections:
            if isinstance(collection, mcoll.PathCollection) and collection.get_picker() is not None:
                n_pts = len(collection.get_offsets())
                if n_pts == 0:
                    continue
                edge_colors = np.tile([1.0, 1.0, 1.0, 1.0], (n_pts, 1))
                line_widths = np.full(n_pts, 0.5)
                for i in range(n_pts):
                    if (offset + i) in selected_set:
                        edge_colors[i] = [0.0, 0.0, 0.0, 1.0]
                        line_widths[i] = 2.0
                collection.set_edgecolors(edge_colors)
                collection.set_linewidths(line_widths)
                offset += n_pts

        # Emit signal
        self.pointsSelected.emit(indices)

        # Redraw with highlighting
        self._canvas.draw()

    # =========================================================================
    # New Analysis Plot Methods
    # =========================================================================

    def plot_allometry(self, result: Any) -> None:
        """
        Plot allometry scatter plot with regression line and confidence bands.

        Parameters:
            result: AllometryResult with centroid_sizes, log_centroid_sizes,
                   regression_coefficients, regression_intercept, r_squared, residuals
        """
        self._record_plot_call("plot_allometry", result)
        self._current_plot_type = "allometry"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        log_cs = result.log_centroid_sizes
        coef = result.regression_coefficients
        intercept = result.regression_intercept
        r_squared = result.r_squared
        residuals = result.residuals

        self._scores = np.column_stack([log_cs, residuals]) if residuals.ndim > 1 else np.column_stack([log_cs, np.zeros_like(log_cs)])
        self._labels = [f"S{i}" for i in range(len(log_cs))]
        self._group_labels = np.zeros(len(log_cs), dtype=int)

        # Scatter plot
        self._ax.scatter(log_cs, residuals if residuals.ndim == 1 else residuals[:, 0], c="#2C3E50", s=80, alpha=0.7, edgecolors="white")

        # Regression line
        x_range = np.linspace(log_cs.min(), log_cs.max(), 100)
        y_pred = coef * x_range + intercept if coef.ndim == 0 else coef[0] * x_range + intercept[0]
        self._ax.plot(x_range, y_pred, "r-", linewidth=2, label="Regression")

        # Confidence band
        n = len(log_cs)
        if n > 2:
            from scipy import stats
            x_mean = np.mean(log_cs)
            ss_x = np.sum((log_cs - x_mean) ** 2)
            mse = np.sum(residuals ** 2) / (n - 2) if residuals.ndim == 1 else np.sum(residuals[:, 0] ** 2) / (n - 2)
            se = np.sqrt(mse)
            t_val = stats.t.ppf(0.975, n - 2)
            se_line = se * np.sqrt(1 / n + (x_range - x_mean) ** 2 / ss_x)
            ci_lower = y_pred - t_val * se_line
            ci_upper = y_pred + t_val * se_line
            self._ax.fill_between(x_range, ci_lower, ci_upper, alpha=0.2, color="red", label="95% CI")

        self._ax.set_xlabel("Log Centroid Size", fontsize=10)
        self._ax.set_ylabel("Shape Score", fontsize=10)
        self._ax.set_title(f"Allometry: Size-Shape Relationship (R² = {r_squared:.4f})", fontsize=12, fontweight="bold")
        self._ax.legend(loc="best")
        self._ax.grid(True, linestyle="--", alpha=0.3)

    def plot_evolution_rate(self, result: Any) -> None:
        """
        Plot evolution rate phenogram showing trait evolution over time.

        Parameters:
            result: EvolutionRateResult with best_model, rate_estimate,
                   aic_weights, trait_mean, trait_variance, trait_series
        """
        self._record_plot_call("plot_evolution_rate", result)
        self._current_plot_type = "evolution_rate"
        self._figure.clear()

        # Main phenogram subplot
        self._ax = self._figure.add_subplot(211)

        # Use actual trait_series from result if available
        if result.trait_series is not None and len(result.trait_series) > 0:
            trait_series = np.asarray(result.trait_series)
            n_points = len(trait_series)
        else:
            # Fallback: show message if no data available
            self._ax.text(0.5, 0.5, "No trait series data available",
                         ha="center", va="center", fontsize=12)
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_title("Phenogram: No Data Available", fontsize=12, fontweight="bold")
            self._figure.tight_layout()
            return

        time_points = np.arange(n_points)

        self._ax.plot(time_points, trait_series, "b-o", linewidth=2, markersize=8, label="Trait evolution")

        # Confidence band based on rate estimate
        if result.rate_estimate > 0:
            variance = result.rate_estimate * time_points
            std = np.sqrt(variance)
            self._ax.fill_between(
                time_points,
                trait_series - 1.96 * std,
                trait_series + 1.96 * std,
                alpha=0.2,
                color="blue",
                label="95% CI",
            )

        # Model info
        model_label = f"Best model: {result.best_model.upper()}\nRate: {result.rate_estimate:.6f}"
        self._ax.text(0.02, 0.98, model_label, transform=self._ax.transAxes, fontsize=9,
                      verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        self._ax.set_xlabel("Stratigraphic Position", fontsize=10)
        self._ax.set_ylabel("Trait Value", fontsize=10)
        self._ax.set_title("Phenogram: Morphological Evolution Over Time", fontsize=12, fontweight="bold")
        self._ax.legend(loc="best")
        self._ax.grid(True, linestyle="--", alpha=0.3)

        # Model comparison subplot
        if result.aic_weights:
            ax2 = self._figure.add_subplot(212)
            models = list(result.aic_weights.keys())
            weights = list(result.aic_weights.values())
            colors = ["#E74C3C" if m == result.best_model else "#3498DB" for m in models]
            bars = ax2.barh(models, weights, color=colors, alpha=0.7)
            ax2.set_xlabel("AIC Weight", fontsize=10)
            ax2.set_title("Model Comparison", fontsize=10)
            ax2.set_xlim(0, 1)
            for bar, w in zip(bars, weights):
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center")

        self._figure.tight_layout()

    def plot_extinction_ranges(self, result: Any) -> None:
        """
        Plot stratigraphic range chart with extinction confidence intervals.

        Parameters:
            result: ExtinctionIntervalResult with lad_positions, ci_lower, ci_upper,
                   confidence_interval_lower, confidence_interval_upper, method
        """
        self._record_plot_call("plot_extinction_ranges", result)
        self._current_plot_type = "extinction"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        lad_positions = result.lad_positions
        ci_lower = result.confidence_interval_lower
        ci_upper = result.confidence_interval_upper

        n_taxa = len(lad_positions)
        taxon_names = [f"Taxon {i + 1}" for i in range(n_taxa)]

        # Sort by LAD (oldest at top)
        sorted_indices = np.argsort(lad_positions)[::-1]
        lad_sorted = lad_positions[sorted_indices]
        ci_lower_sorted = ci_lower[sorted_indices]
        ci_upper_sorted = ci_upper[sorted_indices]
        names_sorted = [taxon_names[i] for i in sorted_indices]

        # Plot ranges
        for i, (lad, _, ci_u, name) in enumerate(zip(lad_sorted, ci_lower_sorted, ci_upper_sorted, names_sorted)):
            # Observed range (solid line from top to LAD)
            self._ax.plot([0.3, 0.7], [0, lad], "b-", linewidth=3, solid_capstyle="butt")
            self._ax.plot([0.2, 0.8], [lad, lad], "b-", linewidth=2)

            # CI whiskers (extending upward)
            self._ax.plot([0.5, 0.5], [ci_u, lad], "r--", linewidth=1.5)

            # CI box
            from matplotlib.patches import Rectangle
            rect = Rectangle((0.25, ci_u), 0.5, lad - ci_u, linewidth=1, edgecolor="red",
                           facecolor="red", alpha=0.2, linestyle="--")
            self._ax.add_patch(rect)

            # Taxon label
            self._ax.text(0.85, lad, name, fontsize=9, va="center", ha="left")

        self._ax.set_xlim(0, 1)
        max_lad = max(lad_sorted) if len(lad_sorted) > 0 else 10
        self._ax.set_ylim(-1, max_lad + 2)
        self._ax.invert_yaxis()
        self._ax.set_xlabel("Taxonomic Range", fontsize=10)
        self._ax.set_ylabel("Stratigraphic Height (layers from top)", fontsize=10)
        self._ax.set_title(f"Extinction Confidence Intervals ({result.method.upper()}, {int(result.confidence_level * 100)}% CI)",
                          fontsize=12, fontweight="bold")
        self._ax.set_xticks([])

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=3, label="Observed LAD"),
            Line2D([0], [0], color="red", linewidth=1.5, linestyle="--", label=f"{int(result.confidence_level * 100)}% CI"),
        ]
        self._ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        self._ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    def plot_beta_diversity(self, result: Any) -> None:
        """
        Plot beta diversity decomposition heatmap.

        Parameters:
            result: BetaDiversityResult with total_beta, turnover_component,
                   nestedness_component, sample_names
        """
        self._record_plot_call("plot_beta_diversity", result)
        self._current_plot_type = "beta_diversity"
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)

        # Plot heatmap of total beta diversity
        matrix = result.total_beta
        n = result.n_samples

        im = self._ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        self._figure.colorbar(im, ax=self._ax, label="Beta Diversity")

        # Labels
        sample_names = result.sample_names if hasattr(result, "sample_names") else [f"S{i}" for i in range(n)]
        self._ax.set_xticks(np.arange(n))
        self._ax.set_yticks(np.arange(n))
        self._ax.set_xticklabels(sample_names, rotation=45, ha="right")
        self._ax.set_yticklabels(sample_names)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                self._ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        self._ax.set_title(f"Beta Diversity Decomposition ({result.decomposition_type.upper()})", fontsize=12, fontweight="bold")

    def plot_null_model(self, result: Any) -> None:
        """
        Plot null model analysis results.

        Parameters:
            result: NullModelResult with observed_score, simulated_scores,
                   mean_simulated, standardized_effect_size, p_value
        """
        self._record_plot_call("plot_null_model", result)
        self._current_plot_type = "null_model"
        self._figure.clear()

        # Histogram of simulated scores
        self._ax = self._figure.add_subplot(111)
        simulated = result.simulated_scores
        self._ax.hist(simulated, bins=50, color="#3498DB", alpha=0.7, edgecolor="black", label="Simulated")

        # Observed score line
        self._ax.axvline(result.observed_score, color="red", linewidth=2, linestyle="--",
                        label=f"Observed = {result.observed_score:.4f}")
        self._ax.axvline(result.mean_simulated, color="green", linewidth=2, linestyle="-",
                        label=f"Mean = {result.mean_simulated:.4f}")

        # SES annotation
        sig = "***" if result.p_value < 0.001 else ("**" if result.p_value < 0.01 else ("*" if result.p_value < 0.05 else ""))
        self._ax.text(0.02, 0.98, f"SES = {result.standardized_effect_size:.2f}\np = {result.p_value:.4f} {sig}",
                     transform=self._ax.transAxes, fontsize=10, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        self._ax.set_xlabel(f"{result.metric.upper()} Score", fontsize=10)
        self._ax.set_ylabel("Frequency", fontsize=10)
        self._ax.set_title(f"Null Model Analysis ({result.algorithm.upper()}, {result.n_permutations} permutations)",
                          fontsize=12, fontweight="bold")
        self._ax.legend(loc="best")

    def _record_plot_call(self, method_name: str, *args, **kwargs) -> None:
        """Record the last plot call for replotting."""
        self._last_plot_call = (method_name, args, kwargs)

    def _replot_current(self) -> None:
        """Replot using the last recorded plot call."""
        if self._last_plot_call is None:
            return
        method_name, args, kwargs = self._last_plot_call
        method = getattr(self, method_name, None)
        if method is not None:
            method(*args, **kwargs)

    def _toggle_labels(self, checked: bool) -> None:
        """Toggle label visibility."""
        self._ax.cla()
        self._replot_current()

    def _toggle_ellipses(self, checked: bool) -> None:
        """Toggle confidence ellipse visibility."""
        self._ax.cla()
        self._replot_current()

    def _zoom_in(self) -> None:
        """Zoom in."""
        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()

        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2

        x_range = (x_lim[1] - x_lim[0]) / 1.2
        y_range = (y_lim[1] - y_lim[0]) / 1.2

        self._ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self._ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])

        self._canvas.draw()

    def _zoom_out(self) -> None:
        """Zoom out."""
        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()

        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2

        x_range = (x_lim[1] - x_lim[0]) * 1.2
        y_range = (y_lim[1] - y_lim[0]) * 1.2

        self._ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self._ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])

        self._canvas.draw()

    def _reset_view(self) -> None:
        """Reset view to auto-scale."""
        self._ax.autoscale()
        self._canvas.draw()

    def _export_plot(self) -> None:
        """
        Export plot with high resolution.

        Mathematical Context:
            Export formats and DPI settings:
                - Screen: 72-100 DPI
                - Print/Publication: 300 DPI
                - High-quality: 600 DPI
                - Vector: SVG, PDF, EPS (infinite resolution)
        """
        # Show save dialog
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, _("Export Plot"), "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;TIFF (*.tiff)"
        )

        if not filepath:
            return

        # Get DPI from format
        dpi_map = {"png": 300, "pdf": 300, "svg": 72, "eps": 300, "tiff": 600}

        ext = filepath.split(".")[-1].lower()
        dpi = dpi_map.get(ext, 300)

        # Get size
        size_dialog = QMessageBox(self)
        size_dialog.setWindowTitle(_("Export Size"))
        size_dialog.setText(_("Select export size:"))
        size_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Calculate size based on DPI
        width_inches = self._figure.get_figwidth()
        height_inches = self._figure.get_figheight()
        width_cm = width_inches * 2.54
        height_cm = height_inches * 2.54

        # Save figure
        try:
            self._figure.savefig(
                filepath, dpi=dpi, bbox_inches="tight", facecolor=self._figure.get_facecolor(), edgecolor="none"
            )

            QMessageBox.information(
                self,
                _("Export Successful"),
                _("Plot saved to:\n{0}\n\nResolution: {1} DPI\nSize: {2:.1f} x {3:.1f} cm").format(
                    filepath, dpi, width_cm, height_cm
                ),
            )
        except Exception as e:
            QMessageBox.critical(self, _("Export Error"), _("Failed to export plot:\n{0}").format(str(e)))

    # =========================================================================
    # Public API
    # =========================================================================

    def get_figure(self) -> Figure:
        """Get the matplotlib figure."""
        return self._figure

    def get_selected_indices(self) -> list[int]:
        """Get currently selected point indices."""
        return self._selected_indices.copy()

    def clear_selection(self) -> None:
        """Clear current selection."""
        self._selected_indices = []
        self._canvas.draw()

    def set_data(
        self,
        scores: np.ndarray,
        labels: list[str],
        groups: np.ndarray | None = None,
        eigenvalues: np.ndarray | None = None,
    ) -> None:
        """
        Set plot data.

        Args:
            scores: Ordination scores matrix (n_samples, n_dims)
            labels: Point labels
            groups: Group assignments (n_samples,)
            eigenvalues: Eigenvalues for variance explanation
        """
        self._scores = scores
        self._labels = labels
        self._group_labels = groups
        self._eigenvalues = eigenvalues

        if groups is not None:
            unique_groups = np.unique(groups)
            self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}
        else:
            self._groups = {}
            self._group_labels = np.zeros(len(labels), dtype=int)
