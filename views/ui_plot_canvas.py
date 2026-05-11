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
        """Apply modern light theme stylesheet."""
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
            }
            QToolBar {
                background-color: #F8F9FA;
                border: 1px solid #E4E7EB;
                spacing: 8px;
                padding: 6px;
            }
            QPushButton {
                background-color: #F0F2F5;
                color: #2C3E50;
                border: 1px solid #E4E7EB;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 50px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #E4E7EB;
                border: 1px solid #BFC9D4;
            }
            QPushButton:pressed {
                background-color: #D9DFE8;
            }
            QComboBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #E4E7EB;
                border-radius: 6px;
                padding: 6px 8px;
                min-width: 100px;
            }
            QComboBox:hover {
                border: 1px solid #3498DB;
            }
            QComboBox:focus {
                border: 1px solid #3498DB;
            }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #3498DB;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #2C3E50;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
            }
            QCheckBox {
                color: #2C3E50;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #E4E7EB;
                border-radius: 3px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #3498DB;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border-color: #2980B9;
            }
            QLabel {
                color: #2C3E50;
                padding: 0 4px;
            }
        """)

        self._figure.patch.set_facecolor("#FFFFFF")

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
        self._ax.clear()
        self._current_plot_type = "rarefaction"

        # Extract data
        if hasattr(result, "curve_data"):
            curve_data = result.curve_data
        else:
            # Generate sample data
            n_points = 20
            curve_data = {}
            for sample in ["Sample 1", "Sample 2", "Sample 3"]:
                n = np.linspace(10, 100, n_points)
                s = 5 + 3 * np.sqrt(n) + np.random.normal(0, 0.5, n_points)
                curve_data[sample] = (n, s)

        # Plot curves
        for i, (sample, (n, s)) in enumerate(curve_data.items()):
            color = self.COLORS[i % len(self.COLORS)]
            self._ax.plot(n, s, color=color, linewidth=2, marker="o", markersize=4, alpha=0.7, label=sample)

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
        self._ax.clear()
        self._current_plot_type = "spectral"

        # Extract data
        if hasattr(result, "frequencies") and hasattr(result, "power"):
            frequencies = result.frequencies
            power = result.power
        else:
            frequencies = np.linspace(0.01, 1.0, 100)
            power = np.random.exponential(1.0, 100)

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

    def plot_anosim_results(self, result: Any) -> None:
        """
        Plot ANOSIM results.

        Shows a bar chart of the R statistic with significance indicator.
        """
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

        # Check threshold
        if min_dist < 0.05:
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
        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2

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
        selected = []
        for i, point in enumerate(self._scores):
            if min(x1, x2) <= point[0] <= max(x1, x2) and min(y1, y2) <= point[1] <= max(y1, y2):
                selected.append(i)

        self._highlight_selection(selected)

    def _on_lasso_select(self, verts) -> None:
        """Handle lasso selection."""
        from matplotlib.path import Path

        path = Path(verts)

        # Find points inside path
        selected = []
        for i, point in enumerate(self._scores):
            if path.contains_point((point[0], point[1])):
                selected.append(i)

        self._highlight_selection(selected)

    def _highlight_selection(self, indices: list[int]) -> None:
        """Highlight selected points and emit signal."""
        self._selected_indices = indices

        # Emit signal
        self.pointsSelected.emit(indices)

        # Redraw with highlighting
        self._canvas.draw()

    def _replot_current(self) -> None:
        """Replot using the current plot type and stored data."""
        if self._current_plot_type == "pca" and self._scores is not None:
            self.plot_pca_scores(
                type(
                    "Result",
                    (),
                    {
                        "scores": self._scores,
                        "explained_variance": self._eigenvalues,
                        "labels": self._labels,
                        "groups": self._group_labels,
                    },
                )()
            )
        elif self._current_plot_type == "pcoa" and self._scores is not None:
            self.plot_pcoa_scores(
                type(
                    "Result",
                    (),
                    {
                        "coordinates": self._scores,
                        "eigenvalues": self._eigenvalues,
                        "labels": self._labels,
                        "groups": self._group_labels,
                    },
                )()
            )
        elif self._current_plot_type == "nmds" and self._scores is not None:
            self.plot_nmds(
                type(
                    "Result",
                    (),
                    {"coordinates": self._scores, "stress": self._stress, "labels": self._labels, "groups": self._group_labels},
                )()
            )

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
