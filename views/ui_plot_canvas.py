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
version: 1.0.1
"""

import contextlib
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
    QDialog,
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
    with contextlib.suppress(OSError):
        plt.style.use("seaborn-whitegrid")
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
        # ``_scores`` is a property (see below): assigning to it also
        # re-clamps the dimension indices used by the hover tooltip and the
        # rectangle/lasso selectors.
        self._scores_storage: np.ndarray | None = None
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

    @property
    def _scores(self) -> np.ndarray | None:
        """Score matrix currently shown (see setter for the invariant)."""
        return getattr(self, "_scores_storage", None)

    @_scores.setter
    def _scores(self, value: np.ndarray | None) -> None:
        # Every plot method assigns the new score matrix; re-clamping here
        # means the hover/selection code can never index a dimension the
        # new result does not have (e.g. PCA with 5 PCs -> 1-D NMDS/LDA,
        # which used to raise IndexError on the first mouse move).
        self._scores_storage = value
        self._clamp_dims()

    @property
    def _current_dim1(self) -> int:
        return getattr(self, "_dim1_storage", 0)

    @_current_dim1.setter
    def _current_dim1(self, value: int) -> None:
        self._dim1_storage = int(value)
        self._clamp_dims()

    @property
    def _current_dim2(self) -> int:
        return getattr(self, "_dim2_storage", 1)

    @_current_dim2.setter
    def _current_dim2(self, value: int) -> None:
        self._dim2_storage = int(value)
        self._clamp_dims()

    def _clamp_dims(self) -> None:
        """
        Keep the hovered dimension pair inside the current score matrix.

        ``_on_motion``, the rectangle/lasso selectors and the tooltip all
        index ``self._scores[:, self._current_dim1]``.  The pair is written
        straight to the backing stores so this stays recursion-free.
        """
        scores = getattr(self, "_scores_storage", None)
        if scores is None or getattr(scores, "ndim", 0) != 2 or scores.shape[1] == 0:
            return
        n_dims = scores.shape[1]
        d1 = min(max(int(getattr(self, "_dim1_storage", 0)), 0), n_dims - 1)
        d2 = min(max(int(getattr(self, "_dim2_storage", 1)), 0), n_dims - 1)
        if d1 == d2 and n_dims > 1:
            d2 = 1 if d1 == 0 else 0
        self._dim1_storage = d1
        self._dim2_storage = d2

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
        self._figure = Figure(figsize=(8, 6), facecolor=self.theme_colors()["figure_bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Setup axes
        self._ax = self._figure.add_subplot(111)
        self._apply_axes_theme(self._ax)

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

        # Sync matplotlib theme.  The palette is the single source of truth
        # (see ``theme_colors``), so a canvas created *after* the switch
        # inherits the same colours through rcParams.
        t = self.theme_colors()
        plt.rcParams.update(
            {
                "axes.facecolor": t["axes_bg"],
                "figure.facecolor": t["figure_bg"],
                "axes.edgecolor": t["border"],
                "axes.labelcolor": t["text"],
                "xtick.color": t["text"],
                "ytick.color": t["text"],
                "text.color": t["text"],
                "grid.color": t["grid"],
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        # Update figure background
        c = get_palette(is_dark)
        self._figure.patch.set_facecolor(c.bg_primary)

        # Redraw if there's an active plot
        if self._current_plot_type and self._ax is not None:
            self._apply_axes_theme()
            self._canvas.draw_idle()

    # =========================================================================
    # Theme helpers
    # =========================================================================

    def theme_colors(self) -> dict[str, str]:
        """
        Matplotlib-facing colours for the currently selected theme.

        Derived from the application design-system palette so the Qt chrome
        and the matplotlib surfaces can never drift apart.  ``plot_*``
        methods must read their surfaces/text colours from here instead of
        hard-coding the light-theme literals, otherwise a canvas switched to
        the dark theme keeps painting white backgrounds.
        """
        c = get_palette(self._is_dark_theme)
        return {
            "text": c.text_primary,
            "text_muted": c.text_secondary,
            "axes_bg": c.bg_secondary,
            "figure_bg": c.bg_primary,
            "border": c.border_light,
            "grid": c.border_light,
            "legend_bg": c.bg_primary,
            "legend_edge": c.border_light,
            "reference": c.text_secondary,
        }

    def _legend_kwargs(self, loc: str = "upper right", **overrides) -> dict:
        """Legend keyword arguments tinted with the current theme."""
        t = self.theme_colors()
        kwargs = {
            "loc": loc,
            "framealpha": 0.95,
            "facecolor": t["legend_bg"],
            "edgecolor": t["legend_edge"],
            "labelcolor": t["text"],
        }
        kwargs.update(overrides)
        return kwargs

    def _apply_axes_theme(self, ax=None) -> None:
        """
        Re-tint an axes (and its figure) to the current theme.

        Only *colours* are touched - grids, spines visibility and reference
        lines are left exactly as the plotting method arranged them.

        ``ax.texts`` is part of the job, not an afterthought. The sample
        labels are drawn with ``annotate()``, which resolves ``text.color``
        from rcParams *when the artist is created*; updating rcParams later
        does not retroactively repaint an existing ``Text``. So a figure
        drawn before a theme switch kept its light-theme label colour and
        became invisible on the dark axes face - measured at roughly
        2:1 against a 4.5:1 requirement, which is why the labels had to be
        walked explicitly here.
        """
        target = self._ax if ax is None else ax
        t = self.theme_colors()
        if target is not None:
            with contextlib.suppress(Exception):
                target.set_facecolor(t["axes_bg"])
                target.tick_params(colors=t["text"])
                target.xaxis.label.set_color(t["text"])
                target.yaxis.label.set_color(t["text"])
                target.title.set_color(t["text"])
                for spine in target.spines.values():
                    spine.set_color(t["border"])
                # Annotations: sample labels, bar values, empty-state notes.
                for artist in target.texts:
                    artist.set_color(t["text"])
                # Legend text is a separate artist list from ax.texts.
                legend = target.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_color(t["text"])
                    if legend.get_frame() is not None:
                        legend.get_frame().set_facecolor(t["figure_bg"])
                        legend.get_frame().set_edgecolor(t["border"])
        with contextlib.suppress(Exception):
            self._figure.patch.set_facecolor(t["figure_bg"])

    # =========================================================================
    # Axes lifecycle helpers
    # =========================================================================

    def _adopt_axes(self, ax):
        """
        Make ``ax`` the axes every interaction handler operates on.

        Tooltips (:meth:`_on_motion`), scroll zoom, the toolbar zoom/reset
        buttons, the label & ellipse toggles and the rectangle/lasso
        selectors all read ``self._ax``.  A plotting method that creates a
        fresh axes without adopting it leaves all of them pointed at an
        axes that is no longer part of the figure.
        """
        self._ax = ax
        selector = getattr(self, "_selector", None)
        if selector is not None:
            # ``update_axes`` exists on both selectors since matplotlib 3.6;
            # suppress so a custom/older widget cannot break the redraw.
            with contextlib.suppress(Exception):
                selector.update_axes(ax)
        return ax

    def _reset_axes(self, *args, **kwargs):
        """
        Clear the figure and create (and adopt) a single fresh axes.

        Equivalent to the historic ``self._figure.clear()`` +
        ``self._figure.add_subplot(111)`` pair, but keeps ``self._ax`` in
        sync so interactivity survives a replot.
        """
        self._figure.clear()
        if not args and not kwargs:
            return self._adopt_axes(self._figure.add_subplot(111))
        return self._adopt_axes(self._figure.add_subplot(*args, **kwargs))

    def _adopt_first_axes(self):
        """
        Adopt the first axes of a multi-panel figure.

        Used by the grid plots, which create one axes per panel: the
        interaction handlers only understand a single axes, so panel one is
        adopted (better a live axes than a destroyed one).
        """
        if self._figure.axes:
            return self._adopt_axes(self._figure.axes[0])
        return None

    def _finalise_grid_plot(self) -> None:
        """
        Post-process a multi-panel figure.

        Adopts panel one for the interaction handlers and re-tints every
        panel with the active theme (the historical code only ever painted
        light-theme surfaces).
        """
        self._adopt_first_axes()
        for ax in self._figure.axes:
            self._apply_axes_theme(ax)

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self._selection_combo.currentIndexChanged.connect(self._on_selection_mode_changed)

    # =========================================================================
    # PCA Plotting Methods
    # =========================================================================

    def plot_pca_scores(
        self,
        result: Any,
        pc1: int = 0,
        pc2: int = 1,
        groups: list[int] | np.ndarray | None = None,
        labels: list[str] | None = None,
        group_names: list[str] | None = None,
        annotate_offset: bool = True,
        show_ellipses: bool | None = None,
    ) -> None:
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

        Parameters:
            result: PCAResult (or any object with ``scores`` /
                ``explained_variance`` attributes) or a raw ``(n, k)`` score
                matrix.
            pc1, pc2: Principal-component indices to plot (0-indexed).
            groups: Optional per-sample group labels. When supplied, the
                points are coloured and (with ``>=2`` distinct groups) a
                95% confidence ellipse is drawn around each group. When
                ``None``, the canvas looks for ``result.groups``; failing
                that, every point is treated as group 0.
            labels: Optional per-sample text labels. When ``None``, falls
                back to ``result.labels``; otherwise to ``S1..Sn``.
            group_names: Optional human-readable group names (one per
                distinct value in ``groups``). Falls back to ``Group <n>``.
            annotate_offset: When True, dodge overlapping point labels
                vertically (cheap iterative repel). Pass False to keep the
                historic straight-above placement.
            show_ellipses: ``True`` forces ellipses, ``False`` suppresses
                them. ``None`` (default) draws them automatically when
                there are >=2 groups OR when the toolbar ellipse toggle
                is on.
        """
        self._record_plot_call(
            "plot_pca_scores",
            result,
            pc1=pc1,
            pc2=pc2,
            groups=groups,
            labels=labels,
            group_names=group_names,
            annotate_offset=annotate_offset,
            show_ellipses=show_ellipses,
        )
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

        labels, groups = self._resolve_metadata(result, labels, groups, scores)
        group_names = self._resolve_group_names(groups, group_names)

        # Store data. NOTE: use the clamped dimensions for the actual
        # drawing below, not the raw arguments. ``_clamp_dims`` exists
        # precisely because a re-plot can be handed stale indices (5-PC PCA
        # tab -> 1-D NMDS result, say), but the property setters only clamp
        # the stored copy -- the old code went on to index ``scores[:, pc1]``
        # and ``eigenvalues[pc1]`` with the *unclamped* argument, so the guard
        # did nothing and the call raised IndexError.
        self._scores = scores
        self._eigenvalues = eigenvalues
        self._labels = labels
        self._group_labels = groups
        self._current_dim1 = pc1
        self._current_dim2 = pc2
        pc1 = self._current_dim1
        pc2 = self._current_dim2

        # Calculate variance explained. A zero-variance input (every column
        # constant) makes every eigenvalue 0, so the ratio is 0/0: guard it
        # instead of emitting a RuntimeWarning and "nan%" in the title.
        total_var = float(np.sum(eigenvalues))
        if total_var > 0:
            var_pc1 = eigenvalues[pc1] / total_var * 100
            var_pc2 = eigenvalues[pc2] / total_var * 100
        else:
            var_pc1 = var_pc2 = 0.0

        # Set up groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot points by group
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            display_name = group_names.get(group, f"Group {group + 1}")
            self._ax.scatter(
                scores[idx, pc1],
                scores[idx, pc2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=display_name if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Add labels if enabled
        if self._show_labels_check.isChecked():
            xs = scores[:, pc1]
            ys = scores[:, pc2]
            if annotate_offset:
                self._annotate_with_offset(xs, ys, labels)
            else:
                for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
                    self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8, ha="center", va="bottom")

        # Add ellipses if enabled (auto-on when >=2 groups).
        if self._should_draw_ellipses(show_ellipses, unique_groups):
            self._add_confidence_ellipses(scores, groups, pc1, pc2)

        # Labels and title
        self._ax.set_xlabel(_("PC{0} ({1:.1f}% variance)").format(pc1 + 1, var_pc1))
        self._ax.set_ylabel(_("PC{0} ({1:.1f}% variance)").format(pc2 + 1, var_pc2))
        self._ax.set_title(_("PCA Scores Plot"))

        # Style: shared theme helper (grid + zero reference lines included)
        if len(unique_groups) > 1:
            self._ax.legend(**self._legend_kwargs("upper right"))
        self._apply_axis_style()

        self._canvas.draw()

    # =========================================================================
    # PCoA Plotting Methods
    # =========================================================================

    def plot_pcoa_scores(
        self,
        result: Any,
        coord1: int = 0,
        coord2: int = 1,
        groups: list[int] | np.ndarray | None = None,
        labels: list[str] | None = None,
        group_names: list[str] | None = None,
        annotate_offset: bool = True,
        show_ellipses: bool | None = None,
    ) -> None:
        """
        Plot PCoA scores.

        Mathematical Context:
            PCoA coordinates: X = U Λ^(1/2)

            where:
                U = eigenvector matrix
                Λ = diagonal eigenvalue matrix

        Parameters: see :meth:`plot_pca_scores` for ``groups``, ``labels``,
            ``group_names``, ``annotate_offset`` and ``show_ellipses``.
        """
        self._record_plot_call(
            "plot_pcoa_scores",
            result,
            coord1=coord1,
            coord2=coord2,
            groups=groups,
            labels=labels,
            group_names=group_names,
            annotate_offset=annotate_offset,
            show_ellipses=show_ellipses,
        )
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

        labels, groups = self._resolve_metadata(result, labels, groups, coords)
        group_names = self._resolve_group_names(groups, group_names)

        # Store data
        self._scores = coords
        self._eigenvalues = eigenvalues
        self._labels = labels
        self._group_labels = groups
        self._current_dim1 = coord1
        self._current_dim2 = coord2

        # Calculate variance explained
        total_var = np.sum(np.abs(eigenvalues))
        var_1 = np.abs(eigenvalues[coord1]) / total_var * 100 if total_var > 0 else 0.0
        var_2 = np.abs(eigenvalues[coord2]) / total_var * 100 if total_var > 0 else 0.0

        # Setup groups
        unique_groups = np.unique(groups)
        self._groups = {g: np.where(groups == g)[0].tolist() for g in unique_groups}

        # Plot
        for i, group in enumerate(unique_groups):
            idx = self._groups[group]
            color = self.COLORS[i % len(self.COLORS)]

            display_name = group_names.get(group, f"Group {group + 1}")
            self._ax.scatter(
                coords[idx, coord1],
                coords[idx, coord2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=display_name if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Labels
        if self._show_labels_check.isChecked():
            xs = coords[:, coord1]
            ys = coords[:, coord2]
            if annotate_offset:
                self._annotate_with_offset(xs, ys, labels)
            else:
                for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
                    self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Ellipses (auto-on when >=2 groups).
        if self._should_draw_ellipses(show_ellipses, unique_groups):
            self._add_confidence_ellipses(coords, groups, coord1, coord2)

        # Axis labels
        self._ax.set_xlabel(_("PCo{0} ({1:.1f}% variance)").format(coord1 + 1, var_1))
        self._ax.set_ylabel(_("PCo{0} ({1:.1f}% variance)").format(coord2 + 1, var_2))
        self._ax.set_title(_("PCoA Scores Plot"))

        # Style
        self._apply_axis_style()
        if len(unique_groups) > 1:
            self._ax.legend(**self._legend_kwargs("upper right"))

        self._canvas.draw()

    # =========================================================================
    # CCA/RDA Plotting Methods (Triplot)
    # =========================================================================

    def plot_cca_triplot(
        self,
        result: Any,
        ax1: int = 0,
        ax2: int = 1,
        groups: list[int] | np.ndarray | None = None,
        labels: list[str] | None = None,
        group_names: list[str] | None = None,
        annotate_offset: bool = True,
    ) -> None:
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

        Parameters: see :meth:`plot_pca_scores` for ``groups``, ``labels``,
            ``group_names`` and ``annotate_offset``.
        """
        self._record_plot_call(
            "plot_cca_triplot",
            result,
            ax1=ax1,
            ax2=ax2,
            groups=groups,
            labels=labels,
            group_names=group_names,
            annotate_offset=annotate_offset,
        )
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

        if hasattr(result, "method"):
            method = result.method.upper()
        else:
            method = "CCA"

        labels, groups = self._resolve_metadata(result, labels, groups, site_scores)
        group_label_map = self._resolve_group_names(groups, group_names)

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

            display_name = group_label_map.get(group, f"Group {group + 1}")
            self._ax.scatter(
                site_scores[idx, ax1],
                site_scores[idx, ax2],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                marker="o",
                label=display_name if len(unique_groups) > 1 else _("Samples"),
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
            xs = site_scores[:, ax1]
            ys = site_scores[:, ax2]
            if annotate_offset:
                self._annotate_with_offset(xs, ys, labels)
            else:
                for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
                    self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Axis labels
        self._ax.set_xlabel(_("{0}1 ({1:.1f}% variance)").format(method, var_ax1))
        self._ax.set_ylabel(_("{0}2 ({1:.1f}% variance)").format(method, var_ax2))
        self._ax.set_title(_("{0} Triplot").format(method))

        # Style
        self._apply_axis_style()

        # Legend
        if len(unique_groups) > 1 or species_scores is not None or biplot_scores is not None:
            self._ax.legend(**self._legend_kwargs("upper right"))

        # Reference lines
        self._ax.axhline(y=0, color=self.theme_colors()["reference"], linestyle="--", linewidth=0.5, alpha=0.5)
        self._ax.axvline(x=0, color=self.theme_colors()["reference"], linestyle="--", linewidth=0.5, alpha=0.5)

        # Grid
        self._ax.grid(True, alpha=0.3, color=self.theme_colors()["grid"])

        self._canvas.draw()

    # =========================================================================
    # TPS Deformation Grid
    # =========================================================================

    def plot_tps_deformation_grid(
        self, tps_result: Any, grid_shape: tuple[int, int] = (15, 15), show_vectors: bool = True
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
        self._record_plot_call(
            "plot_tps_deformation_grid", tps_result, grid_shape=grid_shape, show_vectors=show_vectors
        )
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
            self._ax.plot(
                xx_orig[i, :],
                yy_orig[i, :],
                color=self.theme_colors()["reference"],
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )
        for j in range(grid_shape[1]):
            self._ax.plot(
                xx_orig[:, j],
                yy_orig[:, j],
                color=self.theme_colors()["reference"],
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )

        # Plot warped grid - colored lines
        for i in range(grid_shape[0]):
            self._ax.plot(warped_grid[i, :, 0], warped_grid[i, :, 1], color="#3498DB", linewidth=1.5, alpha=0.8)
        for j in range(grid_shape[1]):
            self._ax.plot(warped_grid[:, j, 0], warped_grid[:, j, 1], color="#3498DB", linewidth=1.5, alpha=0.8)

        # Plot source landmarks
        self._ax.scatter(
            source[:, 0],
            source[:, 1],
            c="#E74C3C",
            s=80,
            marker="o",
            edgecolors="white",
            linewidths=1,
            label=_("Source"),
            zorder=5,
        )

        # Plot target landmarks
        self._ax.scatter(
            target[:, 0],
            target[:, 1],
            c="#27AE60",
            s=80,
            marker="s",
            edgecolors="white",
            linewidths=1,
            label=_("Target"),
            zorder=5,
        )

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
        self._ax.legend(
            loc="upper right",
            framealpha=0.95,
            facecolor=self.theme_colors()["figure_bg"],
            edgecolor=self.theme_colors()["border"],
        )

        # Aspect ratio
        self._ax.set_aspect("equal", adjustable="box")

        self._canvas.draw()

    # =========================================================================
    # NMDS Plotting Methods
    # =========================================================================

    def plot_nmds(
        self,
        result: Any,
        groups: list[int] | np.ndarray | None = None,
        labels: list[str] | None = None,
        group_names: list[str] | None = None,
        annotate_offset: bool = True,
        show_ellipses: bool | None = None,
    ) -> None:
        """
        Plot NMDS ordination.

        Mathematical Context:
            NMDS minimizes the stress function:

            Stress = √(Σ(d_ij - d̂_ij)² / Σd_ij²)

            where:
                d_ij = original dissimilarity
                d̂_ij = ordination distance

        Parameters: see :meth:`plot_pca_scores` for ``groups``, ``labels``,
            ``group_names``, ``annotate_offset`` and ``show_ellipses``.
        """
        self._record_plot_call(
            "plot_nmds",
            result,
            groups=groups,
            labels=labels,
            group_names=group_names,
            annotate_offset=annotate_offset,
            show_ellipses=show_ellipses,
        )
        self._ax.clear()
        self._current_plot_type = "nmds"

        # Extract data
        if hasattr(result, "coordinates"):
            coords = result.coordinates
        else:
            coords = result

        stress = getattr(result, "stress", 0)

        labels, groups = self._resolve_metadata(result, labels, groups, coords)
        group_label_map = self._resolve_group_names(groups, group_names)

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

            display_name = group_label_map.get(group, f"Group {group + 1}")
            self._ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c=color,
                s=100,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=display_name if len(unique_groups) > 1 else None,
                picker=True,
            )

        # Labels
        if self._show_labels_check.isChecked():
            xs = coords[:, 0]
            ys = coords[:, 1]
            if annotate_offset:
                self._annotate_with_offset(xs, ys, labels)
            else:
                for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
                    self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)

        # Ellipses (auto-on when >=2 groups).
        if self._should_draw_ellipses(show_ellipses, unique_groups):
            self._add_confidence_ellipses(coords, groups, 0, 1)

        # Axis labels and title
        self._ax.set_xlabel(_("NMDS1"))
        self._ax.set_ylabel(_("NMDS2"))
        self._ax.set_title(_("NMDS Ordination (Stress = {0:.4f})").format(stress))

        # Style
        self._apply_axis_style()
        if len(unique_groups) > 1:
            self._ax.legend(**self._legend_kwargs("upper right"))

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

        The result is a ``models.diversity_result.DiversityResult`` whose
        ``indices`` attribute is a ``dict[str, DiversityIndexResult]`` (each
        carrying ``index_name`` / ``value``). The old implementation looked for
        ``.values`` and ``.labels`` attributes that this object does not have,
        silently fell back to the hard-coded placeholder list
        ``[1.5, 0.8, 2.5]``, and then crashed anyway because the bar positions
        were sized from the 6-entry ``indices`` dict while the placeholder had 3
        entries::

            ValueError: shape mismatch: ... arg 0 with shape (6,) and arg 1 with shape (3,)

        So running Diversity from the ribbon always failed. Note the indices
        have deliberately different units (H' ~ 3, 1-D ~ 0.94, Chao-1 ~ S),
        so Chao-1 dominates the axis; every bar is annotated with its exact
        value so the small ones stay readable.
        """
        self._record_plot_call("plot_diversity_summary", result)
        self._ax.clear()
        self._current_plot_type = "diversity"

        # Extract data. Accept either the real DiversityResult
        # (.indices -> dict of DiversityIndexResult) or a plain mapping of
        # name -> number, and fall back to explicit scalars.
        raw = getattr(result, "indices", None)
        if raw is None:
            raw = getattr(result, "values", None)

        labels: list[str] = []
        values: list[float] = []
        if isinstance(raw, dict):
            for key, item in raw.items():
                labels.append(str(getattr(item, "index_name", key)))
                val = getattr(item, "value", item)
                values.append(float(val) if np.isscalar(val) or isinstance(val, (int, float)) else np.nan)
        elif isinstance(raw, (list, tuple, np.ndarray)):
            values = [float(v) for v in raw]
            labels = (
                [str(getattr(result, "labels", None)[i]) for i in range(len(values))]
                if getattr(result, "labels", None) is not None
                else [str(i + 1) for i in range(len(values))]
            )
        else:
            raise ValueError(
                f"plot_diversity_summary expects a DiversityResult (or a result "
                f"with .indices/.values); got {type(result).__name__}."
            )

        if not values:
            self._ax.text(
                0.5, 0.5, _("No diversity indices available"), ha="center", va="center", transform=self._ax.transAxes
            )
            self._figure.tight_layout()
            self._canvas.draw()
            return

        # Bar chart
        x_pos = np.arange(len(values))
        bars = self._ax.bar(x_pos, values, color=self.COLORS[: len(values)], alpha=0.7, edgecolor="white", linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values, strict=False):
            height = bar.get_height()
            self._ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=self.theme_colors()["text"],
                fontweight="bold",
            )

        # Labels
        self._ax.set_xticks(x_pos)
        self._ax.set_xticklabels(labels, color=self.theme_colors()["text"])
        self._ax.set_ylabel(_("Index Value"), color=self.theme_colors()["text"])
        self._ax.set_title(_("Biodiversity Indices"), color=self.theme_colors()["text"])

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
            sample_names = getattr(result, "sample_names", [f"Sample {i + 1}" for i in range(len(coverage_levels))])
            getattr(result, "confidence_lower", None)
            getattr(result, "confidence_upper", None)

            # Build curve_data dict for consistent plotting
            curve_data = {}
            for i, name in enumerate(sample_names):
                curve_data[name] = (coverage_levels, expected_richness)
        elif hasattr(result, "curve_data"):
            # Legacy format
            curve_data = result.curve_data
        else:
            # No data available
            self._ax.text(0.5, 0.5, _("No rarefaction data available"), ha="center", va="center", fontsize=12)
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
        self._ax.set_xlabel(_("Number of Individuals"), color=self.theme_colors()["text"])
        self._ax.set_ylabel(_("Expected Species Richness"), color=self.theme_colors()["text"])
        self._ax.set_title(_("Rarefaction Curves"), color=self.theme_colors()["text"])

        # Legend
        self._ax.legend(
            loc="lower right",
            framealpha=0.9,
            facecolor=self.theme_colors()["text"],
            edgecolor="#34495E",
            labelcolor="#ECF0F1",
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
            self._ax.text(0.5, 0.5, _("No spectral data available"), ha="center", va="center", fontsize=12)
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
            self._ax.axvline(
                x=peak_frequency,
                color=self.COLORS[3],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"{_('Peak')}: f={peak_frequency:.4f}, T={peak_period:.2f}"
                if peak_period
                else f"{_('Peak')}: f={peak_frequency:.4f}",
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
            with contextlib.suppress(Exception):
                self._colorbar.remove()
            self._colorbar = None

        # Extract data
        time = getattr(result, "time", None)
        frequencies = getattr(result, "frequencies", None)
        power = getattr(result, "power", None)
        coi = getattr(result, "coi", None)
        wavelet = getattr(result, "wavelet", "Morlet")

        # Check if required data is available
        if time is None or frequencies is None or power is None:
            self._ax.text(0.5, 0.5, _("No wavelet data available"), ha="center", va="center", fontsize=12)
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_title(_("Wavelet Scalogram: No Data"), fontsize=12, fontweight="bold")
            self._canvas.draw()
            return

        # Use pcolormesh for the scalogram
        time_grid, freq_grid = np.meshgrid(time, frequencies)

        # Plot power as colormap
        self._pcolormesh = self._ax.pcolormesh(
            time_grid,
            freq_grid,
            power,
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
        self._ax.bar([0], [R], color=color, alpha=0.7, edgecolor="white", width=0.5)

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
            color=self.theme_colors()["text"],
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
            0,
            -0.1 * max(abs(R), 0.1),
            info,
            ha="center",
            va="top",
            fontsize=9,
            color="#7F8C8D",
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
            _wedges, _texts = self._ax.pie(
                sizes, labels=labels_pie, colors=colors, startangle=90, textprops={"fontsize": 9}
            )

            # Add center text with F and p (pie axes centered at origin)
            sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            center_text = f"F = {F:.3f}{sig}\np = {p_value:.4f}"
            self._ax.text(
                0,
                0,
                center_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=self.theme_colors()["text"],
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
                0.5,
                0.5,
                info_text,
                ha="center",
                va="center",
                fontsize=11,
                color=self.theme_colors()["text"],
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
        # Use twiny() for cumulative contribution line on top
        self._ax = self._reset_axes()
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

        for bar, val in zip(bars, values, strict=False):
            self._ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=8,
                color=self.theme_colors()["text"],
            )

        # Cumulative contribution line on top axis
        ax_top.plot(cumulative, y_pos, color=self.COLORS[3], linewidth=2, marker="o", markersize=4, alpha=0.8)
        ax_top.set_xlabel(_("Cumulative Contribution (%)"), color=self.COLORS[3])
        ax_top.tick_params(axis="x", colors=self.COLORS[3])
        ax_top.set_xlim([0, 100])

        # Style
        self._ax.set_facecolor(self.theme_colors()["axes_bg"])
        self._ax.spines["top"].set_color(self.COLORS[0])
        ax_top.spines["top"].set_color(self.COLORS[3])

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_anova_boxplot(
        self, data: np.ndarray, groups: list[Any], variable_name: str = "", group_names: list[str] | None = None
    ) -> None:
        """Plot boxplot comparing groups for ONE variable.

        ``data`` is the full (n_samples, n_variables) matrix, so the column
        to display must be selected: pass a 0-based column index as
        ``variable_name`` for a plain array, or a field name for a structured
        array. The old body ignored ``variable_name`` except for the axis
        label and built ``plot_data`` as a list of 2-D row-blocks::

            plot_data.append(data[mask])          # (n_in_group, n_vars)!

        matplotlib then rejected the list outright::

            ValueError: X must have 2 or fewer dimensions

        so this method could not run for any multi-variable matrix — the only
        shape it survived was a single-column input, and even then string group
        labels crashed on ``f"Group {g + 1}"``.
        """
        self._record_plot_call("plot_anova_boxplot", data, groups, variable_name=variable_name, group_names=group_names)
        self._current_plot_type = "anova_boxplot"
        self._ax = self._reset_axes()

        data = np.asarray(data)
        # Structured arrays keep their field structure, so they must NOT be
        # reshaped to (n, 1) below -- doing so silently drops ``dtype.names``
        # and the name lookup then fails.
        is_structured = data.dtype.names is not None
        if not is_structured:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.ndim != 2:
                raise ValueError(f"plot_anova_boxplot expects a 2-D (samples x variables) array, got {data.ndim}-D")

        # Resolve which column to plot. ``variable_name`` is a *name* only
        # for a structured array; for a plain (n, p) matrix it is taken as a
        # 0-based column index. Anything else used to fall back to column 0
        # silently, so asking for "v1" drew the "v0" boxplot.
        n_vars = data.shape[1] if data.ndim > 1 else 1
        names = [str(c) for c in data.dtype.names] if is_structured else None
        if names:
            if not variable_name:
                raise ValueError(f"plot_anova_boxplot needs a variable name; available: {names}")
            if variable_name not in names:
                raise ValueError(f"Unknown variable {variable_name!r}; available: {names}")
            # Structured arrays are indexed by field name, not by position.
            values_all = np.asarray(data[variable_name], dtype=float)
        else:
            if data.ndim == 1:
                values_all = data
            else:
                col = 0
                if variable_name:
                    try:
                        col = int(variable_name)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"plot_anova_boxplot got variable_name={variable_name!r} for a plain "
                            f"(samples x variables) array with {n_vars} column(s). Pass a 0-based "
                            f"column index, or a structured array to use names."
                        ) from None
                    if not 0 <= col < n_vars:
                        raise ValueError(f"column {col} out of range for {n_vars} column(s)")
                values_all = data[:, col]

        unique_groups = sorted(set(groups))
        if not unique_groups:
            raise ValueError("plot_anova_boxplot needs at least one group")
        if group_names is None:
            # Works for both int and str labels (the old f"Group {g + 1}" raised
            # TypeError on any string group name).
            group_names = [str(g) for g in unique_groups]

        plot_data = []
        for g in unique_groups:
            mask = np.array([gr == g for gr in groups])
            plot_data.append(values_all[mask])

        # matplotlib renamed the boxplot ``labels`` kwarg to ``tick_labels``
        # in 3.9 (the project supports >=3.7), so try the new name first.
        try:
            bp = self._ax.boxplot(plot_data, tick_labels=group_names[: len(unique_groups)], patch_artist=True)
        except TypeError:
            bp = self._ax.boxplot(plot_data, labels=group_names[: len(unique_groups)], patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(self.COLORS[i % len(self.COLORS)])
            patch.set_alpha(0.7)

        self._ax.set_ylabel(variable_name)
        self._ax.set_title(f"{_('Group Comparison')}: {variable_name}")
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_lda_scores(
        self,
        result: Any,
        groups: list[int] | np.ndarray | None = None,
        labels: list[str] | None = None,
        group_names: list[str] | None = None,
        annotate_offset: bool = True,
        show_ellipses: bool | None = None,
    ) -> None:
        """Plot LDA scatter plot with confidence ellipses.

        Parameters: see :meth:`plot_pca_scores` for ``groups``, ``labels``,
            ``group_names``, ``annotate_offset`` and ``show_ellipses``.
        """
        self._record_plot_call(
            "plot_lda_scores",
            result,
            groups=groups,
            labels=labels,
            group_names=group_names,
            annotate_offset=annotate_offset,
            show_ellipses=show_ellipses,
        )
        self._current_plot_type = "lda"
        self._ax = self._reset_axes()

        scores = result.scores
        n_dims = scores.shape[1]

        labels, groups = self._resolve_metadata(result, labels, groups, scores)
        group_label_map = self._resolve_group_names(groups, group_names)

        # Store data for replotting
        self._scores = scores
        self._group_labels = groups
        self._labels = labels

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

        unique_groups = np.unique(groups)

        for i, g in enumerate(unique_groups):
            mask = groups == g
            color = self.COLORS[i % len(self.COLORS)]
            display_name = group_label_map.get(int(g), f"Group {int(g) + 1}")
            self._ax.scatter(
                x_data[mask],
                y_data[mask],
                c=color,
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                label=display_name,
            )

        # Add labels if enabled
        if self._show_labels_check.isChecked():
            if annotate_offset:
                self._annotate_with_offset(x_data, y_data, labels)
            else:
                for i, (x, y) in enumerate(zip(x_data, y_data, strict=False)):
                    self._ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.8, ha="center", va="bottom")

        # Add ellipses if enabled (auto-on when >=2 groups).
        if n_dims >= 2 and self._should_draw_ellipses(show_ellipses, unique_groups):
            self._add_confidence_ellipses(scores, groups, pc1, pc2)

        if hasattr(result, "explained_variance_ratio") and len(result.explained_variance_ratio) >= 2:
            x_label = f"LD1 ({result.explained_variance_ratio[0]:.1%})"
            y_label = f"LD2 ({result.explained_variance_ratio[1]:.1%})"

        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        self._ax.set_title(_("Linear Discriminant Analysis"))

        # Style
        self._apply_axes_theme()

        if len(unique_groups) > 1:
            self._ax.legend(**self._legend_kwargs("upper right"))

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_dendrogram(self, result: Any, labels: list[str] | None = None) -> None:
        """Plot hierarchical clustering dendrogram."""
        self._record_plot_call("plot_dendrogram", result, labels=labels)
        self._current_plot_type = "dendrogram"
        self._ax = self._reset_axes()

        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        scipy_dendrogram(
            result.linkage_matrix,
            labels=labels,
            ax=self._ax,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=result.linkage_matrix[-(result.n_clusters - 1), 2] if result.n_clusters > 1 else 0,
        )

        self._ax.set_title(f"{_('Hierarchical Clustering')} (cophenetic r={result.cophenetic_corr:.3f})")
        self._ax.set_ylabel(_("Distance"))
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_rose_diagram(self, bin_centers: np.ndarray, counts: np.ndarray, mean_direction_deg: float = 0.0) -> None:
        """Plot rose diagram for directional data."""
        self._record_plot_call("plot_rose_diagram", bin_centers, counts, mean_direction_deg=mean_direction_deg)
        self._current_plot_type = "rose"
        self._ax = self._reset_axes(projection="polar")

        n_bins = len(counts)
        if n_bins == 0:
            # Previously ``2 * np.pi / 0`` raised ZeroDivisionError.
            self._ax.text(0.5, 0.5, _("No directional data"), ha="center", va="center", transform=self._ax.transAxes)
            self._ax.set_title(_("Rose Diagram"), pad=20)
            self._canvas.draw()
            return
        bin_width = 2 * np.pi / n_bins
        # Convert degree centers to radians if values > 2*pi
        centers = np.deg2rad(bin_centers) if np.max(bin_centers) > 2 * np.pi else bin_centers
        if len(centers) != n_bins:
            # The production caller passes ``bin_rose_diagram``'s centres, which
            # already match ``counts`` one-for-one. Guard anyway so a caller
            # that hands over bin *edges* (n+1) gets a clear error instead of a
            # numpy broadcast ValueError from deep inside matplotlib.
            raise ValueError(
                f"plot_rose_diagram needs one centre per count; got {len(centers)} centres for {n_bins} counts."
            )

        self._ax.bar(centers, counts, width=bin_width * 0.8, color=self.COLORS[0], alpha=0.7, edgecolor="white")

        # Mean direction arrow
        max_count = max(counts) if len(counts) > 0 else 1
        mean_rad = np.deg2rad(mean_direction_deg)
        self._ax.annotate(
            "",
            xy=(mean_rad, max_count * 0.9),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=self.COLORS[3], lw=2),
        )

        self._ax.set_title(_("Rose Diagram"), pad=20)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_efa_contours(self, original: np.ndarray, reconstructed: np.ndarray, title: str = "") -> None:
        """Plot original vs reconstructed EFA contours."""
        self._record_plot_call("plot_efa_contours", original, reconstructed, title=title)
        self._current_plot_type = "efa"
        self._ax = self._reset_axes()

        self._ax.plot(
            original[:, 0], original[:, 1], "o-", color=self.COLORS[0], markersize=2, linewidth=1, label=_("Original")
        )
        self._ax.plot(
            reconstructed[:, 0], reconstructed[:, 1], "-", color=self.COLORS[3], linewidth=1.5, label=_("Reconstructed")
        )

        self._ax.set_aspect("equal")
        self._ax.legend(fontsize=8)
        self._ax.set_title(title or _("Elliptic Fourier Analysis"))
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_gpa_aligned(self, coords: np.ndarray, title: str = "") -> None:
        """
        Plot GPA-aligned landmark configurations as an overlay scatter.

        Parameters:
            coords: either a single configuration ``(n_landmarks, 2)`` or a
                stack of aligned configurations ``(n_specimens, n_landmarks, 2)``
            title: axes title

        This replaces the previous call site that passed a single positional
        argument to :meth:`plot_efa_contours` (which needs both an original
        *and* a reconstructed contour), which raised ``TypeError`` as soon as
        a GPA result was 2-D.
        """
        self._record_plot_call("plot_gpa_aligned", coords, title=title)
        self._current_plot_type = "gpa_aligned"
        self._ax = self._reset_axes()

        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)

        if coords.ndim == 2:
            # A single aligned configuration.
            self._ax.plot(
                coords[:, 0],
                coords[:, 1],
                "-o",
                color=self.COLORS[0],
                markersize=4,
                linewidth=1,
                label=_("Specimen"),
            )
        elif coords.ndim == 3:
            for specimen in coords:
                self._ax.plot(
                    specimen[:, 0],
                    specimen[:, 1],
                    "-o",
                    color=self.COLORS[0],
                    alpha=0.3,
                    markersize=3,
                )
            mean_shape = coords.mean(axis=0)
            self._ax.plot(
                mean_shape[:, 0],
                mean_shape[:, 1],
                "-o",
                color=self.COLORS[3],
                linewidth=2.0,
                markersize=5,
                label=_("Mean shape"),
            )
        else:
            raise ValueError(f"plot_gpa_aligned expects a 2-D or 3-D coordinate array, got {coords.ndim}-D")

        self._ax.set_aspect("equal")
        self._ax.legend(**self._legend_kwargs("best"))
        self._ax.grid(True, linestyle="--", alpha=0.3, color=self.theme_colors()["grid"])
        self._ax.set_title(title or _("GPA Aligned Landmarks"))
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_she_curve(self, result: Any) -> None:
        """Plot SHE analysis curves (S, H, E vs sample size)."""
        self._record_plot_call("plot_she_curve", result)
        self._current_plot_type = "she"
        ax1 = self._reset_axes()
        ax2 = ax1.twinx()

        ax1.plot(result.sample_sizes, result.s_values, "o-", color=self.COLORS[0], markersize=3, label="S (Richness)")
        ax2.plot(result.sample_sizes, result.e_values, "s-", color=self.COLORS[2], markersize=3, label="E (Evenness)")

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

    def plot_ripley_k(self, result: Any, show_points: bool = True) -> None:
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
            r_values, envelope_lower, envelope_upper, color="#3498DB", alpha=0.2, label=_("95% Envelope")
        )

        # Plot L(r) curve
        self._ax.plot(r_values, l_values, color="#E74C3C", linewidth=2.5, label=_("L(r) - r"))

        # Plot zero reference line
        self._ax.axhline(y=0, color=self.theme_colors()["text"], linestyle="--", linewidth=1, alpha=0.7)

        # Labels
        self._ax.set_xlabel(_("Distance (r)"))
        self._ax.set_ylabel(_("L(r) - r"))
        self._ax.set_title(_("Ripley's K Spatial Point Pattern Analysis"))

        # Legend
        self._ax.legend(
            loc="upper left",
            framealpha=0.95,
            facecolor=self.theme_colors()["figure_bg"],
            edgecolor=self.theme_colors()["border"],
        )

        # Style
        self._apply_axis_style()
        self._ax.grid(True, alpha=0.3, color=self.theme_colors()["grid"])

        # Interpretation text
        interp = result.interpretation
        self._ax.text(
            0.98,
            0.02,
            interp,
            transform=self._ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor=self.theme_colors()["figure_bg"],
                edgecolor=self.theme_colors()["border"],
                alpha=0.9,
            ),
        )

        self._canvas.draw()

    def plot_abundance_models(self, results: dict) -> None:
        """Plot rank-abundance curves with fitted models."""
        self._record_plot_call("plot_abundance_models", results)
        self._current_plot_type = "abundance_models"
        self._ax = self._reset_axes()

        for i, (name, fit) in enumerate(results.items()):
            obs = np.sort(fit.observed)[::-1]
            pred = np.sort(fit.predicted)[::-1]
            n = min(len(obs), len(pred))
            ranks = np.arange(1, n + 1)

            if i == 0:
                self._ax.scatter(
                    ranks, obs, s=20, color=self.theme_colors()["text"], alpha=0.6, label=_("Observed"), zorder=5
                )

            self._ax.plot(
                ranks,
                pred,
                "-",
                color=self.COLORS[i % len(self.COLORS)],
                linewidth=1.5,
                label=f"{fit.model_name} (R²={fit.r_squared:.3f})",
            )

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
        """Apply the *current theme* plus grid and zero reference lines."""
        t = self.theme_colors()
        self._apply_axes_theme()

        self._ax.grid(True, alpha=0.3, color=t["grid"])
        self._ax.axhline(y=0, color=t["reference"], linestyle="--", linewidth=0.5, alpha=0.5)
        self._ax.axvline(x=0, color=t["reference"], linestyle="--", linewidth=0.5, alpha=0.5)

    # ------------------------------------------------------------------
    # Shared metadata resolution (labels, groups, group_names)
    #
    # The four ordination-style plotters (PCA, PCoA, NMDS, CCA, LDA) used
    # to read ``result.labels`` / ``result.groups`` and otherwise drop the
    # real sample names (``S1..Sn``) and the habitat groups (single
    # colour, no 95% ellipses). The legacy fallback order was also hidden
    # inside each method. These helpers make the resolution a single
    # source of truth so the call site ``plot_pca_scores(result, groups=g,
    # labels=l)`` wins, the dataclass ``result.groups`` wins next, and the
    # ugly ``S1..Sn`` placeholder is only reached as a last resort.
    # ------------------------------------------------------------------

    def _resolve_metadata(
        self,
        result: Any,
        labels: list[str] | None,
        groups: list[int] | np.ndarray | None,
        scores: np.ndarray | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Return ``(labels, groups)`` with the same length as ``scores``.

        Priority:
            1. explicit ``labels`` / ``groups`` argument
            2. ``result.labels`` / ``result.groups``
            3. synthetic defaults (``S1..Sn`` and a single group).

        The returned ``groups`` is always an ``np.ndarray`` of ints.

        ``scores`` is the (n, k) score matrix the caller is about to
        plot — passed in explicitly so this helper does not have to
        rely on ``self._scores_storage`` being populated *before* the
        call (the plotting method sets it *after* resolving metadata).
        """
        n = None
        if scores is not None and getattr(scores, "ndim", 0) == 2:
            n = scores.shape[0]
        if n is None:
            stored = getattr(self, "_scores_storage", None)
            if stored is not None and getattr(stored, "ndim", 0) == 2:
                n = stored.shape[0]

        if labels is None:
            labels = getattr(result, "labels", None)
        if labels is None:
            if n is None:
                labels = []
            else:
                labels = [f"S{i + 1}" for i in range(n)]
        if isinstance(labels, np.ndarray):
            labels = [str(x) for x in labels.tolist()]

        if groups is None:
            raw_groups = getattr(result, "groups", None)
            if raw_groups is None:
                if n is None:
                    raw_groups = []
                else:
                    raw_groups = np.zeros(n, dtype=int)
        else:
            raw_groups = groups
        groups_arr = np.asarray(raw_groups, dtype=int).ravel()
        if n is not None and groups_arr.size != n:
            # Mismatched length: degrade to a single group rather than
            # silently dropping samples or crashing on a downstream
            # ``np.unique(groups)`` call.
            groups_arr = np.zeros(n, dtype=int)
        return list(labels), groups_arr

    def _resolve_group_names(
        self,
        groups: np.ndarray,
        group_names: list[str] | None,
    ) -> dict[int, str]:
        """Map each distinct group id to a display name.

        If ``group_names`` was passed (one name per unique group, in
        sorted-id order) it wins. Otherwise every group falls back to
        ``Group <n+1>``.
        """
        unique = sorted(np.unique(groups).tolist())
        names: dict[int, str] = {}
        if group_names is not None and len(group_names) >= len(unique):
            for i, g in enumerate(unique):
                names[g] = str(group_names[i])
            return names
        for g in unique:
            names[g] = f"Group {int(g) + 1}"
        return names

    def _should_draw_ellipses(
        self,
        show_ellipses: bool | None,
        unique_groups: np.ndarray,
    ) -> bool:
        """Decide whether 95% confidence ellipses should be drawn.

        ``show_ellipses`` wins if it is not ``None``. Otherwise the new
        default — "auto" — draws them whenever there are >=2 groups, OR
        whenever the toolbar toggle is on (preserves the historic
        one-group ellipse use case).
        """
        if show_ellipses is not None:
            return bool(show_ellipses)
        if len(unique_groups) >= 2:
            return True
        return bool(self._show_ellipses_check.isChecked())

    def _annotate_with_offset(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        labels: list[str],
    ) -> None:
        """Annotate points with a cheap iterative repel.

        ``adjustText`` is intentionally NOT a dependency. The algorithm
        works in *pixel* space — every point is projected through the
        axes transform, pairs whose bounding boxes overlap get pushed
        apart vertically, and the pixel offsets are converted back to
        data units so the annotation's ``xytext`` lands where the
        viewport shows it. A handful of passes is enough to disentangle
        the dense-cluster artefacts that broke the historic straight-
        above placement (S19/S20/S21 stacking).

        ``ha="center"`` so the offset stays symmetric.
        """
        if len(xs) == 0:
            return
        n = len(xs)

        # Project every (finite) point to display pixels. Skip non-finite
        # points — they cannot be hit-tested anyway and would crash the
        # transform.
        finite_mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite_mask):
            return
        pix = self._ax.transData.transform
        inv = self._ax.transData.inverted().transform
        pts_pix: list[tuple[float, float]] = []
        for i in range(n):
            if finite_mask[i]:
                pts_pix.append(pix((float(xs[i]), float(ys[i]))))
            else:
                pts_pix.append((0.0, 0.0))

        # Pixel-space repel. Threshold is roughly "two labels worth of
        # vertical space" (12 px) and "a short label's width" (30 px).
        offsets_px = np.zeros((n, 2), dtype=float)
        for _pass in range(8):
            moved = False
            for i in range(n):
                if not finite_mask[i]:
                    continue
                for j in range(i + 1, n):
                    if not finite_mask[j]:
                        continue
                    dx = pts_pix[j][0] - pts_pix[i][0] + (offsets_px[j, 0] - offsets_px[i, 0])
                    dy = pts_pix[j][1] - pts_pix[i][1] + (offsets_px[j, 1] - offsets_px[i, 1])
                    if abs(dx) > 30:
                        continue
                    if abs(dy) < 12:
                        # Push j above i (downward in screen space).
                        offsets_px[j, 1] = offsets_px[i, 1] + 12
                        moved = True
            if not moved:
                break

        for i in range(n):
            if not finite_mask[i]:
                continue
            x = float(xs[i])
            y = float(ys[i])
            if offsets_px[i, 0] == 0 and offsets_px[i, 1] == 0:
                # No dodge needed — keep the historic "straight above"
                # placement so a single label looks identical to the
                # pre-fix canvas.
                self._ax.annotate(
                    labels[i],
                    (x, y),
                    fontsize=8,
                    alpha=0.8,
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                )
                continue
            # Convert the pixel dodge back into data units. Annotate's
            # ``xytext`` is in data coords here so the offset stays put
            # even when the user zooms or pans.
            target_x, target_y = inv(
                (pts_pix[i][0] + offsets_px[i, 0], pts_pix[i][1] + offsets_px[i, 1])
            )
            va = "center"
            self._ax.annotate(
                labels[i],
                (target_x, target_y),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va=va,
                xytext=(0, 0),
                textcoords="data",
            )

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

        if getattr(self._scores, "ndim", 0) != 2:
            # Not a score matrix (a 1-D series): no hover geometry to hit-test.
            return

        # Find nearest point
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        d1, d2 = self._current_dim1, self._current_dim2

        # Optimization: filter to candidate points within extended bounding box first
        # to avoid O(n) distance calculation on every mouse move
        x_range = self._ax.get_xlim()
        y_range = self._ax.get_ylim()
        # Extend search box by 5% margin for better UX
        x_margin = (x_range[1] - x_range[0]) * 0.05
        y_margin = (y_range[1] - y_range[0]) * 0.05
        candidates_mask = (
            (self._scores[:, d1] >= x_range[0] - x_margin)
            & (self._scores[:, d1] <= x_range[1] + x_margin)
            & (self._scores[:, d2] >= y_range[0] - y_margin)
            & (self._scores[:, d2] <= y_range[1] + y_margin)
        )
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) == 0:
            return

        # Calculate distances only for filtered candidates
        scores_d1 = self._scores[candidate_indices, d1]
        scores_d2 = self._scores[candidate_indices, d2]
        distances = np.sqrt((scores_d1 - x) ** 2 + (scores_d2 - y) ** 2)
        nearest_cand_idx = np.argmin(distances)
        nearest_idx = candidate_indices[nearest_cand_idx]
        min_dist = distances[nearest_cand_idx]

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
            # ``_group_labels`` can be shorter than the score matrix (a
            # result may carry group assignments for only part of the
            # samples), so bound-check before indexing.
            group_labels = self._group_labels
            if group_labels is not None and nearest_idx < len(group_labels):
                group = group_labels[nearest_idx]
            else:
                group = 0

            tooltip = f"{label}\n"
            tooltip += f"X: {self._scores[nearest_idx, d1]:.4f}\n"
            tooltip += f"Y: {self._scores[nearest_idx, d2]:.4f}\n"
            if group_labels is not None and nearest_idx < len(group_labels):
                tooltip += f"Group: {group + 1}"

            # Draw annotation
            self._hover_annotation = self._ax.annotate(
                tooltip,
                xy=(self._scores[nearest_idx, d1], self._scores[nearest_idx, d2]),
                xytext=(20, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="white", lw=1),
                bbox=dict(boxstyle="round", facecolor=self.theme_colors()["text"], alpha=0.9),
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
        if index in (1, 2) and hasattr(self, "_selector"):
            self._selector.disconnect()
            del self._selector

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
        if self._scores is None or eclick.xdata is None or erelease.xdata is None:
            return

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
        if self._scores is None:
            return

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
        self._ax = self._reset_axes()

        log_cs = result.log_centroid_sizes
        coef = result.regression_coefficients
        intercept = result.regression_intercept
        r_squared = result.r_squared
        residuals = result.residuals

        self._scores = (
            np.column_stack([log_cs, residuals])
            if residuals.ndim > 1
            else np.column_stack([log_cs, np.zeros_like(log_cs)])
        )
        self._labels = [f"S{i}" for i in range(len(log_cs))]
        self._group_labels = np.zeros(len(log_cs), dtype=int)

        # Scatter plot
        self._ax.scatter(
            log_cs,
            residuals if residuals.ndim == 1 else residuals[:, 0],
            c=self.theme_colors()["text"],
            s=80,
            alpha=0.7,
            edgecolors="white",
        )

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
            mse = np.sum(residuals**2) / (n - 2) if residuals.ndim == 1 else np.sum(residuals[:, 0] ** 2) / (n - 2)
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
        # Repaint: without this the canvas keeps showing the previous plot
        # even though the axes objects were rewritten.
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Two-Block PLS (morphological integration)
    #
    # Pre-fix the canvas silently swallowed the payload: ``ui_main_window``
    # sent ``result.to_dict()`` to ``plot.plot_pls``, the canvas had no
    # such method, the ``else: return`` branch ran, and the analysis
    # vanished with no warning. The fix is twofold:
    #   1. ``plot_pls`` here accepts either a ``PLSResult`` dataclass OR a
    #      ``dict`` (so the existing dialog payload works without
    #      rewriting the controller).
    #   2. ``plot_pls_results`` is registered as an alias so the legacy
    #      ``hasattr(plot, "plot_pls_results")`` branch in the main window
    #      stops falling through to ``return``.
    # Rendering is delegated to ``AllometryPlotter.plot_pls_scores`` (the
    # standalone publication plotter) so the canvas and the script path
    # stay visually consistent.
    # ------------------------------------------------------------------

    def plot_pls(
        self,
        result: Any,
        groups: list[int] | np.ndarray | None = None,
        group_names: list[str] | None = None,
    ) -> None:
        """Plot 2-Block PLS (morphological integration) scores.

        Parameters:
            result: ``PLSResult`` dataclass, OR a ``dict`` produced by
                ``PLSResult.to_dict()`` (the PLSDialog payload). Lists of
                arrays round-trip back to ``np.ndarray`` automatically.
            groups: Optional per-specimen group labels for colouring. When
                ``None``, all specimens are drawn in a single colour.
            group_names: Optional display names for each unique group
                (sorted-id order). Falls back to ``Group <n>``.
        """
        self._record_plot_call("plot_pls", result, groups=groups, group_names=group_names)
        self._current_plot_type = "pls"
        self._ax = self._reset_axes()

        payload = self._normalise_pls_payload(result)
        left_scores = np.asarray(payload["left_scores"], dtype=np.float64)
        right_scores = np.asarray(payload["right_scores"], dtype=np.float64)
        correlations = np.asarray(payload["pls_correlations"], dtype=np.float64)
        integration_index = float(payload["integration_index"])

        n_specimens = left_scores.shape[0]
        if groups is None:
            groups_arr = np.zeros(n_specimens, dtype=int)
        else:
            groups_arr = np.asarray(groups, dtype=int).ravel()
            if groups_arr.size != n_specimens:
                groups_arr = np.zeros(n_specimens, dtype=int)
        unique_groups = np.unique(groups_arr)
        group_label_map = self._resolve_group_names(groups_arr, group_names)

        # Reuse the AllometryPlotter just for visual consistency — we
        # then draw our own copy on ``self._ax`` because matplotlib
        # artists belong to exactly one figure, and the canvas must own
        # them for the zoom / reset / lasso toolbar buttons to work.
        # Rendering matches ``AllometryPlotter.plot_pls_scores`` 1:1
        # (scatter by group + regression line + integration title).
        x = left_scores[:, 0]
        y = right_scores[:, 0]

        if len(unique_groups) <= 1:
            self._ax.scatter(
                x, y,
                c="#2C3E50", s=80, alpha=0.7,
                edgecolors="white", linewidths=0.5,
            )
        else:
            sorted_groups = sorted(unique_groups.tolist())
            for i, group_id in enumerate(sorted_groups):
                mask = groups_arr == group_id
                self._ax.scatter(
                    x[mask], y[mask],
                    c=[self.COLORS[i % len(self.COLORS)]],
                    label=group_label_map.get(int(group_id), f"Group {group_id + 1}"),
                    s=80, alpha=0.7,
                    edgecolors="white", linewidths=0.5,
                )

        # Regression line: numpy raises if x has only one unique value,
        # but that case cannot occur for a PLS result with n>=3 specimens.
        if x.size >= 2 and np.unique(x).size >= 2:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            self._ax.plot(x_line, np.polyval(coef, x_line), "r--", linewidth=1.5, alpha=0.7)

        r1 = float(correlations[0]) if correlations.size > 0 else 0.0
        self._ax.set_xlabel(_("Block A PLS Score (Comp 1)"))
        self._ax.set_ylabel(_("Block B PLS Score (Comp 1)"))
        self._ax.set_title(
            _("Two-Block PLS (r₁ = {0:.4f}, integration = {1:.4f})").format(r1, integration_index)
        )

        self._apply_axes_theme()
        if len(unique_groups) > 1:
            self._ax.legend(**self._legend_kwargs("best"))
        self._ax.grid(True, linestyle="--", alpha=0.3, color=self.theme_colors()["grid"])

        # Stash the metadata so the hover tooltip / selectors stay useful.
        if left_scores.shape[1] >= 2:
            self._scores = left_scores[:, :2]
        else:
            self._scores = left_scores[:, :1]
        self._labels = [f"S{i + 1}" for i in range(n_specimens)]
        self._group_labels = groups_arr

        self._figure.tight_layout()
        self._canvas.draw()

    # Back-compat alias — ``ui_main_window._on_run_pls`` checks for either
    # name before giving up on the result. Keeping both stops that branch
    # from falling through to ``return``.
    plot_pls_results = plot_pls

    @staticmethod
    def _normalise_pls_payload(result: Any) -> dict[str, Any]:
        """Coerce a ``PLSResult`` (or its ``to_dict()``) into a plain dict.

        The PLSDialog payload is already a dict, but the controller
        may also pass the dataclass (e.g. from automated tests). The
        PLSResult carries ``left_scores`` / ``right_scores`` as
        ``np.ndarray``s; the dict carries Python lists. Both shapes
        must reach the plotter intact.
        """
        if isinstance(result, dict):
            payload = dict(result)
        elif hasattr(result, "to_dict") and callable(result.to_dict):
            payload = result.to_dict()
        else:
            payload = {
                k: getattr(result, k)
                for k in (
                    "left_scores", "right_scores", "pls_correlations",
                    "integration_index", "rv_coefficient", "pls1_pvalue",
                    "pls1_z", "singular_values", "covariance_explained",
                    "cumulative_covariance", "n_components", "n_specimens",
                )
                if hasattr(result, k)
            }
        # Some old payloads serialise correlations under ``rv_coefficients``.
        if "pls_correlations" not in payload and "rv_coefficients" in payload:
            payload["pls_correlations"] = payload["rv_coefficients"]
        return payload

    def plot_evolution_rate(self, result: Any) -> None:
        """
        Plot evolution rate phenogram showing trait evolution over time.

        Parameters:
            result: EvolutionRateResult with best_model, rate_estimate,
                   aic_weights, trait_mean, trait_variance, trait_series
        """
        self._record_plot_call("plot_evolution_rate", result)
        self._current_plot_type = "evolution_rate"

        # Main phenogram subplot
        self._ax = self._reset_axes(2, 1, 1)

        # Use actual trait_series from result if available
        if result.trait_series is not None and len(result.trait_series) > 0:
            trait_series = np.asarray(result.trait_series)
            n_points = len(trait_series)
        else:
            # Fallback: show message if no data available
            self._ax.text(0.5, 0.5, "No trait series data available", ha="center", va="center", fontsize=12)
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
        self._ax.text(
            0.02,
            0.98,
            model_label,
            transform=self._ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

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
            for bar, w in zip(bars, weights, strict=False):
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center")

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_extinction_ranges(self, result: Any) -> None:
        """
        Plot stratigraphic range chart with extinction confidence intervals.

        Parameters:
            result: ExtinctionIntervalResult with lad_positions, ci_lower, ci_upper,
                   confidence_interval_lower, confidence_interval_upper, method
        """
        self._record_plot_call("plot_extinction_ranges", result)
        self._current_plot_type = "extinction"
        self._ax = self._reset_axes()

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
        for i, (lad, _ci_l, ci_u, name) in enumerate(
            zip(lad_sorted, ci_lower_sorted, ci_upper_sorted, names_sorted, strict=False)
        ):
            # Observed range (solid line from top to LAD)
            self._ax.plot([0.3, 0.7], [0, lad], "b-", linewidth=3, solid_capstyle="butt")
            self._ax.plot([0.2, 0.8], [lad, lad], "b-", linewidth=2)

            # CI whiskers (extending upward)
            self._ax.plot([0.5, 0.5], [ci_u, lad], "r--", linewidth=1.5)

            # CI box
            from matplotlib.patches import Rectangle

            rect = Rectangle(
                (0.25, ci_u), 0.5, lad - ci_u, linewidth=1, edgecolor="red", facecolor="red", alpha=0.2, linestyle="--"
            )
            self._ax.add_patch(rect)

            # Taxon label
            self._ax.text(0.85, lad, name, fontsize=9, va="center", ha="left")

        self._ax.set_xlim(0, 1)
        max_lad = max(lad_sorted) if len(lad_sorted) > 0 else 10
        self._ax.set_ylim(-1, max_lad + 2)
        self._ax.invert_yaxis()
        self._ax.set_xlabel("Taxonomic Range", fontsize=10)
        self._ax.set_ylabel("Stratigraphic Height (layers from top)", fontsize=10)
        self._ax.set_title(
            f"Extinction Confidence Intervals ({result.method.upper()}, {int(result.confidence_level * 100)}% CI)",
            fontsize=12,
            fontweight="bold",
        )
        self._ax.set_xticks([])

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=3, label="Observed LAD"),
            Line2D(
                [0], [0], color="red", linewidth=1.5, linestyle="--", label=f"{int(result.confidence_level * 100)}% CI"
            ),
        ]
        self._ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        self._ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        self._canvas.draw()

    def plot_beta_diversity(self, result: Any) -> None:
        """
        Plot beta diversity decomposition heatmap.

        Parameters:
            result: BetaDiversityResult with total_beta, turnover_component,
                   nestedness_component, sample_names
        """
        self._record_plot_call("plot_beta_diversity", result)
        self._current_plot_type = "beta_diversity"
        self._ax = self._reset_axes()

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

        self._ax.set_title(
            f"Beta Diversity Decomposition ({result.decomposition_type.upper()})", fontsize=12, fontweight="bold"
        )
        self._canvas.draw()

    def plot_null_model(self, result: Any) -> None:
        """
        Plot null model analysis results.

        Parameters:
            result: NullModelResult with observed_score, simulated_scores,
                   mean_simulated, standardized_effect_size, p_value
        """
        self._record_plot_call("plot_null_model", result)
        self._current_plot_type = "null_model"

        # Histogram of simulated scores
        self._ax = self._reset_axes()
        simulated = result.simulated_scores
        self._ax.hist(simulated, bins=50, color="#3498DB", alpha=0.7, edgecolor="black", label="Simulated")

        # Observed score line
        self._ax.axvline(
            result.observed_score,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Observed = {result.observed_score:.4f}",
        )
        self._ax.axvline(
            result.mean_simulated,
            color="green",
            linewidth=2,
            linestyle="-",
            label=f"Mean = {result.mean_simulated:.4f}",
        )

        # SES annotation
        sig = (
            "***"
            if result.p_value < 0.001
            else ("**" if result.p_value < 0.01 else ("*" if result.p_value < 0.05 else ""))
        )
        self._ax.text(
            0.02,
            0.98,
            f"SES = {result.standardized_effect_size:.2f}\np = {result.p_value:.4f} {sig}",
            transform=self._ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        self._ax.set_xlabel(f"{result.metric.upper()} Score", fontsize=10)
        self._ax.set_ylabel("Frequency", fontsize=10)
        self._ax.set_title(
            f"Null Model Analysis ({result.algorithm.upper()}, {result.n_permutations} permutations)",
            fontsize=12,
            fontweight="bold",
        )
        self._ax.legend(loc="best")
        self._canvas.draw()

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

    # ------------------------------------------------------------------
    # Public tool API. These are thin aliases for the private helpers
    # above; keeping the public names explicit gives callers a stable
    # contract that does not silently disappear when the private
    # implementation is renamed.
    # ------------------------------------------------------------------

    def export_plot(self, options: object | None = None) -> None:
        """Public entry point for the Save action.

        When ``options`` is ``None`` (legacy call sites) the canvas falls
        back to the legacy "ask for path + write at default DPI" flow. When
        ``options`` is provided (typically a
        :class:`plot_export.PlotExportOptions` instance) the canvas
        delegates to :func:`plot_export.export_figure` so callers
        can fully control format / DPI / background / colour / size.
        """
        if options is None:
            self._export_plot()
            return
        # New path: use the unified facade. The dialog already showed
        # the Save dialog, so we write directly without further
        # prompting.
        try:
            from plot_export import PlotExportOptions, export_figure

            if not isinstance(options, PlotExportOptions):
                raise TypeError("export_plot(options=...) expects a PlotExportOptions")
            path = options.metadata.pop("_target_path", None) if options.metadata else None
            if not path:
                # Fall back to asking the user when no path was threaded
                # through. The PlotExportDialog always sets one. Build
                # the filter from the requested format instead of a
                # hardcoded PNG filter (svg/pdf exports previously
                # showed "PNG (*.png)" only).
                fmt = str(getattr(options, "format", "png") or "png")
                path, _selected_filter = QFileDialog.getSaveFileName(
                    self, _("Export Plot"), "", f"{fmt.upper()} (*.{fmt})"
                )
                if not path:
                    return
            export_figure(self._figure, path, options)
            QMessageBox.information(
                self,
                _("Export Successful"),
                _("Plot saved to:\n{0}").format(path),
            )
        except Exception as e:
            QMessageBox.critical(self, _("Export Error"), _("Failed to export plot:\n{0}").format(str(e)))

    def zoom_in(self) -> None:
        """Public entry point for the Zoom action."""
        self._zoom_in()

    def reset_view(self) -> None:
        """Public entry point for the Reset action."""
        self._reset_view()

    def _export_plot(self) -> None:
        """Legacy export flow kept for backward compatibility.

        Modern callers should use :meth:`export_plot` with a
        :class:`plot_export.PlotExportOptions` instance.
        """
        from plot_export import export_figure

        from views.ui_plot_export_dialog import PlotExportDialog

        dialog = PlotExportDialog("plot.png", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            options = dialog.get_options()
        except ValueError as exc:
            QMessageBox.critical(self, _("Export Error"), str(exc))
            return
        path = dialog.get_path()
        try:
            export_figure(self._figure, path, options)
            width_cm = self._figure.get_figwidth() * 2.54
            height_cm = self._figure.get_figheight() * 2.54
            QMessageBox.information(
                self,
                _("Export Successful"),
                _("Plot saved to:\n{0}\n\nResolution: {1} DPI\nSize: {2:.1f} x {3:.1f} cm").format(
                    path, options.dpi, width_cm, height_cm
                ),
            )
        except Exception as e:
            QMessageBox.critical(self, _("Export Error"), _("Failed to export plot:\n{0}").format(str(e)))

    # =========================================================================
    # Univariate Statistics Plots (P0: text → visualization)
    # =========================================================================

    def plot_summary_statistics(
        self, data: np.ndarray, col_names: list[str], stats_list: list[Any] | None = None
    ) -> None:
        """Plot summary statistics as a panel of histograms + boxplots.

        Parameters:
            data: Data matrix (n_samples, n_variables)
            col_names: Column/variable names
            stats_list: Optional list of ColumnStats objects for annotation
        """
        self._record_plot_call("plot_summary_statistics", data, col_names, stats_list)
        self._current_plot_type = "summary_statistics"
        self._figure.clear()

        n_vars = min(data.shape[1], 12)  # Limit to 12 panels
        ncols = min(4, n_vars)
        nrows = (n_vars + ncols - 1) // ncols

        for i in range(n_vars):
            ax = self._figure.add_subplot(nrows, ncols, i + 1)
            col_data = data[:, i]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                ax.text(0.5, 0.5, "All NaN", transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_title(col_names[i] if i < len(col_names) else f"Var {i + 1}", fontsize=9)
                continue

            ax.hist(valid, bins=min(30, max(5, len(valid) // 3)), color="#3498DB", alpha=0.7, edgecolor="white")
            name = col_names[i] if i < len(col_names) else f"Var {i + 1}"
            ax.set_title(name, fontsize=9, fontweight="bold")

            if stats_list and i < len(stats_list):
                s = stats_list[i]
                info = f"μ={s.mean:.2f} σ={s.std:.2f}\nmed={s.median:.2f}"
                ax.text(
                    0.97,
                    0.95,
                    info,
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    ha="right",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )

            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        self._figure.suptitle(_("Descriptive Statistics"), fontsize=13, fontweight="bold")
        self._figure.tight_layout(rect=[0, 0, 1, 0.95])
        self._finalise_grid_plot()
        self._canvas.draw()

    def plot_normality_qq(
        self, data: np.ndarray, col_names: list[str], normality_results: list[Any] | None = None
    ) -> None:
        """Plot Q-Q plots for normality assessment.

        Parameters:
            data: Data matrix (n_samples, n_variables)
            col_names: Column names
            normality_results: Optional list of NormalityResult for annotation
        """
        self._record_plot_call("plot_normality_qq", data, col_names, normality_results)
        self._current_plot_type = "normality_qq"
        self._figure.clear()

        from scipy import stats as sp_stats

        n_vars = min(data.shape[1], 9)
        ncols = min(3, n_vars)
        nrows = (n_vars + ncols - 1) // ncols

        for i in range(n_vars):
            ax = self._figure.add_subplot(nrows, ncols, i + 1)
            col_data = data[:, i]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) < 3:
                ax.text(0.5, 0.5, "n < 3", transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_title(col_names[i] if i < len(col_names) else f"Var {i + 1}", fontsize=9)
                continue

            sp_stats.probplot(valid, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor("#3498DB")
            ax.get_lines()[0].set_markeredgecolor("white")
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[1].set_color("#E74C3C")

            name = col_names[i] if i < len(col_names) else f"Var {i + 1}"
            title = name
            if normality_results and i < len(normality_results):
                nr = normality_results[i]
                sig = "✓ Normal" if nr.is_normal_shapiro else "✗ Non-normal"
                title += f"\nW={nr.shapiro_stat:.3f}, p={nr.shapiro_p:.3f} {sig}"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

        self._figure.suptitle(_("Normality Test (Q-Q Plots)"), fontsize=13, fontweight="bold")
        self._figure.tight_layout(rect=[0, 0, 1, 0.95])
        self._finalise_grid_plot()
        self._canvas.draw()

    def plot_group_comparison(
        self,
        data: np.ndarray,
        groups: list[int] | np.ndarray,
        col_names: list[str],
        test_name: str = "t-test",
        p_values: list[float] | None = None,
    ) -> None:
        """Plot boxplots comparing groups across variables.

        Parameters:
            data: Data matrix (n_samples, n_variables)
            groups: Group assignment per sample
            col_names: Variable names
            test_name: Name of the statistical test (for title)
            p_values: Optional per-variable p-values for annotation
        """
        self._record_plot_call("plot_group_comparison", data, groups, col_names, test_name, p_values)
        self._current_plot_type = "group_comparison"
        self._figure.clear()

        groups_arr = np.asarray(groups)
        unique_groups = sorted(set(groups_arr.tolist()))
        n_vars = min(data.shape[1], 9)
        ncols = min(3, n_vars)
        nrows = (n_vars + ncols - 1) // ncols

        colors = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12", "#9B59B6", "#1ABC9C"]

        for i in range(n_vars):
            ax = self._figure.add_subplot(nrows, ncols, i + 1)
            group_data = []
            group_labels = []
            for gi, g in enumerate(unique_groups):
                mask = groups_arr == g
                vals = data[mask, i]
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    group_data.append(valid)
                    group_labels.append(str(g))

            if not group_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=8)
                continue

            bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True, widths=0.6)
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[j % len(colors)])
                patch.set_alpha(0.7)

            name = col_names[i] if i < len(col_names) else f"Var {i + 1}"
            title = name
            if p_values and i < len(p_values):
                pv = p_values[i]
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
                title += f"\np={pv:.4f} {sig}"
                if pv < 0.05:
                    ax.set_title(title, fontsize=8, fontweight="bold", color="#E74C3C")
                else:
                    ax.set_title(title, fontsize=8, fontweight="bold")
            else:
                ax.set_title(title, fontsize=8, fontweight="bold")

            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        self._figure.suptitle(_("{0} — Group Comparison").format(test_name), fontsize=13, fontweight="bold")
        self._figure.tight_layout(rect=[0, 0, 1, 0.95])
        self._finalise_grid_plot()
        self._canvas.draw()

    def plot_coniss_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        n_zones: int,
        sample_names: list[str] | None = None,
        zone_boundaries: list[int] | None = None,
    ) -> None:
        """Plot CONISS dendrogram with zone boundaries.

        Parameters:
            linkage_matrix: scipy linkage matrix
            n_zones: Number of zones
            sample_names: Optional sample labels
            zone_boundaries: Optional zone boundary indices
        """
        self._record_plot_call("plot_coniss_dendrogram", linkage_matrix, n_zones, sample_names, zone_boundaries)
        self._current_plot_type = "coniss_dendrogram"

        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        ax = self._reset_axes()

        # Use sample names as labels if available
        labels = sample_names if sample_names and len(sample_names) == linkage_matrix.shape[0] + 1 else None

        scipy_dendrogram(
            linkage_matrix,
            ax=ax,
            labels=labels,
            color_threshold=linkage_matrix[-(n_zones - 1), 2] if n_zones > 1 else 0,
            leaf_rotation=90 if labels and len(labels) > 10 else 0,
            leaf_font_size=8,
        )

        # Draw zone boundaries
        if zone_boundaries:
            for zb in zone_boundaries:
                ax.axhline(y=zb, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_title(_("CONISS Zonation ({0} zones)").format(n_zones), fontsize=12, fontweight="bold")
        ax.set_xlabel(_("Samples"), fontsize=10)
        ax.set_ylabel(_("Distance"), fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_scree(
        self, eigenvalues: np.ndarray, explained_var: np.ndarray, cumulative_var: np.ndarray, method: str = "PCA"
    ) -> None:
        """Plot scree diagram with individual and cumulative variance.

        Parameters:
            eigenvalues: Raw eigenvalues
            explained_var: Per-component explained variance (%)
            cumulative_var: Cumulative explained variance (%)
            method: Analysis method name for title
        """
        self._record_plot_call("plot_scree", eigenvalues, explained_var, cumulative_var, method)
        self._current_plot_type = "scree"

        ax1 = self._reset_axes()
        n = len(eigenvalues)
        components = np.arange(1, n + 1)

        # Bar chart: individual variance
        ax1.bar(components, explained_var, color="#3498DB", alpha=0.8, label=_("Individual"))
        ax1.set_xlabel(_("Component"), fontsize=10)
        ax1.set_ylabel(_("Variance Explained (%)"), fontsize=10, color="#3498DB")
        ax1.tick_params(axis="y", labelcolor="#3498DB")
        ax1.set_xticks(components)

        # Line chart: cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(components, cumulative_var, "o-", color="#E74C3C", linewidth=2, markersize=5, label=_("Cumulative"))
        ax2.set_ylabel(_("Cumulative (%)"), fontsize=10, color="#E74C3C")
        ax2.tick_params(axis="y", labelcolor="#E74C3C")
        ax2.set_ylim(0, 105)

        # Kaiser criterion line (eigenvalue > 1 for correlation PCA)
        ax1.axhline(
            y=100.0 / n, color="gray", linestyle=":", linewidth=1, alpha=0.5, label=f"Kaiser ({100.0 / n:.1f}%)"
        )

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

        ax1.set_title(_("{0} Scree Plot").format(method), fontsize=12, fontweight="bold")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_phylo_tree(
        self, tree: Any, trait_values: dict[str, float] | None = None, title: str = "Phylogenetic Tree"
    ) -> None:
        """Plot a phylogenetic tree with optional trait values.

        Parameters:
            tree: PhyloTree or PhyloNode with Newick structure
            trait_values: Optional {taxon_name: value} for coloring tips
            title: Plot title
        """
        # Bug fix: the replot record must carry every argument the method
        # consumes - trait_values drives the tip colours and title the
        # caption, and dropping them made _replot_current() (theme switch,
        # label/ellipse toggles) repaint a different picture.
        self._record_plot_call("plot_phylo_tree", tree, trait_values=trait_values, title=title)
        self._current_plot_type = "phylo_tree"

        ax = self._reset_axes()

        root = tree.root if hasattr(tree, "root") else tree
        if root is None:
            ax.text(0.5, 0.5, "Empty tree", transform=ax.transAxes, ha="center", fontsize=12)
            self._canvas.draw()
            return

        # Layout: compute y positions (leaves evenly spaced) and x positions (depth)
        leaves = root.get_leaves()
        if not leaves:
            ax.text(0.5, 0.5, "No leaves", transform=ax.transAxes, ha="center", fontsize=12)
            self._canvas.draw()
            return

        leaf_y = {id(leaf): i for i, leaf in enumerate(leaves)}

        def assign_y(node):
            if node.is_leaf:
                return leaf_y[id(node)]
            child_ys = [assign_y(c) for c in node.children]
            return sum(child_ys) / len(child_ys) if child_ys else 0

        def get_depth(node, d=0):
            return max((get_depth(c, d + (c.branch_length or 1.0)) for c in node.children), default=d)

        assign_y(root)
        max_depth = get_depth(root)

        # Draw tree
        def draw_node(node, x_start):
            if node.is_leaf:
                y = leaf_y[id(node)]
                ax.plot([x_start, 0], [y, y], color=self.theme_colors()["text"], linewidth=1)
                label = node.name or ""
                color = "#E74C3C" if trait_values and label in trait_values else "#2C3E50"
                ax.text(-0.02 * max_depth, y, f" {label}", va="center", fontsize=8, color=color)
                if trait_values and label in trait_values:
                    val = trait_values[label]
                    ax.plot(0, y, "o", color="#3498DB", markersize=6, alpha=0.7)
                    ax.text(0.02 * max_depth, y, f"{val:.3f}", va="center", fontsize=7, color="#3498DB")
                return y

            child_ys = []
            for child in node.children:
                bl = child.branch_length or 1.0
                child_x = x_start - bl
                cy = draw_node(child, child_x)
                child_ys.append(cy)
                # Horizontal line from parent to child x
                ax.plot([x_start, child_x], [cy, cy], color=self.theme_colors()["text"], linewidth=1)

            # Vertical connector
            if child_ys:
                y_min, y_max = min(child_ys), max(child_ys)
                ax.plot([x_start, x_start], [y_min, y_max], color=self.theme_colors()["text"], linewidth=1)
                mid_y = (y_min + y_max) / 2
                # Support value
                if node.support is not None and node.support > 0:
                    ax.text(x_start, mid_y + 0.3, f"{node.support:.0f}", ha="center", fontsize=6, color="#888")
                return mid_y
            return 0

        draw_node(root, max_depth)

        ax.set_xlim(-0.1 * max_depth, max_depth * 1.3)
        ax.set_ylim(-0.5, len(leaves) - 0.5)
        ax.set_xlabel("Branch length", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_eigenshape_scores(
        self, scores: np.ndarray, explained_var: np.ndarray, specimen_labels: list[str] | None = None
    ) -> None:
        """Plot Eigenshape score scatter plot (ES1 vs ES2).

        Parameters:
            scores: Score matrix (n_specimens, n_components)
            explained_var: Explained variance per component
            specimen_labels: Optional specimen labels
        """
        self._record_plot_call("plot_eigenshape_scores", scores, explained_var, specimen_labels)
        self._current_plot_type = "eigenshape_scores"

        ax = self._reset_axes()
        x = scores[:, 0]
        y = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(x)

        ax.scatter(x, y, c="#9B59B6", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)

        if specimen_labels:
            for i, label in enumerate(specimen_labels):
                ax.annotate(label, (x[i], y[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")

        var1 = explained_var[0] * 100 if len(explained_var) > 0 else 0
        var2 = explained_var[1] * 100 if len(explained_var) > 1 else 0
        ax.set_xlabel(f"ES1 ({var1:.1f}%)", fontsize=10)
        ax.set_ylabel(f"ES2 ({var2:.1f}%)" if scores.shape[1] > 1 else "ES2", fontsize=10)
        ax.set_title(_("Eigenshape Score Plot"), fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, linestyle="--", alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    def plot_markov_heatmap(
        self, transition_matrix: np.ndarray, facies_names: list[str], chi2_stat: float = 0.0, p_value: float = 1.0
    ) -> None:
        """Plot Markov chain transition probability matrix as heatmap.

        Parameters:
            transition_matrix: Raw transition counts (n_states, n_states)
            facies_names: Names for each state/facies
            chi2_stat: Chi-squared statistic
            p_value: P-value for chi-squared test
        """
        self._record_plot_call("plot_markov_heatmap", transition_matrix, facies_names, chi2_stat, p_value)
        self._current_plot_type = "markov_heatmap"

        ax = self._reset_axes()

        # Normalize to probabilities (row-wise)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob_matrix = transition_matrix / row_sums

        n = len(facies_names)
        im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                val = prob_matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(facies_names, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(facies_names, fontsize=9)
        ax.set_xlabel(_("To (next state)"), fontsize=10)
        ax.set_ylabel(_("From (current state)"), fontsize=10)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ax.set_title(
            _("Markov Transition Probabilities\nχ²={0:.1f}, p={1:.4f} {2}").format(chi2_stat, p_value, sig),
            fontsize=12,
            fontweight="bold",
        )

        self._figure.colorbar(im, ax=ax, shrink=0.8, label=_("Probability"))
        self._figure.tight_layout()
        self._canvas.draw()

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

    # ------------------------------------------------------------------
    # Macroevolution plots
    #
    # The macroevolution/ and morpho3d/ engines had no entry point, so the
    # README's "FBD Process", "Cohort Survivorship", "Diversity Dynamics" and
    # "2D/3D GPA" were unreachable from the running application. These plot
    # methods plus the StatisticsController dispatchers are the missing path.
    # ------------------------------------------------------------------

    def plot_cohort_survivorship(self, result: Any) -> None:
        """Foote cohort survivorship per interval, with the confidence band.

        Args:
            result: SurvivorshipResult from macroevolution.cohort.
        """
        self._record_plot_call("plot_cohort_survivorship", result)
        self._current_plot_type = "cohort_survivorship"
        self._ax = self._reset_axes()

        centers, labels = [], []
        for iv in result.intervals:
            centers.append((iv.t_start + iv.t_end) / 2.0)
            labels.append(f"{iv.t_start:.1f}-{iv.t_end:.1f}")

        survival = np.asarray(result.survival_rates, dtype=float)
        self._ax.plot(centers, survival, "o-", color=self.COLORS[0], label=_("Survival probability"))
        if result.confidence_intervals:
            lo = np.array([c[0] for c in result.confidence_intervals], dtype=float)
            hi = np.array([c[1] for c in result.confidence_intervals], dtype=float)
            if lo.size == len(centers):
                self._ax.fill_between(centers, lo, hi, color=self.COLORS[0], alpha=0.18, label=_("Confidence interval"))

        for x, y in zip(centers, survival):
            if np.isfinite(y):
                self._ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=self.theme_colors()["text"],
                )

        self._ax.set_xticks(centers)
        self._ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        self._ax.set_ylim(-0.05, 1.08)
        self._ax.set_xlabel(_("Interval (Ma)"))
        self._ax.set_ylabel(_("Cohort survivorship"))
        self._ax.set_title(_("Foote Cohort Survivorship"))
        self._ax.legend(loc="lower left", fontsize=8)
        self._ax.grid(True, alpha=0.25)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_diversity_dynamics(self, result: Any) -> None:
        """Richness through time with origination / extinction / turnover.

        Args:
            result: DiversityCurve from macroevolution.diversity.
        """
        self._record_plot_call("plot_diversity_dynamics", result)
        self._current_plot_type = "diversity_dynamics"
        self._ax = self._reset_axes()

        t = np.asarray(result.times, dtype=float)
        richness = np.asarray(result.richness, dtype=float)
        ax2 = self._ax.twinx()
        ax2.patch.set_visible(False)

        self._ax.plot(t, richness, "s-", color=self.COLORS[0], label=_("Richness"))
        self._ax.set_xlabel(_("Time"))
        self._ax.set_ylabel(_("Taxonomic richness"), color=self.COLORS[0])
        self._ax.tick_params(axis="y", labelcolor=self.COLORS[0])

        for values, color, name in (
            (result.origination_rates, self.COLORS[2], _("Origination rate")),
            (result.extinction_rates, self.COLORS[3], _("Extinction rate")),
            (result.turnover_rate, self.COLORS[5], _("Turnover")),
        ):
            v = np.asarray(values, dtype=float)
            if v.size == t.size and np.any(np.isfinite(v)):
                ax2.plot(t, v, "^-", color=color, label=name, linewidth=1.1, alpha=0.85)

        ax2.set_ylabel(_("Per-capita rate"), color=self.theme_colors()["text"])
        self._ax.set_title(_("Diversity Dynamics"))

        h1, l1 = self._ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        self._ax.grid(True, alpha=0.25)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_survival_curve(self, result: Any) -> None:
        """Kaplan-Meier step function with confidence band and risk table.

        Args:
            result: SurvivalResult from macroevolution.survival.
        """
        self._record_plot_call("plot_survival_curve", result)
        self._current_plot_type = "survival"
        self._ax = self._reset_axes()

        t = np.asarray(result.times_full, dtype=float)
        s = np.asarray(result.survival_prob, dtype=float)
        self._ax.step(t, s, where="post", color=self.COLORS[0], label=_("Survival function"))

        if result.lower_ci is not None and result.upper_ci is not None:
            lo = np.asarray(result.lower_ci, dtype=float)
            hi = np.asarray(result.upper_ci, dtype=float)
            if lo.size == t.size:
                self._ax.fill_between(
                    t, lo, hi, step="post", color=self.COLORS[0], alpha=0.18, label=_("Confidence interval")
                )

        if result.median_survival is not None and np.isfinite(result.median_survival):
            self._ax.axvline(
                result.median_survival,
                color=self.COLORS[3],
                linestyle="--",
                linewidth=1,
                label=f"{_('Median')} = {result.median_survival:.2f}",
            )

        if result.n_at_risk is not None:
            risk = np.asarray(result.n_at_risk, dtype=float)
            if risk.size == t.size:
                ax2 = self._ax.twinx()
                ax2.patch.set_visible(False)
                ax2.step(t, risk, where="post", color=self.COLORS[7], linewidth=1, linestyle=":", label=_("At risk"))
                ax2.set_ylabel(_("Number at risk"), color=self.COLORS[7])
                ax2.tick_params(axis="y", labelcolor=self.COLORS[7])
                ax2.set_ylim(0, max(1.0, float(np.nanmax(risk)) * 1.15))

        n_events = int(np.sum(result.events)) if result.events is not None else 0
        self._ax.set_xlabel(_("Time"))
        self._ax.set_ylabel(_("Survival probability"))
        self._ax.set_ylim(-0.02, 1.05)
        self._ax.set_title(f"{_('Kaplan-Meier')}  (n={int(t.size)}, events={n_events})")
        self._ax.legend(loc="lower left", fontsize=8)
        self._ax.grid(True, alpha=0.25)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_fbd_diversity(self, replicates: list[Any]) -> None:
        """Standing-diversity trajectories from FBD replicates.

        Args:
            replicates: list of FBDSimulationResult.
        """
        self._record_plot_call("plot_fbd_diversity", replicates)
        self._current_plot_type = "fbd"
        self._ax = self._reset_axes()

        drew = 0
        for k, rep in enumerate(replicates):
            curve = getattr(rep, "diversity_curve", None)
            if curve is None:
                continue
            arr = np.asarray(curve, dtype=float)
            if arr.ndim != 1 or arr.size < 2:
                continue
            xs = np.linspace(0.0, 1.0, arr.size)
            self._ax.plot(
                xs,
                arr,
                color=self.COLORS[k % len(self.COLORS)],
                alpha=0.8 if len(replicates) < 6 else 0.5,
                linewidth=1.2,
            )
            drew += 1

        self._ax.set_xlabel(_("Time (normalised)"))
        self._ax.set_ylabel(_("Standing diversity"))
        self._ax.set_title(f"{_('FBD Simulation')}  ({drew} replicate(s))")
        self._ax.grid(True, alpha=0.25)
        self._figure.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # 3-D morphometrics plots
    # ------------------------------------------------------------------

    def plot_gpa3d_aligned(
        self, aligned: np.ndarray, consensus: np.ndarray, specimen_labels: list[str] | None = None, title: str = ""
    ) -> None:
        """GPA alignment of 3-D landmark configurations.

        A 2-D canvas cannot show a 3-D scatter honestly, so the first two
        principal dimensions carry the layout and the third drives the
        per-specimen colour.

        Args:
            aligned: (n_specimens, n_landmarks, 3) aligned configurations.
            consensus: (n_landmarks, 3) consensus shape.
            specimen_labels: Optional per-specimen labels.
            title: Optional title override.
        """
        self._record_plot_call("plot_gpa3d_aligned", aligned)
        self._current_plot_type = "gpa3d"
        self._ax = self._reset_axes()

        arr = np.asarray(aligned, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                "plot_gpa3d_aligned expects (n_specimens, n_landmarks, 3); "
                f"got shape {arr.shape}. Use plot_gpa_aligned for 2-D data."
            )

        cmap = plt.get_cmap("viridis")
        lo3 = float(np.min(arr[:, 0, 2]))
        span = float(np.max(arr[:, 0, 2]) - lo3) or 1.0
        for i in range(arr.shape[0]):
            frac = (arr[i, 0, 2] - lo3) / span
            xy = arr[i, :, :2]
            self._ax.plot(xy[:, 0], xy[:, 1], "o-", markersize=3, linewidth=0.8, color=cmap(frac), alpha=0.85)
            if specimen_labels and i < len(specimen_labels):
                self._ax.annotate(
                    specimen_labels[i], (xy[0, 0], xy[0, 1]), fontsize=6, color=self.theme_colors()["text"]
                )

        cons = np.asarray(consensus, dtype=float)
        have_cons = cons.ndim == 2 and cons.shape == (arr.shape[1], arr.shape[2])
        if have_cons:
            self._ax.plot(
                cons[:, 0], cons[:, 1], "s--", color="#E74C3C", markersize=4, linewidth=1.6, label=_("Consensus")
            )

        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.set_xlabel(_("Dim 1"))
        self._ax.set_ylabel(_("Dim 2"))
        self._ax.set_title(title or _("3-D GPA Alignment"))
        if have_cons:
            self._ax.legend(loc="best", fontsize=8)
        self._ax.grid(True, alpha=0.2)
        self._figure.tight_layout()
        self._canvas.draw()
