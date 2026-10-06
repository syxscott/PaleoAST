# views/ui_imputation_dialog.py
"""
Missing Value Imputation Dialog for PaleoAST

Provides interactive UI for handling missing values in data matrices.

Author: PaleoAST Development Team
version: 1.2.0
"""

import logging
from typing import Any

import numpy as np
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import get_palette
from config.i18n import _
from views.ui_dialogs import BaseAnalysisDialog

logger = logging.getLogger(__name__)


class MissingValueReportWidget(QWidget):
    """Widget displaying missing value analysis results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary label
        self.summary_label = QLabel()
        self.summary_label.setFont(QFont("Consolas", 10))
        layout.addWidget(self.summary_label)

        # NaN distribution heatmap representation
        self.distribution_label = QLabel()
        self.distribution_label.setFont(QFont("Consolas", 9))
        self.distribution_label.setWordWrap(True)
        layout.addWidget(self.distribution_label)

        # Statistics
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        layout.addWidget(self.stats_text)

    def set_report(
        self,
        total_nan: int,
        nan_proportion: float,
        rows_with_nan: int,
        cols_with_nan: int,
        nan_by_row: np.ndarray,
        nan_by_col: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> None:
        """Update the report display."""
        self.summary_label.setText(
            f"<b>{_('Missing-value summary')}:</b> {total_nan} NaN ({nan_proportion * 100:.1f}%)"
        )

        # Show rows and columns with NaN
        self.distribution_label.setText(
            _("Rows with NaN: {0}/{1} | Columns with NaN: {2}/{3}").format(rows_with_nan, n_rows, cols_with_nan, n_cols)
        )

        # Statistics
        stats_lines = [
            "=" * 40,
            _("NaN counts by row (first 10):"),
            str(nan_by_row[:10].tolist()),
            "",
            _("NaN counts by column:"),
            str(nan_by_col.tolist()),
        ]
        self.stats_text.setText("\n".join(stats_lines))


class ImputationConfigWidget(QWidget):
    """Widget for configuring imputation options."""

    # Method indices
    METHOD_MEAN = 0
    METHOD_MEDIAN = 1
    METHOD_KNN = 2
    METHOD_REMOVE_ROWS = 3
    METHOD_REMOVE_COLUMNS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Method selection
        method_group = QGroupBox(_("Imputation method"))
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                _("Column-mean imputation (Mean)"),
                _("Column-median imputation (Median)"),
                _("K-Nearest-Neighbour imputation (KNN)"),
                _("Remove rows containing NaN"),
                _("Remove columns containing NaN"),
            ]
        )
        method_layout.addWidget(self.method_combo)

        # KNN options
        self.knn_options = QWidget()
        knn_layout = QHBoxLayout(self.knn_options)
        knn_layout.addWidget(QLabel(_("K value:")))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(5)
        knn_layout.addWidget(self.k_spin)
        knn_layout.addStretch()
        self.knn_options.setVisible(False)
        method_layout.addWidget(self.knn_options)

        layout.addWidget(method_group)

        # Preview button
        self.preview_btn = QPushButton(_("Preview processed result"))
        layout.addWidget(self.preview_btn)

        layout.addStretch()

        # Connect signals
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

    def _on_method_changed(self, index: int) -> None:
        """Show/hide KNN options based on method selection."""
        self.knn_options.setVisible(index == self.METHOD_KNN)

    def get_method(self) -> str:
        """Get selected imputation method."""
        methods = ["mean", "median", "knn", "remove_rows", "remove_columns"]
        return methods[self.method_combo.currentIndex()]

    def get_k(self) -> int:
        """Get K value for KNN."""
        return self.k_spin.value()


class ImputationDialog(BaseAnalysisDialog):
    """
    Dialog for analyzing and handling missing values.

    Provides:
        - Missing value analysis report
        - Multiple imputation strategies
        - Preview before applying
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        nan_count: int = 0,
        rows_with_nan: int = 0,
        cols_with_nan: int = 0,
        nan_by_row: np.ndarray | None = None,
        nan_by_col: np.ndarray | None = None,
        n_rows: int = 0,
        n_cols: int = 0,
        nan_proportion: float = 0.0,
    ) -> None:
        """
        Initialize the imputation dialog.

        Parameters:
            parent: Parent widget
            nan_count: Total number of NaN values
            rows_with_nan: Number of rows containing NaN
            cols_with_nan: Number of columns containing NaN
            nan_by_row: NaN count per row
            nan_by_col: NaN count per column
            n_rows: Total number of rows
            n_cols: Total number of columns
            nan_proportion: Proportion of data that is NaN
        """
        super().__init__(_("Missing-Value Centre"), parent)

        self.nan_count = nan_count
        self.rows_with_nan = rows_with_nan
        self.cols_with_nan = cols_with_nan
        self.nan_by_row = nan_by_row if nan_by_row is not None else np.array([])
        self.nan_by_col = nan_by_col if nan_by_col is not None else np.array([])
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nan_proportion = nan_proportion
        self._is_dark_theme = False

        self._setup_parameters()
        self._update_report()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self.result_label.setStyleSheet(f"color: {get_palette(is_dark).text_secondary}; padding: 8px;")

    def _setup_parameters(self) -> None:
        """Setup the dialog UI."""
        # Header
        header = QLabel(
            _("<h2>Missing-Value Centre</h2><p>Missing values were detected in the data — choose a strategy.</p>")
        )
        header.setWordWrap(True)
        self.layout().addWidget(header)

        # Report widget
        self.report_widget = MissingValueReportWidget()
        self.layout().addWidget(self.report_widget)

        # Config widget
        self.config_widget = ImputationConfigWidget()
        self.layout().addWidget(self.config_widget)

        # Result preview
        result_group = QGroupBox(_("Processing preview"))
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel(_('Press "Preview processed result" to view the projected data.'))
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #666; padding: 8px;")
        result_layout.addWidget(self.result_label)

        self.layout().addWidget(result_group)

        # Buttons
        self.preview_btn = self.config_widget.preview_btn
        self.preview_btn.clicked.connect(self._on_preview)

    def _update_report(self) -> None:
        """Update the missing value report display."""
        self.report_widget.set_report(
            total_nan=self.nan_count,
            nan_proportion=self.nan_proportion,
            rows_with_nan=self.rows_with_nan,
            cols_with_nan=self.cols_with_nan,
            nan_by_row=self.nan_by_row,
            nan_by_col=self.nan_by_col,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )

    def _on_preview(self) -> None:
        """Handle preview button click."""
        method = self.config_widget.get_method()
        k = self.config_widget.get_k()

        method_names = {
            "mean": _("Mean imputation"),
            "median": _("Median imputation"),
            "knn": _("KNN imputation (k={0})").format(k),
            "remove_rows": _("Remove rows"),
            "remove_columns": _("Remove columns"),
        }

        # Generate impact description
        if method == "remove_rows":
            remaining_rows = self.n_rows - self.rows_with_nan
            impact = _("Will remove {0} rows, leaving {1}").format(self.rows_with_nan, remaining_rows)
            preview_note = f"\n\n<i>{_('Preview: row count will change from')}: {self.n_rows} → {remaining_rows}</i>"
        elif method == "remove_columns":
            remaining_cols = self.n_cols - self.cols_with_nan
            impact = _("Will remove {0} columns, leaving {1}").format(self.cols_with_nan, remaining_cols)
            preview_note = f"\n\n<i>{_('Preview: column count will change from')}: {self.n_cols} → {remaining_cols}</i>"
        else:
            impact = _("Will impute {0} NaN values").format(self.nan_count)
            # Show sample of rows with NaN
            rows_with_nan_indices = np.where(self.nan_by_row > 0)[0]
            if len(rows_with_nan_indices) > 0:
                sample_rows = rows_with_nan_indices[:3]  # Show first 3
                preview_note = f"\n\n<i>{_('Preview: first 3 rows containing NaN')}: {sample_rows.tolist()}...</i>"
            else:
                preview_note = ""

        self.result_label.setText(
            f"<b>{_('Selected method')}:</b> {method_names.get(method, method)}<br/>"
            f"<b>{_('Impact')}:</b> {impact}{preview_note}"
        )

    def get_parameters(self) -> dict[str, Any]:
        """Get imputation parameters."""
        self._parameters = {
            "method": self.config_widget.get_method(),
            "k": self.config_widget.get_k(),
        }
        return self._parameters

    def _get_help_text(self) -> str:
        """Return help text for the dialog."""
        return _(
            """
<h2>Missing-Value Methods</h2>

<h3>Column-mean imputation (Mean)</h3>
<p>Replace each NaN with the mean of the non-NaN values in its column. Fast, but reduces the column variance.</p>

<h3>Column-median imputation (Median)</h3>
<p>Replace each NaN with the median of the non-NaN values in its column. More robust against outliers.</p>

<h3>K-Nearest-Neighbour imputation (KNN)</h3>
<p>For every NaN cell, find the K most similar samples and use their values to fill it in. Captures local structure, but is slower.</p>

<h3>Remove rows</h3>
<p>Drop every row that contains any NaN. Reduces the sample count but preserves data integrity.</p>

<h3>Remove columns</h3>
<p>Drop every column that contains any NaN. May discard important features.</p>
"""
        )
