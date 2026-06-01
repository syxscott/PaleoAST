# views/ui_imputation_dialog.py
"""
Missing Value Imputation Dialog for PaleoAST

Provides interactive UI for handling missing values in data matrices.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
        n_cols: int
    ) -> None:
        """Update the report display."""
        self.summary_label.setText(
            f"<b>缺失值统计:</b> {total_nan} 个 NaN ({nan_proportion*100:.1f}%)"
        )

        # Show rows and columns with NaN
        self.distribution_label.setText(
            f"含 NaN 的行: {rows_with_nan}/{n_rows} | 含 NaN 的列: {cols_with_nan}/{n_cols}"
        )

        # Statistics
        stats_lines = [
            "=" * 40,
            "逐行 NaN 数量 (前10行):",
            str(nan_by_row[:10].tolist()),
            "",
            "逐列 NaN 数量:",
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
        method_group = QGroupBox(_("填充方法"))
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            _("列均值填充 (Mean)"),
            _("列中位数填充 (Median)"),
            _("K近邻填充 (KNN)"),
            _("删除含NaN的行"),
            _("删除含NaN的列"),
        ])
        method_layout.addWidget(self.method_combo)

        # KNN options
        self.knn_options = QWidget()
        knn_layout = QHBoxLayout(self.knn_options)
        knn_layout.addWidget(QLabel(_("K 值:")))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(5)
        knn_layout.addWidget(self.k_spin)
        knn_layout.addStretch()
        self.knn_options.setVisible(False)
        method_layout.addWidget(self.knn_options)

        layout.addWidget(method_group)

        # Preview button
        self.preview_btn = QPushButton(_("预览处理结果"))
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
        nan_proportion: float = 0.0
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
        super().__init__(_("缺失值处理中心"), parent)

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
        self.result_label.setStyleSheet(
            f"color: {get_palette(is_dark).text_secondary}; padding: 8px;"
        )

    def _setup_parameters(self) -> None:
        """Setup the dialog UI."""
        # Header
        header = QLabel(
            _("<h2>缺失值处理中心</h2>"
              "<p>检测到数据中存在缺失值，请选择处理方式。</p>")
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
        result_group = QGroupBox(_("处理预览"))
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel(_("点击\"预览处理结果\"查看处理后的数据预览"))
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
            n_cols=self.n_cols
        )

    def _on_preview(self) -> None:
        """Handle preview button click."""
        method = self.config_widget.get_method()
        k = self.config_widget.get_k()

        method_names = {
            "mean": "均值填充",
            "median": "中位数填充",
            "knn": f"KNN填充 (k={k})",
            "remove_rows": "删除行",
            "remove_columns": "删除列"
        }

        # Generate impact description
        if method == "remove_rows":
            remaining_rows = self.n_rows - self.rows_with_nan
            impact = f"将删除 {self.rows_with_nan} 行，剩余 {remaining_rows} 行"
            preview_note = f"\n\n<i>预览: 数据将从 {self.n_rows} 行变为 {remaining_rows} 行</i>"
        elif method == "remove_columns":
            remaining_cols = self.n_cols - self.cols_with_nan
            impact = f"将删除 {self.cols_with_nan} 列，剩余 {remaining_cols} 列"
            preview_note = f"\n\n<i>预览: 数据将从 {self.n_cols} 列变为 {remaining_cols} 列</i>"
        else:
            impact = f"将填充 {self.nan_count} 个 NaN 值"
            # Show sample of rows with NaN
            rows_with_nan_indices = np.where(self.nan_by_row > 0)[0]
            if len(rows_with_nan_indices) > 0:
                sample_rows = rows_with_nan_indices[:3]  # Show first 3
                preview_note = f"\n\n<i>预览: 前3个含NaN的行: {sample_rows.tolist()}...</i>"
            else:
                preview_note = ""

        self.result_label.setText(
            f"<b>选择的方法:</b> {method_names.get(method, method)}\n"
            f"<b>影响:</b> {impact}{preview_note}"
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
        return _("""
<h2>缺失值处理方法</h2>

<h3>列均值填充 (Mean)</h3>
<p>用每列的非NaN值的均值填充该列的NaN。简单快速，但会降低数据方差。</p>

<h3>列中位数填充 (Median)</h3>
<p>用每列的非NaN值的中位数填充。对异常值更稳健。</p>

<h3>K近邻填充 (KNN)</h3>
<p>对于每个NaN，找到与该样本最相似的K个邻居，利用邻居的值进行填充。
考虑数据的局部结构，更加精确但计算较慢。</p>

<h3>删除行</h3>
<p>直接删除包含任何NaN的行。会减少样本数量，但保持数据完整性。</p>

<h3>删除列</h3>
<p>直接删除包含任何NaN的列。可能丢失重要特征。</p>
        """)
