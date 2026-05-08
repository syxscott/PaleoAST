# =============================================================================
# FILE: views/ui_dialogs.py
# =============================================================================
"""
Analysis Configuration Dialogs for PaleoAST

This module implements comprehensive configuration dialogs for all statistical
analyses, with detailed parameter controls and mathematical descriptions.

Design Patterns:
    - Factory Pattern: Dialog factory for different analysis types
    - Strategy Pattern: Different parameter strategies per analysis
    - Observer Pattern: Dialogs observe state for data preview

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QGroupBox, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QTextEdit, QScrollArea, QTabWidget, QLineEdit, QRadioButton,
    QButtonGroup, QProgressBar, QFrame, QSlider, QTreeWidget,
    QTreeWidgetItem, QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QSize
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette
)

from config.i18n import _


class BaseAnalysisDialog(QDialog):
    """
    Base class for all analysis configuration dialogs.
    
    Provides common functionality:
        - Parameter storage
        - Data validation
        - Preview area
        - Help text display
    """
    
    parametersChanged = pyqtSignal(dict)
    
    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.BaseAnalysisDialog")
        self._logger.info(f"Dialog opened: '{title}'")

        self.setWindowTitle(title)
        self.setMinimumSize(600, 500)
        self.setModal(True)

        # Parameter storage
        self._parameters: Dict[str, Any] = {}

        # Setup UI
        self._setup_ui()

        # Apply styling
        self._apply_stylesheet()
    
    def _setup_ui(self) -> None:
        """Setup base UI structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title label
        self._title_label = QLabel(self.windowTitle())
        title_font = QFont("Arial", 14, QFont.Weight.Bold)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)
        
        # Main content area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        content_widget = QWidget()
        self._content_layout = QVBoxLayout(content_widget)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._content_layout.setContentsMargins(0, 8, 0, 8)
        
        self._scroll_area.setWidget(content_widget)
        layout.addWidget(self._scroll_area, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._help_button = QPushButton(_("Help"))
        self._help_button.clicked.connect(self._show_help)
        button_layout.addWidget(self._help_button)

        self._cancel_button = QPushButton(_("Cancel"))
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)

        self._run_button = QPushButton(_("Run"))
        self._run_button.clicked.connect(self._on_run)
        self._run_button.setDefault(True)
        button_layout.addWidget(self._run_button)
        
        layout.addLayout(button_layout)
    
    def _apply_stylesheet(self) -> None:
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A2E;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 12px;
            }
            QGroupBox {
                color: #3498DB;
                font-weight: bold;
                border: 1px solid #34495E;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
            }
            QPushButton {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #2C3E50;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3D566E;
                border: 1px solid #3498DB;
            }
            QPushButton:pressed {
                background-color: #2C3E50;
            }
            QPushButton[default="true"] {
                background-color: #3498DB;
                border: 1px solid #2980B9;
            }
            QPushButton[default="true"]:hover {
                background-color: #2980B9;
            }
            QComboBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 150px;
            }
            QComboBox:hover {
                border: 1px solid #3498DB;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #95A5A6;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 6px;
                min-width: 80px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #3498DB;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                border: none;
                background-color: transparent;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                border: none;
                background-color: transparent;
            }
            QCheckBox {
                color: #ECF0F1;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #34495E;
                border-radius: 3px;
                background-color: #2C3E50;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border-color: #2980B9;
            }
            QCheckBox::indicator:hover {
                border-color: #3498DB;
            }
            QRadioButton {
                color: #ECF0F1;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #34495E;
                border-radius: 9px;
                background-color: #2C3E50;
            }
            QRadioButton::indicator:checked {
                background-color: #3498DB;
                border-color: #2980B9;
            }
            QTextEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 8px;
            }
            QListWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
            }
            QListWidget::item:selected {
                background-color: #3498DB;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #34495E;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                background-color: #3498DB;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background-color: #2980B9;
            }
        """)
    
    def _on_run(self) -> None:
        """Validate parameters and accept dialog."""
        if self._validate_parameters():
            params = self.get_parameters()
            self._logger.info(f"Dialog accepted with parameters: {params}")
            self.accept()
    
    def _validate_parameters(self) -> bool:
        """Validate parameters before running. Override in subclasses."""
        return True
    
    def _show_help(self) -> None:
        """Show help dialog with mathematical description."""
        help_text = self._get_help_text()
        
        help_dialog = QMessageBox(self)
        help_dialog.setWindowTitle(_("{0} - Help").format(self.windowTitle()))
        help_dialog.setText(help_text)
        help_dialog.setIcon(QMessageBox.Icon.Information)
        help_dialog.exec()
    
    def _get_help_text(self) -> str:
        """Get help text for the analysis. Override in subclasses."""
        return _("No help available for this analysis.")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def add_parameter_group(self, title: str) -> QGroupBox:
        """Add a parameter group box."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        group.setLayout(layout)
        self._content_layout.addWidget(group)
        return group
    
    def add_form_group(
        self,
        title: str,
        labels: List[str],
        widgets: List[QWidget]
    ) -> QGroupBox:
        """Add a form-style parameter group."""
        group = QGroupBox(title)
        layout = QFormLayout(group)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        for label, widget in zip(labels, widgets):
            layout.addRow(label, widget)
        
        group.setLayout(layout)
        self._content_layout.addWidget(group)
        return group


class PCADialog(BaseAnalysisDialog):
    """
    Principal Component Analysis Configuration Dialog.
    
    Mathematical Background:
        PCA finds orthogonal axes that maximize variance:
        
        Given data matrix X ∈ ℝ^(n×p):
        
        1. Center data: Z = X - μ, where μ_j = (1/n) Σᵢ x_ij
        
        2. Compute covariance: C = (1/(n-1)) Z^T Z ∈ ℝ^(p×p)
        
        3. Eigendecomposition: C v_j = λ_j v_j
        
        4. Project: PC_scores = Z @ V
        
        Variance explained by PC_j: r²_j = λ_j / Σλ_i
    
    Parameters:
        n_components: Number of principal components (default: 2)
        method: Correlation vs. covariance matrix
        scaling: Standardize variables before PCA
        rotation: Varimax rotation option
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(_("Principal Component Analysis"), parent)
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup PCA-specific parameters."""
        # Method selection
        method_group = self.add_parameter_group(_("Analysis Method"))
        method_layout = QVBoxLayout(method_group)
        
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            _("Correlation Matrix (Standardize)"),
            _("Covariance Matrix (Center only)"),
            _("SPCS (Specialized)")
        ])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(QLabel(_("Similarity Matrix:")))
        method_layout.addWidget(self._method_combo)
        
        # Number of components
        components_group = self.add_parameter_group(_("Components"))
        components_layout = QVBoxLayout(components_group)
        
        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(2, 100)
        self._n_components_spin.setValue(3)
        self._n_components_spin.setPrefix(_("Number of components: "))
        
        self._min_variance_spin = QDoubleSpinBox()
        self._min_variance_spin.setRange(0, 100)
        self._min_variance_spin.setValue(5.0)
        self._min_variance_spin.setSuffix(" %")
        self._min_variance_spin.setPrefix(_("Minimum variance: "))
        
        components_layout.addWidget(self._n_components_spin)
        components_layout.addWidget(self._min_variance_spin)
        
        # Display options
        display_group = self.add_parameter_group(_("Display Options"))
        display_layout = QVBoxLayout(display_group)
        
        self._show_loadings_check = QCheckBox(_("Show loadings table"))
        self._show_loadings_check.setChecked(True)
        display_layout.addWidget(self._show_loadings_check)
        
        self._show_scores_check = QCheckBox(_("Show scores table"))
        self._show_scores_check.setChecked(True)
        display_layout.addWidget(self._show_scores_check)
        
        self._show_scree_check = QCheckBox(_("Show scree plot"))
        self._show_scree_check.setChecked(True)
        display_layout.addWidget(self._show_scree_check)
        
        self._show_biplot_check = QCheckBox(_("Show biplot"))
        display_layout.addWidget(self._show_biplot_check)
        
        # Biplot options (enabled when biplot is checked)
        biplot_layout = QHBoxLayout()
        biplot_layout.addWidget(QLabel(_("Scaling factor:")))
        self._biplot_scale_spin = QDoubleSpinBox()
        self._biplot_scale_spin.setRange(0.1, 10.0)
        self._biplot_scale_spin.setValue(1.0)
        self._biplot_scale_spin.setSingleStep(0.1)
        biplot_layout.addWidget(self._biplot_scale_spin)
        biplot_layout.addStretch()
        display_layout.addLayout(biplot_layout)
        
        # Advanced options
        advanced_group = self.add_parameter_group(_("Advanced Options"))
        advanced_layout = QVBoxLayout(advanced_group)
        
        self._use_correlation_check = QCheckBox(_("Use correlation matrix (Z-score standardization)"))
        self._use_correlation_check.setChecked(True)
        advanced_layout.addWidget(self._use_correlation_check)
        
        self._impute_missing_check = QCheckBox(_("Impute missing values (pairwise deletion)"))
        advanced_layout.addWidget(self._impute_missing_check)
        
        self._parallel_check = QCheckBox(_("Use parallel computation"))
        self._parallel_check.setChecked(True)
        advanced_layout.addWidget(self._parallel_check)
    
    def _on_method_changed(self, index: int) -> None:
        """Handle method selection change."""
        if index == 0:  # Correlation
            self._use_correlation_check.setChecked(True)
        elif index == 1:  # Covariance
            self._use_correlation_check.setChecked(False)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get PCA parameters."""
        self._parameters = {
            'n_components': self._n_components_spin.value(),
            'method': 'correlation' if self._use_correlation_check.isChecked() else 'covariance',
            'min_variance': self._min_variance_spin.value() / 100.0,
            'show_loadings': self._show_loadings_check.isChecked(),
            'show_scores': self._show_scores_check.isChecked(),
            'show_scree': self._show_scree_check.isChecked(),
            'show_biplot': self._show_biplot_check.isChecked(),
            'biplot_scale': self._biplot_scale_spin.value(),
            'impute_missing': self._impute_missing_check.isChecked(),
            'parallel': self._parallel_check.isChecked(),
        }
        return self._parameters
    
    def _get_help_text(self) -> str:
        return f"""
<h2>{_("Principal Component Analysis (PCA)")}</h2>

<p>{_("PCA is a dimension reduction technique that finds orthogonal axes (maximum variance directions) in multidimensional data.")}</p>

<h3>{_("Mathematical Formulation:")}</h3>

<p>{_("Given data matrix")} X ∈ ℝ<sup>n×p</sup>:</p>

<ol>
<li><b>{_("Centering")}:</b> Z = X - μ, {_("where")} μ<sub>j</sub> = (1/n) Σ<sub>i</sub> x<sub>ij</sub></li>
<li><b>{_("Covariance")}:</b> C = (1/(n-1)) Z<sup>T</sup> Z</li>
<li><b>{_("Eigendecomposition")}:</b> C v<sub>j</sub> = λ<sub>j</sub> v<sub>j</sub></li>
<li><b>{_("Project")}:</b> PC_scores = Z @ [v<sub>1</sub>, v<sub>2</sub>, ...]</li>
</ol>

<h3>{_("Parameters:")}</h3>
<ul>
<li><b>{_("Correlation vs. Covariance: Use correlation for standardized data")}</b></li>
<li><b>{_("Components: Number of PCs to retain")}</b></li>
<li><b>{_("Minimum variance: Eigenvalue threshold")}</b></li>
</ul>

<h3>{_("Interpretation:")}</h3>
<p>{_("PC1 captures the direction of maximum variance, PC2 the second most, etc.")}</p>
        """


class PCoADialog(BaseAnalysisDialog):
    """
    Principal Coordinate Analysis Configuration Dialog.
    
    Mathematical Background:
        PCoA is a metric MDS technique that finds coordinates
        that best represent a given distance matrix.
        
        Given distance matrix D ∈ ℝ^(n×n):
        
        1. Square distances: D²
        
        2. Gower centering: B = -½ J D² J
           where J = I - (1/n)11^T
        
        3. Eigen-decomposition: B = U Λ U^T
        
        4. Coordinates: X = U Λ^(1/2)
        
        For non-Euclidean distances, negative eigenvalues
        may occur - these can be set aside or corrected.
    
    Parameters:
        metric: Distance metric (Bray-Curtis, Jaccard, Euclidean, etc.)
        n_components: Number of coordinates (default: 2)
        correction: Negative eigenvalue correction method
    """
    
    DISTANCE_METRICS = [
        "Bray-Curtis",
        "Jaccard",
        "Euclidean",
        "Manhattan",
        "Canberra",
        "Sorensen",
        "Kulczynski",
        "Hamming"
    ]
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(_("Principal Coordinate Analysis"), parent)
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup PCoA-specific parameters."""
        # Distance metric
        metric_group = self.add_parameter_group(_("Distance Metric"))
        metric_layout = QVBoxLayout(metric_group)
        
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self.DISTANCE_METRICS)
        self._metric_combo.setCurrentText("Bray-Curtis")
        metric_layout.addWidget(QLabel(_("Distance measure:")))
        metric_layout.addWidget(self._metric_combo)
        
        metric_help = QLabel(
            "Bray-Curtis: Good for abundance data\n"
            "Jaccard: Presence/absence data\n"
            "Euclidean: Continuous variables"
        )
        metric_help.setStyleSheet("color: #95A5A6; font-size: 10px;")
        metric_layout.addWidget(metric_help)
        
        # Number of components
        components_group = self.add_parameter_group(_("Coordinates"))
        components_layout = QVBoxLayout(components_group)
        
        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(2, 50)
        self._n_components_spin.setValue(3)
        self._n_components_spin.setPrefix(_("Number of coordinates: "))
        components_layout.addWidget(self._n_components_spin)
        
        # Eigenvalue correction
        correction_group = self.add_parameter_group(_("Negative Eigenvalue Handling"))
        correction_layout = QVBoxLayout(correction_group)
        
        self._correction_group = QButtonGroup()
        
        self._correction_none = QRadioButton(_("Keep all eigenvalues (may produce complex coordinates)"))
        self._correction_none.setChecked(True)
        self._correction_group.addButton(self._correction_none, 0)
        correction_layout.addWidget(self._correction_none)
        
        self._correction_wc = QRadioButton(_("Wickoff correction (add constant to squared distances)"))
        self._correction_group.addButton(self._correction_wc, 1)
        correction_layout.addWidget(self._correction_wc)
        
        self._correction_torg = QRadioButton(_("Torgerson correction (approximate Euclidean)"))
        self._correction_group.addButton(self._correction_torg, 2)
        correction_layout.addWidget(self._correction_torg)
        
        self._correction_majorization = QRadioButton(_("Classical MDS with majorization"))
        self._correction_group.addButton(self._correction_majorization, 3)
        correction_layout.addWidget(self._correction_majorization)
        
        # Display options
        display_group = self.add_parameter_group(_("Display"))
        display_layout = QVBoxLayout(display_group)

        self._show_eigenvalues_check = QCheckBox(_("Show eigenvalues"))
        self._show_eigenvalues_check.setChecked(True)
        display_layout.addWidget(self._show_eigenvalues_check)
        
        self._show_vectors_check = QCheckBox(_("Show distance vectors"))
        display_layout.addWidget(self._show_vectors_check)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get PCoA parameters."""
        corrections = ['none', 'wickoff', 'torgerson', 'majorization']
        
        self._parameters = {
            'metric': self._metric_combo.currentText().lower().replace('-', '_'),
            'n_components': self._n_components_spin.value(),
            'correction': corrections[self._correction_group.checkedId()],
            'show_eigenvalues': self._show_eigenvalues_check.isChecked(),
            'show_vectors': self._show_vectors_check.isChecked(),
        }
        return self._parameters
    
    def _get_help_text(self) -> str:
        return f"""
<h2>{_("Principal Coordinate Analysis (PCoA)")}</h2>

<p>{_("PCoA (also called Classical MDS) finds coordinates that represent a given distance matrix as accurately as possible in lower dimensions.")}</p>

<h3>{_("Mathematical Formulation:")}</h3>

<p>{_("Given distance matrix")} D ∈ ℝ<sup>n×n</sup>:</p>

<ol>
<li><b>{_("Square")}:</b> D² ({_("element-wise")})</li>
<li><b>{_("Gower centering")}:</b> B = -½ J D² J, {_("where")} J = I - (1/n)11<sup>T</sup></li>
<li><b>{_("Eigen-decomposition")}:</b> B = U Λ U<sup>T</sup></li>
<li><b>{_("Coordinates")}:</b> X = U Λ<sup>1/2</sup></li>
</ol>

<h3>{_("Negative Eigenvalues:")}</h3>
<p>{_("Non-Euclidean distances may produce negative eigenvalues.")}</p>
        """


class NMDSOptionsDialog(BaseAnalysisDialog):
    """
    Non-metric Multidimensional Scaling Configuration Dialog.
    
    Mathematical Background:
        NMDS finds a low-dimensional representation that preserves
        the rank order of distances, minimizing stress:
        
        Stress = √(Σ(d_ij - d̂_ij)² / Σd_ij²)
        
        where d_ij is original distance, d̂_ij is ordination distance.
        
        Algorithm: SMACOF (Scaling by MAjorizing a COmplicated Function)
        
        Uses iterative majorization to minimize stress,
        typically with 500+ random restarts to avoid local minima.
    
    Parameters:
        metric: Distance metric
        n_dimensions: Ordination dimensions (2-3)
        n_restarts: Random restarts for global optimum
        max_iterations: Maximum iterations per run
        stress_tolerance: Convergence threshold
    """
    
    DISTANCE_METRICS = [
        "Bray-Curtis",
        "Jaccard",
        "Euclidean",
        "Manhattan",
        "Canberra",
        "Chi-square",
        "Wisconsin"
    ]
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(_("Non-metric MDS (NMDS)"), parent)
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup NMDS-specific parameters."""
        # Distance metric
        metric_group = self.add_parameter_group(_("Dissimilarity Index"))
        metric_layout = QVBoxLayout(metric_group)
        
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self.DISTANCE_METRICS)
        self._metric_combo.setCurrentText("Bray-Curtis")
        metric_layout.addWidget(QLabel(_("Distance measure:")))
        metric_layout.addWidget(self._metric_combo)
        
        # Dimensions
        dims_group = self.add_parameter_group(_("Ordination Dimensions"))
        dims_layout = QVBoxLayout(dims_group)
        
        self._n_dims_spin = QSpinBox()
        self._n_dims_spin.setRange(2, 4)
        self._n_dims_spin.setValue(2)
        self._n_dims_spin.setPrefix(_("Dimensions: "))
        dims_layout.addWidget(self._n_dims_spin)
        
        dims_layout.addWidget(QLabel(
            _("2D: Best for visualization\n"
              "3D: May reveal additional structure")
        ))
        
        # Optimization
        opt_group = self.add_parameter_group(_("Optimization"))
        opt_layout = QVBoxLayout(opt_group)
        
        self._n_restarts_spin = QSpinBox()
        self._n_restarts_spin.setRange(1, 100)
        self._n_restarts_spin.setValue(20)
        self._n_restarts_spin.setPrefix(_("Random restarts: "))
        opt_layout.addWidget(self._n_restarts_spin)
        
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(100, 10000)
        self._max_iter_spin.setValue(500)
        self._max_iter_spin.setPrefix(_("Max iterations: "))
        opt_layout.addWidget(self._max_iter_spin)
        
        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(0.0001, 0.1)
        self._tolerance_spin.setValue(0.001)
        self._tolerance_spin.setDecimals(4)
        self._tolerance_spin.setPrefix(_("Convergence tolerance: "))
        opt_layout.addWidget(self._tolerance_spin)
        
        # Display
        display_group = self.add_parameter_group(_("Display"))
        display_layout = QVBoxLayout(display_group)

        self._show_stress_check = QCheckBox(_("Show stress plot"))
        self._show_stress_check.setChecked(True)
        display_layout.addWidget(self._show_stress_check)
        
        self._show_shepard_check = QCheckBox(_("Show Shepard diagram"))
        display_layout.addWidget(self._show_shepard_check)
        
        self._show_points_check = QCheckBox(_("Show points with labels"))
        display_layout.addWidget(self._show_points_check)
        
        self._show_confidence_check = QCheckBox(_("Show 95% confidence ellipses"))
        display_layout.addWidget(self._show_confidence_check)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get NMDS parameters."""
        self._parameters = {
            'metric': self._metric_combo.currentText().lower().replace('-', '_'),
            'n_dimensions': self._n_dims_spin.value(),
            'n_restarts': self._n_restarts_spin.value(),
            'max_iterations': self._max_iter_spin.value(),
            'tolerance': self._tolerance_spin.value(),
            'show_stress': self._show_stress_check.isChecked(),
            'show_shepard': self._show_shepard_check.isChecked(),
            'show_points': self._show_points_check.isChecked(),
            'show_confidence': self._show_confidence_check.isChecked(),
        }
        return self._parameters
    
    def _get_help_text(self) -> str:
        return f"""
<h2>{_("Non-metric Multidimensional Scaling (NMDS)")}</h2>

<p>{_("NMDS is an ordination technique that finds coordinates preserving the rank order of distances.")}</p>

<h3>{_("Mathematical Formulation:")}</h3>

<p>{_("The stress function measures rank distortion:")}</p>
<p>Stress = √(Σ(d̂<sub>ij</sub> - d<sub>ij</sub>)² / Σd<sub>ij</sub>²)</p>

<h3>{_("Algorithm (SMACOF):")}</h3>
<ol>
<li>{_("Initialize random configuration")}</li>
<li>{_("Compute distances in current configuration")}</li>
<li>{_("Apply monotone regression to get disparities")}</li>
<li>{_("Find optimal configuration for disparities")}</li>
<li>{_("Repeat until convergence")}</li>
</ol>

<h3>{_("Interpretation:")}</h3>
<ul>
<li>{_("Stress < 0.05: Excellent")}</li>
<li>{_("Stress < 0.10: Good")}</li>
<li>{_("Stress < 0.15: Acceptable")}</li>
<li>{_("Stress > 0.20: Poor")}</li>
</ul>
        """


class DiversityDialog(BaseAnalysisDialog):
    """
    Biodiversity Analysis Configuration Dialog.
    
    Mathematical Background:
        Diversity indices quantify species richness and evenness.
        
        Species Richness: S = number of species
        
        Shannon Index: H' = -Σ p_i ln(p_i)
        
        Simpson Index: D = 1 - Σ p_i²
        
        Fisher's Alpha: Solves N = α ln(1 + N/α)
        
        where p_i = n_i / N is proportion of species i
    
    Parameters:
        sample_name: Sample identifier
        indices: Which diversity indices to compute
        confidence: Confidence interval method
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(_("Biodiversity Analysis"), parent)
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup diversity analysis parameters."""
        # Sample selection
        sample_group = self.add_parameter_group(_("Sample"))
        sample_layout = QVBoxLayout(sample_group)
        
        self._sample_name_edit = QLineEdit()
        self._sample_name_edit.setPlaceholderText(_("Enter sample name..."))
        sample_layout.addWidget(QLabel(_("Sample name:")))
        sample_layout.addWidget(self._sample_name_edit)
        
        # Diversity indices
        indices_group = self.add_parameter_group(_("Diversity Indices"))
        indices_layout = QVBoxLayout(indices_group)
        
        self._richness_check = QCheckBox(_("Species Richness (S)"))
        self._richness_check.setChecked(True)
        indices_layout.addWidget(self._richness_check)
        
        self._shannon_check = QCheckBox(_("Shannon Index (H')"))
        self._shannon_check.setChecked(True)
        indices_layout.addWidget(self._shannon_check)
        
        self._simpson_check = QCheckBox(_("Simpson Index (1-D)"))
        self._simpson_check.setChecked(True)
        indices_layout.addWidget(self._simpson_check)
        
        self._fisher_check = QCheckBox(_("Fisher's Alpha"))
        self._fisher_check.setChecked(True)
        indices_layout.addWidget(self._fisher_check)
        
        self._chao_check = QCheckBox(_("Chao1 Richness Estimator"))
        indices_layout.addWidget(self._chao_check)
        
        self._evenness_check = QCheckBox(_("Pielou's Evenness (J')"))
        indices_layout.addWidget(self._evenness_check)
        
        # Options
        options_group = self.add_parameter_group(_("Options"))
        options_layout = QVBoxLayout(options_group)
        
        self._log_base_combo = QComboBox()
        self._log_base_combo.addItems([_("Natural log (e)"), _("Log base 2"), _("Log base 10")])
        options_layout.addWidget(QLabel(_("Shannon log base:")))
        options_layout.addWidget(self._log_base_combo)
        
        self._ci_check = QCheckBox(_("Calculate confidence intervals (bootstrap)"))
        self._ci_check.setChecked(False)
        options_layout.addWidget(self._ci_check)
        
        self._ci_iterations_spin = QSpinBox()
        self._ci_iterations_spin.setRange(100, 9999)
        self._ci_iterations_spin.setValue(1000)
        self._ci_iterations_spin.setPrefix(_("Bootstrap iterations: "))
        options_layout.addWidget(self._ci_iterations_spin)
        
        self._ci_level_spin = QDoubleSpinBox()
        self._ci_level_spin.setRange(90, 99)
        self._ci_level_spin.setValue(95)
        self._ci_level_spin.setSuffix(" %")
        self._ci_level_spin.setPrefix(_("Confidence level: "))
        options_layout.addWidget(self._ci_level_spin)
        
        # Display
        display_group = self.add_parameter_group(_("Display"))
        display_layout = QVBoxLayout(display_group)

        self._bar_chart_check = QCheckBox(_("Bar chart comparison"))
        display_layout.addWidget(self._bar_chart_check)
        
        self._radar_chart_check = QCheckBox(_("Radar chart"))
        display_layout.addWidget(self._radar_chart_check)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get diversity analysis parameters."""
        log_bases = [np.e, 2, 10]
        
        self._parameters = {
            'sample_name': self._sample_name_edit.text(),
            'richness': self._richness_check.isChecked(),
            'shannon': self._shannon_check.isChecked(),
            'simpson': self._simpson_check.isChecked(),
            'fisher': self._fisher_check.isChecked(),
            'chao1': self._chao_check.isChecked(),
            'evenness': self._evenness_check.isChecked(),
            'log_base': log_bases[self._log_base_combo.currentIndex()],
            'confidence_intervals': self._ci_check.isChecked(),
            'ci_iterations': self._ci_iterations_spin.value(),
            'ci_level': self._ci_level_spin.value() / 100.0,
            'bar_chart': self._bar_chart_check.isChecked(),
            'radar_chart': self._radar_chart_check.isChecked(),
        }
        return self._parameters
    
    def _get_help_text(self) -> str:
        return f"""
<h2>{_("Biodiversity Analysis")}</h2>

<p>{_("Biodiversity indices quantify species richness and evenness in ecological communities.")}</p>

<h3>{_("Common Indices:")}</h3>
<ul>
<li><b>{_("Species Richness (S):")}</b> S = {_("number of observed species")}</li>
<li><b>{_("Shannon Index (H'):")}</b> H' = -Σ p<sub>i</sub> ln(p<sub>i</sub>), {_("where")} p<sub>i</sub> = n<sub>i</sub> / N</li>
<li><b>{_("Simpson Index (1-D):")}</b> D = 1 - Σ p<sub>i</sub>²</li>
<li><b>{_("Fisher's Alpha:")}</b> S = α ln(1 + N/α)</li>
<li><b>{_("Chao1 Estimator:")}</b> S<sub>Chao1</sub> = S<sub>obs</sub> + f₁² / (2f₂)</li>
</ul>

<h3>{_("Interpretation:")}</h3>
<p>{_("Higher values generally indicate greater diversity. Compare across samples to assess patterns.")}</p>
        """


class RarefactionDialog(BaseAnalysisDialog):
    """
    Rarefaction Curve Configuration Dialog.
    
    Mathematical Background:
        Rarefaction standardizes samples to common size,
        accounting for sampling effort differences.
        
        Expected species at n individuals:<br/>
        E(S_n) = Σ[1 - C(N-n_i, n) / C(N, n)]
        
        where N = total individuals, n_i = individuals of species i
    
    Parameters:
        sample_names: Samples to include
        max_individuals: Maximum to rarefy to
        confidence: Show confidence intervals
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(_("Rarefaction Analysis"), parent)
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup rarefaction parameters."""
        # Sample selection
        sample_group = self.add_parameter_group(_("Select Samples"))
        sample_layout = QVBoxLayout(sample_group)
        
        self._sample_list = QListWidget()
        self._sample_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._sample_list.addItems([
            _("Sample 1"), _("Sample 2"), _("Sample 3"), _("Sample 4"), _("Sample 5")
        ])
        sample_layout.addWidget(self._sample_list)
        
        # Settings
        settings_group = self.add_parameter_group(_("Settings"))
        settings_layout = QVBoxLayout(settings_group)
        
        self._max_n_spin = QSpinBox()
        self._max_n_spin.setRange(10, 10000)
        self._max_n_spin.setValue(100)
        self._max_n_spin.setPrefix(_("Maximum individuals: "))
        settings_layout.addWidget(self._max_n_spin)
        
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 100)
        self._step_spin.setValue(5)
        self._step_spin.setPrefix(_("Step size: "))
        settings_layout.addWidget(self._step_spin)
        
        self._ci_check = QCheckBox(_("Show 95% confidence intervals"))
        self._ci_check.setChecked(True)
        settings_layout.addWidget(self._ci_check)
        
        # Display
        display_group = self.add_parameter_group(_("Display"))
        display_layout = QVBoxLayout(display_group)

        self._separate_plots_check = QCheckBox(_("Separate plots per sample"))
        display_layout.addWidget(self._separate_plots_check)
        
        self._grid_check = QCheckBox(_("Show grid"))
        display_layout.addWidget(self._grid_check)
        self._grid_check.setChecked(True)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get rarefaction parameters."""
        selected = [item.text() for item in self._sample_list.selectedItems()]
        
        self._parameters = {
            'samples': selected,
            'max_n': self._max_n_spin.value(),
            'step': self._step_spin.value(),
            'confidence_intervals': self._ci_check.isChecked(),
            'separate_plots': self._separate_plots_check.isChecked(),
            'grid': self._grid_check.isChecked(),
        }
        return self._parameters
    
    def _get_help_text(self) -> str:
        return f"""
<h2>{_("Rarefaction Analysis")}</h2>

<p>{_("Rarefaction standardizes species counts to a common sampling effort.")}</p>

<h3>{_("Mathematical Formulation:")}</h3>

<p>{_("Expected species at n individuals:")}</p>
<p>E(S<sub>n</sub>) = Σ<sub>i=1</sub><sup>S</sup> [1 - C(N-n<sub>i</sub>, n) / C(N, n)]</p>

<p>{_("where")}:</p>
<ul>
<li>N = {_("total individuals in sample")}</li>
<li>n<sub>i</sub> = {_("individuals of species")} i</li>
<li>C(a, b) = {_("binomial coefficient")}</li>
</ul>

<h3>{_("Interpretation:")}</h3>
<p>{_("Rarefaction curves that plateau indicate adequate sampling.")}</p>
        """


class ImportDialog(QDialog):
    """
    Data Import Configuration Dialog.
    
    Configures data import from various file formats
    with preview and validation.
    
    Signals:
        dataImported: Emitted with (data, metadata) when import succeeds
    """
    
    dataImported = pyqtSignal(object, dict)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.setWindowTitle(_("Import Data"))
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup import dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # File selection
        file_group = QGroupBox(_("File Selection"))
        file_layout = QVBoxLayout(file_group)
        
        file_select_layout = QHBoxLayout()
        self._file_path_edit = QLineEdit()
        self._file_path_edit.setPlaceholderText(_("Select file..."))
        file_select_layout.addWidget(self._file_path_edit)
        
        browse_btn = QPushButton(_("Browse..."))
        browse_btn.clicked.connect(self._browse_file)
        file_select_layout.addWidget(browse_btn)
        file_layout.addLayout(file_select_layout)
        
        self._format_combo = QComboBox()
        self._format_combo.addItems([
            _("CSV (Comma Separated)"),
            _("CSV (Tab Separated)"),
            _("Text (Space Separated)"),
            _("Excel (.xlsx)")
        ])
        file_layout.addWidget(QLabel(_("File format:")))
        file_layout.addWidget(self._format_combo)
        
        layout.addWidget(file_group)
        
        # Options
        options_group = QGroupBox(_("Import Options"))
        options_layout = QGridLayout(options_group)
        
        self._header_check = QCheckBox(_("First row contains headers"))
        self._header_check.setChecked(True)
        options_layout.addWidget(self._header_check, 0, 0)
        
        self._row_labels_check = QCheckBox(_("First column contains row labels"))
        self._row_labels_check.setChecked(True)
        options_layout.addWidget(self._row_labels_check, 0, 1)
        
        self._skip_rows_spin = QSpinBox()
        self._skip_rows_spin.setRange(0, 100)
        self._skip_rows_spin.setValue(0)
        options_layout.addWidget(QLabel(_("Rows to skip:")), 1, 0)
        options_layout.addWidget(self._skip_rows_spin, 1, 1)
        
        self._na_values_edit = QLineEdit()
        self._na_values_edit.setPlaceholderText(_("NA, NaN, -, empty"))
        options_layout.addWidget(QLabel(_("NA values:")), 2, 0)
        options_layout.addWidget(self._na_values_edit, 2, 1)
        
        layout.addWidget(options_group)
        
        # Preview
        preview_group = QGroupBox(_("Data Preview"))
        preview_layout = QVBoxLayout(preview_group)
        
        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self._preview_text)
        
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        import_btn = QPushButton(_("Import"))
        import_btn.clicked.connect(self._import_data)
        button_layout.addWidget(import_btn)
        
        layout.addLayout(button_layout)
    
    def _browse_file(self) -> None:
        """Browse for file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            _("Select Data File"),
            "",
            _("Data Files (*.csv *.txt *.xlsx);;All Files (*)")
        )
        
        if filepath:
            self._file_path_edit.setText(filepath)
            self._load_preview()
    
    def _load_preview(self) -> None:
        """Load file preview."""
        filepath = self._file_path_edit.text()
        if not filepath:
            return
        
        try:
            delimiter = ','
            if self._format_combo.currentIndex() == 1:
                delimiter = '\t'

            try:
                import pandas as pd
            except ImportError:
                self._preview_text.setText(
                    _("pandas is required for data preview. Install with: pip install pandas")
                )
                return
            df = pd.read_csv(filepath, delimiter=delimiter, nrows=5)
            self._preview_text.setText(df.head().to_string())
        except Exception as e:
            self._preview_text.setText(_("Error loading preview:\n{0}").format(str(e)))
    
    def _import_data(self) -> None:
        """Import data from file."""
        filepath = self._file_path_edit.text()
        if not filepath:
            QMessageBox.warning(self, _("No File"), _("Please select a file to import."))
            return
        
        try:
            try:
                import pandas as pd
            except ImportError:
                QMessageBox.critical(
                    self, _("Import Error"),
                    _("pandas is required for data import. Install with: pip install pandas")
                )
                return

            delimiter = ','
            if self._format_combo.currentIndex() == 1:
                delimiter = '\t'
            
            na_values = [v.strip() for v in self._na_values_edit.text().split(',')]
            if not na_values or na_values == ['']:
                na_values = ['NA', 'NaN', '-', '']
            
            df = pd.read_csv(
                filepath,
                delimiter=delimiter,
                header=0 if self._header_check.isChecked() else None,
                index_col=0 if self._row_labels_check.isChecked() else False,
                skiprows=self._skip_rows_spin.value(),
                na_values=na_values
            )
            
            data = df.values.astype(float)
            row_labels = list(df.index.astype(str))
            col_labels = list(df.columns.astype(str))
            
            metadata = {
                'row_labels': row_labels,
                'col_labels': col_labels
            }
            
            self.dataImported.emit(data, metadata)
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, _("Import Error"), _("Failed to import data:\n{0}").format(str(e)))
