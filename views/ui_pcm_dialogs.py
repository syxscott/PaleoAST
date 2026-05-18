# =============================================================================
# FILE: views/ui_pcm_dialogs.py
# =============================================================================
"""
Phylogenetic Comparative Methods Dialogs for PaleoAST

Provides dialogs for:
    - Phylogenetic Independent Contrasts (PIC)
    - Ancestral State Reconstruction (ASR)
    - Blomberg's K phylogenetic signal
    - Phylogenetic ANOVA

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, BorderRadius, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class PCMBaseDialog(QDialog):
    """
    Base dialog for PCM analyses.

    Provides common UI structure:
        - Tree input (Newick text)
        - Trait selection
        - Group assignment (for Phylo-ANOVA)
        - Results display
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.{title}")
        self._is_dark_theme = False

        self.setWindowTitle(title)
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self._setup_ui()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        t = Typography()
        self.setStyleSheet(f"QDialog {{ background-color: {c.bg_primary}; color: {c.text_primary}; }}"
                           f"QLabel {{ color: {c.text_primary}; font-size: {t.body_size}px; }}"
                           f"QGroupBox {{ color: {c.text_primary}; font-weight: {t.medium}; "
                           f"border: 1px solid {c.border_light}; border-radius: 4px; }}"
                           f"QTextEdit {{ background-color: {c.bg_primary}; color: {c.text_primary}; "
                           f"border: 1px solid {c.border_light}; border-radius: 4px; "
                           f"font-family: 'Consolas', monospace; font-size: {t.body_sm_size}px; }}")

    def _setup_ui(self) -> None:
        """Setup common UI structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        t = Typography()
        title_label = QLabel(self.windowTitle())
        title_font = QFont(t.family_primary, t.h4_size, QFont.Weight.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Tree input
        tree_group = QGroupBox(_("Phylogenetic Tree (Newick format)"))
        tree_layout = QVBoxLayout(tree_group)

        self._tree_input = QTextEdit()
        self._tree_input.setPlaceholderText(_("(A:0.1,B:0.2)C:0.3;"))
        self._tree_input.setMaximumHeight(80)
        tree_layout.addWidget(self._tree_input)

        load_tree_btn = QPushButton(_("Load from file..."))
        load_tree_btn.clicked.connect(self._load_tree_file)
        tree_layout.addWidget(load_tree_btn)

        layout.addWidget(tree_group)

        # Trait data input
        trait_group = QGroupBox(_("Trait Data"))
        trait_layout = QVBoxLayout(trait_group)

        trait_info = QLabel(
            _("Enter one trait value per line: TaxonName=TraitValue\n"
              "Example:\n"
              "Homo_sapiens=2.45\n"
              "Pan_troglodytes=3.12\n"
              "Gorilla_gorilla=4.05")
        )
        trait_info.setStyleSheet(f"color: {get_palette().text_secondary}; font-size: 11px;")
        trait_layout.addWidget(trait_info)

        self._trait_input = QTextEdit()
        self._trait_input.setPlaceholderText("Homo_sapiens=2.45\nPan_troglodytes=3.12\nGorilla_gorilla=4.05")
        self._trait_input.setMaximumHeight(100)
        trait_layout.addWidget(self._trait_input)

        layout.addWidget(trait_group)

        # Method-specific content (subclasses override)
        self._method_widget = QWidget()
        method_layout = QVBoxLayout(self._method_widget)
        layout.addWidget(self._method_widget, 1)

        # Results
        results_group = QGroupBox(_("Results"))
        results_layout = QVBoxLayout(results_group)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(120)
        results_layout.addWidget(self._results_text)

        layout.addWidget(results_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._run_button = QPushButton(_("Run Analysis"))
        self._run_button.clicked.connect(self._on_run)
        button_layout.addWidget(self._run_button)

        self._close_button = QPushButton(_("Close"))
        self._close_button.clicked.connect(self.accept)
        button_layout.addWidget(self._close_button)

        layout.addLayout(button_layout)

    def _load_tree_file(self) -> None:
        """Load Newick tree from file."""
        from PyQt6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getOpenFileName(
            self, _("Open Newick Tree"),
            "", _("Newick Files (*.tre *.tree *.nwck *.newick);;All Files (*)")
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                self._tree_input.setText(content)
                self._logger.info(f"Loaded tree from {filepath}")
            except Exception as e:
                QMessageBox.critical(self, _("Error"), _("Failed to load tree: {0}").format(str(e)))

    def _parse_trait_input(self) -> dict[str, float]:
        """Parse trait input text into {taxon: value} dict."""
        lines = self._trait_input.toPlainText().strip().split("\n")
        result: dict[str, float] = {}
        errors: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                parts = line.split("=", 1)
                name = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    result[name] = value
                except ValueError:
                    errors.append(line)
            elif ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    result[name] = value
                except ValueError:
                    errors.append(line)
        if errors:
            self._logger.warning(f"Failed to parse {len(errors)} trait lines")
        return result

    def _on_run(self) -> None:
        """Run the analysis. Subclasses implement specific logic."""
        raise NotImplementedError


class PICDialog(PCMBaseDialog):
    """
    Phylogenetic Independent Contrasts dialog.

    Computes standardized contrasts at each internal node:
        IC = (x_child1 - x_child2) / sqrt(v1 + v2)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Phylogenetic Independent Contrasts (PIC)"), parent)

        # Method-specific options
        method_layout = self._method_widget.layout()

        opts_group = QGroupBox(_("Options"))
        opts_layout = QHBoxLayout(opts_group)

        self._check_branch_lengths = QCheckBox(_("Use branch lengths"))
        self._check_branch_lengths.setChecked(True)
        opts_layout.addWidget(self._check_branch_lengths)

        opts_layout.addStretch()
        method_layout.addWidget(opts_group)

    def _on_run(self) -> None:
        """Run PIC analysis."""
        tree_text = self._tree_input.toPlainText().strip()
        trait_values = self._parse_trait_input()

        if not tree_text:
            QMessageBox.warning(self, _("Input Error"), _("Please enter a phylogenetic tree in Newick format."))
            return
        if len(trait_values) < 3:
            QMessageBox.warning(self, _("Input Error"), _("Please enter at least 3 trait values."))
            return

        try:
            from controllers.statistics_controller import StatisticsController
            ctrl = StatisticsController()
            result = ctrl.analyze_pic(tree_text, trait_values)
            self._results_text.setPlainText(result.summary())
            self._logger.info(f"PIC completed: {result.n_contrasts} contrasts")
        except Exception as e:
            self._logger.error(f"PIC failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "PIC"))


class AncestralStateDialog(PCMBaseDialog):
    """
    Ancestral State Reconstruction dialog.

    Reconstructs trait values at internal nodes via weighted squared-change parsimony.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Ancestral State Reconstruction (ASR)"), parent)

        method_layout = self._method_widget.layout()

        model_group = QGroupBox(_("Evolution Model"))
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel(_("Model:")))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Brownian Motion (BM)", "Ornstein-Uhlenbeck (OU)"])
        model_layout.addWidget(self._model_combo)
        model_layout.addStretch()

        method_layout.addWidget(model_group)

    def _on_run(self) -> None:
        """Run ancestral state reconstruction."""
        tree_text = self._tree_input.toPlainText().strip()
        trait_values = self._parse_trait_input()

        if not tree_text:
            QMessageBox.warning(self, _("Input Error"), _("Please enter a phylogenetic tree in Newick format."))
            return
        if not trait_values:
            QMessageBox.warning(self, _("Input Error"), _("Please enter trait values."))
            return

        model = "bm" if self._model_combo.currentIndex() == 0 else "ou"

        try:
            from controllers.statistics_controller import StatisticsController
            ctrl = StatisticsController()
            result = ctrl.analyze_ancestral_states(tree_text, trait_values, model=model)

            # Format results
            lines = [result.summary(), "", _("Reconstructed internal node states:")]
            for node_name, state in sorted(result.node_states.items()):
                lines.append(f"  {node_name}: {state:.4f}")
            self._results_text.setPlainText("\n".join(lines))
            self._logger.info(f"ASR completed: {len(result.node_states)} nodes")
        except Exception as e:
            self._logger.error(f"ASR failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "祖先状态重建"))


class PhyloSignalDialog(PCMBaseDialog):
    """
    Blomberg's K phylogenetic signal dialog.

    Measures phylogenetic signal:
        K < 1: convergence
        K ≈ 1: Brownian motion
        K > 1: phylogenetic niche conservatism
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Phylogenetic Signal (Blomberg's K)"), parent)

        method_layout = self._method_widget.layout()

        # Permutations
        perm_group = QGroupBox(_("Significance Testing"))
        perm_layout = QHBoxLayout(perm_group)

        perm_layout.addWidget(QLabel(_("Permutations:")))
        self._n_perm_spin = QSpinBox()
        self._n_perm_spin.setRange(99, 9999)
        self._n_perm_spin.setValue(999)
        self._n_perm_spin.setSingleStep(100)
        perm_layout.addWidget(self._n_perm_spin)
        perm_layout.addStretch()

        method_layout.addWidget(perm_group)

    def _on_run(self) -> None:
        """Run phylogenetic signal analysis."""
        tree_text = self._tree_input.toPlainText().strip()
        trait_values = self._parse_trait_input()

        if not tree_text:
            QMessageBox.warning(self, _("Input Error"), _("Please enter a phylogenetic tree in Newick format."))
            return
        if len(trait_values) < 3:
            QMessageBox.warning(self, _("Input Error"), _("Please enter at least 3 trait values."))
            return

        try:
            from controllers.statistics_controller import StatisticsController
            ctrl = StatisticsController()
            result = ctrl.analyze_phylogenetic_signal(
                tree_text, trait_values,
                n_randomizations=self._n_perm_spin.value()
            )
            self._results_text.setPlainText(result.summary())
            self._logger.info(f"Blomberg's K = {result.k:.4f}")
        except Exception as e:
            self._logger.error(f"Phylogenetic signal failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "系统发育信号"))


class PhyloANOVADialog(PCMBaseDialog):
    """
    Phylogenetic ANOVA dialog.

    Tests for trait differences between groups while accounting for
    phylogenetic non-independence.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Phylogenetic ANOVA"), parent)

        method_layout = self._method_widget.layout()

        # Group input
        group_group = QGroupBox(_("Group Assignments"))
        group_layout = QVBoxLayout(group_group)

        group_info = QLabel(
            _("Enter group assignments (one per line): TaxonName=GroupName\n"
              "Example:\n"
              "Homo_sapiens=Human\n"
              "Pan_troglodytes=GreatApes\n"
              "Gorilla_gorilla=GreatApes")
        )
        group_info.setStyleSheet(f"color: {get_palette().text_secondary}; font-size: 11px;")
        group_layout.addWidget(group_info)

        self._group_input = QTextEdit()
        self._group_input.setPlaceholderText("Homo_sapiens=Human\nPan_troglodytes=GreatApes\nGorilla_gorilla=GreatApes")
        self._group_input.setMaximumHeight(100)
        group_layout.addWidget(self._group_input)

        # Permutations
        perm_layout = QHBoxLayout()
        perm_layout.addWidget(QLabel(_("Permutations:")))
        self._n_perm_spin = QSpinBox()
        self._n_perm_spin.setRange(99, 9999)
        self._n_perm_spin.setValue(999)
        self._n_perm_spin.setSingleStep(100)
        perm_layout.addWidget(self._n_perm_spin)
        perm_layout.addStretch()

        group_layout.addLayout(perm_layout)
        method_layout.addWidget(group_group)

    def _parse_group_input(self) -> dict[str, str]:
        """Parse group input text into {taxon: group} dict."""
        lines = self._group_input.toPlainText().strip().split("\n")
        result: dict[str, str] = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                parts = line.split("=", 1)
                name = parts[0].strip()
                group = parts[1].strip()
                result[name] = group
        return result

    def _on_run(self) -> None:
        """Run phylogenetic ANOVA."""
        tree_text = self._tree_input.toPlainText().strip()
        trait_values = self._parse_trait_input()
        group_labels = self._parse_group_input()

        if not tree_text:
            QMessageBox.warning(self, _("Input Error"), _("Please enter a phylogenetic tree in Newick format."))
            return
        if len(trait_values) < 3:
            QMessageBox.warning(self, _("Input Error"), _("Please enter at least 3 trait values."))
            return
        if not group_labels:
            QMessageBox.warning(self, _("Input Error"), _("Please enter group assignments."))
            return

        # Check that all traits have groups
        missing_groups = set(trait_values.keys()) - set(group_labels.keys())
        if missing_groups:
            QMessageBox.warning(
                self, _("Input Error"),
                _("Missing group assignments for: {0}").format(", ".join(sorted(missing_groups)))
            )
            return

        try:
            from controllers.statistics_controller import StatisticsController
            ctrl = StatisticsController()
            result = ctrl.analyze_phylo_anova(
                tree_text, trait_values, group_labels,
                n_permutations=self._n_perm_spin.value()
            )
            self._results_text.setPlainText(result.summary())
            self._logger.info(f"Phylo-ANOVA: F={result.f_statistic:.4f}")
        except Exception as e:
            self._logger.error(f"Phylo-ANOVA failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "系统发育方差分析"))
