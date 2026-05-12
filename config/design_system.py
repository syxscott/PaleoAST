# =============================================================================
# FILE: config/design_system.py
# =============================================================================
"""
Modern Design System for PaleoAST

Defines a unified, professional design language for all UI components.
Based on Material Design 3 + modern flat design principles.

Color Palette:
    - Primary: #3498DB (Professional Blue)
    - Success: #27AE60 (Fresh Green)
    - Warning: #F39C12 (Golden)
    - Error: #E74C3C (Soft Red)
    - Neutral: #ECF0F1, #BDC3C7, #95A5A6

Spacing: Based on 4px grid system
Typography: Segoe UI + fallbacks
"""

from dataclasses import dataclass

# =============================================================================
# Color Scheme (Light Theme - Modern)
# =============================================================================


@dataclass
class ColorPalette:
    """Modern light color palette."""

    # Primary colors
    primary = "#3498DB"  # Professional blue
    primary_light = "#5DADE2"  # Light blue (hover)
    primary_dark = "#2980B9"  # Dark blue (active)

    # Semantic colors
    success = "#27AE60"
    warning = "#F39C12"
    error = "#E74C3C"
    info = "#16A085"

    # Neutral colors
    bg_primary = "#FFFFFF"  # Main background
    bg_secondary = "#F8F9FA"  # Secondary surface
    bg_tertiary = "#F0F2F5"  # Tertiary surface

    text_primary = "#2C3E50"  # Main text
    text_secondary = "#7F8C8D"  # Secondary text
    text_disabled = "#BDC3C7"  # Disabled text

    border_light = "#E4E7EB"  # Light border
    border_medium = "#D0D5DD"  # Medium border
    border_dark = "#BDC3C7"  # Dark border

    # Hover/Active states
    hover_overlay = "rgba(52, 152, 219, 0.08)"
    active_overlay = "rgba(52, 152, 219, 0.15)"

    # Shadows
    shadow_sm = "0 1px 2px rgba(0,0,0,0.05)"
    shadow_md = "0 4px 6px rgba(0,0,0,0.07)"
    shadow_lg = "0 10px 15px rgba(0,0,0,0.10)"
    shadow_xl = "0 20px 25px rgba(0,0,0,0.15)"


# =============================================================================
# Spacing System (4px grid)
# =============================================================================


@dataclass
class Spacing:
    """Spacing constants (4px base unit)."""

    xs = 4  # Extra small
    sm = 8  # Small
    md = 12  # Medium
    lg = 16  # Large
    xl = 24  # Extra large
    xxl = 32  # Double extra large

    # Combinations for common patterns
    padding_compact = f"{sm}px"
    padding_normal = f"{md}px"
    padding_generous = f"{lg}px"

    margin_compact = f"{sm}px"
    margin_normal = f"{lg}px"
    margin_generous = f"{xxl}px"


# =============================================================================
# Typography System
# =============================================================================


@dataclass
class Typography:
    """Typography scale."""

    family_primary = "'Segoe UI', 'Microsoft YaHei', -apple-system, BlinkMacSystemFont, sans-serif"
    family_monospace = "'Consolas', 'Monaco', 'Courier New', monospace"

    # Font sizes
    h1_size = 32
    h2_size = 28
    h3_size = 24
    h4_size = 20
    h5_size = 16

    body_lg_size = 16
    body_size = 14
    body_sm_size = 13
    caption_size = 12

    # Font weights
    thin = 100
    light = 300
    normal = 400
    medium = 500
    semibold = 600
    bold = 700

    # Line heights
    line_height_tight = 1.2
    line_height_normal = 1.5
    line_height_relaxed = 1.75


# =============================================================================
# Radius System
# =============================================================================


@dataclass
class BorderRadius:
    """Border radius presets."""

    none = "0px"
    sm = "2px"
    md = "4px"
    lg = "6px"
    xl = "8px"
    full = "9999px"


# =============================================================================
# Global StyleSheet Generator
# =============================================================================


def get_modern_stylesheet() -> str:
    """Generate comprehensive modern stylesheet for entire application."""
    colors = ColorPalette()
    spacing = Spacing()
    typo = Typography()
    radius = BorderRadius()

    return f"""
/* =============================================================================
   GLOBAL STYLES
   ============================================================================= */

* {{
    font-family: {typo.family_primary};
}}

QWidget {{
    background-color: {colors.bg_primary};
    color: {colors.text_primary};
}}

QMainWindow {{
    background-color: {colors.bg_primary};
}}

/* =============================================================================
   MENU BAR & MENUS
   ============================================================================= */

QMenuBar {{
    background-color: {colors.bg_primary};
    border-bottom: 1px solid {colors.border_light};
    padding: {spacing.sm}px 0;
}}

QMenuBar::item {{
    padding: 6px 16px;
    background: transparent;
    border-radius: {radius.md};
    margin: 2px 4px;
}}

QMenuBar::item:selected {{
    background-color: {colors.hover_overlay};
    color: {colors.primary};
    font-weight: {typo.medium};
}}

QMenu {{
    background-color: {colors.bg_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.lg};
    padding: 4px 0;
    box-shadow: {colors.shadow_md};
}}

QMenu::item {{
    padding: 8px 24px 8px 28px;
    border-radius: {radius.md};
    margin: 2px 4px;
}}

QMenu::item:selected {{
    background-color: {colors.hover_overlay};
    color: {colors.primary};
}}

QMenu::item:disabled {{
    color: {colors.text_disabled};
}}

/* =============================================================================
   BUTTONS (MODERN FLAT WITH SUBTLE SHADOW)
   ============================================================================= */

QPushButton {{
    background-color: {colors.bg_secondary};
    color: {colors.text_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.lg};
    padding: 10px 18px;
    min-width: 80px;
    min-height: 36px;
    font-size: {typo.body_size}px;
    font-weight: {typo.medium};
    transition: all 200ms ease-out;
}}

QPushButton:hover {{
    background-color: {colors.bg_tertiary};
    border-color: {colors.primary};
    color: {colors.primary};
}}

QPushButton:pressed {{
    background-color: {colors.primary};
    color: white;
    border-color: {colors.primary_dark};
}}

QPushButton:disabled {{
    background-color: {colors.bg_tertiary};
    color: {colors.text_disabled};
    border-color: {colors.border_light};
}}

QPushButton[default="true"] {{
    background-color: {colors.primary};
    color: white;
    border-color: {colors.primary_dark};
    font-weight: {typo.semibold};
}}

QPushButton[default="true"]:hover {{
    background-color: {colors.primary_light};
    border-color: {colors.primary};
}}

QPushButton[default="true"]:pressed {{
    background-color: {colors.primary_dark};
}}

/* =============================================================================
   INPUT FIELDS (CLEAN & MODERN)
   ============================================================================= */

QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {colors.bg_primary};
    color: {colors.text_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.md};
    padding: 8px 12px;
    selection-background-color: {colors.primary};
    selection-color: white;
}}

QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {{
    border: 1px solid {colors.border_medium};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {colors.primary};
    outline: 0;
}}

/* =============================================================================
   COMBO BOX & SPINBOX
   ============================================================================= */

QComboBox {{
    background-color: {colors.bg_primary};
    color: {colors.text_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.md};
    padding: 8px 12px;
    min-height: 36px;
}}

QComboBox:hover {{
    border: 1px solid {colors.primary};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {colors.text_secondary};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {colors.bg_primary};
    color: {colors.text_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.md};
    padding: 8px 12px;
    min-height: 36px;
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border: 1px solid {colors.primary};
}}

/* =============================================================================
   CHECKBOXES & RADIO BUTTONS
   ============================================================================= */

QCheckBox, QRadioButton {{
    color: {colors.text_primary};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {colors.border_medium};
    border-radius: {radius.sm};
    background-color: {colors.bg_primary};
}}

QCheckBox::indicator:hover {{
    border: 1px solid {colors.primary};
}}

QCheckBox::indicator:checked {{
    background-color: {colors.primary};
    border-color: {colors.primary};
    image: url(none);
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {colors.border_medium};
    border-radius: 50%;
    background-color: {colors.bg_primary};
}}

QRadioButton::indicator:checked {{
    background: qradial-gradient(circle, {colors.primary} 40%, {colors.bg_primary} 50%);
    border-color: {colors.primary};
}}

/* =============================================================================
   GROUPBOXES & FRAMES
   ============================================================================= */

QGroupBox {{
    color: {colors.text_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.lg};
    margin-top: 12px;
    padding-top: 12px;
    background-color: {colors.bg_secondary};
    font-weight: {typo.medium};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 8px;
    color: {colors.primary};
}}

QFrame {{
    background-color: {colors.bg_primary};
    border: none;
}}

QFrame[frameShape="4"] {{
    background-color: {colors.border_light};
    max-height: 1px;
    margin: 8px 0;
}}

/* =============================================================================
   SCROLLBARS
   ============================================================================= */

QScrollBar:vertical {{
    background: {colors.bg_secondary};
    width: 10px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:vertical {{
    background: {colors.border_dark};
    min-height: 20px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:vertical:hover {{
    background: {colors.border_medium};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none;
    height: 0px;
}}

QScrollBar:horizontal {{
    background: {colors.bg_secondary};
    height: 10px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:horizontal {{
    background: {colors.border_dark};
    min-width: 20px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:horizontal:hover {{
    background: {colors.border_medium};
}}

/* =============================================================================
   TREEVIEW & LISTWIDGET
   ============================================================================= */

QTreeView, QListView {{
    background-color: {colors.bg_primary};
    border: 1px solid {colors.border_light};
    border-radius: {radius.lg};
    outline: 0;
}}

QTreeView::item {{
    padding: 6px 4px;
    border-radius: {radius.md};
}}

QTreeView::item:hover {{
    background-color: {colors.hover_overlay};
}}

QTreeView::item:selected {{
    background-color: {colors.active_overlay};
    color: {colors.primary};
    font-weight: {typo.medium};
}}

QListWidget::item {{
    padding: 8px 12px;
    border-radius: {radius.md};
}}

QListWidget::item:hover {{
    background-color: {colors.hover_overlay};
}}

QListWidget::item:selected {{
    background-color: {colors.primary};
    color: white;
    font-weight: {typo.medium};
}}

/* =============================================================================
   TABS
   ============================================================================= */

QTabBar::tab {{
    background-color: {colors.bg_secondary};
    color: {colors.text_secondary};
    border: none;
    padding: 10px 20px;
    margin: 0 2px;
    border-radius: {radius.lg} {radius.lg} 0 0;
}}

QTabBar::tab:hover {{
    background-color: {colors.bg_tertiary};
}}

QTabBar::tab:selected {{
    background-color: {colors.primary};
    color: white;
    font-weight: {typo.medium};
}}

/* =============================================================================
   SLIDERS
   ============================================================================= */

QSlider::groove:horizontal {{
    height: 6px;
    background: {colors.border_light};
    border-radius: {radius.sm};
}}

QSlider::handle:horizontal {{
    width: 18px;
    background: {colors.primary};
    border-radius: 50%;
    margin: -6px 0;
}}

QSlider::handle:horizontal:hover {{
    background: {colors.primary_light};
}}

/* =============================================================================
   STATUS BAR
   ============================================================================= */

QStatusBar {{
    background-color: {colors.bg_secondary};
    border-top: 1px solid {colors.border_light};
    color: {colors.text_secondary};
    padding: 4px 8px;
}}

/* =============================================================================
   DIALOGS
   ============================================================================= */

QDialog {{
    background-color: {colors.bg_primary};
}}

QLabel {{
    color: {colors.text_primary};
}}

QLabel[class="label-primary"] {{
    font-size: {typo.body_lg_size}px;
    font-weight: {typo.bold};
    color: {colors.text_primary};
}}

QLabel[class="label-secondary"] {{
    font-size: {typo.body_sm_size}px;
    color: {colors.text_secondary};
}}

QLabel[class="label-success"] {{
    color: {colors.success};
}}

QLabel[class="label-error"] {{
    color: {colors.error};
}}
"""


# Export instances for convenient access
colors = ColorPalette()
spacing = Spacing()
typography = Typography()
radius = BorderRadius()
