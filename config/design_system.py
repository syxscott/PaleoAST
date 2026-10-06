# =============================================================================
# FILE: config/design_system.py
# =============================================================================
"""
Design tokens for PaleoAST.

WHAT IS ACTUALLY AUTHORITATIVE
------------------------------
This module, not the docstring of any other file. The colours below are the
single source of truth; the QSS built by :func:`get_modern_stylesheet` and the
ribbon widgets in views/ui_main_window.py both read them.

Colour roles
------------
``primary`` and friends are *accents*. They are used two different ways, and
the two need different values:

  * as a **foreground** on a surface (menu selection text, a group title, a
    focus ring) -- must be legible against ``bg_*``;
  * as a **fill**, with ``on_primary`` as the foreground painted on top.

Keeping those two roles on one token is what let the dark theme inherit
``primary = #1E40AF`` -- a blue chosen for white backgrounds -- and then paint
dark-blue text on dark-blue fills at 1.44:1. The dark palette therefore
overrides every accent, and ``on_primary`` exists so that "text on the accent"
never has to be hardcoded as white.

Spacing is a 4px grid. It is only worth claiming if the stylesheet actually
uses it, which is why the QSS interpolates ``{spacing.*}`` rather than
hardcoding pixels.
"""


# =============================================================================
# Color Scheme (Light Theme - Modern)
# =============================================================================


class ColorPalette:
    """Modern light color palette with professional scientific styling."""

    # Primary colors - Professional deep blue for scientific data
    primary = "#1E40AF"  # Deep professional blue
    primary_light = "#3B82F6"  # Lighter blue (hover)
    primary_dark = "#1E3A8A"  # Darker blue (active)

    # Secondary - Supporting blue tones
    secondary = "#64748B"  # Slate for secondary elements

    # Semantic colors. Nudged one step darker than the stock Tailwind values so
    # they clear 4.5:1 on bg_tertiary, the hardest light surface -- a colour
    # that passes on white but fails on a grey surface is still a failure.
    success = "#04805A"  # Emerald
    warning = "#A95D05"  # Amber
    error = "#DC2626"  # Red
    info = "#077A96"  # Cyan

    # Foreground painted ON an accent fill. White, because every light-theme
    # accent is dark enough to carry it.
    on_primary = "#FFFFFF"

    # Neutral colors - Clean grays
    bg_primary = "#FFFFFF"  # Main background
    bg_secondary = "#F8FAFC"  # Secondary surface
    bg_tertiary = "#F1F5F9"  # Tertiary surface
    bg_hover = "#E2E8F0"  # Hover background

    text_primary = "#0F172A"  # Main text (near black)
    text_secondary = "#617187"  # Secondary text
    text_disabled = "#94A3B8"  # Disabled text

    border_light = "#E2E8F0"  # Light border
    border_medium = "#CBD5E1"  # Medium border
    border_focus = "#3B82F6"  # Focus border

    # Hover/Active states
    hover_overlay = "rgba(30, 64, 175, 0.08)"
    active_overlay = "rgba(30, 64, 175, 0.12)"
    selected_overlay = "rgba(59, 130, 246, 0.15)"

    # Shadows - Subtle and modern
    shadow_sm = "0 1px 2px rgba(0,0,0,0.05)"
    shadow_md = "0 4px 6px -1px rgba(0,0,0,0.1)"
    shadow_lg = "0 10px 15px -3px rgba(0,0,0,0.1)"
    shadow_xl = "0 20px 25px -5px rgba(0,0,0,0.1)"


class ColorPaletteDark(ColorPalette):
    """Dark theme color palette.

    Every accent is overridden, not just the surfaces. The accents in the
    light palette were chosen to be *dark* so that white text would sit on
    them; inheriting them here painted dark-blue text on dark-blue fills
    (ribbon tab selection measured 1.44:1 against a 4.5:1 requirement) and
    dark-blue group titles on a dark ribbon (2.05:1).

    The values are the same ramp stepped lighter -- blue-500/400/300,
    emerald-400, amber-400, red-400, cyan-400 -- so an accent stays the same
    hue in both themes and only its lightness moves.
    """

    bg_primary = "#0F172A"
    bg_secondary = "#1E293B"
    # Darker than the other surfaces on purpose: this is the one both
    # text_secondary and the primary accent have to sit on, and at #334155
    # they measured 4.04 and 4.07. Dropping the surface fixes both without
    # washing the accents out to chase a number.
    bg_tertiary = "#2A3546"
    bg_hover = "#475569"

    text_primary = "#F1F5F9"
    text_secondary = "#94A3B8"
    text_disabled = "#64748B"

    border_light = "#334155"
    border_medium = "#475569"
    border_focus = "#60A5FA"

    # Accents, re-stepped for a dark background.
    primary = "#60A5FA"
    primary_light = "#93C5FD"
    primary_dark = "#3B82F6"
    success = "#34D399"
    warning = "#FBBF24"
    error = "#F87171"
    info = "#22D3EE"

    # The accents are light now, so the text painted on them must be dark.
    on_primary = "#0F172A"

    hover_overlay = "rgba(59, 130, 246, 0.15)"
    active_overlay = "rgba(59, 130, 246, 0.25)"
    selected_overlay = "rgba(59, 130, 246, 0.3)"


# =============================================================================
# Spacing System (4px grid)
# =============================================================================


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

    body_lg_size = 15
    body_size = 13
    body_sm_size = 12
    caption_size = 11

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


def get_modern_stylesheet(palette: ColorPalette | None = None) -> str:
    """Generate comprehensive modern stylesheet for entire application."""
    colors = palette if palette is not None else ColorPalette()
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
    padding: {spacing.sm}px {spacing.lg}px;
    background: transparent;
    border-radius: {radius.md};
    margin: {spacing.xs}px {spacing.xs}px;
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
    padding: {spacing.xs}px 0;
    /* No `box-shadow` here on purpose -- Qt's style sheets have no shadow
       property, so the declaration was dropped with "Unknown property
       box-shadow" and the menu had no elevation. A real drop shadow needs a
       QGraphicsDropShadowEffect on the widget, not QSS. See
       tests/test_qss_supported_properties.py, which fails the build if an
       unsupported property is reintroduced. */
}}

QMenu::item {{
    padding: {spacing.sm}px {spacing.xl}px {spacing.sm}px 28px;
    border-radius: {radius.md};
    margin: {spacing.xs}px {spacing.xs}px;
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
    padding: {spacing.md}px {spacing.lg}px;
    min-width: 80px;    min-height: 36px;
    font-size: {typo.body_size}px;
    font-weight: {typo.medium};
    /* No `transition` here on purpose. Qt's style-sheet reference has no
       transition property -- writing one makes Qt print
       "Unknown property transition" and ignore it, so hover/press states
       have always snapped instantly rather than eased. Animating them for
       real needs a QPropertyAnimation on a widget property, not QSS; see
       tests/test_qss_supported_properties.py, which fails the build if an
       unsupported property is reintroduced here. */
}}

QPushButton:hover {{
    background-color: {colors.bg_tertiary};
    border-color: {colors.primary};
    color: {colors.primary};
}}

QPushButton:pressed {{
    background-color: {colors.primary};
    color: {colors.on_primary};
    border-color: {colors.primary_dark};
}}

QPushButton:disabled {{
    background-color: {colors.bg_tertiary};
    color: {colors.text_disabled};
    border-color: {colors.border_light};
}}

QPushButton[default="true"] {{
    background-color: {colors.primary};
    color: {colors.on_primary};
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
    padding: {spacing.sm}px {spacing.md}px;
    selection-background-color: {colors.primary};
    selection-color: {colors.on_primary};
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
    padding: {spacing.sm}px {spacing.md}px;
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
    padding: {spacing.sm}px {spacing.md}px;
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
    width: 20px;
    height: 20px;
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
    width: 20px;
    height: 20px;
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
    left: {spacing.lg}px;
    padding: 0 {spacing.sm}px;
    color: {colors.primary};
}}

QFrame {{
    background-color: {colors.bg_primary};
    border: none;
}}

QFrame[frameShape="4"] {{
    background-color: {colors.border_light};
    max-height: 1px;
    margin: {spacing.sm}px 0;
}}

/* =============================================================================
   SCROLLBARS
   ============================================================================= */

QScrollBar:vertical {{
    background: {colors.bg_secondary};
    width: {spacing.sm}px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:vertical {{
    background: {colors.border_medium};
    min-height: 20px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:vertical:hover {{
    background: {colors.text_secondary};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none;
    height: 0px;
}}

QScrollBar:horizontal {{
    background: {colors.bg_secondary};
    height: {spacing.sm}px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:horizontal {{
    background: {colors.border_medium};
    min-width: 20px;
    border-radius: {radius.sm};
}}

QScrollBar::handle:horizontal:hover {{
    background: {colors.text_secondary};
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
    padding: {spacing.sm}px {spacing.xs}px;
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
    padding: {spacing.sm}px {spacing.md}px;
    border-radius: {radius.md};
}}

QListWidget::item:hover {{
    background-color: {colors.hover_overlay};
}}

QListWidget::item:selected {{
    background-color: {colors.primary};
    color: {colors.on_primary};
    font-weight: {typo.medium};
}}

/* =============================================================================
   TABS
   ============================================================================= */

QTabBar::tab {{
    background-color: {colors.bg_secondary};
    color: {colors.text_secondary};
    border: none;
    padding: {spacing.md}px {spacing.xl}px;
    margin: 0;
    border-radius: {radius.lg} {radius.lg} 0 0;
}}

QTabBar::tab:hover {{
    background-color: {colors.bg_tertiary};
}}

QTabBar::tab:selected {{
    background-color: {colors.primary};
    color: {colors.on_primary};
    font-weight: {typo.medium};
}}

/* =============================================================================
   SLIDERS
   ============================================================================= */

QSlider::groove:horizontal {{
    height: {spacing.sm}px;
    background: {colors.border_light};
    border-radius: {radius.sm};
}}

QSlider::handle:horizontal {{
    /* 20px, not 18px: 18 is off the 4px grid this stylesheet keeps. The size
       is also chosen so the overhang matches the margin exactly -- handle 20,
       groove 8, so it hangs (20 - 8) / 2 = 6px past each edge, which is what
       the -6px below says. Declaring the height explicitly keeps that
       arithmetic checkable instead of leaving it to Qt's auto-sizing. */
    width: 20px;
    height: 20px;
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
    padding: {spacing.xs}px {spacing.sm}px;
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


def get_palette(dark: bool = False) -> ColorPalette:
    """Get the appropriate color palette for the current theme."""
    return ColorPaletteDark() if dark else ColorPalette()


def get_stylesheet(dark: bool = False) -> str:
    """Get the complete stylesheet for the current theme."""
    return get_modern_stylesheet(get_palette(dark))
