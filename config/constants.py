# =============================================================================
# FILE: config/constants.py
# =============================================================================
"""
Global Constants and Configuration Parameters for PaleoAST

This module defines all global constants used throughout the application,
including application metadata, GUI settings, and computational parameters.

Mathematical Notation Reference:
- n: Number of samples (rows)
- p: Number of variables (columns)
- k: Number of components/dimensions
- N: Total number of elements

Author: PaleoAST Development Team
version: 1.0.1
"""

from typing import Final

# =============================================================================
# APPLICATION METADATA
# =============================================================================

APP_NAME: Final[str] = "PaleoAST"
"""Application name - Paleontological Advanced Statistical Toolkit"""

APP_VERSION: Final[str] = "1.0.1"
"""Current version string following semantic versioning (MAJOR.MINOR.PATCH)"""

APP_AUTHOR: Final[str] = "PaleoAST Development Team"
"""Primary author or development team name"""

APP_DESCRIPTION: Final[str] = (
    "Professional desktop application for paleontological and paleoecological "
    "data analysis and visualization. Supports multivariate statistics, "
    "geometric morphometrics, biodiversity analysis, and stratigraphic methods."
)
"""Comprehensive application description"""


# =============================================================================
# GUI CONFIGURATION CONSTANTS
# =============================================================================

DEFAULT_FONT_FAMILY: Final[str] = "Arial"
"""
Default font family for the application UI.
Follows Nature/Science journal standards which recommend Arial or Helvetica.
"""

DEFAULT_FONT_SIZE: Final[int] = 10
"""
Default font size in points for regular UI text.
"""

WINDOW_MIN_WIDTH: Final[int] = 1280
"""
Minimum window width in pixels.
Ensures the application remains usable on smaller displays.
"""

WINDOW_MIN_HEIGHT: Final[int] = 720
"""
Minimum window height in pixels.
Standard 16:9 aspect ratio consideration.
"""

NAVIGATION_PANEL_WIDTH: Final[int] = 250
"""
Width of the left navigation tree panel in pixels.
"""

STATUSBAR_HEIGHT: Final[int] = 28
"""
Height of the bottom status bar in pixels.
"""

RIBBON_HEIGHT: Final[int] = 120
"""
Height of the ribbon menu bar in pixels when fully expanded.
"""


# =============================================================================
# SPREADSHEET CONFIGURATION
# =============================================================================

SPREADSHEET_MAX_ROWS: Final[int] = 1000000
"""
Maximum number of rows supported in the scientific spreadsheet.
Set to 1 million for handling large paleontological datasets.
"""

SPREADSHEET_MAX_COLS: Final[int] = 10000
"""
Maximum number of columns supported in the scientific spreadsheet.
Supports extensive variable sets in multivariate analyses.
"""

DEFAULT_CELL_WIDTH: Final[int] = 100
"""
Default column width in pixels for spreadsheet cells.
"""

DEFAULT_ROW_HEIGHT: Final[int] = 25
"""
Default row height in pixels for spreadsheet cells.
"""

HEADER_HEIGHT: Final[int] = 30
"""
Height of the header row in pixels.
"""


# =============================================================================
# COMPUTATIONAL PARAMETERS
# =============================================================================

MAX_WORKERS: Final[int] = 8
"""
Maximum number of worker threads for parallel computation.
Optimized for modern multi-core processors.
"""

PERMUTATION_TESTS: Final[int] = 9999
"""
Default number of permutations for statistical significance testing.
Using 9999 allows for precise p-value calculations:
p = (number of permuted statistics >= observed) / 10000
This provides 0.0001 precision in p-value estimation.
"""

NMDS_MAX_ITERATIONS: Final[int] = 500
"""
Maximum number of iterations for NMDS optimization per random restart.
SMACOF algorithm convergence criterion: stress improvement < 1e-6.
"""

NMDS_RANDOM_RESTARTS: Final[int] = 50
"""
Number of random initialization restarts for NMDS.
Multiple restarts help escape local minima in non-convex optimization.
"""

GPA_CONVERGENCE_TOLERANCE: Final[float] = 1e-8
"""
Convergence tolerance for Generalized Procrustes Analysis.
Iteration stops when maximum Procrustes distance change < tolerance.
"""

GPA_MAX_ITERATIONS: Final[int] = 100
"""
Maximum number of iterations for GPA convergence.
Typical convergence usually occurs within 20-30 iterations.
"""

EIGENVALUE_TOLERANCE: Final[float] = 1e-10
"""
Tolerance for considering an eigenvalue as effectively zero.
Used in PCA and factor analysis for determining rank deficiency.
"""

SVD_TOLERANCE: Final[float] = 1e-12
"""
Tolerance for singular value decomposition computations.
Determines numerical stability in matrix operations.
"""


# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

DPI_STANDARD: Final[int] = 300
"""
Standard DPI for publication-ready figure exports.
Nature and Science journals typically require 300 DPI minimum.
"""

DPI_SCREEN: Final[int] = 96
"""
Standard screen DPI for on-screen display rendering.
"""

FIGURE_WIDTH_STANDARD: Final[float] = 7.0
"""
Standard figure width in inches for single-column layouts.
Corresponds to Nature journal single column width.
"""

FIGURE_HEIGHT_STANDARD: Final[float] = 5.5
"""
Standard figure height in inches.
Maintains approximately 4:3 aspect ratio for optimal viewing.
"""

LINE_WIDTH_DEFAULT: Final[float] = 1.5
"""
Default line width for plot axes and curves in points.
"""

MARKER_SIZE_DEFAULT: Final[float] = 60
"""
Default marker size for scatter plots in points^2.
Area proportional: size parameter in matplotlib scatter().
"""

FONT_SIZE_TITLE: Final[int] = 14
"""
Font size for figure titles in points.
"""

FONT_SIZE_AXIS_LABEL: Final[int] = 12
"""
Font size for axis labels in points.
"""

FONT_SIZE_TICK_LABELS: Final[int] = 10
"""
Font size for axis tick labels in points.
"""

CONFIDENCE_ELLIPSE_ALPHA: Final[float] = 0.1
"""
Transparency (alpha) value for confidence ellipse fill.
0.0 = fully transparent, 1.0 = fully opaque.
"""

CONFIDENCE_LEVEL_DEFAULT: Final[float] = 0.95
"""
Default confidence level for statistical ellipses and intervals.
Corresponds to 95% confidence level (α = 0.05).
"""

GRID_ALPHA: Final[float] = 0.3
"""
Transparency of background grid lines in plots.
"""


# =============================================================================
# FILE FORMAT CONSTANTS
# =============================================================================

SUPPORTED_DATA_EXTENSIONS: Final[tuple] = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".mat")
"""
Tuple of supported data file extensions for import.
"""

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple] = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tiff", ".tif")
"""
Tuple of supported image export formats.
"""

EXPORT_DPI_OPTIONS: Final[list] = [72, 150, 300, 600]
"""
Common DPI options for figure export dialog.
"""


# =============================================================================
# DATA TYPE CONSTANTS
# =============================================================================


class DataType:
    """
    Enumeration of supported data types in the spreadsheet.

    These types follow standard statistical classification:
    - Nominal: Categorical without order (e.g., species names, sites)
    - Ordinal: Categorical with order (e.g., abundance scales, stages)
    - Continuous: Quantitative measurements (e.g., measurements, ratios)
    """

    NOMINAL: Final[str] = "nominal"
    """Nominal/categorical data type - unordered categories"""

    ORDINAL: Final[str] = "ordinal"
    """Ordinal data type - ordered categories with meaningful rank"""

    CONTINUOUS: Final[str] = "continuous"
    """Continuous data type - real-valued measurements"""

    BINARY: Final[str] = "binary"
    """Binary data type - presence/absence (0/1)"""

    COUNT: Final[str] = "count"
    """Count data type - non-negative integers"""


class DistanceMetric:
    """
    Enumeration of supported distance/similarity metrics.

    Mathematical definitions:
    - Euclidean: L2 norm distance
    - Manhattan: L1 norm distance
    - Bray-Curtis: Compositional dissimilarity (ecological)
    - Jaccard: Set-based similarity for presence/absence data
    """

    EUCLIDEAN: Final[str] = "euclidean"
    """Euclidean (L2) distance: d(x,y) = ||x-y||_2"""

    MANHATTAN: Final[str] = "manhattan"
    """Manhattan (L1) distance: d(x,y) = ||x-y||_1"""

    BRAY_CURTIS: Final[str] = "bray_curtis"
    """Bray-Curtis dissimilarity: (Σ|x_i-y_i|)/(Σ|x_i+y_i|)"""

    JACCARD: Final[str] = "jaccard"
    """Jaccard dissimilarity: 1 - |A∩B|/|A∪B|"""

    CANBERRA: Final[str] = "canberra"
    """Canberra distance: Σ|x_i-y_i|/(|x_i|+|y_i|)"""

    CHEBYCHEV: Final[str] = "chebychev"
    """Chebychev (L∞) distance: max|x_i-y_i|"""


# =============================================================================
# ERROR MESSAGE CONSTANTS
# =============================================================================

ERR_MATRIX_DIMENSION_MISMATCH: Final[str] = "Matrix dimension mismatch: expected shape {expected}, got {actual}"

ERR_MATRIX_NOT_SQUARE: Final[str] = "Matrix must be square for this operation. Got shape {shape}"

ERR_MATRIX_SINGULAR: Final[str] = "Matrix is singular or near-singular (determinant ≈ 0). Cannot compute inverse."

ERR_DIVISION_BY_ZERO: Final[str] = "Division by zero encountered in computation."

ERR_EIGENVALUE_CONVERGENCE: Final[str] = "Eigenvalue computation did not converge within maximum iterations."

ERR_DATA_EMPTY: Final[str] = "Input data is empty or contains only missing values."

ERR_INVALID_DATATYPE: Final[str] = (
    "Invalid data type specified: {dtype}. Allowed types: nominal, ordinal, continuous, binary, count"
)

ERR_FILE_NOT_FOUND: Final[str] = "File not found: {filepath}"

ERR_FILE_FORMAT_UNSUPPORTED: Final[str] = "File format not supported: {extension}"
