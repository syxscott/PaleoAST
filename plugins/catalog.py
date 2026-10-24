# =============================================================================
# FILE: plugins/catalog.py
# =============================================================================
"""
The catalog of first-party analyses, and an adapter that makes each one
reachable through the plugin registry.

WHY THIS EXISTS
~~~~~~~~~~~~~~~
The registry had a read side and no write side. The
StatisticsController's ``list_analyses()`` already merges the names of
its own ``analyze_*`` methods with whatever the registry holds, and
``run_plugin()`` already dispatches into it -- but nothing ever
registered anything, so ``list_plugins()`` returned ``[]`` on every call.
A scripting layer over that registry would have had nothing to script.

``plugins/loader.py`` carried a 7-entry ``_BUILTIN_PLUGINS`` tuple that
no code read, and whose comment records that it had previously been
documented as a complete inventory when it named 7 of 23. This module
replaces it with a table that is actually consumed, and
``tests/plugins/test_catalog.py`` asserts the table against the
analyzers that exist in the source tree -- so a new analyzer that
nobody added here fails a test rather than silently disappearing from
the scripting surface.

THE ADAPTER
~~~~~~~~~~~
None of the existing analyzers are ``AnalysisPlugin``
subclasses, and forcing them to be would mean changing 47 classes and
their constructors. Instead each catalog entry names a class AND a
method, and :class:`BuiltinAnalysisPlugin` wraps the pair. The method
part is not decoration: many analyzers have no ``analyze`` at all.
``DesignTestAnalyzer`` exposes ``two_way_anova`` and
``repeated_measures_anova``; ``MantelAnalyzer`` adds ``analyze_partial``
alongside ``analyze``; ``ARMAAnalyzer`` and ``CoxPHAnalyzer`` only have
``fit``. A catalog that only knew about ``analyze`` would reach roughly
half of them.

Imports are deferred to the first call, so filling the registry at
startup does not drag in matplotlib, scipy.spatial and every analysis
module before the main window exists.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any

from plugins.base import AnalysisPlugin, AnalysisResult, data_slot_names, sheet_columns
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisEntry:
    """One first-party analysis, addressable by name.

    Attributes:
        name: Stable identifier used by the registry and by scripts.
        module: Importable module path.
        class_name: The analyzer class inside it, or ``None`` when the
            target is a module-level function.
        method: The method or function to call. Not always ``analyze``.
        category: Grouping for menus and ``list_categories``.
        description: One line, for a menu tooltip or a script listing.
    """

    name: str
    module: str
    class_name: str | None
    method: str
    category: str
    description: str
    #: The parameters the sheet supplies, and in which order.  ``None``
    #: means the automatic rule applies: the sheet goes, whole, into one
    #: parameter -- which is right for an analysis that wants a MATRIX.
    #:
    #: A list is for an analysis that wants SERIES: the first name takes
    #: the sheet's first column, the second its second column, and so on.
    #: ``analyze(time, values)`` has TWO array parameters, and the
    #: automatic rule picked the first one -- so the whole sheet went in
    #: as the TIME axis, the values were never passed, and the analysis
    #: could not run at all.
    data_slots: tuple[str, ...] | None = None


def _e(
    name: str,
    module: str,
    class_name: str | None,
    method: str,
    category: str,
    description: str,
    data_slots: tuple[str, ...] | None = None,
) -> AnalysisEntry:
    """Terse constructor, so the table below stays readable."""
    return AnalysisEntry(name, module, class_name, method, category, description, data_slots=data_slots)


def _f(
    name: str,
    module: str,
    function: str,
    category: str,
    description: str,
    data_slots: tuple[str, ...] | None = None,
) -> AnalysisEntry:
    """Catalog entry for a module-level function rather than a class."""
    return AnalysisEntry(name, module, None, function, category, description, data_slots=data_slots)


# The catalog. Every entry is checked against the source tree by
# tests/plugins/test_catalog.py; do not add a name here that does not
# resolve.
BUILTIN_ANALYSES: tuple[AnalysisEntry, ...] = (
    # -- ordination ------------------------------------------------------
    _e("pca", "stats.pca", "PCAAnalyzer", "analyze", "ordination", "Principal component analysis"),
    _e("pcoa", "stats.pcoa", "PCoAAnalyzer", "analyze", "ordination", "Principal coordinate analysis"),
    _e("nmds", "stats.nmds", "NMDSAnalyzer", "analyze", "ordination", "Non-metric multidimensional scaling"),
    _e(
        "cca",
        "stats.cca",
        "CCAAnalyzer",
        "analyze",
        "ordination",
        "Canonical correspondence analysis / redundancy analysis",
    ),
    _e("lda", "stats.lda", "LDAAnalyzer", "analyze", "ordination", "Linear discriminant analysis"),
    _e(
        "ca",
        "stats.detriding",
        "DetrendedCAAnalyzer",
        "correspondence",
        "ordination",
        "Correspondence analysis of a contingency table",
    ),
    _e("dca", "stats.detriding", "DetrendedCAAnalyzer", "analyze", "ordination", "Detrended correspondence analysis"),
    # -- group comparison ------------------------------------------------
    _e("anosim", "stats.anosim", "ANOSIMAnalyzer", "analyze", "group-comparison", "ANOSIM group comparison"),
    _e("permanova", "stats.permanova", "PERMANOVAAnalyzer", "analyze", "group-comparison", "PERMANOVA"),
    _e(
        "simper",
        "stats.simper",
        "SimperAnalyzer",
        "analyze",
        "group-comparison",
        "SIMPER contribution of variables to group dissimilarity",
    ),
    _e("procd", "stats.procD_lm", None, "procD_lm", "group-comparison", "ProcD and sequential model tests"),
    _e(
        "phylo_anova",
        "stats.pcm",
        "PCMAnalyzer",
        "phylogenetic_anova",
        "group-comparison",
        "Phylogenetic ANOVA on a tree",
    ),
    _e(
        "phylogenetic_signal",
        "stats.pcm",
        "PCMAnalyzer",
        "compute_phylogenetic_signal",
        "group-comparison",
        "Blomberg K phylogenetic signal",
    ),
    # -- spatial association ---------------------------------------------
    _e("mantel", "stats.mantel", "MantelAnalyzer", "analyze", "spatial", "Mantel test between two distance matrices"),
    _e(
        "partial_mantel",
        "stats.mantel",
        "MantelAnalyzer",
        "analyze_partial",
        "spatial",
        "Partial Mantel test controlling for a third matrix",
    ),
    _e("ripley_k", "stats.spatial", "RipleyKAnalyzer", "analyze", "spatial", "Ripley's K point pattern analysis"),
    _e(
        "minimum_spanning_tree",
        "stats.geometry",
        "GeometryAnalyzer",
        "minimum_spanning_tree",
        "spatial",
        "Minimum spanning tree of a distance matrix",
    ),
    _e(
        "morphospace_disparity",
        "stats.geometry",
        "GeometryAnalyzer",
        "morphospace_disparity",
        "spatial",
        "Disparity through time of a morphospace",
    ),
    _e(
        "morans_i",
        "stats.spatial_stats",
        "SpatialStatsAnalyzer",
        "morans_i",
        "spatial",
        "Global spatial autocorrelation (Moran's I)",
        data_slots=("values",),
    ),
    _e(
        "grid_interpolate",
        "stats.spatial_stats",
        "SpatialStatsAnalyzer",
        "grid_interpolate",
        "spatial",
        "Interpolate scattered sites onto a regular grid",
        data_slots=("values",),
    ),
    _e(
        "nearest_neighbour",
        "stats.spatial_stats",
        "SpatialStatsAnalyzer",
        "nearest_neighbour_stats",
        "spatial",
        "Nearest-neighbour point pattern statistics",
    ),
    _e(
        "spherical",
        "stats.spatial_stats",
        "SpatialStatsAnalyzer",
        "spherical_stats",
        "spatial",
        "Mean direction and concentration on the sphere",
    ),
    # -- clustering -------------------------------------------------------
    _e(
        "hierarchical_clustering",
        "stats.clustering",
        "ClusteringAnalyzer",
        "analyze",
        "clustering",
        "Agglomerative hierarchical clustering",
    ),
    _e("kmeans", "stats.clustering", "ClusteringAnalyzer", "analyze_kmeans", "clustering", "K-means partitioning"),
    _e(
        "kmeans_elbow",
        "stats.clustering",
        "ClusteringAnalyzer",
        "elbow_curve",
        "clustering",
        "Within-cluster sum of squares against k",
    ),
    # -- univariate and designs -------------------------------------------
    _e(
        "one_way_anova",
        "stats.univariate",
        "UnivariateAnalyzer",
        "one_way_anova",
        "univariate",
        "One-way ANOVA with Tukey post-hoc",
    ),
    _e(
        "kruskal_wallis",
        "stats.univariate",
        "UnivariateAnalyzer",
        "kruskal_wallis",
        "univariate",
        "Kruskal-Wallis rank test",
    ),
    _e("mann_whitney", "stats.univariate", "UnivariateAnalyzer", "mann_whitney", "univariate", "Mann-Whitney U test"),
    _e("normality", "stats.univariate", "UnivariateAnalyzer", "normality_test", "univariate", "Normality tests"),
    _e(
        "summary_statistics",
        "stats.univariate",
        "UnivariateAnalyzer",
        "summary_statistics",
        "univariate",
        "Descriptive statistics",
    ),
    _e(
        "two_way_anova",
        "stats.design_tests",
        "DesignTestAnalyzer",
        "two_way_anova",
        "univariate",
        "Two-way factorial ANOVA with interaction",
    ),
    _e(
        "repeated_measures_anova",
        "stats.design_tests",
        "DesignTestAnalyzer",
        "repeated_measures_anova",
        "univariate",
        "Repeated-measures ANOVA",
    ),
    _e(
        "paired_rank_test",
        "stats.univariate",
        "UnivariateAnalyzer",
        "paired_rank_test",
        "univariate",
        "Paired sign or Wilcoxon signed-rank test",
    ),
    _e(
        "intraclass_correlation",
        "stats.design_tests",
        "DesignTestAnalyzer",
        "intraclass_correlation",
        "univariate",
        "Intraclass correlation coefficient",
    ),
    _e(
        "contingency_chi_square",
        "stats.design_tests",
        "DesignTestAnalyzer",
        "contingency_chi_square",
        "univariate",
        "Contingency table chi-square with Cramer's V",
    ),
    # -- diversity ---------------------------------------------------------
    _e("diversity", "ecology.diversity", "DiversityAnalyzer", "analyze_sample", "diversity", "Alpha diversity indices"),
    _e(
        "beta_diversity",
        "ecology.beta_diversity",
        "BetaDiversityAnalyzer",
        "decompose_beta_diversity",
        "diversity",
        "Beta diversity decomposition",
    ),
    _e(
        "rarefaction",
        "ecology.rarefaction",
        "RarefactionAnalyzer",
        "analyze",
        "diversity",
        "Individual and sample rarefaction",
    ),
    _e(
        "coverage_rarefaction",
        "ecology.beta_diversity",
        "CoverageRarefactionAnalyzer",
        "analyze",
        "diversity",
        "Sample-based coverage rarefaction",
    ),
    _e(
        "null_models",
        "ecology.null_models",
        "NullModelAnalyzer",
        "analyze",
        "diversity",
        "Null model randomisations of a community matrix",
    ),
    _e("she", "ecology.advanced", "SHEAnalyzer", "analyze", "diversity", "SHE analysis for diversity"),
    _e(
        "abundance_models",
        "ecology.advanced",
        "AbundanceModelFitter",
        "fit_all",
        "diversity",
        "Abundance model fitting: log-series, log-normal, geometric and broken-stick, ranked by AIC",
    ),
    _e(
        "dtw",
        "ecology.dtw",
        "DTWAnalyzer",
        "distance_matrix",
        "diversity",
        "Dynamic time warping and LB_Keogh distances",
    ),
    # -- stratigraphy ------------------------------------------------------
    _e("ua", "stratigraphy.biostratigraphy", "UAAnalyzer", "analyze", "stratigraphy", "Unitary associations"),
    _e(
        "rasc",
        "stratigraphy.biostratigraphy",
        "RASCAnalyzer",
        "analyze",
        "stratigraphy",
        "Ranging ancilla time scaling",
    ),
    _e(
        "coniss",
        "stratigraphy.coniss",
        "CONISSAnalyzer",
        "analyze",
        "stratigraphy",
        "CONISS constrained cluster analysis",
    ),
    _e(
        "stratigraphic_correlation",
        "stratigraphy.correlation",
        "StratigraphicCorrelationAnalyzer",
        "analyze",
        "stratigraphy",
        "Composite standard section correlation",
    ),
    _e(
        "age_model", "stratigraphy.correlation", "AgeModelAnalyzer", "build_model", "stratigraphy", "Depth to age model"
    ),
    _e(
        "sedimentation_rate",
        "stratigraphy.correlation",
        "SedimentationRateAnalyzer",
        "calculate",
        "stratigraphy",
        "Sedimentation rate from an age model",
    ),
    _e(
        "spectral",
        "stratigraphy.spectral_analysis",
        "SpectralAnalyzer",
        "analyze",
        "time-series",
        "Periodogram with an AR(1) red-noise test",
        data_slots=("time", "values"),
    ),
    _e(
        "wavelet",
        "stratigraphy.spectral_analysis",
        "SpectralAnalyzer",
        "wavelet_transform",
        "time-series",
        "Continuous wavelet transform",
        data_slots=("time", "values"),
    ),
    _e(
        "spectral_peaks",
        "stratigraphy.spectral_analysis",
        "SpectralAnalyzer",
        "find_significant_peaks",
        "time-series",
        "Significant spectral periods",
    ),
    _e(
        "acf",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "autocorrelation",
        "time-series",
        "Autocorrelation function with Bartlett bands",
        data_slots=("values",),
    ),
    _e(
        "redfit",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "redfit",
        "time-series",
        "REDFIT spectral analysis against an AR(1) null",
        data_slots=("time", "values"),
    ),
    _e(
        "multitaper",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "multitaper",
        "time-series",
        "Multitaper spectral estimation",
        data_slots=("values",),
    ),
    _e(
        "prewhiten",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "prewhiten",
        "time-series",
        "AR(1) prewhitening of a series",
        data_slots=("values",),
    ),
    _e(
        "cross_correlation",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "cross_correlation",
        "time-series",
        "Cross-correlation with a signed lag axis",
        data_slots=("x", "y"),
    ),
    _e(
        "autoassociation",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "autoassociation",
        "time-series",
        "Autoassociation function for cyclostratigraphy",
        data_slots=("time", "values"),
    ),
    _e(
        "mann_kendall",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "mann_kendall",
        "time-series",
        "Mann-Kendall trend test with Sen's slope",
        data_slots=("values",),
    ),
    _e(
        "runs_test",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "runs",
        "time-series",
        "Runs test for randomness of a series",
        data_slots=("values",),
    ),
    _e(
        "orbital_forcing",
        "stratigraphy.cycles",
        None,
        "orbital_forcing",
        "time-series",
        "Milankovitch periods and relative amplitudes",
    ),
    _e(
        "insolation_series",
        "stratigraphy.cycles",
        None,
        "insolation_series",
        "time-series",
        "Insolation forcing sampled on a time axis",
        data_slots=("times_ma",),
    ),
    _e(
        "arma",
        "stratigraphy.arma",
        "ARMAAnalyzer",
        "fit",
        "time-series",
        "Autoregressive moving average model",
        data_slots=("times", "values"),
    ),
    _e("markov", "stratigraphy.markov", "MarkovAnalyzer", "analyze", "time-series", "Markov chain state transitions"),
    _e(
        "extinction",
        "stratigraphy.extinction",
        "ExtinctionIntervalAnalyzer",
        "analyze",
        "stratigraphy",
        "Extinction interval estimation",
        data_slots=("lad_positions",),
    ),
    _e(
        "directional",
        "stratigraphy.directional",
        "DirectionalAnalyzer",
        "analyze",
        "directional",
        "Circular and axial directional statistics",
        data_slots=("angles_deg",),
    ),
    _e(
        "isotope_paleotemperature",
        "stratigraphy.isotope_analysis",
        "IsotopeAnalyzer",
        "analyze",
        "time-series",
        "Stable isotope paleothermometry",
    ),
    # -- morphometrics ------------------------------------------------------
    _e("gpa", "morphometrics.gpa", "GPAAnalyzer", "analyze", "morphometrics", "Generalised Procrustes analysis"),
    _e(
        "relative_warps",
        "morphometrics.relative_warps",
        "RelativeWarpsAnalyzer",
        "analyze",
        "morphometrics",
        "Relative warps analysis",
    ),
    _e("tps", "morphometrics.tps", "TPSAnalyzer", "analyze", "morphometrics", "Thin-plate spline deformation"),
    _e("efa", "morphometrics.efa", "EFAAnalyzer", "analyze", "morphometrics", "Elliptic Fourier analysis"),
    _e("eigenshape", "morphometrics.efa", "EigenshapeAnalyzer", "analyze", "morphometrics", "Eigenshape analysis"),
    _e(
        "allometry",
        "morphometrics.allometry",
        "AllometryAnalyzer",
        "analyze_allometry",
        "morphometrics",
        "Multivariate allometry",
    ),
    _e(
        "pls",
        "morphometrics.allometry",
        "IntegrationAnalyzer",
        "analyze_pls",
        "morphometrics",
        "Partial least squares integration",
    ),
    _e(
        "evolution_rate",
        "morphometrics.evolution_rate",
        "EvolutionRateAnalyzer",
        "analyze",
        "morphometrics",
        "Evolutionary rate estimation",
        data_slots=("trait_series", "time_intervals"),
    ),
    # -- macroevolution ------------------------------------------------------
    _e(
        "kaplan_meier",
        "macroevolution.survival",
        "KaplanMeierAnalyzer",
        "fit",
        "macroevolution",
        "Kaplan-Meier survivorship",
        data_slots=("times", "events"),
    ),
    _e(
        "cox_ph",
        "macroevolution.survival",
        "CoxPHAnalyzer",
        "fit",
        "macroevolution",
        "Cox proportional hazards model",
        data_slots=("durations", "events"),
    ),
    # -- growth ---------------------------------------------------------------
    _e(
        "growth_model",
        "models.growth_models",
        "GrowthModelAnalyzer",
        "fit",
        "growth",
        "Nonlinear growth curve fitting",
        data_slots=("times", "values"),
    ),
    _e(
        "growth_model_comparison",
        "models.growth_models",
        "GrowthModelAnalyzer",
        "fit_all",
        "growth",
        "Fit several growth curves and rank them by AICc",
        data_slots=("times", "values"),
    ),
    # ---------------------------------------------------------------------------
    # The plot menu.
    #
    # Registered here for one reason that is really two. The catalog is what
    # PaleoAST.spec reads to decide what to freeze, and PyInstaller follows
    # imports rather than names -- so a plotting module reached by name from
    # the UI is absent from the executable, and the failure surfaces on the
    # user's machine as an ImportError from a menu item. That already happened
    # once: six catalogued analyses were missing from the build while every
    # local check passed. Registering them means a plot kind cannot be added
    # without also being frozen.
    #
    # They are reachable by name from the script console as well, because
    # register_builtin_analyses() wraps whatever is catalogued.
    # ---------------------------------------------------------------------------
    _e(
        "xy_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_xy",
        "plot",
        "XY scatter with an optional fitted line and log axes",
    ),
    _e(
        "xy_error_bars",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_xy_with_errorbars",
        "plot",
        "XY scatter with error bars",
    ),
    _e(
        "histogram",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_histogram",
        "plot",
        "Histogram of one column",
    ),
    _e(
        "bar_chart",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_bar_chart",
        "plot",
        "Bar chart of a column, or sums per category",
    ),
    _e(
        "box_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_box_plot",
        "plot",
        "Box plot of one column per group",
    ),
    _e(
        "pie_chart",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_pie_chart",
        "plot",
        "Pie chart of a column's values or labels",
    ),
    _e(
        "stacked_chart",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_stacked_chart",
        "plot",
        "Stacked bars over the rows of a matrix",
    ),
    _e(
        "percentiles",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_percentiles",
        "plot",
        "Percentile plot: a box plot without outliers",
    ),
    _e(
        "normal_probability_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_normal_probability",
        "plot",
        "Normal probability (Q-Q) plot",
    ),
    _e(
        "ternary_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_ternary",
        "plot",
        "Ternary plot of three components",
    ),
    _e(
        "bubble_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_bubble",
        "plot",
        "Scatter with a third variable as point area",
    ),
    _e(
        "matrix_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_matrix",
        "plot",
        "Matrix plot of a 2-D array",
    ),
    _e(
        "mosaic_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_mosaic",
        "plot",
        "Mosaic plot of a contingency table",
    ),
    _e(
        "venn_diagram",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_venn",
        "plot",
        "Venn diagram for two or three sets",
    ),
    _e(
        "radar_chart",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_radar",
        "plot",
        "Radar chart: one polygon per matrix row",
    ),
    _e(
        "polar_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_polar",
        "plot",
        "Polar plot of one column",
    ),
    _e(
        "vector_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_vectors",
        "plot",
        "Vector (quiver) field",
    ),
    _e(
        "network_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_network",
        "plot",
        "Network plot from an edge list",
    ),
    _e(
        "plot_3d",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_3d",
        "plot",
        "3-D scatter, line or bubble plot",
    ),
    _e(
        "surface_plot",
        "visualization.plot_menu",
        "PlotMenuAnalyzer",
        "plot_surface",
        "plot",
        "3-D surface over a grid",
    ),
    # -- the univariate extra tests, through their facade --------------------
    # The analyzers themselves take numpy arrays, which the generated form
    # cannot fill: it would hand the whole sheet to the first sample and the
    # form's column number to the second. The facade takes the sheet and
    # column numbers and does the extraction, so these entries are the ones
    # the analysis runner can actually drive.
    _e(
        "brunner_munzel",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "brunner_munzel",
        "univariate",
        "Brunner-Munzel test for stochastic equality of two samples",
    ),
    _e(
        "mood_median",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "mood_median",
        "univariate",
        "Mood's median test for a common median",
    ),
    _e(
        "fligner_kileen",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "fligner_kileen",
        "univariate",
        "Fliegler-Kileen test for equality of dispersion between groups",
    ),
    _e(
        "outlier_screen",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "outlier_screen",
        "univariate",
        "Outlier screening by z-score or by Grubbs' test",
    ),
    _e(
        "cochran_mantel_haenszel",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "cochran_mantel_haenszel",
        "univariate",
        "Cochran-Mantel-Haenszel test on a 2x2 table of four counts",
    ),
    _e(
        "proportion_test",
        "stats.univariate_menu",
        "UnivariateExtraMenu",
        "proportion_test",
        "univariate",
        "Exact binomial test of one proportion against a null value",
    ),
    # -- the two-sample distribution tests, through their facade -------------
    _e(
        "ks_2samp",
        "stats.distribution_menu",
        "DistributionTestMenu",
        "ks_2samp",
        "distribution",
        "Kolmogorov-Smirnov test of two samples",
    ),
    _e(
        "epps_singleton",
        "stats.distribution_menu",
        "DistributionTestMenu",
        "epps_singleton",
        "distribution",
        "Epps-Singleton test of two samples",
    ),
    # -- the k-sample omnibus test: one population, or several? -----------
    _f(
        "anderson_darling_ksample",
        "stats.anderson_darling",
        "anderson_darling_ksample",
        "distribution",
        "Anderson-Darling test of whether several samples share one population",
        data_slots=("values", "groups"),
    ),
    # -- the diversity sampling tools, through their facade ------------------
    _e(
        "sqs_subsample",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "subsample",
        "diversity",
        "Shareholder-quorum subsample of one sample",
    ),
    _e(
        "sample_coverage",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "coverage",
        "diversity",
        "Sample coverage of a samples-by-taxa table",
    ),
    _e(
        "sample_rarefaction_tao",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "rarefaction",
        "diversity",
        "Tao rarefaction of a samples-by-taxa table",
    ),
    _e(
        "diversity_profile",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "profile",
        "diversity",
        "Diversity indices of one sample across sampling depths",
    ),
    _e(
        "diversity_t_test",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "t_test",
        "diversity",
        "t-test of the diversity indices between groups of samples",
    ),
    _e(
        "diversity_permutation_test",
        "ecology.diversity_menu",
        "DiversityExtraMenu",
        "permutation_test",
        "diversity",
        "Permutation test of the diversity indices between groups of samples",
    ),
    # ---------------------------------------------------------------------------
    # Cyclostratigraphy and astrochronology.
    #
    # These are the analyses that turn a measured record into an age scale:
    # what is periodic in it, how fast it accumulated, and how that ties to
    # the astronomical record.  They want SERIES, so each one declares the
    # columns it reads; see ``AnalysisEntry.data_slots``.
    # ---------------------------------------------------------------------------
    _f(
        "evolutionary_spectrum",
        "stratigraphy.evolutionary",
        "evolutionary_spectrum",
        "time-series",
        "Sliding-window spectrum: how the periodic content moves through a record",
        data_slots=("time", "values"),
    ),
    _f(
        "evolutionary_lomb_scargle",
        "stratigraphy.evolutionary",
        "evolutionary_lomb_scargle",
        "time-series",
        "Sliding-window Lomb-Scargle spectrum for unevenly spaced records",
        data_slots=("time", "values"),
    ),
    _f(
        "evolutionary_multitaper",
        "stratigraphy.evolutionary",
        "evolutionary_multitaper",
        "time-series",
        "Sliding-window multitaper spectrum",
        data_slots=("time", "values"),
    ),
    _f(
        "amplitude_modulation",
        "stratigraphy.amplitude_modulation",
        "amplitude_modulation",
        "time-series",
        "Instantaneous amplitude envelope of a record (Tanner-Hilbert)",
        data_slots=("time", "values"),
    ),
    _f(
        "tanner_hilbert",
        "stratigraphy.amplitude_modulation",
        "tanner_hilbert",
        "time-series",
        "Tanner-Hilbert transform: envelope and instantaneous phase",
        data_slots=("time", "values"),
    ),
    _f(
        "gaussian_filter",
        "stratigraphy.filtering",
        "gaussian_filter",
        "time-series",
        "Gaussian low-pass filter",
        data_slots=("values",),
    ),
    _f(
        "taner_filter",
        "stratigraphy.filtering",
        "taner_filter",
        "time-series",
        "Taner weighted band-pass filter (Kodama & Hinnov 2015)",
        data_slots=("values",),
    ),
    _f(
        "bandpass_filter",
        "stratigraphy.filtering",
        "bandpass_filter",
        "time-series",
        "Band-pass filter between two periods",
        data_slots=("values",),
    ),
    _f(
        "change_points",
        "stratigraphy.change_points",
        "detect_change_points",
        "time-series",
        "Bayesian change-point detection (Ruggieri 2013)",
        data_slots=("values",),
    ),
    _f(
        "emd",
        "stratigraphy.emd",
        "emd",
        "time-series",
        "Empirical mode decomposition",
        data_slots=("values",),
    ),
    _f(
        "eemd",
        "stratigraphy.emd",
        "eemd",
        "time-series",
        "Ensemble empirical mode decomposition",
        data_slots=("values",),
    ),
    _f(
        "pda",
        "stratigraphy.pda",
        "power_decomposition",
        "time-series",
        "Power decomposition analysis into orbital-period bands (Li et al. 2016)",
        data_slots=("time", "values"),
    ),
    _f(
        "interpolate_series",
        "stratigraphy.series_tools",
        "interpolate_series",
        "time-series",
        "Interpolate a record onto a new time axis",
        data_slots=("time", "values"),
    ),
    _f(
        "select_intervals",
        "stratigraphy.series_tools",
        "select_intervals",
        "time-series",
        "Take the parts of a record that fall inside given intervals",
        data_slots=("time", "values"),
    ),
    _f(
        "add_gaps",
        "stratigraphy.series_tools",
        "add_gaps",
        "time-series",
        "Insert gaps into a record",
        data_slots=("time", "values"),
    ),
    _f(
        "remove_peaks",
        "stratigraphy.series_tools",
        "remove_peaks",
        "time-series",
        "Remove peaks above a threshold",
        data_slots=("time", "values"),
    ),
    _f(
        "derivative",
        "stratigraphy.series_tools",
        "derivative",
        "time-series",
        "Numerical derivative of a record",
        data_slots=("time", "values"),
    ),
    _f(
        "clip_values",
        "stratigraphy.series_tools",
        "clip_values",
        "time-series",
        "Clip a record to a range",
        data_slots=("values",),
    ),
    _f(
        "sort_unique",
        "stratigraphy.series_tools",
        "sort_unique",
        "time-series",
        "Sort a column and drop its duplicates",
        data_slots=("values",),
    ),
    _f(
        "generate_signal",
        "stratigraphy.signal_generator",
        "generate_signal",
        "time-series",
        "Synthesise a signal from components plus optional noise",
        data_slots=("time",),
    ),
    _f(
        "coco",
        "stratigraphy.coco",
        "coco",
        "stratigraphy",
        "Correlation coefficient: the most likely sedimentation rate (Li et al. 2018)",
        data_slots=("time", "values"),
    ),
    _f(
        "ecoco",
        "stratigraphy.coco",
        "ecoco",
        "stratigraphy",
        "Evolutionary correlation coefficient: a sedimentation rate that changes",
        data_slots=("time", "values"),
    ),
    _f(
        "time_opt",
        "stratigraphy.tuning",
        "time_opt",
        "stratigraphy",
        "TimeOpt astrochronological tuning (Meyers 2015)",
        data_slots=("depths", "values"),
    ),
    _f(
        "etime_opt",
        "stratigraphy.tuning",
        "etime_opt",
        "stratigraphy",
        "eTimeOpt astrochronological tuning with a changing sedimentation rate (Meyers 2019)",
        data_slots=("depths", "values"),
    ),
    _f(
        "age_scale",
        "stratigraphy.tuning",
        "build_age_scale",
        "stratigraphy",
        "Build an age scale from depth anchors",
        data_slots=("depths", "anchors", "anchor_ages_ma"),
    ),
    _f(
        "sedimentation_rate_age_model",
        "stratigraphy.tuning",
        "sedimentation_rate_to_age_model",
        "stratigraphy",
        "Turn a sedimentation rate into an age model",
        data_slots=("depths",),
    ),
    _f(
        "spectral_moments",
        "stratigraphy.spectral_moments",
        "spectral_moments",
        "time-series",
        "Spectral moments: the shape of a power spectrum",
        data_slots=("power", "frequencies"),
    ),
    _f(
        "estimate_ar1_rho",
        "stratigraphy.spectral_moments",
        "estimate_ar1_rho",
        "time-series",
        "The AR(1) coefficient of a record",
        data_slots=("values",),
    ),
    _f(
        "minimum_rho",
        "stratigraphy.spectral_moments",
        "minimum_rho",
        "time-series",
        "The smallest AR(1) coefficient that explains a record",
        data_slots=("values",),
    ),
    _f(
        "dynamic_filter",
        "stratigraphy.dynamic_filter",
        "dynamic_filter",
        "time-series",
        "Adaptive filter: the noise level decides how hard each point is smoothed",
        data_slots=("values",),
    ),
    _f(
        "fixed_bandwidth_summary",
        "stratigraphy.moving_summary",
        "fixed_bandwidth_summary",
        "time-series",
        "Mean, spread and extremes over a window of fixed time width",
        data_slots=("time", "values"),
    ),
    _f(
        "fixed_count_summary",
        "stratigraphy.moving_summary",
        "fixed_count_summary",
        "time-series",
        "Mean, spread and extremes over a window of fixed sample count",
        data_slots=("time", "values"),
    ),
    # -- the astronomical solution ------------------------------------------
    _f(
        "astronomical_solution",
        "stratigraphy.astronomy",
        "load_astronomical_solution",
        "stratigraphy",
        "The astronomical solution: orbital elements through deep time",
    ),
    _e(
        "insolation",
        "stratigraphy.astronomy",
        "InsolationAnalyzer",
        "analyze",
        "stratigraphy",
        "Daily-mean insolation at a latitude, from the astronomical solution",
        data_slots=("times_ma",),
    ),
    _f(
        "astronomical_target_curve",
        "stratigraphy.astronomy",
        "astronomical_target_curve",
        "stratigraphy",
        "A synthetic astronomical target curve at one Milankovitch period",
        data_slots=("times_ma",),
    ),
    _f(
        "milankovitch_periods",
        "stratigraphy.astronomy",
        "milankovitch_periods",
        "stratigraphy",
        "The Milankovitch periods and their nominal amplitudes",
    ),
    # -- coherence between two records, and the spectrum of an angle ------
    _f(
        "coherence",
        "stratigraphy.coherence",
        "coherence",
        "time-series",
        "Coherence and phase between two records",
        data_slots=("time", "first", "second"),
    ),
    # -- the bispectrum, and its surrogate threshold ---------------------
    _f(
        "bispectrum",
        "stratigraphy.bispectrum",
        "bispectrum",
        "time-series",
        "The bispectrum of a record, and the phase of its triples",
        data_slots=("values",),
    ),
    _f(
        "bicoherence_significance",
        "stratigraphy.bispectrum",
        "bicoherence_significance",
        "time-series",
        "How strong the triple coupling has to be to be more than chance",
        data_slots=("values",),
    ),
    # -- the AR(1) coefficient of a record, and a spectrum that survives it
    _f(
        "evolutionary_rho1",
        "stratigraphy.evolutionary_rho",
        "evolutionary_rho1",
        "time-series",
        "The AR(1) coefficient of a record, tracked through a sliding window",
        data_slots=("time", "values"),
    ),
    _f(
        "robust_lomb_scargle",
        "stratigraphy.robust_lomb",
        "robust_lomb_scargle",
        "time-series",
        "A Lomb-Scargle spectrum that a red-noise background cannot explain away",
        data_slots=("time", "values"),
    ),
    # -- the recurrence plot, and what it says about a record ------------
    _f(
        "recurrence_plot",
        "stratigraphy.recurrence",
        "recurrence_plot",
        "time-series",
        "The recurrence plot of a record, and the numbers read off it",
        data_slots=("values",),
    ),
    # -- the age-depth model, with its uncertainty band -------------------
    _f(
        "age_depth_uncertainty",
        "stratigraphy.age_depth_uncertainty",
        "undatable_age_depth_model",
        "stratigraphy",
        "An age-depth model with the band its dating points imply",
        data_slots=("depths", "ages"),
    ),
    # -- the rank-correlation variant of COCO, and its interval ------------
    _f(
        "coco_spearman",
        "stratigraphy.coco_variants",
        "coco_spearman",
        "stratigraphy",
        "COCO with the rank correlation instead of the Pearson one",
        data_slots=("time", "values"),
    ),
    _f(
        "correlation_confidence_interval",
        "stratigraphy.coco_variants",
        "correlation_confidence_interval",
        "stratigraphy",
        "The confidence interval of a correlation coefficient",
        data_slots=("x", "y"),
    ),
    _f(
        "ecoco_row_summary",
        "stratigraphy.coco_variants",
        "ecoco_row_summary",
        "stratigraphy",
        "The eCOCO grid, summarised window by window",
        data_slots=("time", "values"),
    ),
    # -- the nine principal orbital frequencies ---------------------------
    _e(
        "orbital_frequencies",
        "stratigraphy.orbital_frequencies",
        "OrbitalFrequenciesAnalyzer",
        "analyze",
        "stratigraphy",
        "The nine principal orbital frequencies at an age",
        data_slots=("age_ma",),
    ),
    # -- the radiocarbon calibration ---------------------------------------
    _e(
        "radiocarbon_calibration",
        "stratigraphy.radiocarbon",
        "RadiocarbonCalibrationAnalyzer",
        "analyze",
        "stratigraphy",
        "A radiocarbon determination turned into a calendar age",
        data_slots=("lab_det", "lab_err"),
    ),
    # -- the wavelet coherence between two records -------------------------
    _f(
        "wavelet_coherence",
        "stratigraphy.wavelet_coherence",
        "wavelet_coherence",
        "time-series",
        "The wavelet coherence between two records, and its phase",
        data_slots=("values_x", "values_y", "scales"),
    ),
    # -- the small, independent tools -------------------------------------
    _f(
        "chi_square_confidence_interval",
        "stratigraphy.small_tools",
        "chi_square_confidence_interval",
        "time-series",
        "The confidence band of a red-noise spectrum",
        data_slots=("observed",),
    ),
    _f(
        "cosine_taper",
        "stratigraphy.small_tools",
        "cosine_taper",
        "time-series",
        "The cosine taper that takes the edge off a record",
    ),
    _f(
        "date_to_day_number",
        "stratigraphy.small_tools",
        "date_to_day_number",
        "stratigraphy",
        "A calendar date as a day number",
    ),
    _f(
        "detrend_polynomial_pp",
        "stratigraphy.small_tools",
        "detrend_polynomial_pp",
        "time-series",
        "A polynomial baseline fitted in a sliding window",
        data_slots=("values",),
    ),
    _f(
        "domain_transform",
        "stratigraphy.small_tools",
        "domain_transform",
        "time-series",
        "The analytic signal of a record, or its cumulative integral",
        data_slots=("values",),
    ),
    _f(
        "find_n_peaks",
        "stratigraphy.small_tools",
        "find_n_peaks",
        "time-series",
        "The positions of the local maxima of a record",
        data_slots=("values",),
    ),
    _f(
        "fixed_bandwidth_moving_average",
        "stratigraphy.small_tools",
        "fixed_bandwidth_moving_average",
        "time-series",
        "A moving average over a fixed span of time rather than of samples",
        data_slots=("values",),
    ),
    _f(
        "gaussian_matched_filter",
        "stratigraphy.small_tools",
        "gaussian_matched_filter",
        "time-series",
        "Where a known shape best fits a record",
        data_slots=("values", "target"),
    ),
    _f(
        "pad_series",
        "stratigraphy.small_tools",
        "pad_series",
        "time-series",
        "A record padded out to a given length",
        data_slots=("values",),
    ),
    _f(
        "phase_diagram",
        "stratigraphy.small_tools",
        "phase_diagram",
        "time-series",
        "The trajectory of a record in the complex plane",
        data_slots=("values_x",),
    ),
    _f(
        "random_phase_resample",
        "stratigraphy.small_tools",
        "random_phase_resample",
        "time-series",
        "A surrogate of a record, with its phases randomised",
        data_slots=("values",),
    ),
    _f(
        "robust_linear_fit",
        "stratigraphy.small_tools",
        "robust_linear_fit",
        "time-series",
        "A straight line through points that include outliers",
        data_slots=("x", "y"),
    ),
    _f(
        "tobit_regression",
        "stratigraphy.small_tools",
        "tobit_regression",
        "time-series",
        "A regression with a censored response",
        data_slots=("x", "y"),
    ),
    _f(
        "wavelet_smoothing",
        "stratigraphy.small_tools",
        "wavelet_smoothing",
        "time-series",
        "A record denoised in the wavelet domain",
        data_slots=("values",),
    ),
    # -- the spectral background, and the tests that use it ---------------
    _f(
        "weedon_periodogram",
        "stratigraphy.spectral_significance",
        "weedon_periodogram",
        "time-series",
        "A periodogram with the degrees of freedom the false-discovery rate needs",
        data_slots=("time", "values"),
    ),
    _f(
        "swa_background",
        "stratigraphy.spectral_significance",
        "swa_background",
        "time-series",
        "The smoothed-window-average background of a spectrum",
        data_slots=("frequencies", "power"),
    ),
    _f(
        "swa_fdr",
        "stratigraphy.spectral_significance",
        "swa_fdr",
        "time-series",
        "Which peaks of a spectrum survive the false-discovery rate",
        data_slots=("frequencies", "power", "swa"),
    ),
    _f(
        "bayesian_periodogram",
        "stratigraphy.spectral_significance",
        "bayesian_periodogram",
        "time-series",
        "The probability that a period in a record is real",
        data_slots=("time", "values"),
    ),
    # -- the second Acycle pass: the small, independent capabilities ---
    _f(
        "box_cox_transform",
        "stratigraphy.boxcox",
        "box_cox_transform",
        "time-series",
        "Box-Cox transform, with the lambda solved for",
        data_slots=("values",),
    ),
    _f(
        "depth_to_time",
        "stratigraphy.depth_time",
        "depth_to_time",
        "stratigraphy",
        "Depth to age, through a set of age anchors",
        data_slots=("depths", "age_anchors", "anchor_ages_ma"),
    ),
    _f(
        "dispersion_statistic",
        "stratigraphy.dispersion",
        "dispersion_statistic",
        "time-series",
        "How far a set of counts departs from what was expected",
        data_slots=("observed",),
    ),
    _f(
        "inverse_percentile",
        "stratigraphy.inverse_percentile",
        "inverse_percentile",
        "time-series",
        "The value below which a given fraction of a sample falls",
        data_slots=("values",),
    ),
    _f(
        "lead_lag_rmse",
        "stratigraphy.lead_lag",
        "lead_lag_rmse",
        "time-series",
        "The lag that makes two records agree best, by least squares",
        data_slots=("first", "second"),
    ),
    _f(
        "moving_median",
        "stratigraphy.moving_median",
        "moving_median",
        "time-series",
        "Moving median filter",
        data_slots=("values",),
    ),
    _f(
        "multitaper_taper_count",
        "stratigraphy.mtm_tuning",
        "select_taper_count",
        "time-series",
        "The number of tapers a multitaper estimate should use",
    ),
    _f(
        "tuned_multitaper_spectrum",
        "stratigraphy.mtm_tuning",
        "tuned_multitaper_spectrum",
        "time-series",
        "Multitaper spectrum with the tapers and the degrees of freedom chosen",
        data_slots=("values",),
    ),
    _f(
        "test_uniform_sampling",
        "stratigraphy.sampling",
        "test_uniform_sampling",
        "time-series",
        "Whether a record was sampled at even intervals",
        data_slots=("time",),
    ),
    # -- the advanced COCO modes, and the orbital bookkeeping ------------
    _f(
        "coco_adaptive",
        "stratigraphy.coco_advanced",
        "adaptive_coco",
        "stratigraphy",
        "COCO with a target spectrum that adapts to the record",
        data_slots=("time", "values"),
    ),
    _f(
        "coco_crossfit",
        "stratigraphy.coco_advanced",
        "crossfit_coco",
        "stratigraphy",
        "COCO that estimates and tests on separate halves of the record",
        data_slots=("time", "values"),
    ),
    _f(
        "coco_interleaved",
        "stratigraphy.coco_advanced",
        "interleaved_coco",
        "stratigraphy",
        "COCO over interleaved short segments, for records with gaps",
        data_slots=("time", "values"),
    ),
    _f(
        "coco_orbit_tracking",
        "stratigraphy.coco_advanced",
        "coco_orbit_tracking",
        "stratigraphy",
        "The orbital peaks of a COCO result, followed through the record",
        data_slots=("time", "values"),
    ),
    _f(
        "fixed_target_weights",
        "stratigraphy.coco_advanced",
        "fixed_target_weights",
        "stratigraphy",
        "The weights a fixed target spectrum gives its periods",
        data_slots=("periods_kyr",),
    ),
    _f(
        "num_orbits",
        "stratigraphy.coco_advanced",
        "num_orbits",
        "stratigraphy",
        "How many orbital cycles a record can hold",
        data_slots=("periods_kyr",),
    ),
    _f(
        "orbit_groups",
        "stratigraphy.coco_advanced",
        "orbit_groups",
        "stratigraphy",
        "The orbital periods, grouped into the orbits they belong to",
        data_slots=("periods_kyr",),
    ),
    # -- the sliding spectrum, and the rate it gives back ----------------
    _f(
        "sliding_spectral_moments",
        "stratigraphy.sliding_spectral_moments",
        "sliding_spectral_moments",
        "time-series",
        "The shape of the spectrum, measured in a sliding window",
        data_slots=("time", "values"),
    ),
    # -- the frequency band that follows a drifting cycle -----------------
    _f(
        "frequency_stabilization",
        "stratigraphy.frequency_stabilization",
        "frequency_stabilization_filter",
        "stratigraphy",
        "A band-pass filter whose passband follows a drifting cycle",
        data_slots=("coordinate", "values"),
    ),
    # -- the Milankovitch frequencies, and how uncertain they are ---------
    _e(
        "milankovitch_uncertainty",
        "stratigraphy.milankovitch_uncertainty",
        "MilankovitchUncertaintyAnalyzer",
        "analyze",
        "stratigraphy",
        "The orbital frequencies at an age, with the interval they lie in",
        data_slots=("ages",),
    ),
    _f(
        "circular_spectrum",
        "stratigraphy.circular_spectrum",
        "circular_spectrum",
        "time-series",
        "The spectrum of an angular variable: its harmonics and mean direction",
        data_slots=("angles_deg",),
    ),
    # ---------------------------------------------------------------------------
    # The last of the Acycle family: the sedimentary noise model, the length of
    # day, the LR04 stack, the smoothing and detrending tools, the figure reader
    # and the batch runner.
    # ---------------------------------------------------------------------------
    _f(
        "dynot",
        "stratigraphy.dynot",
        "sedimentary_noise",
        "stratigraphy",
        "The sedimentary noise in a record, against an AR(1) or white baseline",
        data_slots=("time", "values"),
    ),
    _f(
        "rho1_sedimentation_ratio",
        "stratigraphy.dynot",
        "rho1_sedimentation_ratio",
        "stratigraphy",
        "The rho1 ratio: the record's noise power against the chosen baseline",
        data_slots=("time", "values"),
    ),
    _f(
        "lod_dimension",
        "stratigraphy.lod",
        "lod_dimension",
        "stratigraphy",
        "Length of day through deep time (Laskar et al. 2004)",
        data_slots=("time", "values"),
    ),
    _f(
        "lr04_stack",
        "stratigraphy.lr04",
        "load_lr04_stack",
        "stratigraphy",
        "The LR04 benthic d18O stack (Lisiecki & Raymo 2005)",
    ),
    _f(
        "optimal_span_smooth",
        "stratigraphy.smoothing",
        "optimal_span_smooth",
        "time-series",
        "Smoothing with the span chosen by cross-validation, plus a bootstrap band",
        data_slots=("time", "values"),
    ),
    _f(
        "moving_average_smooth",
        "stratigraphy.smoothing",
        "moving_average_smooth",
        "time-series",
        "Smoothing with a moving average of a given span",
        data_slots=("time", "values"),
    ),
    _f(
        "standardize",
        "stratigraphy.transforms",
        "standardize",
        "time-series",
        "Standardise a series (z-score or min-max)",
        data_slots=("values",),
    ),
    _f(
        "log_transform",
        "stratigraphy.transforms",
        "log_transform",
        "time-series",
        "Log-transform a series",
        data_slots=("values",),
    ),
    _f(
        "apply_function",
        "stratigraphy.transforms",
        "apply_function",
        "time-series",
        "Apply a simple function to a series",
        data_slots=("values",),
    ),
    _f(
        "detrend_series",
        "stratigraphy.detrending",
        "detrend_series",
        "time-series",
        "Detrend with a mean, a polynomial, a smoother or the EEMD residual",
        data_slots=("time", "values"),
    ),
    _f(
        "remove_part",
        "stratigraphy.series_edit",
        "remove_part",
        "time-series",
        "Remove one interval from a record",
        data_slots=("time", "values"),
    ),
    _f(
        "remove_range",
        "stratigraphy.series_edit",
        "remove_range",
        "time-series",
        "Keep one interval of a record and drop the rest",
        data_slots=("time", "values"),
    ),
    _f(
        "plot_digitizer",
        "stratigraphy.plot_digitizer",
        "digitize_curve",
        "time-series",
        "Read a curve off a published figure",
    ),
    _f(
        "mini_robot",
        "stratigraphy.mini_robot",
        "run_robot",
        "time-series",
        "Replay a recorded sequence of analyses over a batch of data",
    ),
)


class BuiltinAnalysisPlugin(AnalysisPlugin):
    """Adapts an existing analyzer class to the plugin interface.

    The wrapped analyzer keeps its own constructor, its own result
    dataclass and its own exceptions; this only supplies the three
    properties and the ``analyze`` signature the registry expects, and
    normalises the returned dataclass into an :class:`AnalysisResult`.
    """

    def __init__(self, entry: AnalysisEntry) -> None:
        """Wrap a catalog entry without importing anything yet."""
        self._entry = entry
        self._analyzer: Any = None
        self._logger = logging.getLogger(f"{__name__}.{entry.name}")

    @property
    def name(self) -> str:
        """The catalog name."""
        return self._entry.name

    @property
    def description(self) -> str:
        """The catalog description."""
        return self._entry.description

    @property
    def category(self) -> str:
        """The catalog category."""
        return self._entry.category

    @property
    def cache_key(self) -> str | None:
        """Cache under the catalog name."""
        return f"{self._entry.name}_result"

    def _resolve(self) -> Any:
        """Import and instantiate the wrapped analyzer, once.

        A ``class_name`` of ``None`` means the target is a module-level
        function; the module itself stands in for the instance.
        """
        if self._analyzer is not None:
            return self._analyzer
        # import_module rather than ``from package import module``: in
        # stats/procD_lm.py the function procD_lm shadows the submodule of
        # the same name, so the attribute form would hand back the
        # function before the catalog had even looked for a class.
        module = importlib.import_module(self._entry.module)
        if self._entry.class_name is None:
            self._analyzer = module
            return self._analyzer
        cls = getattr(module, self._entry.class_name)
        self._analyzer = cls()
        return self._analyzer

    def _callable(self) -> Any:
        """The bound method, or the function, the catalog names."""
        target = self._resolve()
        method = getattr(target, self._entry.method, None)
        if not callable(method):
            where = (
                self._entry.module
                if self._entry.class_name is None
                else f"{self._entry.module}.{self._entry.class_name}"
            )
            raise AttributeError(f"{where} has no callable {self._entry.method!r}; the catalog is out of date")
        return method

    def _invoke(self, method: Any, data: Any, kwargs: dict[str, Any]) -> Any:
        """Call the target with the data on the parameter that wants it.

        The contract is ``analyze(data, **kwargs)`` with ``data`` the whole
        spreadsheet, but the target's first positional parameter is not
        always the data: ``growth_model(model, times, values)`` takes a
        model name first, and ``orbital_forcing(period_kyr, ...)`` takes no
        data at all. Handing the sheet to the first positional parameter
        regardless gave the sheet to ``model`` and to ``period_kyr``.

        The parameter is found from the signature -- see
        :func:``plugins.base.data_slot_name`` -- and the data is passed
        there by keyword. A target with no such parameter gets the form's
        arguments alone.

        ``data`` may also be a mapping of slot name to value, which is
        what the runner sends for an analysis that wants a SERIES: the
        sheet is cut into columns there, and each column is handed to
        the parameter that asked for it.
        """
        if isinstance(data, dict):
            # The caller already cut the sheet into slots.
            for name in data:
                if name in kwargs:
                    raise ValidationError(
                        f"{name} is the data parameter and was also given explicitly; it cannot be set twice."
                    )
            return method(**data, **kwargs)

        slots = data_slot_names(inspect.signature(method), self._entry.data_slots)
        if not slots:
            return method(**kwargs)
        for name in slots:
            if name in kwargs:
                raise ValidationError(
                    f"{name} is the data parameter and was also given explicitly; it cannot be set twice."
                )
        if self._entry.data_slots is None:
            # No declaration: the sheet goes, whole, into the one slot.
            return method(**{slots[0]: data}, **kwargs)
        # Declared: the sheet is a set of columns, and each slot takes one
        # of them.  The first is the time axis, the second the
        # measurement -- which is the order the sheet's columns are in.
        columns = sheet_columns(data, len(slots))
        return method(**dict(zip(slots, columns, strict=True)), **kwargs)

    def analyze(self, data: Any, **kwargs: Any) -> AnalysisResult:
        """Run the wrapped analysis.

        Parameters
        ----------
        data:
            Whatever the wrapped method takes as its first argument --
            a matrix, a list of observations, a table.
        **kwargs:
            Passed straight through. The wrapped methods disagree about
            parameter names, so this does not attempt to normalise them.
        """
        method = self._callable()
        # Deliberately not wrapped in a try/except. The analyzers raise the
        # project's own exception types, which callers match on; catching
        # them here to re-raise a generic one would lose that.
        outcome = self._invoke(method, data, kwargs)

        metadata: dict[str, Any] = {
            "module": self._entry.module,
            "class": self._entry.class_name or "<module function>",
            "method": self._entry.method,
            "category": self._entry.category,
        }
        summary = getattr(outcome, "summary", None)
        if callable(summary):
            try:
                metadata["summary"] = summary()
            except Exception:  # a broken summary must not lose the result
                self._logger.debug("summary() failed for %s", self._entry.name)
        return AnalysisResult(name=self._entry.name, data=outcome, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of the entry."""
        return {
            "name": self._entry.name,
            "module": self._entry.module,
            "class": self._entry.class_name,
            "method": self._entry.method,
            "category": self._entry.category,
            "description": self._entry.description,
        }


def register_builtin_analyses(*, registry: Any = None, replace: bool = False) -> list[str]:
    """Fill the plugin registry from the catalog.

    Idempotent: a name already present is skipped rather than raising.
    ``AnalysisPluginRegistry.register`` raises on a duplicate, which is
    the right behaviour for a plugin author who registered twice by
    mistake but the wrong behaviour for a catalog being loaded at
    startup more than once -- which is exactly what happens when a test
    imports the application and then the process re-enters main().

    Parameters
    ----------
    registry:
        Target registry; defaults to the singleton.
    replace:
        Overwrite an existing entry instead of skipping it. For a test
        that has edited the catalog.

    Returns
    -------
    list of str
        The names registered by this call.
    """
    if registry is None:
        from plugins.registry import get_plugin_registry

        registry = get_plugin_registry()

    registered: list[str] = []
    for entry in BUILTIN_ANALYSES:
        if registry.get(entry.name) is not None:
            if not replace:
                logger.debug("Plugin %r already registered; skipping", entry.name)
                continue
            registry.unregister(entry.name)
        try:
            registry.register(BuiltinAnalysisPlugin(entry))
        except (TypeError, ValueError) as exc:
            logger.warning("Could not register %r: %s", entry.name, exc)
            continue
        registered.append(entry.name)
    logger.info("Registered %d of %d built-in analyses", len(registered), len(BUILTIN_ANALYSES))
    return registered


__all__ = [
    "BUILTIN_ANALYSES",
    "AnalysisEntry",
    "BuiltinAnalysisPlugin",
    "register_builtin_analyses",
]
