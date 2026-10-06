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
import logging
from dataclasses import dataclass
from typing import Any

from plugins.base import AnalysisPlugin, AnalysisResult

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


def _e(
    name: str,
    module: str,
    class_name: str | None,
    method: str,
    category: str,
    description: str,
) -> AnalysisEntry:
    """Terse constructor, so the table below stays readable."""
    return AnalysisEntry(name, module, class_name, method, category, description)


def _f(
    name: str,
    module: str,
    function: str,
    category: str,
    description: str,
) -> AnalysisEntry:
    """Catalog entry for a module-level function rather than a class."""
    return AnalysisEntry(name, module, None, function, category, description)


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
    ),
    _e(
        "grid_interpolate",
        "stats.spatial_stats",
        "SpatialStatsAnalyzer",
        "grid_interpolate",
        "spatial",
        "Interpolate scattered sites onto a regular grid",
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
    ),
    _e(
        "wavelet",
        "stratigraphy.spectral_analysis",
        "SpectralAnalyzer",
        "wavelet_transform",
        "time-series",
        "Continuous wavelet transform",
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
    ),
    _e(
        "redfit",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "redfit",
        "time-series",
        "REDFIT spectral analysis against an AR(1) null",
    ),
    _e(
        "multitaper",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "multitaper",
        "time-series",
        "Multitaper spectral estimation",
    ),
    _e(
        "prewhiten",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "prewhiten",
        "time-series",
        "AR(1) prewhitening of a series",
    ),
    _e(
        "cross_correlation",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "cross_correlation",
        "time-series",
        "Cross-correlation with a signed lag axis",
    ),
    _e(
        "autoassociation",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "autoassociation",
        "time-series",
        "Autoassociation function for cyclostratigraphy",
    ),
    _e(
        "mann_kendall",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "mann_kendall",
        "time-series",
        "Mann-Kendall trend test with Sen's slope",
    ),
    _e(
        "runs_test",
        "stratigraphy.cycles",
        "CycleAnalyzer",
        "runs",
        "time-series",
        "Runs test for randomness of a series",
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
    ),
    _e("arma", "stratigraphy.arma", "ARMAAnalyzer", "fit", "time-series", "Autoregressive moving average model"),
    _e("markov", "stratigraphy.markov", "MarkovAnalyzer", "analyze", "time-series", "Markov chain state transitions"),
    _e(
        "extinction",
        "stratigraphy.extinction",
        "ExtinctionIntervalAnalyzer",
        "analyze",
        "stratigraphy",
        "Extinction interval estimation",
    ),
    _e(
        "directional",
        "stratigraphy.directional",
        "DirectionalAnalyzer",
        "analyze",
        "directional",
        "Circular and axial directional statistics",
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
    ),
    # -- macroevolution ------------------------------------------------------
    _e(
        "kaplan_meier",
        "macroevolution.survival",
        "KaplanMeierAnalyzer",
        "fit",
        "macroevolution",
        "Kaplan-Meier survivorship",
    ),
    _e("cox_ph", "macroevolution.survival", "CoxPHAnalyzer", "fit", "macroevolution", "Cox proportional hazards model"),
    # -- growth ---------------------------------------------------------------
    _e(
        "growth_model", "models.growth_models", "GrowthModelAnalyzer", "fit", "growth", "Nonlinear growth curve fitting"
    ),
    _e(
        "growth_model_comparison",
        "models.growth_models",
        "GrowthModelAnalyzer",
        "fit_all",
        "growth",
        "Fit several growth curves and rank them by AICc",
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
        outcome = method(data, **kwargs)

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
