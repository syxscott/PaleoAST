# =============================================================================
# FILE: stratigraphy/__init__.py
# =============================================================================
"""
PaleoAST Biostratigraphy Package

This package implements biological stratigraphy and time series analysis.

Modules:
    - biostratigraphy: UA and RASC methods for stratigraphic correlation
    - spectral_analysis: Lomb-Scargle periodogram and wavelet CWT for time series
    - time_bins: geologic time binning (palaeoverse-style) and occurrence binning
    - astronomy, coco, tuning: astrochronology -- from a measured record to
      an age scale

Author: PaleoAST Development Team
version: 1.1.0
"""

from .amplitude_modulation import AmplitudeModulationResult, amplitude_modulation, tanner_hilbert
from .arma import ARMAAnalyzer, ARMAResult, ForecastResult
from .astronomy import (
    AstronomicalSolution,
    astronomical_target_curve,
    daily_mean_insolation,
    insolation,
    load_astronomical_solution,
    milankovitch_periods,
)
from .biostratigraphy import BioeventResult, RASCAnalyzer, UAAnalyzer, Zone
from .change_points import ChangePointResult, detect_change_points
from .circular_spectrum import CircularSpectrumResult, circular_spectrum
from .coco import CocoResult, ECocoResult, berger1989_periods, coco, ecoco, milankovitch_target_spectrum
from .coherence import CoherenceResult, coherence, coherence_significance
from .correlation import (
    AgeModelAnalyzer,
    AgeModelResult,
    StratigraphicCorrelationAnalyzer,
    StratigraphicCorrelationResult,
    StratigraphicSection,
)
from .cycles import (
    ACFResult,
    CycleAnalyzer,
    RedfitResult,
    ar1_prewhiten,
    autoassociation,
    autocorrelation,
    cross_correlation,
    insolation_series,
    mann_kendall,
    multitaper_spectrum,
    orbital_forcing,
    redfit,
    runs_test,
)
from .detrending import DetrendResult, available_detrending_methods, detrend_series
from .dynamic_filter import DynamicFilterResult, dynamic_filter
from .dynot import DynotResult, rho1_sedimentation_ratio, sedimentary_noise
from .emd import EMDResult, eemd, emd
from .evolutionary import (
    EvolutionarySpectrumResult,
    evolutionary_lomb_scargle,
    evolutionary_multitaper,
    evolutionary_spectrum,
)
from .extinction import ExtinctionIntervalAnalyzer, ExtinctionIntervalResult
from .filtering import FilterResult, bandpass_filter, gaussian_filter, taner_filter
from .isotope_analysis import (
    Excursion,
    IsotopeAnalyzer,
    IsotopeData,
    IsotopeResult,
    IsotopeTrend,
)
from .lod import LODResult, lod_day_lengthening_rate, lod_days_in_year, lod_dimension, lod_length_of_day
from .lr04 import LR04Stack, load_lr04_stack, lr04_target_curve
from .mini_robot import RobotResult, RobotStep, robot_from_preset, run_robot, run_robot_batch, save_robot
from .moving_summary import MovingSummaryResult, fixed_bandwidth_summary, fixed_count_summary
from .pda import PDABand, PDAResult, power_decomposition
from .plot_digitizer import AxisCalibration, DigitizedSeries, calibrate_axes, digitize_curve, digitize_profile
from .series_edit import remove_part, remove_range
from .series_tools import (
    add_gaps,
    clip_values,
    derivative,
    interpolate_series,
    merge_series,
    remove_peaks,
    select_intervals,
    sort_unique,
)
from .signal_generator import (
    generate_pink_noise,
    generate_red_noise,
    generate_signal,
    generate_sine,
    generate_white_noise,
)
from .smoothing import SmoothingResult, moving_average_smooth, optimal_span_smooth
from .spectral_analysis import SpectralAnalyzer, SpectralResult, WaveletResult
from .spectral_moments import (
    SpectralMomentsResult,
    estimate_ar1_rho,
    minimum_rho,
    spectral_moments,
)
from .time_bins import (
    bin_time,
    get_scale,
    tax_expand_time,
    tax_range_time,
    time_bins,
)
from .timeaxis import TimeAxis
from .transforms import apply_function, log_transform, standardize
from .tuning import (
    AgeScaleResult,
    ETimeOptResult,
    TimeOptResult,
    build_age_scale,
    etime_opt,
    sedimentation_rate_to_age_model,
    time_opt,
)

__all__ = [
    "ACFResult",
    "ARMAAnalyzer",
    "ARMAResult",
    "AgeModelAnalyzer",
    "AgeModelResult",
    "AgeScaleResult",
    "AmplitudeModulationResult",
    "AstronomicalSolution",
    "AxisCalibration",
    "BioeventResult",
    "ChangePointResult",
    "CircularSpectrumResult",
    "CocoResult",
    "CoherenceResult",
    "CycleAnalyzer",
    "DetrendResult",
    "DigitizedSeries",
    "DynamicFilterResult",
    "DynotResult",
    "ECocoResult",
    "EMDResult",
    "ETimeOptResult",
    "EvolutionarySpectrumResult",
    "Excursion",
    "ExtinctionIntervalAnalyzer",
    "ExtinctionIntervalResult",
    "FilterResult",
    "ForecastResult",
    "IsotopeAnalyzer",
    "IsotopeData",
    "IsotopeResult",
    "IsotopeTrend",
    "LODResult",
    "LR04Stack",
    "MovingSummaryResult",
    "PDABand",
    "PDAResult",
    "RASCAnalyzer",
    "RedfitResult",
    "RobotResult",
    "RobotStep",
    "SmoothingResult",
    "SpectralAnalyzer",
    "SpectralMomentsResult",
    "SpectralResult",
    "StratigraphicCorrelationAnalyzer",
    "StratigraphicCorrelationResult",
    "StratigraphicSection",
    "TimeAxis",
    "TimeOptResult",
    "UAAnalyzer",
    "WaveletResult",
    "Zone",
    "add_gaps",
    "amplitude_modulation",
    "apply_function",
    "ar1_prewhiten",
    "astronomical_target_curve",
    "autoassociation",
    "autocorrelation",
    "available_detrending_methods",
    "bandpass_filter",
    "berger1989_periods",
    "bin_time",
    "build_age_scale",
    "calibrate_axes",
    "change_points",
    "circular_spectrum",
    "clip_values",
    "coco",
    "coherence",
    "coherence_significance",
    "cross_correlation",
    "daily_mean_insolation",
    "derivative",
    "detect_change_points",
    "detrend_series",
    "digitize_curve",
    "digitize_profile",
    "dynamic_filter",
    "ecoco",
    "eemd",
    "emd",
    "estimate_ar1_rho",
    "etime_opt",
    "evolutionary_lomb_scargle",
    "evolutionary_multitaper",
    "evolutionary_spectrum",
    "fixed_bandwidth_summary",
    "fixed_count_summary",
    "gaussian_filter",
    "generate_pink_noise",
    "generate_red_noise",
    "generate_signal",
    "generate_sine",
    "generate_white_noise",
    "get_scale",
    "insolation",
    "insolation_series",
    "interpolate_series",
    "load_astronomical_solution",
    "load_lr04_stack",
    "lod_day_lengthening_rate",
    "lod_days_in_year",
    "lod_dimension",
    "lod_length_of_day",
    "log_transform",
    "lr04_target_curve",
    "mann_kendall",
    "merge_series",
    "milankovitch_periods",
    "milankovitch_target_spectrum",
    "minimum_rho",
    "moving_average_smooth",
    "multitaper_spectrum",
    "optimal_span_smooth",
    "orbital_forcing",
    "power_decomposition",
    "redfit",
    "remove_part",
    "remove_peaks",
    "remove_range",
    "rho1_sedimentation_ratio",
    "robot_from_preset",
    "run_robot",
    "run_robot_batch",
    "runs_test",
    "save_robot",
    "sedimentary_noise",
    "sedimentation_rate_to_age_model",
    "select_intervals",
    "sort_unique",
    "spectral_moments",
    "standardize",
    "taner_filter",
    "tanner_hilbert",
    "tax_expand_time",
    "tax_range_time",
    "time_bins",
    "time_opt",
]
