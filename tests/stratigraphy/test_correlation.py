# tests/stratigraphy/test_correlation.py
"""Tests for stratigraphy correlation module."""

import numpy as np

from stratigraphy.correlation import (
    AgeModelAnalyzer,
    SedimentationRateAnalyzer,
    StratigraphicCorrelationAnalyzer,
    StratigraphicSection,
)


def test_stratigraphic_section():
    """Test StratigraphicSection dataclass."""
    sec = StratigraphicSection(
        name="Test Section",
        heights=np.array([0, 10, 20, 30]),
        thicknesses=np.array([10, 10, 10, 10]),
        lithologies=["sandstone", "mudstone", "sandstone", "mudstone"],
    )

    assert sec.name == "Test Section"
    assert len(sec.heights) == 4
    assert sec.to_dict()["name"] == "Test Section"


def test_correlation_analyzer_dtw():
    """Test DTW correlation between two sections."""
    sec1 = StratigraphicSection(
        name="Section 1",
        heights=np.array([0, 10, 20, 30, 40]),
        thicknesses=np.array([10, 10, 10, 10, 10]),
        lithologies=["A"] * 5,
    )

    # Section 2 is similar (slightly different spacing)
    sec2 = StratigraphicSection(
        name="Section 2",
        heights=np.array([0, 12, 22, 32, 42]),
        thicknesses=np.array([12, 10, 10, 10, 10]),
        lithologies=["A"] * 5,
    )

    analyzer = StratigraphicCorrelationAnalyzer()
    result = analyzer.analyze([sec1, sec2], method="dtw")

    assert len(result.sections) == 2
    assert result.correlation_matrix.shape == (2, 2)
    assert 0 <= result.correlation_matrix[0, 1] <= 1
    assert result.correlation_matrix[0, 0] == 1.0  # Self-correlation


def test_correlation_analyzer_euclidean():
    """Test Euclidean correlation between sections."""
    sec1 = StratigraphicSection(
        name="Section 1",
        heights=np.array([0, 10, 20]),
        thicknesses=np.array([10, 10, 10]),
        lithologies=["A"] * 3,
    )

    sec2 = StratigraphicSection(
        name="Section 2",
        heights=np.array([0, 10, 20]),
        thicknesses=np.array([10, 10, 10]),
        lithologies=["A"] * 3,
    )

    analyzer = StratigraphicCorrelationAnalyzer()
    result = analyzer.analyze([sec1, sec2], method="euclidean")

    assert result.correlation_matrix[0, 1] == 1.0  # Identical sections


def test_correlation_with_ages():
    """Test correlation when sections have age data."""
    sec = StratigraphicSection(
        name="Section with ages",
        heights=np.array([0, 10, 20, 30]),
        thicknesses=np.array([10, 10, 10, 10]),
        lithologies=["A"] * 4,
        ages=np.array([100.0, 95.0, 90.0, 85.0]),
        age_errors=np.array([0.5, 0.5, 0.5, 0.5]),
    )

    analyzer = StratigraphicCorrelationAnalyzer()
    result = analyzer.analyze([sec], method="dtw")

    assert len(result.sections) == 1
    assert result.sections[0].ages is not None


def test_age_model_linear():
    """Test linear age model building."""
    sec = StratigraphicSection(
        name="Test Section",
        heights=np.array([0, 10, 20, 30, 40]),
        thicknesses=np.array([10, 10, 10, 10, 10]),
        lithologies=["A"] * 5,
    )

    constraints = [
        (0, 100.0, 0.5),
        (40, 80.0, 0.5),
    ]

    analyzer = AgeModelAnalyzer()
    result = analyzer.build_model(sec, constraints, model_type="linear")

    assert len(result.modeled_ages) == 5
    assert result.modeled_ages[0] == 100.0
    assert result.modeled_ages[-1] == 80.0
    assert len(result.sedimentation_rates) == 5


def test_age_model_spline():
    """Test spline age model building with sufficient points."""
    sec = StratigraphicSection(
        name="Test Section",
        heights=np.array([0, 5, 10, 15, 20, 25, 30, 35, 40]),
        thicknesses=np.array([5] * 9),
        lithologies=["A"] * 9,
    )

    # Need at least 4 points for cubic spline
    constraints = [
        (0, 100.0, 0.5),
        (13, 93.0, 0.5),
        (27, 86.5, 0.5),
        (40, 80.0, 0.5),
    ]

    analyzer = AgeModelAnalyzer()
    result = analyzer.build_model(sec, constraints, model_type="spline")

    assert len(result.modeled_ages) == 9
    # First age should be close to 100
    assert abs(result.modeled_ages[0] - 100.0) < 1.0


def test_sedimentation_rate():
    """Test sedimentation rate calculation."""
    sec = StratigraphicSection(
        name="Test Section",
        heights=np.array([0, 10, 20, 30, 40]),
        thicknesses=np.array([10, 10, 10, 10, 10]),
        lithologies=["A"] * 5,
        ages=np.array([100.0, 95.0, 90.0, 85.0, 80.0]),
    )

    analyzer = SedimentationRateAnalyzer()
    heights, rates, smoothed = analyzer.calculate(sec, smooth=False)

    assert len(heights) == 5
    assert len(rates) == 5
    assert len(smoothed) == 5


def test_sedimentation_rate_with_smoothing():
    """Test sedimentation rate with LOWESS smoothing."""
    sec = StratigraphicSection(
        name="Test Section",
        heights=np.array([0, 5, 10, 15, 20, 25, 30, 35, 40]),
        thicknesses=np.array([5] * 9),
        lithologies=["A"] * 9,
        ages=np.array([100.0, 97.5, 95.0, 92.5, 90.0, 87.5, 85.0, 82.5, 80.0]),
    )

    analyzer = SedimentationRateAnalyzer()
    heights, rates, smoothed = analyzer.calculate(sec, smooth=True, frac=0.3)

    assert len(heights) == 9
    assert len(rates) == 9
    assert len(smoothed) == 9


def test_empty_section_handling():
    """Test handling of empty sections."""
    sec1 = StratigraphicSection(
        name="Empty 1",
        heights=np.array([]),
        thicknesses=np.array([]),
        lithologies=[],
    )

    sec2 = StratigraphicSection(
        name="Empty 2",
        heights=np.array([]),
        thicknesses=np.array([]),
        lithologies=[],
    )

    analyzer = StratigraphicCorrelationAnalyzer()
    result = analyzer.analyze([sec1, sec2], method="dtw")

    # Should handle gracefully
    assert len(result.sections) == 2


def test_best_matches():
    """Test best matches finding."""
    sec1 = StratigraphicSection(
        name="Section 1",
        heights=np.array([0, 10, 20]),
        thicknesses=np.array([10, 10, 10]),
        lithologies=["A"] * 3,
    )

    sec2 = StratigraphicSection(
        name="Section 2",
        heights=np.array([0, 15, 30]),
        thicknesses=np.array([15, 15, 15]),
        lithologies=["A"] * 3,
    )

    sec3 = StratigraphicSection(
        name="Section 3",
        heights=np.array([0, 5, 10]),
        thicknesses=np.array([5, 5, 5]),
        lithologies=["A"] * 3,
    )

    analyzer = StratigraphicCorrelationAnalyzer()
    result = analyzer.analyze([sec1, sec2, sec3], method="dtw")

    assert len(result.best_matches) <= 3  # At most 3 pairs
    # Best matches should be sorted by correlation (descending)
    if len(result.best_matches) > 1:
        assert result.best_matches[0][2] >= result.best_matches[1][2]
