# =============================================================================
# FILE: tests/golden/test_detrending.py
# =============================================================================
"""
Tests for ``stratigraphy.detrending`` (the Acycle detrending option set).

Every numeric expectation here is either

* a property of the *algorithm* the module implements -- a least-squares
  polynomial reproduces a polynomial exactly; a local polynomial fit
  reproduces a local polynomial exactly; and, above all, the definition
  ``detrended == values - trend`` that Acycle's ``detrending.m`` fixes
  (``detr.<model> = datay - trend``);
* a property of the *window semantics* -- ``span`` is a fraction of the
  record's time span, so a narrow window follows a local feature and a wide
  window smooths it away;
* or a property of the *underlying implementation* -- the ``eemd`` method
  must use the residual that :func:`stratigraphy.emd.eemd` computes, and a
  fixed ``random_seed`` must make that residual repeatable.

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  Where the underlying EMD is only an approximation (its
ensemble noise seeding makes the residual sensitive to the noise realisation),
the tolerance is stated explicitly and generously, and the *contract* is
pinned exactly instead.

References cited in the assertions:
    Cleveland, W.S. 1979. J. Am. Stat. Assoc. 74: 829-836. (LOWESS.)
    Cleveland, W.S. et al. 1990. J. Off. Stat. 6: 3-73. (LOESS.)
    Wu, Z. & Huang, N.E. 2009. Adv. Data Sci. Eng. 1: 1-41. (EEMD.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"detrend_series"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.detrending import available_detrending_methods, detrend_series
from stratigraphy.emd import eemd
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的长度。取整百是为了让"跨度百分比"和"点数"的关系看得清楚。
N_SAMPLES = 400

#: 记录的时间跨度（时间坐标的长度）。
TIME_SPAN = 100.0

#: 局部趋势断点的位置（占记录跨度的比例）。
KINK_POSITION = 0.5

#: EEMD 用例的采样点数。EMD 需要足够多的点才能把振荡与趋势分开。
N_EEMD = 800

#: EEMD 用例里线性趋势的斜率（每样本）。
EEMD_TREND_SLOPE = 0.002

#: EEMD 用例里振荡分量的振幅。
EEMD_AMPLITUDE = 10.0

#: EEMD 用例里振荡分量的周期（样本数）。
EEMD_PERIOD = 37.0

#: EEMD 集合平均的试验次数。试验越多越慢，但残差越稳定。
EEMD_TRIALS = 10

#: EEMD 试验用的随机种子。
EEMD_SEED = 42

#: 噪声分量的相对幅度（相对于整条记录的标准差）。
NOISE_LEVEL = 0.02


# =============================================================================
# Helpers
# =============================================================================


def _time_axis(n_samples: int = N_SAMPLES, span: float = TIME_SPAN) -> npt.NDArray:
    """A strictly increasing time axis covering ``span``."""
    return np.linspace(0.0, span, n_samples)


def _kink_series() -> tuple[npt.NDArray, npt.NDArray]:
    """A record whose local trend changes slope halfway through.

    The first half rises with slope ``3``, the second half falls with slope
    ``-1``, so the record has one kink at the middle.  A local regression with
    a narrow window must follow both branches.
    """
    times = _time_axis()
    breakpoint = KINK_POSITION * TIME_SPAN
    truth = np.where(times <= breakpoint, 3.0 * times, 3.0 * breakpoint - (times - breakpoint))
    return times, truth


def _eemd_series() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """An oscillation plus a linear trend plus noise.

    Returns:
        ``(times, trend, oscillation, values)``.  The oscillation is much
        stronger than the trend, which is the situation the EMD handles best:
        the oscillation is peeled off into the IMFs and the trend is what is
        left behind.
    """
    times = np.arange(N_EEMD, dtype=float)
    trend = EEMD_TREND_SLOPE * times
    oscillation = EEMD_AMPLITUDE * np.sin(2.0 * np.pi * times / EEMD_PERIOD)
    rng = np.random.default_rng(20261024)
    noise = rng.normal(scale=NOISE_LEVEL * float(np.std(oscillation + trend)), size=N_EEMD)
    return times, trend, oscillation, oscillation + trend + noise


def _relative_error(estimate: npt.NDArray, truth: npt.NDArray) -> float:
    """Maximum absolute deviation, relative to the range of the truth."""
    scale = float(np.ptp(truth))
    if scale <= 0.0:
        return float(np.max(np.abs(np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float))))
    return float(np.max(np.abs(np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float))) / scale)


# =============================================================================
# 方法清单
# =============================================================================


def test_available_methods_list_the_acycle_option_set():
    """The nine trend models of Acycle's detrending tool are all available."""
    methods = available_detrending_methods()

    assert set(methods) == {
        "mean",
        "linear",
        "poly2",
        "poly",
        "lowess",
        "rlowess",
        "loess",
        "rloess",
        "eemd",
    }
    assert len(methods) == 9


# =============================================================================
# 逐个模型的真值
# =============================================================================


def test_mean_trend_is_the_mean_of_the_record():
    """The mean model is a constant, and on a constant record it is exact."""
    times = _time_axis()
    values = np.full(times.size, 7.5)

    result = detrend_series(times, values, method="mean")

    np.testing.assert_allclose(result.trend, 7.5, atol=1e-12)
    np.testing.assert_allclose(result.detrended, 0.0, atol=1e-12)


def test_linear_trend_is_recovered_exactly():
    """A least-squares line reproduces a line exactly."""
    times = _time_axis()
    truth = 2.0 * times + 3.0

    result = detrend_series(times, truth, method="linear")

    assert _relative_error(result.trend, truth) < 1e-9
    np.testing.assert_allclose(result.detrended, 0.0, atol=1e-9)


def test_quadratic_trend_is_recovered_by_the_second_order_model():
    """A least-squares parabola reproduces a parabola exactly."""
    times = _time_axis()
    truth = 0.05 * times**2 - 3.0 * times + 7.0

    result = detrend_series(times, truth, method="poly2")

    assert _relative_error(result.trend, truth) < 1e-9
    np.testing.assert_allclose(result.detrended, 0.0, atol=1e-8)


def test_cubic_trend_is_recovered_by_the_nth_order_model():
    """The ``poly`` model fits the order it is given."""
    times = _time_axis()
    truth = 0.001 * times**3 - 0.02 * times**2 + 2.0 * times

    result = detrend_series(times, truth, method="poly", order=3)

    assert _relative_error(result.trend, truth) < 1e-9
    np.testing.assert_allclose(result.detrended, 0.0, atol=1e-8)


def test_exponential_trend_is_followed_by_the_local_quadratic_model():
    """LOESS follows a smooth, curved trend to well below one percent.

    A local quadratic fit of a smooth function is accurate in the interior of
    the record; the tolerance below (0.1 % of the range of the truth) leaves
    room for the end effects of a one-sided window.
    """
    times = _time_axis(n_samples=200)
    truth = np.exp(times / 40.0)

    result = detrend_series(times, truth, method="loess", span=0.2)

    assert _relative_error(result.trend, truth) < 0.01


def test_detrending_a_linear_trend_leaves_only_the_noise():
    """A linear trend plus noise detrends to the noise and nothing else."""
    times = _time_axis()
    rng = np.random.default_rng(11)
    noise = rng.normal(scale=0.05, size=times.size)
    values = 2.0 * times + 3.0 + noise

    result = detrend_series(times, values, method="linear")

    assert float(np.std(result.detrended)) < 0.1
    assert float(np.max(np.abs(result.detrended))) < 0.5


# =============================================================================
# 局部回归
# =============================================================================


def test_lowess_follows_the_local_trend():
    """A local linear fit follows both branches of a kinked record."""
    times, truth = _kink_series()

    result = detrend_series(times, truth, method="lowess", span=0.2)

    for probe in (0.25 * TIME_SPAN, 0.75 * TIME_SPAN):
        index = int(np.argmin(np.abs(times - probe)))
        assert result.trend[index] == pytest.approx(truth[index], abs=0.5)


def test_loess_follows_the_local_trend():
    """The local quadratic model follows the same kinked record."""
    times, truth = _kink_series()

    result = detrend_series(times, truth, method="loess", span=0.2)

    for probe in (0.25 * TIME_SPAN, 0.75 * TIME_SPAN):
        index = int(np.argmin(np.abs(times - probe)))
        assert result.trend[index] == pytest.approx(truth[index], abs=0.5)


def test_the_window_is_a_fraction_of_the_record_span():
    """``span`` is a percentage of the record's time span, not a point count.

    A narrow window follows the kink of the record; a window that is wide
    compared with the record smooths the kink away.  This is the behaviour
    Acycle's ``smooth_win`` (0.35 by default) has, and it is what separates
    "35 % of the record" from "35 % of the samples".
    """
    times, truth = _kink_series()
    kink_index = int(np.argmin(np.abs(times - KINK_POSITION * TIME_SPAN)))

    narrow = detrend_series(times, truth, method="lowess", span=0.02)
    wide = detrend_series(times, truth, method="lowess", span=0.8)

    narrow_error = abs(float(narrow.trend[kink_index]) - float(truth[kink_index]))
    wide_error = abs(float(wide.trend[kink_index]) - float(truth[kink_index]))

    # A window of 2 % of the record still resolves the kink ...
    assert narrow_error < 0.01 * float(np.ptp(truth))
    # ... while a window of 80 % smooths it away.
    assert wide_error > 10.0 * narrow_error


def test_robust_smoothers_ignore_an_outlier():
    """The robust variants do not let a single spike pull the trend with it."""
    times = _time_axis()
    truth = 2.0 + 0.5 * times
    polluted = truth.copy()
    polluted[times.size // 2] += 20.0

    plain = detrend_series(times, polluted, method="lowess", span=0.35)
    robust = detrend_series(times, polluted, method="rlowess", span=0.35)

    spike = times.size // 2
    assert abs(float(robust.trend[spike]) - float(truth[spike])) < 0.05
    assert abs(float(plain.trend[spike]) - float(truth[spike])) > 10.0 * abs(
        float(robust.trend[spike]) - float(truth[spike])
    )


# =============================================================================
# EEMD
# =============================================================================


def test_eemd_uses_the_eemd_residual_as_the_trend():
    """The EEMD model takes its trend from the EEMD residual, exactly."""
    times, _trend, _oscillation, values = _eemd_series()

    result = detrend_series(times, values, method="eemd", n_trials=EEMD_TRIALS, random_seed=EEMD_SEED)
    reference = eemd(values, n_trials=EEMD_TRIALS, random_seed=EEMD_SEED)

    np.testing.assert_array_equal(result.trend, reference.trend)


def test_eemd_trend_tracks_the_underlying_trend():
    """The EEMD residual follows the trend of a trend-plus-noise record.

    The EMD residual is an approximation: the ensemble noise seeding makes
    it sensitive to the noise realisation, and the ends of the record suffer
    from one-sided windows.  The tolerances below are therefore generous --
    the shape has to agree (correlation), the slope has to be right to
    within a quarter, and the interior has to stay within a quarter of the
    range of the trend.
    """
    times, truth, oscillation, values = _eemd_series()

    result = detrend_series(times, values, method="eemd", n_trials=EEMD_TRIALS, random_seed=EEMD_SEED)

    correlation = float(np.corrcoef(result.trend, truth)[0, 1])
    assert correlation > 0.95, f"trend shape not recovered (correlation {correlation:.4f})"

    slope = float(np.polyfit(times, result.trend, 1)[0])
    assert abs(slope - EEMD_TREND_SLOPE) / EEMD_TREND_SLOPE < 0.25

    interior = slice(int(0.1 * times.size), int(0.9 * times.size))
    assert _relative_error(result.trend[interior], truth[interior]) < 0.25

    # What is left after detrending is the oscillation again.
    assert float(np.std(result.detrended)) == pytest.approx(float(np.std(oscillation)), rel=0.05)
    assert float(np.max(np.abs(result.detrended - oscillation))) < 0.15 * EEMD_AMPLITUDE


def test_eemd_is_reproducible_for_a_fixed_seed():
    """The same seed gives the same EEMD trend, bit for bit."""
    times, _trend, _oscillation, values = _eemd_series()

    first = detrend_series(times, values, method="eemd", n_trials=EEMD_TRIALS, random_seed=EEMD_SEED)
    second = detrend_series(times, values, method="eemd", n_trials=EEMD_TRIALS, random_seed=EEMD_SEED)

    np.testing.assert_array_equal(first.trend, second.trend)
    np.testing.assert_array_equal(first.detrended, second.detrended)


def test_eemd_without_a_seed_warns_that_the_result_is_not_reproducible():
    """An unseeded ensemble says so, because a published curve must repeat."""
    times, _trend, _oscillation, values = _eemd_series()

    with pytest.warns(RuntimeWarning):
        detrend_series(times, values, method="eemd", n_trials=2)


# =============================================================================
# 恒等关系
# =============================================================================


@pytest.mark.parametrize("method", available_detrending_methods())
def test_detrended_is_the_series_minus_the_trend(method):
    """``detrended == values - trend``, point by point, for every model.

    This is the definition Acycle's detrending tool fixes, and it is the
    easiest thing in this module to get wrong: a model that returns a trend
    of the wrong length, or a "detrended" series that was normalised instead
    of subtracted, still looks plausible in a plot.
    """
    times = _time_axis()
    rng = np.random.default_rng(3)
    values = 0.4 * times**2 - 2.0 * times + 5.0 + rng.normal(scale=0.1, size=times.size)

    result = detrend_series(times, values, method=method, order=3, n_trials=3, random_seed=EEMD_SEED)

    assert result.trend.shape == values.shape
    assert result.detrended.shape == values.shape
    np.testing.assert_allclose(result.detrended, values - result.trend, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(result.times, times)
    assert result.method == method


def test_summary_reports_the_method_and_the_samples():
    """The summary is a one-glance report of what was done."""
    times = _time_axis()
    values = 2.0 * times + 1.0

    result = detrend_series(times, values, method="linear")
    summary = result.summary()

    assert "linear" in summary
    assert str(times.size) in summary


# =============================================================================
# Validation
# =============================================================================


def test_invalid_inputs_raise_validation_error():
    """Every documented boundary raises ``ValidationError``.

    ``ValidationError`` is the backwards-compatible alias of
    ``DataValidationError`` in ``utils.exceptions``.
    """
    times = _time_axis()
    values = np.sin(times)

    with pytest.raises(ValidationError):
        detrend_series(times, values, method="not-a-method")
    with pytest.raises(ValidationError):
        detrend_series(times, values, method="poly", order=0)
    with pytest.raises(ValidationError):
        detrend_series(times, values, method="lowess", span=0.0)
    with pytest.raises(ValidationError):
        detrend_series(times, values, method="lowess", span=1.5)
    with pytest.raises(ValidationError):
        detrend_series(times[:3], values[:3], method="linear")
    with pytest.raises(ValidationError):
        detrend_series(np.array([0.0, 1.0, np.nan, 3.0]), values, method="linear")
    with pytest.raises(ValidationError):
        detrend_series(times, values[:-1], method="linear")
    with pytest.raises(ValidationError):
        detrend_series(times, values, method="poly", order=times.size + 1)
