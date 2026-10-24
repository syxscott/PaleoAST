# =============================================================================
# FILE: tests/golden/test_tuning.py
# =============================================================================
"""
Tests for astrochronological tuning in ``stratigraphy.tuning``.

The suite is organised around three ideas:

* **Ground truth.** A synthetic record is generated *at a known sedimentation
  rate* with a 100 kyr eccentricity component and a 21 kyr precession component
  whose amplitude is modulated by the eccentricity (the "bundle" structure of
  Meyers 2015). The tuning must put the record's dominant cycle back onto the
  orbital period it belongs to. The truth is the compression factor 1.0: the
  module anchors its reference rate by matching the record's dominant
  *depth-domain* peak onto the orbital family that peak belongs to (here the
  ~21 kyr precession), so a correct search must return 1.0 rather than some
  arbitrary number. The assertion is deliberately robust -- the score at the
  truth must be at least 80 % of the best score, and the best compression must
  sit in the neighbourhood of the truth -- and it never asserts an exact
  equality.
* **Properties, not frozen numbers.** The age scale is checked for exactness
  *at the anchors* and for monotonicity *between* them; its uncertainty is
  checked for the properties the interpolation actually guarantees (never
  negative, equal to the anchor's own error at a dated horizon, growing only
  where the age model extrapolates) rather than for a monotone growth away
  from the anchors, which no interpolation scheme provides; the
  sedimentation-rate age model is checked for linearity; the Milankovitch
  table is checked against the published periods with a stated tolerance.
* **The frequency-grid bug that was fixed elsewhere in this repository.**
  A periodogram's frequency grid must be set by the length of the record
  (Rayleigh resolution), not by how densely the record is sampled. Denser
  sampling of the *same* record must not move the recovered periods, which is
  what ``test_denser_sampling_does_not_degrade_the_tuning`` pins down.

References cited in the assertions:
    Meyers, S.R., 2015. The evaluation of eccentricity-related amplitude
    modulation and bundling in paleoclimate data. Paleoceanography, 30(10):
    2241-2272. (TimeOpt.)
    Meyers, S.R., 2019. Cyclostratigraphy and the problem of astrochronologic
    testing. Earth-Science Reviews, 190: 190-223. (eTimeOpt.)
    Berger, A., 1978. Long-term climatic variation: astronomical theory.
    In Understanding the Climate. Academic Press. (Orbital periods.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"age_scale", "etime_opt", "sedimentation_rate_age_model", "time_opt"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.tuning import (
    build_age_scale,
    etime_opt,
    milankovitch_periods,
    sedimentation_rate_to_age_model,
    time_opt,
)
from utils.exceptions import ValidationError

#: Sedimentation rate of the synthetic section: 0.05 depth units/kyr.
TRUE_SEDIMENTATION_RATE = 0.05

#: The eccentricity and precession periods used in the synthetic record (kyr).
ECCENTRICITY_KYR = 100.0
PRECESSION_KYR = 21.0


# =============================================================================
# Helpers
# =============================================================================


def _synthetic_record(
    sedimentation_rate: float = TRUE_SEDIMENTATION_RATE,
    n_points: int = 1000,
    noise: float = 0.03,
    seed: int = 7,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    A synthetic stratigraphic section recorded at a known rate.

    The signal carries a 21 kyr precession cycle whose amplitude is modulated
    at the 100 kyr eccentricity period (the bundle), plus a 100 kyr and a
    41 kyr component. Depths are ``time * sedimentation_rate``.

    Returns:
        (depths, values).
    """
    rng = np.random.default_rng(seed)
    time_kyr = np.arange(n_points, dtype=float)
    envelope = 1.0 + 0.8 * np.cos(2.0 * np.pi * time_kyr / ECCENTRICITY_KYR + 0.3)
    values = envelope * np.sin(2.0 * np.pi * time_kyr / PRECESSION_KYR)
    values += 0.5 * np.cos(2.0 * np.pi * time_kyr / ECCENTRICITY_KYR)
    values += 0.2 * np.sin(2.0 * np.pi * time_kyr / 41.0)
    values += noise * rng.standard_normal(n_points)
    depths = time_kyr * sedimentation_rate
    return depths, values


def _score_at(result, compression: float) -> float:
    """The score the search assigned to a given compression factor."""
    index = int(np.argmin(np.abs(result.compression_factors - compression)))
    return float(result.scores[index])


def _dominant_period(times: npt.NDArray, values: npt.NDArray, band_kyr: tuple[float, float]) -> float:
    """The period (kyr) of the strongest spectral line inside a band."""
    dt = float(times[1] - times[0])
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    freqs = np.fft.rfftfreq(len(times), dt)
    mask = (freqs >= 1.0 / band_kyr[1]) & (freqs <= 1.0 / band_kyr[0])
    assert np.any(mask), "the search band contains no frequency bin"
    return float(1.0 / freqs[mask][int(np.argmax(spectrum[mask]))])


# =============================================================================
# TimeOpt: ground truth
# =============================================================================


class TestTimeOptGroundTruth:
    """TimeOpt must return the compression factor of a known section."""

    def test_recovers_true_compression(self):
        """The best compression must sit at the truth (1.0 under the
        module's anchoring convention), robustly rather than exactly."""
        depths, values = _synthetic_record()
        result = time_opt(depths, values, random_seed=11)

        assert abs(result.best_compression - 1.0) <= 0.1, (
            f"best compression {result.best_compression:.4f} is far from the truth (1.0)"
        )

    def test_score_at_truth_is_close_to_the_maximum(self):
        """The robust criterion: the score at the truth must be at least
        80 % of the best score, and the best must be in its neighbourhood."""
        depths, values = _synthetic_record()
        result = time_opt(depths, values, random_seed=11)

        best_score = float(np.max(result.scores))
        truth_score = _score_at(result, 1.0)
        assert truth_score >= 0.8 * best_score, (
            f"score at the truth ({truth_score:.4f}) is far below the best score ({best_score:.4f})"
        )

        truth_index = int(np.argmin(np.abs(result.compression_factors - 1.0)))
        best_index = int(np.argmax(result.scores))
        assert abs(truth_index - best_index) <= 4, (
            "the highest-scoring compression is not in the neighbourhood of the truth"
        )

    def test_search_is_discriminating(self):
        """A flat score curve would satisfy every test above, so the score
        must actually vary across the compression range."""
        depths, values = _synthetic_record()
        result = time_opt(depths, values, random_seed=11)

        scores = result.scores
        assert float(np.max(scores)) > 1.5 * float(np.median(scores)), (
            "the score curve is too flat: the compression search would be meaningless"
        )

    def test_tuned_axis_recovers_the_orbital_periods(self):
        """The tuned time axis must put the record's lines back onto the
        orbital periods they were generated from."""
        depths, values = _synthetic_record()
        result = time_opt(depths, values, random_seed=11)

        eccentricity = _dominant_period(result.tuned_times, values, (ECCENTRICITY_KYR * 0.9, ECCENTRICITY_KYR * 1.1))
        precession = _dominant_period(result.tuned_times, values, (PRECESSION_KYR * 0.9, PRECESSION_KYR * 1.1))
        assert abs(eccentricity - ECCENTRICITY_KYR) / ECCENTRICITY_KYR < 0.1, (
            f"the tuned axis puts the eccentricity line at {eccentricity:.2f} kyr, expected ~100 kyr"
        )
        assert abs(precession - PRECESSION_KYR) / PRECESSION_KYR < 0.1, (
            f"the tuned axis puts the precession line at {precession:.2f} kyr, expected ~21 kyr"
        )

    def test_precession_dominated_record_is_anchored_correctly(self):
        """A record whose strongest cycle is the precession (the usual case in
        cyclostratigraphy) must not be anchored onto the 100 kyr family."""
        rng = np.random.default_rng(11)
        n_points = 1000
        time_kyr = np.arange(n_points, dtype=float)
        values = np.sin(2.0 * np.pi * time_kyr / PRECESSION_KYR)
        values += 0.30 * np.cos(2.0 * np.pi * time_kyr / ECCENTRICITY_KYR)
        values += 0.02 * rng.standard_normal(n_points)
        depths = time_kyr * TRUE_SEDIMENTATION_RATE

        result = time_opt(depths, values, random_seed=3)
        assert abs(result.best_compression - 1.0) <= 0.15, (
            f"best compression {result.best_compression:.4f}: a precession-dominated record "
            "was not anchored onto the precession family"
        )

    def test_tuned_times_align_with_the_input_order(self):
        """``tuned_times`` must be one value per input point, in input order."""
        depths, values = _synthetic_record()
        result = time_opt(depths, values, random_seed=11)

        assert result.tuned_times.shape == depths.shape
        assert np.all(np.diff(result.tuned_times) > 0)

        # A reversed section must give the reversed tuned axis.
        reversed_result = time_opt(depths[::-1].copy(), values[::-1].copy(), random_seed=11)
        assert np.all(np.diff(reversed_result.tuned_times) < 0)

    def test_denser_sampling_does_not_degrade_the_tuning(self):
        """The frequency grid is set by the length of the record, not by the
        sampling density: sampling the same section more finely must not move
        the recovered solution."""
        depths, values = _synthetic_record()
        dense_depths = np.interp(np.linspace(depths[0], depths[-1], 4 * len(depths)), depths, values)
        dense_depths = np.linspace(depths[0], depths[-1], 4 * len(depths))

        coarse = time_opt(depths, values, random_seed=11)
        dense = time_opt(dense_depths, np.interp(dense_depths, depths, values), random_seed=11)

        assert abs(dense.best_compression - coarse.best_compression) <= 0.1, (
            "denser sampling moved the recovered compression factor"
        )
        assert dense.best_score >= 0.8 * coarse.best_score, "denser sampling degraded the tuning score"


# =============================================================================
# eTimeOpt: non-constant sedimentation rate
# =============================================================================


class TestETimeOpt:
    """eTimeOpt optimises a piecewise-constant rate."""

    def test_constant_rate_section_keeps_a_uniform_rate(self):
        """For a section deposited at a constant rate, every window must end
        up with essentially the same compression factor."""
        depths, values = _synthetic_record()
        result = etime_opt(depths, values, n_windows=5, n_compression=20, random_seed=5)

        factors = result.compression_factors
        assert factors.shape == (5,)
        assert float(np.ptp(factors)) <= 0.1, f"the per-window compression factors disagree: {factors}"
        assert abs(float(np.mean(factors)) - 1.0) <= 0.15

    def test_tuned_axis_is_monotone_and_aligned(self):
        depths, values = _synthetic_record()
        result = etime_opt(depths, values, n_windows=5, n_compression=20, random_seed=5)

        assert result.tuned_times.shape == depths.shape
        assert np.all(np.diff(result.tuned_times) > 0)
        assert result.window_centres.shape == (5,)
        assert np.all(np.diff(result.window_centres) > 0)
        assert result.scores.shape == (5,)


# =============================================================================
# Age scale
# =============================================================================


class TestAgeScale:
    """The age scale must be exact at the anchors and monotone between them."""

    ANCHORS = np.array([0.0, 50.0, 100.0])
    ANCHOR_AGES = np.array([120.0, 90.0, 60.0])

    @pytest.mark.parametrize("model", ["linear", "piecewise", "pchip"])
    def test_ages_are_exact_at_the_anchors(self, model: str):
        depths = np.linspace(0.0, 100.0, 21)
        result = build_age_scale(depths, self.ANCHORS, self.ANCHOR_AGES, model=model)

        assert result.depths.shape == depths.shape
        assert result.ages_ma.shape == depths.shape
        assert result.age_uncertainty.shape == depths.shape
        for anchor, age in zip(self.ANCHORS, self.ANCHOR_AGES, strict=True):
            assert np.isclose(result.ages_ma[depths == anchor][0], age, atol=1e-9)

    @pytest.mark.parametrize("model", ["linear", "piecewise", "pchip"])
    def test_ages_are_monotone_between_anchors(self, model: str):
        depths = np.linspace(0.0, 100.0, 51)
        result = build_age_scale(depths, self.ANCHORS, self.ANCHOR_AGES, model=model)

        assert np.all(np.diff(result.ages_ma) < 0), "ages must decrease as depth increases"

    def test_uncertainty_is_non_negative_and_exact_at_the_anchors(self):
        """The 1σ uncertainty must be non-negative everywhere and must equal
        the dated horizon's own error exactly at that horizon.

        Why *not* "the uncertainty grows away from the anchors": the age scale
        propagates the anchor errors with the inverse-distance-weighted blend
        of ``stratigraphy.correlation._depth_varying_half_width``. When the
        anchors carry *equal* errors (the usual case) that blend is a convex
        combination of identical numbers and is therefore **constant** over
        the dated range -- a linear age model genuinely has a constant
        propagated uncertainty there. Monotone growth away from the anchors
        is a property of neither linear nor PCHIP interpolation, so it is not
        a valid criterion. What *is* guaranteed, and what is checked here, is:

        * the uncertainty is never negative;
        * at a dated horizon the uncertainty collapses to that anchor's own
          1σ error (the anchor's weight dominates the blend);
        * the age at a dated horizon is exactly the given age.
        """
        depths = np.linspace(0.0, 100.0, 51)
        errors = np.array([0.5, 0.5, 0.5])
        result = build_age_scale(
            depths,
            self.ANCHORS,
            self.ANCHOR_AGES,
            anchor_uncertainties_ma=errors,
        )

        assert np.all(result.age_uncertainty >= 0.0), "age uncertainty must be non-negative"

        # At every anchor the uncertainty is the anchor's own error, and the
        # age is exactly the given age.
        for anchor, age, error in zip(self.ANCHORS, self.ANCHOR_AGES, errors, strict=True):
            index = int(np.argmin(np.abs(depths - anchor)))
            assert np.isclose(result.ages_ma[index], age, atol=1e-9)
            assert np.isclose(result.age_uncertainty[index], error, rtol=1e-6), (
                f"the uncertainty at the anchor {anchor} must equal the anchor's own error {error}"
            )

        # With unequal anchor errors the uncertainty tracks them: each anchor
        # carries its own error, and no point inside the dated range is claimed
        # to be better known than the best anchor.
        unequal = np.array([0.2, 0.5, 0.9])
        result = build_age_scale(
            depths,
            self.ANCHORS,
            self.ANCHOR_AGES,
            anchor_uncertainties_ma=unequal,
        )
        for anchor, error in zip(self.ANCHORS, unequal, strict=True):
            index = int(np.argmin(np.abs(depths - anchor)))
            assert np.isclose(result.age_uncertainty[index], error, rtol=1e-6)
        assert float(np.min(result.age_uncertainty)) >= 0.2 - 1e-6

    def test_uncertainty_grows_when_extrapolating_beyond_the_anchors(self):
        """Outside the dated range the distance penalty of
        ``stratigraphy.correlation._depth_varying_half_width`` grows the
        uncertainty, so an undated end of the section is never claimed to be
        better known than the nearest dated horizon. This is the one sense in
        which "farther from the anchors" really does mean "less certain"."""
        depths = np.array([-40.0, 0.0, 25.0, 50.0, 100.0, 140.0])
        errors = np.array([0.5, 0.5, 0.5])
        result = build_age_scale(
            depths,
            self.ANCHORS,
            self.ANCHOR_AGES,
            anchor_uncertainties_ma=errors,
        )

        assert result.age_uncertainty[0] > result.age_uncertainty[1], (
            "the uncertainty must grow when extrapolating below the first anchor"
        )
        assert result.age_uncertainty[-1] > result.age_uncertainty[-2], (
            "the uncertainty must grow when extrapolating above the last anchor"
        )
        assert np.all(result.age_uncertainty >= 0.0)

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValidationError):
            build_age_scale(np.linspace(0.0, 100.0, 11), self.ANCHORS, self.ANCHOR_AGES, model="cubic")

    def test_too_few_anchors_is_rejected(self):
        with pytest.raises(ValidationError):
            build_age_scale(np.linspace(0.0, 100.0, 11), np.array([0.0]), np.array([100.0]))


# =============================================================================
# Sedimentation rate -> age model
# =============================================================================


class TestSedimentationRateAgeModel:
    """A constant sedimentation rate gives a linear age-depth relation."""

    def test_age_is_linear_in_depth(self):
        depths = np.linspace(0.0, 100.0, 21)
        rate = 10.0  # depth units / Myr
        datum = 50.0  # Ma

        ages = sedimentation_rate_to_age_model(depths, rate, datum_age_ma=datum)

        assert ages.shape == depths.shape
        assert np.allclose(ages, datum + (depths - depths[0]) / rate)
        assert np.all(np.diff(ages) > 0)

    def test_datum_is_the_age_of_the_first_depth(self):
        depths = np.linspace(0.0, 100.0, 11)
        ages = sedimentation_rate_to_age_model(depths, 5.0, datum_age_ma=12.5)
        assert np.isclose(ages[0], 12.5)

    def test_non_positive_rate_is_rejected(self):
        with pytest.raises(ValidationError):
            sedimentation_rate_to_age_model(np.linspace(0.0, 100.0, 11), 0.0)

    def test_non_monotonic_depths_are_rejected(self):
        with pytest.raises(ValidationError):
            sedimentation_rate_to_age_model(np.array([0.0, 2.0, 1.0, 3.0]), 10.0)


# =============================================================================
# Milankovitch table
# =============================================================================


class TestMilankovitchPeriods:
    """The table must carry the published orbital periods."""

    def test_contains_the_standard_periods(self):
        periods = milankovitch_periods()
        values = list(periods.values())
        for expected in (100.0, 41.0, 21.0):
            assert any(abs(value - expected) / expected <= 0.30 for value in values), (
                f"no Milankovitch period within 30 % of {expected} kyr in {periods}"
            )

    def test_returns_a_copy(self):
        periods = milankovitch_periods()
        periods["eccentricity_100"] = -1.0
        assert milankovitch_periods()["eccentricity_100"] == 100.0


# =============================================================================
# Boundaries
# =============================================================================


class TestBoundaries:
    """Invalid inputs must raise ``ValidationError``, not a bare ValueError."""

    def test_non_monotonic_depths_are_rejected(self):
        depths = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.5, 7.0])
        values = np.sin(np.arange(9, dtype=float))
        with pytest.raises(ValidationError):
            time_opt(depths, values)

    def test_too_few_points_are_rejected(self):
        with pytest.raises(ValidationError):
            time_opt(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        "compression_range",
        [(0.0, 2.0), (-1.0, 2.0), (2.0, 0.5), (1.0, 1.0)],
    )
    def test_invalid_compression_range_is_rejected(self, compression_range: tuple[float, float]):
        depths = np.linspace(0.0, 50.0, 100)
        values = np.sin(depths / 3.0)
        with pytest.raises(ValidationError):
            time_opt(depths, values, compression_range=compression_range)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValidationError):
            time_opt(np.linspace(0.0, 50.0, 100), np.zeros(99))

    def test_non_finite_values_are_rejected(self):
        depths = np.linspace(0.0, 50.0, 100)
        values = np.sin(depths / 3.0)
        values[10] = np.nan
        with pytest.raises(ValidationError):
            time_opt(depths, values)


# =============================================================================
# Reproducibility
# =============================================================================


class TestReproducibility:
    """A fixed seed must give a bit-identical result."""

    def test_time_opt_is_reproducible(self):
        depths, values = _synthetic_record()
        first = time_opt(depths, values, random_seed=42)
        second = time_opt(depths, values, random_seed=42)

        assert np.array_equal(first.compression_factors, second.compression_factors)
        assert np.array_equal(first.scores, second.scores)
        assert np.array_equal(first.phase_offsets, second.phase_offsets)
        assert first.best_compression == second.best_compression
        assert first.best_phase == second.best_phase
        assert np.array_equal(first.tuned_times, second.tuned_times)

    def test_etime_opt_is_reproducible(self):
        depths, values = _synthetic_record()
        first = etime_opt(depths, values, n_windows=4, n_compression=12, random_seed=5)
        second = etime_opt(depths, values, n_windows=4, n_compression=12, random_seed=5)

        assert np.array_equal(first.compression_factors, second.compression_factors)
        assert np.array_equal(first.tuned_times, second.tuned_times)
        assert np.array_equal(first.window_centres, second.window_centres)
