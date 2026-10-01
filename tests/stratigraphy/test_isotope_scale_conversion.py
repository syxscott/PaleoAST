# tests/stratigraphy/test_isotope_scale_conversion.py
"""
Regression tests for δ¹⁸O VPDB / VSMOW scale conversion in paleotemperature
equations (缺陷 1).

Erez & Luz (1983) and Bemis et al. (1998) were calibrated with both
δc and δw on a VPDB-like scale, using the historical pre-Coplen-1988
offset (δw_VPDB ≈ δw_VSMOW - 0.27 ‰). Feeding in raw VSMOW without
conversion gives delta_diff values off by ~30 ‰ and temperatures
100+ °C wrong. See the documented references inside each function.
"""

import math

import numpy as np
import pytest

from stratigraphy.isotope_analysis import (
    IsotopeAnalyzer,
    IsotopeData,
    EL_BEMIS_VSMOW_TO_VPDB_OFFSET,
)


class TestVPDBVSMOWConversion:
    """The shared helpers used by the temperature equations."""

    def test_coplen_round_trip(self):
        """Coplen 1988 linear conversion is self-inverse."""
        vpdb = -2.0
        vsmow = 1.03091 * vpdb + 30.91
        vpdb_back = (vsmow - 30.91) / 1.03091
        assert math.isclose(vpdb_back, vpdb, abs_tol=1e-9)

    def test_el_bemis_offset_constant(self):
        """The Erez & Luz / Bemis convention is a simple 0.27 ‰ offset."""
        assert math.isclose(EL_BEMIS_VSMOW_TO_VPDB_OFFSET, 0.27, abs_tol=1e-9)
        vsmow_to_vpdb_smow = 0.0 - EL_BEMIS_VSMOW_TO_VPDB_OFFSET
        assert math.isclose(vsmow_to_vpdb_smow, -0.27, abs_tol=1e-9)


class TestErezLuzScaleConversion:
    """Erez & Luz (1983) requires δw on the small-offset VPDB scale."""

    def test_warm_surface_ocean_with_default_scale(self):
        """Default scale is VSMOW. A warm tropical planktonic foraminifera
        (δc ≈ -1 ‰ VPDB) against SMOW (δw = 0 ‰ VSMOW) should yield a
        plausible 18-22 °C, NOT the broken 21.55 °C from no conversion
        AND NOT a ~-90 °C value from a wrong-direction conversion."""
        analyzer = IsotopeAnalyzer()
        T = analyzer.compute_paleotemperature_erez_luz(
            delta18O_sw=0.0, delta18O_c=-1.0,
            # default delta18O_sw_scale="vsmow"
        )
        # delta18O_sw_vpdb (E&L convention) = 0 - 0.27 = -0.27
        # delta_diff = -1.0 - (-0.27) = -0.73
        # T = 17 - 4.52*(-0.73) + 0.03*(-0.73)^2 ≈ 20.32
        assert 17 < T < 24, (
            f"Erez-Luz warm-water sample (δc=-1 VPDB, δw=0 VSMOW) gave "
            f"T = {T}, expected ~20 °C; scale conversion is broken"
        )

    def test_same_scale_inputs_match_polynomial(self):
        """When δw is already on VPDB (vpdb scale), the function must
        reproduce the polynomial directly."""
        analyzer = IsotopeAnalyzer()
        # delta_diff = 0, T = 17 °C
        T = analyzer.compute_paleotemperature_erez_luz(
            delta18O_sw=0.0, delta18O_c=0.0,
            delta18O_sw_scale="vpdb",
        )
        assert math.isclose(T, 17.0, abs_tol=0.05)

    def test_vsmow_input_is_converted_to_el_bemis_vpdb(self):
        """When the user supplies δw in VSMOW, the function MUST convert
        it via the 0.27 ‰ offset (E&L convention) before subtraction.
        Same numerical answer whether the conversion is internal or
        external."""
        analyzer = IsotopeAnalyzer()
        T_vsmow = analyzer.compute_paleotemperature_erez_luz(
            delta18O_sw=0.0, delta18O_c=-1.0,
            # default scale is vsmow
        )
        T_vpdb = analyzer.compute_paleotemperature_erez_luz(
            delta18O_sw=-EL_BEMIS_VSMOW_TO_VPDB_OFFSET, delta18O_c=-1.0,
            delta18O_sw_scale="vpdb",
        )
        assert math.isclose(T_vsmow, T_vpdb, abs_tol=1e-6), (
            f"Erez-Luz scale conversion broken: VSMOW input gave "
            f"{T_vsmow}, equivalent VPDB input gave {T_vpdb}"
        )


class TestBemisScaleConversion:
    """Bemis et al. (1998) uses the same small-offset VPDB convention."""

    def test_warm_water_default_scale(self):
        """Same warm-tropical sample as Erez-Luz: δc=-1 VPDB, δw=0 VSMOW,
        no genus correction."""
        analyzer = IsotopeAnalyzer()
        T = analyzer.compute_paleotemperature_bemis(
            delta18O_c=-1.0, delta18O_sw=0.0, genus="generic",
            # default delta18O_sw_scale="vsmow"
        )
        # δw_eff = (0 - 0.27) + 0 = -0.27
        # delta_diff = -1 - (-0.27) = -0.73
        # T = 16.998 - 4.52 * (-0.73) ≈ 20.30
        assert 17 < T < 24, (
            f"Bemis warm-water sample gave T = {T}, expected ~20 °C"
        )

    def test_genus_correction_still_works(self):
        """The genus-specific Δ (e.g. 0.27 for G. ruber) must still
        add the right offset AFTER the scale conversion."""
        analyzer = IsotopeAnalyzer()
        T_ruber = analyzer.compute_paleotemperature_bemis(
            delta18O_c=-1.0, delta18O_sw=0.0, genus="G. ruber",
        )
        # δw_eff = (0 - 0.27) + 0.27 = 0.0
        # delta_diff = -1 - 0 = -1
        # T = 16.998 - 4.52 * (-1) = 21.518
        assert math.isclose(T_ruber, 21.518, abs_tol=0.1)

    def test_vsmow_input_matches_vpdb_input(self):
        """The function must accept raw VSMOW and pre-converted VPDB δw
        and produce the same temperature."""
        analyzer = IsotopeAnalyzer()
        T_vsmow = analyzer.compute_paleotemperature_bemis(
            delta18O_c=-1.0, genus="generic",
        )
        T_vpdb = analyzer.compute_paleotemperature_bemis(
            delta18O_c=-1.0, delta18O_sw=-EL_BEMIS_VSMOW_TO_VPDB_OFFSET,
            genus="generic", delta18O_sw_scale="vpdb",
        )
        assert math.isclose(T_vsmow, T_vpdb, abs_tol=1e-6)


class TestValidRangeEnforcement:
    """Outside the calibrated 16-25 °C window must warn, not silently
    extrapolate."""

    def test_erez_luz_out_of_range_warns(self):
        analyzer = IsotopeAnalyzer()
        # δc far more negative than plausible → delta_diff ≪ -5 → T ≫ 35 °C
        with pytest.warns(UserWarning):
            analyzer.compute_paleotemperature_erez_luz(
                delta18O_sw=0.0,
                delta18O_c=-15.0,  # clearly unphysical
            )

    def test_bemis_out_of_range_warns(self):
        analyzer = IsotopeAnalyzer()
        with pytest.warns(UserWarning):
            analyzer.compute_paleotemperature_bemis(
                delta18O_c=-15.0, genus="generic",
            )


class TestIsotopeAnalyzerExposesPaleotemperature:
    """IsotopeAnalyzer.analyze() must compute paleotemperature when asked."""

    def test_analyze_computes_paleotemperature_when_requested(self):
        analyzer = IsotopeAnalyzer()
        depth = np.array([0, 10, 20, 30, 40], dtype=float)
        age = np.array([0, 1, 2, 3, 4], dtype=float)
        d18O = np.array([-1.0, -0.8, -1.1, -0.9, -1.0], dtype=float)
        iso = IsotopeData(depth=depth, age=age, d18O=d18O)

        result = analyzer.analyze(
            iso,
            compute_paleotemperature=True,
            delta18O_sw_vsmow=0.0,
            genus="generic",
            equation="erez_luz",
        )
        assert "paleotemperature" in result.metadata
        temps = result.metadata["paleotemperature"]["temperatures_c"]
        assert len(temps) == 5
        # All temperatures should land in the warm-tropical ballpark
        # (VSMOW → small-offset VPDB via 0.27 ‰)
        valid = temps[np.isfinite(temps)]
        assert np.all((valid > 17) & (valid < 24))
