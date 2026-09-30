# =============================================================================
# FILE: tests/stratigraphy/test_isotope_genus_offsets.py
# =============================================================================
"""
The Bemis et al. (1998) genus correction must not be dropped silently.

`compute_paleotemperature_bemis` looks up a per-species delta-w offset from a
table keyed on abbreviated names -- "G. ruber", "G. sacculifer". The lookup was a
plain `dict.get(genus, generic)`, so every name a palaeontologist would
actually type fell through to the generic calibration:

    'G. ruber'                T = 21.5180     offset applied
    'Globigerinoides ruber'   T = 20.2976     offset silently dropped
    'globigerinoides ruber'   T = 20.2976     offset silently dropped
    'Globigerinoides ruber '  T = 20.2976     offset silently dropped
    'g. ruber'                T = 20.2976     offset silently dropped

1.22 degrees of paleotemperature, with no warning. For an inter-species
comparison that offset is the entire point of the calibration, so a silently
generic answer is worse than an error.

A genus that genuinely has no tabulated offset is legitimate input and still
uses the generic calibration -- but it now says so.
"""

from __future__ import annotations

import warnings

import pytest

from stratigraphy.isotope_analysis import IsotopeAnalyzer

bemis = IsotopeAnalyzer.compute_paleotemperature_bemis

DELTA_C = -1.0

#: Spellings that all mean G. ruber.
RUBER_SPELLINGS = [
    "G. ruber",
    "G. ruber",
    "g. ruber",
    "Globigerinoides ruber",
    "globigerinoides ruber",
    "GLOBIGERINOIDES RUBER",
    "Globigerinoides ruber ",
    "  Globigerinoides ruber  ",
    "Globigerinoides, ruber",
    "ruber",
]

#: Genera with no tabulated offset; the generic calibration is correct for these.
UNMATCHED = ["S. quadrilateralus", "nonexistent genus", "Orbulina universa"]


def _temperature(genus: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bemis(DELTA_C, 0.0, genus=genus)


class TestGenusOffsetLookup:
    """The offset table is consulted for every realistic spelling."""

    def test_every_spelling_of_ruber_gets_the_same_temperature(self):
        """'Globigerinoides ruber' must give what 'G. ruber' gives."""
        expected = _temperature("G. ruber")
        assert expected != _temperature("generic"), "fixture is degenerate: ruber == generic"

        wrong = {g: _temperature(g) for g in RUBER_SPELLINGS if _temperature(g) != expected}
        assert not wrong, f"these spellings lost the genus offset: {wrong}"

    def test_sacculifer_offset_is_applied_too(self):
        expected = _temperature("G. sacculifer")
        for genus in ("Globigerinoides sacculifer", "sacculifer", "G. sacculifer"):
            assert _temperature(genus) == pytest.approx(expected), genus

    def test_the_two_species_offsets_stay_distinct(self):
        """A guard against a lookup that collapses everything to one value."""
        assert _temperature("G. ruber") != _temperature("G. sacculifer")
        assert _temperature("G. ruber") != _temperature("generic")
        assert _temperature("G. sacculifer") != _temperature("generic")

    def test_matched_names_do_not_warn_about_the_offset(self):
        """A correctly matched name must not produce a spurious warning."""
        for genus in ("G. ruber", "Globigerinoides ruber", "G. sacculifer", "generic"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bemis(DELTA_C, 0.0, genus=genus)
            offset_warnings = [w for w in caught if "offset is tabulated" in str(w.message)]
            assert not offset_warnings, f"{genus!r} warned about a missing offset: {offset_warnings[0].message}"


class TestUnmatchedGenusWarns:
    """A genus with no tabulated offset still works, but says so."""

    @pytest.mark.parametrize("genus", UNMATCHED)
    def test_unmatched_genus_warns(self, genus):
        with pytest.warns(UserWarning, match="offset is tabulated"):
            bemis(DELTA_C, 0.0, genus=genus)

    @pytest.mark.parametrize("genus", UNMATCHED)
    def test_unmatched_genus_still_returns_the_generic_calibration(self, genus):
        assert _temperature(genus) == pytest.approx(_temperature("generic"))

    def test_generic_is_explicit_and_silent(self):
        """Asking for 'generic' is a decision, not a miss."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bemis(DELTA_C, 0.0, genus="generic")
        assert not [w for w in caught if "offset is tabulated" in str(w.message)]
