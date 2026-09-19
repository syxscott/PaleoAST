# =============================================================================
# FILE: tests/golden/test_palaeoverse_values.py
# =============================================================================
"""
Golden numeric values borrowed from the palaeoverse R package.

Sources (paleoast-refs/palaeoverse):
    - tests/testthat/test-time_bins.R  (stage-level time_bins() goldens,
      exercised here through PaleoAST user scales since PaleoAST has no
      built-in stage rank)
    - data/GTS2020.rda (interval colour + font columns, cross-checked
      against the ICS-chart fills shipped in stratigraphy.time_bins)

PaleoAST keeps ICS v0.1 boundaries for its built-in table, so only the
style columns (colour/font/abbr) and the tidy-scale arithmetic are asserted
against palaeoverse numbers here.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from stratigraphy.time_bins import _font_for_colour, get_scale

# palaeoverse::time_bins(interval = "Maastrichtian") stage row
MAASTRICHTIAN = {
    "interval_name": "Maastrichtian",
    "max_ma": 72.1,
    "min_ma": 66.0,
    "abbr": "M",
    "colour": "#F2FA8C",
}
MAASTRICHTIAN_GOLDEN = {
    "bin": 1,
    "mid_ma": 69.05,
    "duration_myr": 6.1,
    "font": "black",
}

# palaeoverse::time_bins(interval = 10) stage row (Tortonian)
TORTONIAN = {
    "interval_name": "Tortonian",
    "max_ma": 11.63,
    "min_ma": 7.246,
    "abbr": "T",
    "colour": "#FFFF66",
}
TORTONIAN_GOLDEN = {
    "bin": 1,
    "mid_ma": 9.438,
    "duration_myr": 4.384,
    "font": "black",
}

# (interval_name, rank, colour, font) exactly as in palaeoverse::GTS2020
GTS2020_STYLE = [
    ("Phanerozoic", "eon", "#9AD9DD", "black"),
    ("Cenozoic", "era", "#F2F91D", "black"),
    ("Mesozoic", "era", "#67C5CA", "black"),
    ("Paleozoic", "era", "#99C08D", "black"),
    ("Quaternary", "period", "#F9F97F", "black"),
    ("Neogene", "period", "#FFE619", "black"),
    ("Paleogene", "period", "#FD9A52", "black"),
    ("Cretaceous", "period", "#7FC64E", "black"),
    ("Jurassic", "period", "#34B2C9", "black"),
    ("Triassic", "period", "#812B92", "white"),
    ("Permian", "period", "#F04028", "white"),
    ("Carboniferous", "period", "#67A599", "black"),
    ("Devonian", "period", "#CB8C37", "black"),
    ("Silurian", "period", "#B3E1B6", "black"),
    ("Ordovician", "period", "#009270", "white"),
    ("Cambrian", "period", "#7FA056", "black"),
    ("Holocene", "epoch", "#FEEBD2", "black"),
    ("Pleistocene", "epoch", "#FFEFAF", "black"),
    ("Pliocene", "epoch", "#FFFF99", "black"),
    ("Miocene", "epoch", "#FFFF00", "black"),
    ("Eocene", "epoch", "#FDB46C", "black"),
    ("Oligocene", "epoch", "#FEC07A", "black"),
    ("Paleocene", "epoch", "#FDA75F", "black"),
]


def _user_scale_row(stage: dict, golden: dict) -> None:
    rows = get_scale(scale=[dict(stage)])
    assert len(rows) == 1
    row = rows[0]
    assert row["interval_name"] == stage["interval_name"]
    assert row["rank"] == "user"
    assert row["bin"] == golden["bin"]
    assert row["max_ma"] == pytest.approx(stage["max_ma"])
    assert row["min_ma"] == pytest.approx(stage["min_ma"])
    assert row["mid_ma"] == pytest.approx(golden["mid_ma"])
    assert row["duration_myr"] == pytest.approx(golden["duration_myr"])
    assert row["abbr"] == stage["abbr"]
    assert row["colour"] == stage["colour"]
    assert row["font"] == golden["font"]


class TestStageUserScaleGoldens:
    def test_maastrichtian(self):
        _user_scale_row(MAASTRICHTIAN, MAASTRICHTIAN_GOLDEN)

    def test_tortonian(self):
        _user_scale_row(TORTONIAN, TORTONIAN_GOLDEN)

    def test_maastrichtian_arithmetic_exact(self):
        row = get_scale(scale=[dict(MAASTRICHTIAN)])[0]
        assert (row["max_ma"] + row["min_ma"]) / 2.0 == pytest.approx(69.05)
        assert row["max_ma"] - row["min_ma"] == pytest.approx(6.1)


class TestGTS2020StyleGoldens:
    @pytest.mark.parametrize(("name", "rank", "colour", "font"), GTS2020_STYLE)
    def test_builtin_row_style(self, name, rank, colour, font):
        rows = get_scale(rank=rank, interval="Phanerozoic")
        row = next(r for r in rows if r["interval_name"] == name)
        assert row["colour"] == colour
        assert row["font"] == font

    @pytest.mark.parametrize(("name", "rank", "colour", "font"), GTS2020_STYLE)
    def test_font_rule_matches_palaeoverse(self, name, rank, colour, font):
        assert _font_for_colour(colour) == font

    def test_stage_colours_are_palaeoverse_fills(self):
        assert MAASTRICHTIAN["colour"] == "#F2FA8C"
        assert TORTONIAN["colour"] == "#FFFF66"


class TestBuiltinScaleStructure:
    def test_periods_contiguous_and_ordered(self):
        rows = get_scale(rank="period")
        assert [r["interval_name"] for r in rows][:2] == ["Cambrian", "Ordovician"]
        assert rows[0]["max_ma"] == pytest.approx(538.8)
        assert rows[-1]["min_ma"] == pytest.approx(0.0)
        for a, b in pairwise(rows):
            assert a["min_ma"] == pytest.approx(b["max_ma"])
            assert a["bin"] + 1 == b["bin"]

    def test_bin_numbering_oldest_first(self):
        rows = get_scale(scale=[dict(MAASTRICHTIAN), dict(TORTONIAN)])
        assert [r["interval_name"] for r in rows] == ["Maastrichtian", "Tortonian"]
        assert [r["bin"] for r in rows] == [1, 2]
