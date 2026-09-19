"""Tests for stratigraphy/time_bins.py — W6 palaeoverse port (DP equal-length bins,
bin_time five methods, FAD/LAD ranges, range-through expansion)."""

import numpy as np
import pytest

from utils.exceptions import DataValidationError
from stratigraphy.time_bins import (
    bin_time,
    get_scale,
    tax_expand_time,
    tax_range_time,
    time_bins,
)
from macroevolution.diversity import interval_count_diversity, range_through_diversity


@pytest.fixture
def user_scale():
    """palaeoverse golden user scale: 5 intervals spanning 0–53 Ma."""
    return [
        {"interval_name": "1", "min_ma": 0, "max_ma": 18},
        {"interval_name": "2", "min_ma": 18, "max_ma": 32},
        {"interval_name": "3", "min_ma": 32, "max_ma": 38},
        {"interval_name": "4", "min_ma": 38, "max_ma": 45},
        {"interval_name": "5", "min_ma": 45, "max_ma": 53},
    ]


@pytest.fixture
def five_bins_10myr():
    """palaeoverse test_bins: five 10-Myr bins, youngest first (bin 1 = 0–10)."""
    return [{"bin": i + 1, "min_ma": 10 * i, "max_ma": 10 * (i + 1)} for i in range(5)]


@pytest.fixture
def edge_occurrences():
    """The 8-occurrence edge-case frame from palaeoverse test-bin_time.R."""
    spans = [(2, 8), (10, 20), (5, 15), (9, 19), (5, 25), (0, 50), (0, 10), (15, 15)]
    return [{"taxon": f"occ{i + 1}", "min_ma": a, "max_ma": b} for i, (a, b) in enumerate(spans)]


class TestTimeBinsDP:
    def test_user_scale_size15_golden(self, user_scale):
        tb = time_bins(scale=user_scale, size=15)
        assert [b["intervals"] for b in tb] == ["5", "4, 3", "2", "1"]
        assert [round(b["duration_myr"], 6) for b in tb] == [8.0, 13.0, 14.0, 18.0]
        assert [b["max_ma"] for b in tb] == [53, 45, 32, 18]
        assert [b["min_ma"] for b in tb] == [45, 32, 18, 0]
        assert [b["bin"] for b in tb] == [1, 2, 3, 4]
        assert all(b["grouping_rank"] == "user" for b in tb)

    def test_n_bins_matches_size_equivalent(self, user_scale):
        a = time_bins(scale=user_scale, size=15)
        b = time_bins(scale=user_scale, n_bins=4)
        assert [x["intervals"] for x in a] == [x["intervals"] for x in b]

    def test_phanerozoic_periods_coarse_binning(self):
        tb = time_bins(rank="period", n_bins=6)
        assert len(tb) == 6
        total_in = sum(b["duration_myr"] for b in tb)
        total_out = sum(b["duration_myr"] for b in get_scale(rank="period"))
        assert total_in == pytest.approx(total_out, abs=1e-9)
        # contiguous coverage from old to young
        assert tb[0]["max_ma"] == pytest.approx(538.8)
        assert tb[-1]["min_ma"] == pytest.approx(0.0)
        for older, younger in zip(tb, tb[1:]):
            assert older["min_ma"] == pytest.approx(younger["max_ma"])

    def test_size_wins_over_n_bins(self, user_scale):
        tb = time_bins(scale=user_scale, size=53, n_bins=4)
        assert len(tb) == 1

    def test_huge_size_clamps_to_one_bin(self, user_scale):
        tb = time_bins(scale=user_scale, size=10000)
        assert len(tb) == 1

    def test_n_bins_too_large_raises(self, user_scale):
        with pytest.raises(DataValidationError, match="must not be greater"):
            time_bins(scale=user_scale, n_bins=99)

    def test_no_args_returns_intervals(self, user_scale):
        tb = time_bins(scale=user_scale)
        assert [b["intervals"] for b in tb] == ["5", "4", "3", "2", "1"]  # oldest first

    def test_unknown_interval_raises(self):
        with pytest.raises(DataValidationError, match="Unknown interval"):
            get_scale(rank="period", interval="Blancan")


class TestBinTimeMethods:
    def test_mid(self, edge_occurrences, five_bins_10myr):
        rows = bin_time(edge_occurrences, five_bins_10myr, method="mid")
        assert [r["bin_assignment"] for r in rows] == [1, 2, None, 2, 2, 3, 1, 2]
        assert rows[2]["n_bins"] == 2  # boundary midpoint still sees overlap count
        assert rows[0]["bin_midpoint"] == 5

    def test_majority_golden(self, edge_occurrences, five_bins_10myr):
        rows = bin_time(edge_occurrences, five_bins_10myr, method="majority")
        expected = {
            1: (1, 100.0), 2: (2, 100.0), 3: (1, 50.0), 4: (2, 90.0),
            5: (2, 50.0), 6: (1, 20.0), 7: (1, 100.0), 8: (2, 100.0),
        }
        for r in rows:
            exp_bin, exp_pct = expected[r["id"]]
            assert r["bin_assignment"] == exp_bin
            assert r["overlap_percentage"] == pytest.approx(exp_pct)

    def test_all(self, edge_occurrences, five_bins_10myr):
        rows = bin_time(edge_occurrences, five_bins_10myr, method="all")
        assert len(rows) == sum(r["n_bins"] for r in bin_time(edge_occurrences, five_bins_10myr, method="mid"))
        ids = [r["id"] for r in rows]
        assert ids.count(6) == 5  # occ6 spans all five bins
        assert ids.count(3) == 2 and ids.count(8) == 1
        per3 = sorted(r["bin_assignment"] for r in rows if r["id"] == 3)
        assert per3 == [1, 2]

    def test_random_returns_reps_datasets(self, edge_occurrences, five_bins_10myr):
        datasets = bin_time(edge_occurrences, five_bins_10myr, method="random", reps=7, seed=3)
        assert len(datasets) == 7
        for ds in datasets:
            assert len(ds) == 8
            for r in ds:
                assert r["bin_assignment"] is None or 1 <= r["bin_assignment"] <= 5
        # a single-overlap occurrence is stable across reps
        assert {ds[0]["bin_assignment"] for ds in datasets} == {1}
        # deterministic with seed
        again = bin_time(edge_occurrences, five_bins_10myr, method="random", reps=7, seed=3)
        assert [[r["bin_assignment"] for r in d] for d in datasets] == [
            [r["bin_assignment"] for r in d] for d in again
        ]

    def test_point_uniform(self, five_bins_10myr):
        occ = [{"taxon": "p", "min_ma": 15, "max_ma": 15}, {"taxon": "q", "min_ma": 0, "max_ma": 50}]
        datasets = bin_time(occ, five_bins_10myr, method="point", reps=5, seed=11)
        assert len(datasets) == 5
        assert all(d[0]["point_estimates"] == 15 and d[0]["bin_assignment"] == 2 for d in datasets)
        for d in datasets:
            est = d[1]["point_estimates"]
            assert 0 <= est <= 50
            b = d[1]["bin_assignment"]
            lo, hi = 10 * (b - 1), 10 * b
            assert lo <= est <= hi

    def test_point_custom_density(self, five_bins_10myr):
        occ = [{"min_ma": 0, "max_ma": 50}]
        datasets = bin_time(
            occ, five_bins_10myr, method="point", reps=20, seed=2,
            fun=lambda x: np.exp(-((x - 0.5) ** 2) / 0.02),
        )
        est = [d[0]["point_estimates"] for d in datasets]
        assert 20 <= np.mean(est) <= 30  # density peaks at midpoint (25 Ma)

    def test_bad_method_raises(self, five_bins_10myr):
        with pytest.raises(DataValidationError, match="mid, majority"):
            bin_time([{"min_ma": 1, "max_ma": 2}], five_bins_10myr, method="scale")

    def test_missing_columns_raise(self, five_bins_10myr):
        with pytest.raises(DataValidationError):
            bin_time([{"min_ma": 1}], five_bins_10myr)
        with pytest.raises(DataValidationError):
            bin_time([{"min_ma": 1, "max_ma": 2}], [{"bin": 1}])

    def test_reversed_range_raises(self, five_bins_10myr):
        with pytest.raises(DataValidationError, match="max_ma must be"):
            bin_time([{"min_ma": 30, "max_ma": 10}], five_bins_10myr)


class TestRanges:
    def test_fad_lad(self):
        occ = [
            {"taxon": "A", "min_ma": 110, "max_ma": 150},
            {"taxon": "A", "min_ma": 120, "max_ma": 130},
            {"taxon": "B", "min_ma": 0, "max_ma": 30},
        ]
        rows = tax_range_time(occ, by="name")
        assert [r["taxon"] for r in rows] == ["A", "B"]
        a, b = rows
        assert a["max_ma"] == 150 and a["min_ma"] == 110 and a["range_myr"] == 40 and a["n_occ"] == 2
        assert b["max_ma"] == 30 and b["min_ma"] == 0 and b["n_occ"] == 1
        rows_fad = tax_range_time(occ, by="FAD")
        assert [r["taxon"] for r in rows_fad] == ["B", "A"]  # ascending FAD age = youngest first
        assert [r["taxon_id"] for r in rows_fad] == [1, 2]

    def test_expand_range_through(self, five_bins_10myr):
        ranges = [
            {"taxon": "A", "min_ma": 22, "max_ma": 37},
            {"taxon": "B", "min_ma": 0, "max_ma": 25},
        ]
        rows = tax_expand_time(ranges, five_bins_10myr)
        a_bins = [r["bin"] for r in rows if r["taxon"] == "A"]
        assert a_bins == [3, 4]  # strict overlap excludes bin 2 (10–20)
        assert [r["bin"] for r in rows if r["taxon"] == "B"] == [1, 2, 3]
        origs = {(r["taxon"], r["bin"]): r["orig"] for r in rows}
        exts = {(r["taxon"], r["bin"]): r["ext"] for r in rows}
        assert origs[("A", 4)] and not origs[("A", 3)]
        assert exts[("A", 3)] and not exts[("A", 4)]
        assert not any(exts[("B", b)] for b in (1, 2, 3))  # B still extant (LAD = 0)

    def test_expand_requires_keys(self, five_bins_10myr):
        with pytest.raises(DataValidationError, match="taxon"):
            tax_expand_time([{"min_ma": 1, "max_ma": 2}], five_bins_10myr)
        with pytest.raises(DataValidationError, match="non-negative"):
            tax_expand_time([{"taxon": "X", "min_ma": -1, "max_ma": 2}], five_bins_10myr)


class TestDiversityCurves:
    def test_range_through_curve(self, five_bins_10myr):
        ranges = [
            {"taxon": "A", "min_ma": 22, "max_ma": 37},
            {"taxon": "B", "min_ma": 0, "max_ma": 25},
        ]
        curve = range_through_diversity(ranges, five_bins_10myr)
        assert [c["richness"] for c in curve] == [1, 1, 2, 1, 0]
        assert sum(c["origination"] for c in curve) == 2
        assert sum(c["extinction"] for c in curve) == 1  # B extant, not counted

    def test_interval_count_curve(self, five_bins_10myr):
        occ = [
            {"taxon": "A", "min_ma": 22, "max_ma": 37},
            {"taxon": "B", "min_ma": 0, "max_ma": 25},
            {"taxon": "C", "min_ma": 12, "max_ma": 18},
        ]
        binned = bin_time(occ, five_bins_10myr, method="all")
        curve = interval_count_diversity(binned, five_bins_10myr)
        assert [c["richness"] for c in curve] == [1, 2, 2, 1, 0]
