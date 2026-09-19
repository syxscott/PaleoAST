"""Tests for presets/registry.py: parameter normalisation and guards."""

import pytest

from presets.registry import (
    ANALYSIS_REGISTRY,
    ParamSpec,
    PresetError,
    check_guards,
    get_spec,
)


class TestRegistry:
    def test_all_ids_registered(self):
        assert sorted(ANALYSIS_REGISTRY) == [
            "anosim",
            "nmds",
            "pca",
            "pcoa",
            "permanova",
            "tps_grid",
        ]

    def test_get_spec_unknown_raises(self):
        with pytest.raises(PresetError, match="Unknown analysis"):
            get_spec("nope")

    def test_defaults_match_params(self):
        for spec in ANALYSIS_REGISTRY.values():
            defaults = spec.defaults()
            assert set(defaults) <= set(spec.params)
            for key, value in defaults.items():
                assert spec.params[key].normalise(value, key, []) == value


class TestParamSpecNormalise:
    def test_bool_strict(self):
        spec = ParamSpec("bool", True)
        assert spec.normalise(False, "x", []) is False
        with pytest.raises(PresetError):
            spec.normalise("yes", "x", [])
        with pytest.raises(PresetError):
            spec.normalise(1, "x", [])  # int is not bool

    def test_int_rejects_fractional(self):
        spec = ParamSpec("int", 2)
        with pytest.raises(PresetError, match="integer"):
            spec.normalise(2.5, "x", [])

    def test_int_accepts_integral_float(self):
        spec = ParamSpec("int", 2)
        assert spec.normalise(3.0, "x", []) == 3

    def test_clamp_min_max_records_warnings(self):
        spec = ParamSpec("int", 2, min=1, max=50)
        warnings: list[str] = []
        assert spec.normalise(0, "params.n", warnings) == 1
        assert spec.normalise(999, "params.n", warnings) == 50
        assert len(warnings) == 2
        assert "clamped to min" in warnings[0]
        assert "clamped to max" in warnings[1]

    def test_rejects_non_numeric(self):
        spec = ParamSpec("float", 0.1)
        with pytest.raises(PresetError, match="number"):
            spec.normalise("0.1", "x", [])

    def test_choice_membership(self):
        spec = ParamSpec("choice", "bray_curtis", choices=("bray_curtis", "euclidean"))
        assert spec.normalise("euclidean", "x", []) == "euclidean"
        with pytest.raises(PresetError, match="is not one of"):
            spec.normalise("hamming", "x", [])

    def test_str_strict(self):
        spec = ParamSpec("str", "abc")
        assert spec.normalise("z", "x", []) == "z"
        with pytest.raises(PresetError):
            spec.normalise(5, "x", [])


class TestCheckGuards:
    def test_all_satisfied(self):
        spec = get_spec("anosim")
        assert check_guards(spec, {"has_data", "groups"}) is None

    def test_first_unmet_reported(self):
        spec = get_spec("anosim")
        assert check_guards(spec, set()) == "has_data"
        assert check_guards(spec, {"has_data"}) == "groups"

    def test_cache_guard(self):
        spec = get_spec("tps_grid")
        assert check_guards(spec, {"cache:gpa_result"}) is None
        assert check_guards(spec, {"cache:other"}) == "cache:gpa_result"
