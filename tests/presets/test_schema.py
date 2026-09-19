"""Tests for presets/schema.py: preset validation and atomic JSON I/O."""

import json
import os

import pytest

from presets.registry import PresetError
from presets.schema import (
    PRESET_FORMAT,
    Preset,
    analysis_ids,
    atomic_write_json,
    load_preset_file,
    looks_like_preset,
    save_preset_file,
    validate_preset_payload,
)


def _payload(**over):
    base = {
        "format": PRESET_FORMAT,
        "name": "Test NMDS",
        "analysis": "nmds",
        "params": {"n_restarts": 25},
        "created_at": "2026-09-19T12:00:00",
        "app_version": "1.0.1",
    }
    base.update(over)
    return base


class TestValidate:
    def test_valid_payload(self):
        preset = validate_preset_payload(_payload())
        assert preset.analysis_id == "nmds"
        assert preset.params["n_restarts"] == 25
        assert preset.warnings == []
        # missing params filled from registry defaults
        assert preset.params["n_dimensions"] == 2

    def test_format_mismatch_rejected(self):
        with pytest.raises(PresetError, match="format"):
            validate_preset_payload(_payload(format="paleoast_preset_v0"))
        with pytest.raises(PresetError, match="format"):
            validate_preset_payload({"analysis": "nmds"})

    def test_unknown_analysis_rejected(self):
        with pytest.raises(PresetError, match="Unknown analysis"):
            validate_preset_payload(_payload(analysis="does_not_exist"))

    def test_unknown_params_dropped_with_warning(self):
        preset = validate_preset_payload(_payload(params={"legacy_flag": True, "n_restarts": 5}))
        assert "legacy_flag" not in preset.params
        assert any("legacy_flag" in w and "dropped" in w for w in preset.warnings)

    def test_out_of_range_clamped_with_warning(self):
        preset = validate_preset_payload(_payload(params={"n_restarts": 100000}))
        assert preset.params["n_restarts"] == 100
        assert any("clamped" in w for w in preset.warnings)

    def test_bad_choice_rejected(self):
        with pytest.raises(PresetError, match="is not one of"):
            validate_preset_payload(_payload(params={"metric": "hamming"}))

    def test_name_falls_back_to_analysis(self):
        preset = validate_preset_payload(_payload(name="   "))
        assert preset.name == "nmds"

    def test_non_dict_rejected(self):
        with pytest.raises(PresetError, match="JSON object"):
            validate_preset_payload([1, 2, 3])

    def test_non_string_params_object_rejected(self):
        with pytest.raises(PresetError, match="params"):
            validate_preset_payload(_payload(params=[1, 2]))


class TestLooksLikePreset:
    def test_positive(self):
        assert looks_like_preset(_payload())

    def test_negative(self):
        assert not looks_like_preset({"a": 1})
        assert not looks_like_preset("x")
        assert not looks_like_preset(_payload(format="other"))


class TestAtomicWrite:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "sub" / "file.json")
        atomic_write_json(path, {"hello": "wörld"})
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == {"hello": "wörld"}
        assert not os.path.exists(path + ".tmp")

    def test_failure_leaves_original_intact(self, tmp_path):
        path = str(tmp_path / "file.json")
        atomic_write_json(path, {"v": 1})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"bad": object()})
        assert not os.path.exists(path + ".tmp")
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == {"v": 1}


class TestFileRoundtrip:
    def test_save_load(self, tmp_path):
        path = str(tmp_path / "p.json")
        preset = Preset(name="My PCA", analysis_id="pca", params={"n_components": 3})
        save_preset_file(path, preset)
        loaded = load_preset_file(path)
        assert loaded.name == "My PCA"
        assert loaded.params["n_components"] == 3

    def test_load_name_falls_back_to_filename(self, tmp_path):
        path = str(tmp_path / "cool_run.json")
        payload = _payload(name="nmds")  # name == analysis -> treated as absent
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        loaded = load_preset_file(path)
        assert loaded.name == "cool_run"

    def test_analysis_ids_sorted(self):
        ids = analysis_ids()
        assert ids == sorted(ids)
        assert "nmds" in ids
