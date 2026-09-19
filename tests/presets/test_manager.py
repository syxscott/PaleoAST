"""Tests for presets/manager.py: directory-backed preset library."""

import json
import os

import pytest

from presets.manager import PresetManager, default_preset_dir, slugify
from presets.registry import PresetError
from presets.schema import PRESET_FORMAT, Preset


def _preset(name="Run A", analysis="pca", **params):
    return Preset(
        name=name,
        analysis_id=analysis,
        params=params or {"n_components": 3},
        created_at="2026-09-19T00:00:00",
        app_version="1.0.1",
    )


class TestSlugify:
    @pytest.mark.parametrize(
        ("name", "slug"),
        [
            ("Bray NMDS 2D x25", "Bray_NMDS_2D_x25"),
            ("weird/../name!", "weird_.._name"),
            ("  spaced  ", "spaced"),
            ("!!!", "preset"),
            ("ok-name_1.2", "ok-name_1.2"),
        ],
    )
    def test_slugs(self, name, slug):
        assert slugify(name) == slug


class TestDefaultDir:
    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PALEOAST_PRESET_DIR", str(tmp_path))
        assert default_preset_dir() == str(tmp_path)

    def test_fallback_home(self, monkeypatch):
        monkeypatch.delenv("PALEOAST_PRESET_DIR", raising=False)
        assert default_preset_dir().endswith(os.path.join(".paleoast", "presets"))


class TestManager:
    def test_save_and_list(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        path = manager.save(_preset("Alpha"))
        assert os.path.isfile(path)
        assert manager.list_names() == ["Alpha"]

    def test_collision_suffixes(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        p1 = manager.save(_preset("Same Name"))
        p2 = manager.save(_preset("Same Name"))
        p3 = manager.save(_preset("Same Name"))
        assert os.path.basename(p1) == "Same_Name.json"
        assert os.path.basename(p2) == "Same_Name_2.json"
        assert os.path.basename(p3) == "Same_Name_3.json"

    def test_load_by_name(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        manager.save(_preset("Beta", analysis="nmds", n_restarts=42))
        loaded = manager.load_by_name("Beta")
        assert loaded.analysis_id == "nmds"
        assert loaded.params["n_restarts"] == 42

    def test_load_by_name_missing(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        with pytest.raises(PresetError, match="No preset named"):
            manager.load_by_name("Ghost")

    def test_list_skips_invalid_files(self, tmp_path):
        (tmp_path / "broken.json").write_text("not json at all", encoding="utf-8")
        (tmp_path / "other.txt").write_text("ignore me", encoding="utf-8")
        manager = PresetManager(str(tmp_path))
        manager.save(_preset("Good"))
        assert manager.list_names() == ["Good"]

    def test_iter_files_empty_dir(self, tmp_path):
        manager = PresetManager(str(tmp_path / "missing"))
        assert manager.iter_files() == []

    def test_delete(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        path = manager.save(_preset("Del"))
        manager.delete(path)
        assert manager.list_names() == []

    def test_save_rejects_unknown_analysis(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        with pytest.raises(PresetError):
            manager.save(Preset(name="X", analysis_id="nope"))
        assert not (tmp_path / "X.json").exists()

    def test_saved_file_is_valid_v1_json(self, tmp_path):
        manager = PresetManager(str(tmp_path))
        path = manager.save(_preset("Shape"))
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["format"] == PRESET_FORMAT
        assert payload["analysis"] == "pca"
