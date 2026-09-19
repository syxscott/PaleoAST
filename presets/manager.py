# =============================================================================
# FILE: presets/manager.py
# =============================================================================
"""
On-disk preset library.

Presets live as individual ``*.json`` files in one directory (default
``~/.paleoast/presets``, overridable with the ``PALEOAST_PRESET_DIR``
environment variable).  :class:`PresetManager` lists/loads/saves/deletes
them, slugifying names into collision-free file names.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import re

from .registry import PresetError
from .schema import Preset, load_preset_file, save_preset_file

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".paleoast", "presets")


def default_preset_dir() -> str:
    return os.environ.get("PALEOAST_PRESET_DIR") or _DEFAULT_DIR


def slugify(name: str) -> str:
    """Filesystem-safe slug; empty results fall back to ``preset``."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return slug or "preset"


class PresetManager:
    """Directory-backed library of analysis presets."""

    def __init__(self, directory: str | None = None) -> None:
        self.directory = directory or default_preset_dir()
        self._logger = logging.getLogger(f"{__name__}.PresetManager")

    # ------------------------------------------------------------------
    def list_names(self) -> list[str]:
        """Preset names in the library, sorted by file name."""
        out = []
        for entry in sorted(os.listdir(self.directory)) if os.path.isdir(self.directory) else []:
            if not entry.endswith(".json"):
                continue
            try:
                out.append(load_preset_file(os.path.join(self.directory, entry)).name)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._logger.warning("Skipping invalid preset file %s: %s", entry, exc)
        out.sort(key=str.lower)
        return out

    def path_for(self, name: str) -> str:
        """Free file path for ``name`` (appends ``_2``, ``_3``... on collision)."""
        base = slugify(name)
        candidate = os.path.join(self.directory, f"{base}.json")
        counter = 2
        while os.path.exists(candidate):
            candidate = os.path.join(self.directory, f"{base}_{counter}.json")
            counter += 1
        return candidate

    def save(self, preset: Preset, path: str | None = None) -> str:
        """Validate and store ``preset``; returns the written path."""
        target = path or self.path_for(preset.name)
        save_preset_file(target, preset)
        self._logger.info("Preset saved: %s", target)
        return target

    def load(self, path: str) -> Preset:
        return load_preset_file(path)

    def load_by_name(self, name: str) -> Preset:
        """Load the preset whose validated name equals ``name``."""
        for entry in self.iter_files():
            preset = load_preset_file(entry)
            if preset.name == name:
                return preset
        raise PresetError(f"No preset named '{name}' in {self.directory}")

    def delete(self, path: str) -> None:
        os.remove(path)

    def iter_files(self) -> list[str]:
        if not os.path.isdir(self.directory):
            return []
        return [
            os.path.join(self.directory, e) for e in sorted(os.listdir(self.directory)) if e.endswith(".json")
        ]
