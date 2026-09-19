# =============================================================================
# FILE: presets/schema.py
# =============================================================================
"""
Preset file format: validation, parsing and atomic writing.

A preset is a small JSON document::

    {
      "format": "paleoast_preset_v1",
      "name": "Bray NMDS 2D x25",
      "analysis": "nmds",
      "params": {"n_restarts": 25},
      "created_at": "2026-09-19T12:00:00",
      "app_version": "1.0.1"
    }

Only parameter keys known to :data:`presets.registry.ANALYSIS_REGISTRY`
survive validation; unknown/legacy keys are dropped with a warning instead
of being forwarded to the analysis (surface-morphometrics-gui convention).
Writes go through a temp file + ``os.replace`` so a crash can never leave a
half-written preset behind.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .registry import ANALYSIS_REGISTRY, PresetError, get_spec

logger = logging.getLogger(__name__)

PRESET_FORMAT = "paleoast_preset_v1"


@dataclass
class Preset:
    """One validated analysis preset."""

    name: str
    analysis_id: str
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    app_version: str = ""
    warnings: list[str] = field(default_factory=list, repr=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": PRESET_FORMAT,
            "name": self.name,
            "analysis": self.analysis_id,
            "params": dict(self.params),
            "created_at": self.created_at,
            "app_version": self.app_version,
        }


def looks_like_preset(obj: Any) -> bool:
    """Duck-type check for a preset payload (used before full validation)."""
    return (
        isinstance(obj, dict)
        and obj.get("format") == PRESET_FORMAT
        and isinstance(obj.get("analysis"), str)
        and isinstance(obj.get("params", {}), dict)
    )


def validate_preset_payload(obj: Any) -> Preset:
    """
    Validate a raw mapping into a :class:`Preset`.

    Missing parameters are filled from the registry defaults; out-of-range
    numbers are clamped (recorded in ``preset.warnings``); unknown keys are
    dropped (also recorded).

    Raises:
        PresetError: If the payload is not a preset, the format version is
            wrong/absent, the analysis id is unknown, or a parameter has a
            wrong type or an illegal choice value.
    """
    if not isinstance(obj, dict):
        raise PresetError(f"Preset must be a JSON object, got {type(obj).__name__}")
    fmt = obj.get("format")
    if fmt != PRESET_FORMAT:
        raise PresetError(f"Preset has format {fmt!r}, expected {PRESET_FORMAT!r}")
    analysis_id = obj.get("analysis")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise PresetError("Preset is missing a non-empty 'analysis' id")
    spec = get_spec(analysis_id)

    name = obj.get("name")
    if name is not None and not isinstance(name, str):
        raise PresetError(f"Preset 'name' must be a string, got {type(name).__name__}")
    name = (name or "").strip() or analysis_id

    raw_params = obj.get("params", {})
    if not isinstance(raw_params, dict):
        raise PresetError("Preset 'params' must be an object")

    warnings: list[str] = []
    params = spec.defaults()
    for key, value in raw_params.items():
        if key not in spec.params:
            warnings.append(f"params.{key}: unknown key for '{analysis_id}', dropped")
            continue
        params[key] = spec.params[key].normalise(value, f"params.{key}", warnings)

    created_at = obj.get("created_at", "")
    app_version = obj.get("app_version", "")
    for key, value in (("created_at", created_at), ("app_version", app_version)):
        if not isinstance(value, str):
            raise PresetError(f"Preset '{key}' must be a string, got {type(value).__name__}")

    return Preset(
        name=name,
        analysis_id=analysis_id,
        params=params,
        created_at=created_at,
        app_version=app_version,
        warnings=warnings,
    )


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON via temp file + rename so readers never see a partial file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_preset_file(path: str) -> Preset:
    """Read and validate one preset JSON file."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    preset = validate_preset_payload(payload)
    if not preset.name or preset.name == preset.analysis_id:
        preset.name = os.path.splitext(os.path.basename(path))[0]
    return preset


def save_preset_file(path: str, preset: Preset) -> None:
    """Validate-then-write a preset to ``path`` (atomic)."""
    checked = validate_preset_payload(preset.to_payload())
    atomic_write_json(path, checked.to_payload())


def analysis_ids() -> list[str]:
    """All registered analysis ids, sorted."""
    return sorted(ANALYSIS_REGISTRY)
