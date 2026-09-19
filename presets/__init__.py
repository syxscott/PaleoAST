"""
================================================================================
PaleoAST Phase W9 - Analysis Presets, Run Queue and Manifests
================================================================================

Borrowed from the surface-morphometrics-gui experiment/config workflow:

- registry: Qt-free declaration of preset-able analyses, their parameter
  types/defaults/bounds and their run guards ("has_data", "groups",
  "cache:gpa_result")
- schema: versioned JSON preset format with explicit legacy-key dropping
  and atomic writes
- manager: directory-backed preset library (~/.paleoast/presets)
- runner: sequential RunQueue state machine with guard skips, single
  running item, and a JSON run manifest

作者: PaleoAST Development Team
版本: 1.0.0
"""

from .manager import PresetManager, default_preset_dir, slugify
from .registry import (
    ANALYSIS_REGISTRY,
    AnalysisSpec,
    ParamSpec,
    PresetError,
    check_guards,
    get_spec,
)
from .runner import (
    ERROR,
    OK,
    PENDING,
    RUNNING,
    SKIPPED,
    RunQueue,
    RunQueueItem,
    build_manifest,
    write_manifest,
)
from .schema import (
    PRESET_FORMAT,
    Preset,
    analysis_ids,
    atomic_write_json,
    load_preset_file,
    looks_like_preset,
    save_preset_file,
    validate_preset_payload,
)

__all__ = [
    "ANALYSIS_REGISTRY",
    "ERROR",
    "OK",
    "PENDING",
    "PRESET_FORMAT",
    "RUNNING",
    "SKIPPED",
    "AnalysisSpec",
    "ParamSpec",
    "Preset",
    "PresetError",
    "PresetManager",
    "RunQueue",
    "RunQueueItem",
    "analysis_ids",
    "atomic_write_json",
    "build_manifest",
    "check_guards",
    "default_preset_dir",
    "get_spec",
    "load_preset_file",
    "looks_like_preset",
    "save_preset_file",
    "slugify",
    "validate_preset_payload",
    "write_manifest",
]
