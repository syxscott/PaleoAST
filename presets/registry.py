# =============================================================================
# FILE: presets/registry.py
# =============================================================================
"""
Registry of preset-able analyses.

An :class:`AnalysisSpec` declares the analysis id used in preset files, the
computational parameters (name, type, default, bounds/choices) and the
state preconditions ("guards") a run needs.  The registry is deliberately
Qt-free so presets, run queues and manifests can be validated headlessly;
``views/`` maps each id onto its ``_execute_*`` method.

Guards are strings evaluated against the application state:
    ``"has_data"``          a data matrix is loaded
    ``"groups"``            sample group assignments exist
    ``"cache:<key>"``       ``<key>`` is present in the result cache
                            (e.g. TPS needs a prior GPA -> ``cache:gpa_result``)

Borrowed conventions (surface-morphometrics-gui): read-with-default /
write-with-setdefault symmetry, and unknown/legacy parameter keys are
dropped explicitly rather than passed through to the analysis.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PresetError(ValueError):
    """Raised for malformed preset payloads or unknown analyses."""


@dataclass(frozen=True)
class ParamSpec:
    """Specification of one computational parameter of an analysis."""

    kind: str  # "int" | "float" | "bool" | "str" | "choice"
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None

    def normalise(self, value: Any, path: str, warnings: list[str]) -> Any:
        """Coerce ``value`` to this parameter's type, clamping to bounds."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise PresetError(f"{path}: expected a boolean, got {value!r}")
            return value
        if self.kind in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PresetError(f"{path}: expected a number, got {value!r}")
            if self.kind == "int":
                if float(value) != int(value):
                    raise PresetError(f"{path}: expected an integer, got {value!r}")
                out: Any = int(value)
            else:
                out = float(value)
            if self.min is not None and out < self.min:
                warnings.append(f"{path}: {out} clamped to min {self.min}")
                out = type(out)(self.min)
            if self.max is not None and out > self.max:
                warnings.append(f"{path}: {out} clamped to max {self.max}")
                out = type(out)(self.max)
            return out
        if self.kind == "choice":
            text = str(value)
            if self.choices and text not in self.choices:
                raise PresetError(f"{path}: {text!r} is not one of {list(self.choices)}")
            return text
        # "str"
        if not isinstance(value, str):
            raise PresetError(f"{path}: expected a string, got {value!r}")
        return value


@dataclass(frozen=True)
class AnalysisSpec:
    """Metadata describing one preset-able analysis."""

    analysis_id: str
    label: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    requires: tuple[str, ...] = ()

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.params.items() if spec.default is not None}


_METRIC_CHOICES = (
    "euclidean",
    "bray_curtis",
    "jaccard",
    "manhattan",
    "canberra",
)

ANALYSIS_REGISTRY: dict[str, AnalysisSpec] = {
    spec.analysis_id: spec
    for spec in (
        AnalysisSpec(
            analysis_id="pca",
            label="PCA",
            params={
                "n_components": ParamSpec("int", 2, min=1, max=50),
                "method": ParamSpec("choice", "covariance", choices=("covariance", "correlation")),
            },
            requires=("has_data",),
        ),
        AnalysisSpec(
            analysis_id="pcoa",
            label="PCoA",
            params={
                "metric": ParamSpec("choice", "bray_curtis", choices=_METRIC_CHOICES),
                "n_components": ParamSpec("int", 10, min=1, max=50),
            },
            requires=("has_data",),
        ),
        AnalysisSpec(
            analysis_id="nmds",
            label="NMDS",
            params={
                "metric": ParamSpec("choice", "bray_curtis", choices=_METRIC_CHOICES),
                "n_dimensions": ParamSpec("int", 2, min=1, max=5),
                "n_restarts": ParamSpec("int", 10, min=1, max=100),
                "max_iterations": ParamSpec("int", 500, min=10, max=100000),
                "tolerance": ParamSpec("float", 1e-5, min=0.0, max=1.0),
            },
            requires=("has_data",),
        ),
        AnalysisSpec(
            analysis_id="anosim",
            label="ANOSIM",
            params={
                "metric": ParamSpec("choice", "bray_curtis", choices=_METRIC_CHOICES),
                "n_permutations": ParamSpec("int", 999, min=99, max=99999),
            },
            requires=("has_data", "groups"),
        ),
        AnalysisSpec(
            analysis_id="permanova",
            label="PERMANOVA",
            params={
                "metric": ParamSpec("choice", "bray_curtis", choices=_METRIC_CHOICES),
                "n_permutations": ParamSpec("int", 999, min=99, max=99999),
            },
            requires=("has_data", "groups"),
        ),
        AnalysisSpec(
            analysis_id="tps_grid",
            label="TPS Grid",
            params={
                "grid_rows": ParamSpec("int", 10, min=2, max=100),
                "grid_cols": ParamSpec("int", 10, min=2, max=100),
            },
            requires=("cache:gpa_result",),
        ),
    )
}


def get_spec(analysis_id: str) -> AnalysisSpec:
    """Return the registry entry or raise :class:`PresetError`."""
    try:
        return ANALYSIS_REGISTRY[analysis_id]
    except KeyError:
        raise PresetError(
            f"Unknown analysis '{analysis_id}'; registered ids: {sorted(ANALYSIS_REGISTRY)}"
        ) from None


def check_guards(spec: AnalysisSpec, available: set[str]) -> str | None:
    """
    Return ``None`` if ``spec``'s preconditions are met, else the first
    unmet guard.  ``available`` lists satisfied guard strings computed by
    the caller (e.g. ``{"has_data", "cache:gpa_result"}``).
    """
    for guard in spec.requires:
        if guard not in available:
            return guard
    return None
