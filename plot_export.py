"""Unified plot export facade for PaleoAST.

Provides a single entry point for rendering Matplotlib ``Figure`` objects
to the four supported formats (SVG, PDF, PNG, JPG) with a common
:class:`PlotExportOptions` for fine-grained control over DPI, size,
background, colour mode and JPEG quality.

The facade is intentionally format-agnostic so callers (PlotCanvas,
FloatingToolbar, MainWindow status-bar menu, batch export) all share
the same validation, error reporting and preset pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib
from matplotlib.figure import Figure

# Optional Pillow backend. Pillow is only required when exporting to
# raster formats; for vector formats we can skip the import. We import
# it lazily to keep the module import-time low.
try:  # pragma: no cover - exercised when Pillow is installed
    from PIL import Image  # noqa: F401
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


ExportFormat = Literal["svg", "pdf", "png", "jpg"]
ColorMode = Literal["color", "grayscale"]
BackgroundChoice = Literal["white", "transparent", "theme"]


@dataclass
class PlotExportOptions:
    """Container for plot export parameters.

    Attributes:
        format: Output file format. ``svg`` and ``pdf`` are vector;
            ``png`` and ``jpg`` are raster.
        dpi: Output resolution for raster formats. Ignored for vector
            formats (where resolution is effectively infinite).
        width_inches: Optional override for figure width in inches.
            When set, the figure is resized before saving.
        height_inches: Optional override for figure height in inches.
        transparent: When ``True`` the page colour is omitted from the
            rendered file. Only meaningful for PNG/SVG.
        background: One of ``white``, ``transparent`` or ``theme``.
            ``theme`` preserves whatever ``figure.patch.get_facecolor``
            is already set to.
        color_mode: ``color`` keeps the existing colour scheme;
            ``grayscale`` rewrites matplotlib colour cycle and face
            colours to grey tones before saving.
        jpeg_quality: Quality for the JPEG encoder (1-100).
        bbox_inches: ``tight`` to crop whitespace, ``standard`` to
            keep the current figure margins.
        embed_fonts: When ``True``, vector formats request the backend
            to embed the text fonts. Falls back silently when the
            backend does not support it.
        metadata: Optional metadata dict passed to ``savefig`` (for
            PDF metadata such as Title/Author).
    """

    format: ExportFormat = "png"
    dpi: int = 300
    width_inches: float | None = None
    height_inches: float | None = None
    transparent: bool = False
    background: BackgroundChoice = "white"
    color_mode: ColorMode = "color"
    jpeg_quality: int = 95
    bbox_inches: Literal["tight", "standard"] = "tight"
    embed_fonts: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# Map between matplotlib backend filenames and our short format names.
_FORMAT_EXTENSIONS: dict[str, str] = {
    "svg": "svg",
    "pdf": "pdf",
    "png": "png",
    "jpg": "jpg",
}

# Legitimate alternative spellings for the canonical extensions above.
# ``.jpeg`` is the official JPEG abbreviation and is accepted as an
# alias of ``jpg`` instead of being rejected by the validator.
_EXTENSION_ALIASES: dict[str, str] = {
    "jpeg": "jpg",
}


def _normalize_path(path: str, fmt: str) -> str:
    """Ensure the file path ends with the correct extension.

    If the user passes a path without an extension, append the one
    matching ``fmt``. If the extension does not match ``fmt`` (allowing
    registered aliases such as ``.jpeg`` for ``jpg``), raise.
    """
    p = Path(path)
    if p.suffix == "":
        return str(p.with_suffix(f".{_FORMAT_EXTENSIONS[fmt]}"))
    suffix = p.suffix.lower().lstrip(".")
    suffix = _EXTENSION_ALIASES.get(suffix, suffix)
    if suffix != _FORMAT_EXTENSIONS[fmt]:
        raise ValueError(
            f"File extension {p.suffix!r} does not match export format {fmt!r}. "
            f"Use .{ _FORMAT_EXTENSIONS[fmt] } or omit the extension."
        )
    return str(p)


def _validate_options(options: PlotExportOptions) -> None:
    """Raise ``ValueError`` for combinations matplotlib cannot honour."""
    if options.format not in _FORMAT_EXTENSIONS:
        raise ValueError(
            f"Unsupported export format: {options.format!r}. "
            f"Use one of {sorted(_FORMAT_EXTENSIONS)}."
        )
    if options.format in {"pdf", "jpg"} and options.transparent:
        # Matplotlib explicitly refuses ``transparent=True`` for PDF
        # (no alpha channel), and JPEG cannot represent transparency
        # at all. We surface this as a clean error rather than letting
        # matplotlib raise a confusing backend exception.
        raise ValueError(
            f"Format {options.format!r} does not support transparent backgrounds."
        )
    if options.dpi <= 0:
        raise ValueError("dpi must be positive")
    if options.width_inches is not None and options.width_inches <= 0:
        raise ValueError("width_inches must be positive")
    if options.height_inches is not None and options.height_inches <= 0:
        raise ValueError("height_inches must be positive")
    if not (1 <= options.jpeg_quality <= 100):
        raise ValueError("jpeg_quality must be in [1, 100]")


def _apply_grayscale(figure: Figure) -> Callable[[], None]:
    """Convert every line/face colour in ``figure`` to greyscale.

    The conversion mutates artists in place, so a zero-argument
    *restore* callable is returned; the caller must invoke it (typically
    from a ``finally`` block) once the export is finished. Lines,
    patches, collections, and images (colour-map + array) are all
    restored to their original state. The previous implementation never
    restored anything, leaving the on-screen figure permanently grey
    after a single grayscale export.
    """
    import numpy as np

    def to_grey(value: Any) -> float:
        rgba = matplotlib.colors.to_rgba(value)
        # ITU-R BT.601 luma weights.
        luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return float(luma)

    # (artist, setter_name, original_value) triples recorded before the
    # mutation so they can be replayed in reverse on restore.
    restore: list[tuple[Any, str, Any]] = []

    for ax in figure.get_axes():
        # Lines.
        for line in ax.get_lines():
            original = line.get_color()
            restore.append((line, "set_color", original))
            line.set_color(str(to_grey(original)))
        # Patches (bars, boxes, pie slices, ...).
        for patch in ax.patches:
            face = patch.get_facecolor()
            if face is not None and len(face) >= 3:
                restore.append((patch, "set_facecolor", face))
                patch.set_facecolor(str(to_grey(face)))
        # Collections (scatter, fill_between, ...): the face colour is
        # an (N, 4) array, so grey it row-wise.
        for coll in ax.collections:
            try:
                face = np.asarray(coll.get_facecolor(), dtype=float)
            except Exception:  # pragma: no cover - defensive
                continue
            if face.ndim == 2 and face.shape[0] > 0 and face.shape[1] >= 3:
                luma = 0.299 * face[:, 0] + 0.587 * face[:, 1] + 0.114 * face[:, 2]
                grey = np.column_stack([luma, luma, luma, face[:, 3]])
                restore.append((coll, "set_facecolor", np.array(face, copy=True)))
                coll.set_facecolor(grey)
        # Image artists such as heatmap imshows.
        for image in ax.images:
            try:
                restore.append((image, "set_cmap", image.get_cmap()))
                original_array = image.get_array()
                if original_array is not None:
                    restore.append((image, "set_array", original_array))
                arr = np.asarray(original_array, dtype=float)
                if arr.ndim >= 3:
                    image.set_array(arr.mean(axis=-1))
                # Reset the colourmap to a grey one if currently a
                # categorical palette so the visual cue survives.
                image.set_cmap("Greys")
            except Exception:  # pragma: no cover - defensive
                pass

    def restore_colors() -> None:
        for artist, setter, original in restore:
            try:
                getattr(artist, setter)(original)
            except Exception:  # pragma: no cover - defensive
                pass

    return restore_colors


def _resolve_background(figure: Figure, options: PlotExportOptions) -> str | None:
    """Translate the ``background`` choice into a savefig argument."""
    if options.background == "transparent" or options.transparent:
        return None
    if options.background == "white":
        return "white"
    # ``theme``: preserve whatever face colour the figure already has.
    try:
        face = figure.patch.get_facecolor()
    except Exception:  # pragma: no cover - defensive
        return "white"
    if face in (None, "none", "None"):
        return "white"
    rgba = matplotlib.colors.to_rgba(face)
    # Matplotlib accepts a colour name; round-trip through to_hex keeps
    # the same visual result.
    return matplotlib.colors.to_hex(rgba)


def export_figure(figure: Figure, path: str, options: PlotExportOptions) -> str:
    """Render ``figure`` to ``path`` using ``options``.

    Returns the absolute path of the written file. Raises ``ValueError``
    for malformed options and re-raises any matplotlib I/O error so the
    caller can present a useful message to the user.
    """
    _validate_options(options)
    fmt = options.format

    out_path = os.path.abspath(_normalize_path(path, fmt))

    # Snapshot figure state BEFORE any mutation — in particular before
    # the optional ``set_size_inches`` below. The previous code took the
    # snapshot after the resize, so the ``finally`` block restored the
    # *export* dimensions and permanently resized the on-screen canvas.
    original_size = (figure.get_figwidth(), figure.get_figheight())
    original_face = figure.patch.get_facecolor()

    # Vector formats honour ``embed_fonts`` through backend rcParams.
    # ``svg.fonttype='path'`` embeds text as glyph outlines while 'none'
    # leaves text editable (i.e. NOT embedded); ``pdf.fonttype=42``
    # ensures TrueType fonts are embedded in PDF output. Original values
    # are restored in the ``finally`` block below.
    rc_backup: dict[str, Any] = {}
    if fmt in {"svg", "pdf"}:
        for key in ("svg.fonttype", "pdf.fonttype"):
            rc_backup[key] = matplotlib.rcParams[key]
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["svg.fonttype"] = "none" if not options.embed_fonts else "path"

    # Resize figure if the user wants a custom canvas size.
    if options.width_inches is not None or options.height_inches is not None:
        w = options.width_inches or figure.get_figwidth()
        h = options.height_inches or figure.get_figheight()
        figure.set_size_inches(w, h)

    restore_colors: Callable[[], None] | None = None
    try:
        if options.color_mode == "grayscale":
            restore_colors = _apply_grayscale(figure)

        facecolor = _resolve_background(figure, options)
        kwargs: dict[str, Any] = {
            "format": fmt,
            "dpi": options.dpi if fmt in {"png", "jpg"} else "figure",
            "bbox_inches": options.bbox_inches,
            "metadata": options.metadata,
        }
        if facecolor is None:
            kwargs["transparent"] = True
        else:
            kwargs["facecolor"] = facecolor

        if fmt == "jpg":
            # JPEG has no alpha: drop any transparent facecolour.
            # matplotlib rejects ``metadata=`` for JPEG so we also
            # strip the metadata kwarg and use ``pil_kwargs`` for the
            # quality encoder option (introduced in matplotlib 3.7).
            kwargs["facecolor"] = "white"
            kwargs.pop("transparent", None)
            kwargs.pop("metadata", None)
            kwargs["pil_kwargs"] = {"quality": options.jpeg_quality}

        figure.savefig(out_path, **kwargs)
        return out_path
    finally:
        # Restore figure state so subsequent in-app interactions are
        # not affected by export-only mutations.
        if restore_colors is not None:
            restore_colors()
        for key, value in rc_backup.items():
            matplotlib.rcParams[key] = value
        figure.set_size_inches(*original_size)
        figure.patch.set_facecolor(original_face)


_PRESETS: dict[str, dict[str, Any]] = {
    # Publication: vector format, 600 dpi, no compression.
    "publication": {"format": "pdf", "dpi": 600, "bbox_inches": "tight"},
    # Vector: SVG/PDF preferred for editable, resizable art.
    "vector": {"format": "svg", "bbox_inches": "tight"},
    # Screen: small PNG suitable for slide decks.
    "screen": {"format": "png", "dpi": 144, "bbox_inches": "tight"},
}


def apply_preset(options: PlotExportOptions, preset: str) -> PlotExportOptions:
    """Return a new :class:`PlotExportOptions` with ``preset`` applied.

    Unknown preset names raise ``ValueError`` so a typo in the UI does
    not silently fall back to defaults.
    """
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown export preset {preset!r}; available: {sorted(_PRESETS)}."
        )
    return replace(options, **_PRESETS[preset])


def parse_options_from_path(path: str, format: str | None = None) -> PlotExportOptions:
    """Infer a sensible :class:`PlotExportOptions` from ``path``.

    ``format`` overrides the extension-based inference. Useful when the
    caller already knows the destination format.
    """
    fmt = format or Path(path).suffix.lower().lstrip(".")
    if not fmt:
        raise ValueError(
            f"Cannot infer export format from path {path!r}: no extension."
        )
    if fmt not in _FORMAT_EXTENSIONS:
        raise ValueError(
            f"Path {path!r} has unsupported extension {fmt!r}. "
            f"Use one of {sorted(_FORMAT_EXTENSIONS)}."
        )
    return PlotExportOptions(format=fmt)  # type: ignore[arg-type]