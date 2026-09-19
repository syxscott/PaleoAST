# =============================================================================
# FILE: visualization/strat_column.py
# =============================================================================
"""
ICS-styled stratigraphic column rendering for PaleoAST.

Borrowed machinery:
    * mplStrater (BSD-3): dual-channel layer rendering -- ``pcolormesh``
      paints each bed with its lithology fill colour, and a second pass
      overlays classic matplotlib hatches through ``PathPatch`` objects, so
      colour and pattern carry independent information.
    * palaeoverse ``axis_geo()`` (re-implemented, GPL-clean): cascading
      geologic-time strips stacked beside the age axis, with shrink-to-fit
      labels that fall back to the ICS abbreviation and are blanked when
      even that does not fit.
    * deeptime ``coord_geo()`` (MIT): black-or-white label colour via the
      BT.601 luminance rule (precomputed in the ``stratigraphy.time_bins``
      rows' ``font`` field), and a log scale admitted only on the axis the
      column runs along (age/depth), never along the strips.

The strips live on the same Axes as the column (in the x range left of 0)
instead of separate shared-y axes: shared axes all mutate one YAxis object,
so per-strip tick handling would fight across them.

matplotlib 3.10 has no registered path hatches, so only the classic hatch
charset is used (``/ \\ | - + x * o O . :`` and repetitions for density).

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.text import Text

from stratigraphy.time_bins import get_scale

logger = logging.getLogger(__name__)

# (fill colour, hatch pattern) per lithology; hatch "" means no overlay.
DEFAULT_LITHOLOGY_STYLES: dict[str, tuple[str, str]] = {
    "conglomerate": ("#E6B34D", "O"),
    "breccia": ("#D55E00", "O"),
    "sandstone": ("#F0E442", "/"),
    "arkose": ("#F0D242", "//"),
    "greywacke": ("#CCB44A", "//"),
    "siltstone": ("#B7B8B3", "-"),
    "claystone": ("#9BB0C4", ":"),
    "mudstone": ("#9BB0C4", "."),
    "shale": ("#708090", "|"),
    "marl": ("#C4B896", "\\"),
    "limestone": ("#7FA8B8", "xx"),
    "dolomite": ("#A8C8D8", "x"),
    "dolostone": ("#A8C8D8", "x"),
    "chalk": ("#E8F0E8", "..."),
    "chert": ("#C0504D", "\\\\"),
    "evaporite": ("#F5E6CC", "oo"),
    "halite": ("#F5E6CC", "o"),
    "gypsum": ("#EFE3D3", "OO"),
    "coal": ("#2F2F2F", ""),
    "peat": ("#6B4423", ""),
    "tuff": ("#DA70D6", "*"),
    "ash": ("#E6A9DF", "+"),
    "bentonite": ("#E6A9DF", "+"),
    "volcanic": ("#B22222", "+++"),
    "basalt": ("#8B4513", "+++"),
    "granite": ("#D98880", "+"),
    "slate": ("#7F8C8D", "\\"),
    "phyllite": ("#95A5A6", "\\\\"),
    "gneiss": ("#C39BD3", "\\\\\\\\"),
}

UNKNOWN_LITHOLOGY_STYLE: tuple[str, str] = ("#D3D3D3", "")

# palaeoverse's autofit_text floor, as a fraction of the base font size:
# below this the label is swapped for the abbreviation, then blanked.
_MIN_LABEL_FRACTION = 0.45

_GREY90 = "#E5E5E5"
_STRIP_UNIT = 1.0  # strip width in data-x units


def _resolve_style(lithology: object, table: Mapping[str, tuple[str, str]]) -> tuple[str, str]:
    """
    Look up a (fill, hatch) pair: exact key first, then the longest key
    occurring in the description ("fine-grained sandstone" -> "sandstone").
    """
    name = str(lithology).strip().lower()
    hit = table.get(name)
    if hit is not None:
        return hit
    best: tuple[str, str] | None = None
    best_len = 0
    for key, value in table.items():
        if len(key) > best_len and key in name:
            best, best_len = value, len(key)
    return best if best is not None else UNKNOWN_LITHOLOGY_STYLE


class StratigraphicColumnPlotter:
    """
    Publication-quality stratigraphic columns with ICS geologic-time strips.

    Produces:
        - single columns versus depth or versus numeric age
        - cascading eon/era/period/epoch strip cascade (age mode)
        - dual-channel fill + hatch lithology rendering
        - log age axis (deeptime-style restriction)
    """

    def __init__(self) -> None:
        """Initialize the stratigraphic column plotter."""
        self._logger = logging.getLogger(f"{__name__}.StratigraphicColumnPlotter")
        self._logger.info("StratigraphicColumnPlotter initialized")
        self._style = "seaborn-v0_8-paper"
        self._figure_size = (8, 10)
        self._dpi = 300
        self._font_size = 10
        self._title_font_size = 12
        self.lithology_styles: dict[str, tuple[str, str]] = dict(DEFAULT_LITHOLOGY_STYLES)

    def set_style(self, style: str) -> None:
        """Set matplotlib style."""
        try:
            plt.style.use(style)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not apply matplotlib style '{style}': {e}")

    # ------------------------------------------------------------------
    # Public plots
    # ------------------------------------------------------------------
    def plot_column(
        self,
        thicknesses: Sequence[float],
        lithologies: Sequence[str],
        *,
        tops: Sequence[float] | None = None,
        layer_bound_ages: Sequence[float] | None = None,
        geo_ranks: Sequence[str] = (),
        log_scale: bool = False,
        show_labels: bool = True,
        title: str = "Stratigraphic Column",
        ylabel: str | None = None,
    ) -> Figure:
        """
        Draw one stratigraphic column.

        Parameters:
            thicknesses: Bed thicknesses, top to bottom (length n).
            lithologies: Lithology description per bed (length n).
            tops: Depth of each bed top (length n, increasing).  Defaults to
                the cumulative thickness from 0.  Ignored in age mode.
            layer_bound_ages: Age (Ma) of every bed boundary, top to bottom
                (length n + 1, strictly increasing).  Gives an age axis and
                enables the geologic-time strips.
            geo_ranks: Cascade ranks, outermost first, e.g.
                ``("eon", "era", "period", "epoch")``; age mode only.
            log_scale: Logarithmic age axis (requires ages > 0).
            show_labels: Print the lithology text beside the column (depth
                mode only, where it does not collide with the strip cascade).
            title: Plot title.
            ylabel: Axis label; defaults to "Depth (m)" or "Age (Ma)".

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: On length mismatches, non-monotonic bounds, or
                ``log_scale`` reaching 0 Ma / applied to a depth axis.
        """
        n_beds = len(thicknesses)
        if len(lithologies) != n_beds:
            raise ValueError(f"lithologies has {len(lithologies)} entries, expected {n_beds}")

        if layer_bound_ages is not None:
            bounds = np.asarray(layer_bound_ages, dtype=np.float64)
            if bounds.size != n_beds + 1:
                raise ValueError(f"layer_bound_ages must have {n_beds + 1} entries, got {bounds.size}")
            if np.any(np.diff(bounds) <= 0):
                raise ValueError("layer_bound_ages must be strictly increasing top to bottom")
            mode = "age"
        else:
            bounds = self._depth_bounds(thicknesses, tops)
            mode = "depth"
            if log_scale:
                raise ValueError("log_scale is only supported with layer_bound_ages (age axis)")
        self._validate_log(bounds, log_scale)

        ranks = tuple(geo_ranks) if mode == "age" else ()
        self.set_style(self._style)
        self._logger.info(f"plot_column: {n_beds} beds, mode={mode}, ranks={list(ranks)}")

        cells = [_resolve_style(lit, self.lithology_styles) for lit in lithologies]
        fig = Figure(figsize=(self._figure_size[0] + 1.1 * len(ranks), self._figure_size[1]))
        ax = fig.add_subplot(111)

        axis_label = ylabel or ("Age (Ma)" if mode == "age" else "Depth (m)")
        annotate = mode == "depth" and show_labels
        x_left = -(len(ranks) + 0.05) * _STRIP_UNIT if ranks else -0.02
        ax.set_xlim(x_left, 3.2 if annotate else 1.02)
        self._orient_axis(ax, bounds, log_scale, axis_label)

        if ranks:
            self._draw_strips(fig, ax, ranks, float(bounds[0]), float(bounds[-1]), log_scale)
        self._draw_beds(ax, bounds, cells, 0.0, 1.0)
        ax.axvline(0.0, color="black", linewidth=0.8)
        if annotate:
            self._annotate_lithologies(ax, bounds, lithologies)
        ax.set_title(title, fontsize=self._title_font_size)

        fig.tight_layout()
        return fig

    def plot_correlated_columns(
        self,
        sections: Sequence[Mapping],
        *,
        geo_ranks: Sequence[str] = ("period", "epoch"),
        log_scale: bool = False,
        title: str = "Correlated Stratigraphic Columns",
        ylabel: str = "Age (Ma)",
    ) -> Figure:
        """
        Draw several columns against one shared age axis + ICS strip cascade.

        Parameters:
            sections: Mappings with keys ``name``, ``bound_ages`` (length
                n + 1, strictly increasing, Ma) and ``lithologies`` (length n).
            geo_ranks: Cascade ranks, outermost first.
            log_scale: Logarithmic age axis.
            title: Figure title.
            ylabel: Shared axis label.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: On empty input, shape mismatches, non-monotonic
                bounds, or ``log_scale`` reaching 0 Ma.
        """
        if not sections:
            raise ValueError("sections must contain at least one column")
        parsed = []
        for sec in sections:
            bounds = np.asarray(sec["bound_ages"], dtype=np.float64)
            lithologies = list(sec["lithologies"])
            if bounds.size != len(lithologies) + 1:
                raise ValueError(f"section '{sec.get('name', '?')}': bound_ages must be lithologies + 1")
            if np.any(np.diff(bounds) <= 0):
                raise ValueError(f"section '{sec.get('name', '?')}': bound_ages must be strictly increasing")
            parsed.append((str(sec.get("name", "")), bounds, lithologies))

        lo = min(float(b[0]) for _, b, _ in parsed)
        hi = max(float(b[-1]) for _, b, _ in parsed)
        self._validate_log(np.array([lo, hi]), log_scale)
        self.set_style(self._style)
        self._logger.info(f"plot_correlated_columns: {len(parsed)} sections, window [{lo}, {hi}] Ma")

        ranks = tuple(geo_ranks)
        span = 1.6  # data units per column slot
        fig = Figure(figsize=(self._figure_size[0] + 2.2 * len(parsed), self._figure_size[1]))
        ax = fig.add_subplot(111)
        ax.set_xlim(-(len(ranks) + 0.05) * _STRIP_UNIT if ranks else -0.02, (len(parsed) - 1) * span + 1.1)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylim(hi, lo)
        ax.set_ylabel(ylabel, fontsize=self._font_size)
        ax.set_xticks([])

        if ranks:
            self._draw_strips(fig, ax, ranks, lo, hi, log_scale)
        for j, (name, bounds, lithologies) in enumerate(parsed):
            x0 = j * span
            self._draw_beds(
                ax, bounds, [_resolve_style(l, self.lithology_styles) for l in lithologies], x0, x0 + 1.0
            )
            self._column_header(ax, x0 + 0.5, name)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=self._title_font_size, y=1.03)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Column drawing internals
    # ------------------------------------------------------------------
    @staticmethod
    def _depth_bounds(thicknesses: Sequence[float], tops: Sequence[float] | None) -> np.ndarray:
        t = np.asarray(thicknesses, dtype=np.float64)
        if np.any(t < 0):
            raise ValueError("thicknesses must be non-negative")
        if tops is None:
            return np.concatenate([[0.0], np.cumsum(t)])
        tops_arr = np.asarray(tops, dtype=np.float64)
        if tops_arr.size != t.size:
            raise ValueError(f"tops must have {t.size} entries, got {tops_arr.size}")
        if np.any(np.diff(tops_arr) < 0) or tops_arr[0] < 0:
            raise ValueError("tops must be non-decreasing, non-negative depths")
        return np.append(tops_arr, tops_arr[-1] + t[-1])

    @staticmethod
    def _validate_log(bounds: np.ndarray, log_scale: bool) -> None:
        if log_scale and float(np.min(bounds)) <= 0:
            raise ValueError("log_scale requires all ages to be strictly positive (no 0 Ma)")

    def _draw_beds(
        self,
        ax: mpl.axes.Axes,
        bounds: np.ndarray,
        cells: list[tuple[str, str]],
        x0: float,
        x1: float,
    ) -> None:
        """
        Dual-channel pass (mplStrater): one pcolormesh carries the fill
        colours, and a PathPatch per bed overlays the classic hatch, so a
        bed is identified by (colour, pattern) rather than colour alone.
        """
        if not cells:
            return
        codes: dict[tuple[str, str], int] = {}
        for cell in cells:
            codes.setdefault(cell, len(codes))
        palette = [None] * len(codes)
        for cell, idx in codes.items():
            palette[idx] = cell
        cell_idx = [codes[cell] for cell in cells]
        grid = np.array([[i + 0.5] for i in cell_idx], dtype=np.float64)

        old_hatch_lw = mpl.rcParams["hatch.linewidth"]
        mpl.rcParams["hatch.linewidth"] = 0.4
        try:
            cmap = ListedColormap([fill for fill, _ in palette])
            mesh = ax.pcolormesh(
                [x0, x1],
                bounds,
                grid,
                cmap=cmap,
                vmin=0.5,
                vmax=len(palette) + 0.5,
                shading="flat",
                edgecolors="black",
                linewidths=0.01,
                zorder=2,
            )
            for path, idx in zip(mesh.get_paths(), cell_idx, strict=True):
                _, hatch = palette[idx]
                if hatch:
                    ax.add_patch(
                        PathPatch(
                            path, facecolor="none", edgecolor="black", hatch=hatch, linewidth=0.0, zorder=3
                        )
                    )
        finally:
            mpl.rcParams["hatch.linewidth"] = old_hatch_lw

    def _orient_axis(
        self,
        ax: mpl.axes.Axes,
        bounds: np.ndarray,
        log_scale: bool,
        axis_label: str,
    ) -> None:
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylim(float(bounds[-1]), float(bounds[0]))  # youngest/shallowest at the top
        ax.set_ylabel(axis_label, fontsize=self._font_size)
        ax.set_xticks([])

    def _annotate_lithologies(self, ax: mpl.axes.Axes, bounds: np.ndarray, lithologies: Sequence[str]) -> None:
        for i, lit in enumerate(lithologies):
            y = (float(bounds[i]) + float(bounds[i + 1])) / 2.0
            ax.text(1.08, y, str(lit), fontsize=self._font_size - 2, va="center", ha="left", zorder=4)

    # ------------------------------------------------------------------
    # Cascading ICS geologic-time strips (same axes, x < 0)
    # ------------------------------------------------------------------
    def _draw_strips(
        self,
        fig: Figure,
        ax: mpl.axes.Axes,
        ranks: Sequence[str],
        age_lo: float,
        age_hi: float,
        log_scale: bool,
    ) -> None:
        """One stacked strip per rank, outermost left; grey90 under-gaps."""
        renderer = self._figure_renderer(fig)
        base_fs = float(self._font_size)
        for i, rank in enumerate(ranks):
            # rank 0 (outermost) sits furthest left
            x0 = -(len(ranks) - i) * _STRIP_UNIT
            ax.add_patch(
                Rectangle((x0, age_lo), _STRIP_UNIT, age_hi - age_lo, facecolor=_GREY90, edgecolor="none", zorder=0)
            )
            rows = get_scale(rank=rank, interval=(age_lo, age_hi))
            for row in rows:
                y0 = max(float(row["min_ma"]), age_lo)
                y1 = min(float(row["max_ma"]), age_hi)
                if y1 <= y0:
                    continue
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        _STRIP_UNIT,
                        y1 - y0,
                        facecolor=row["colour"] or _GREY90,
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=1,
                    )
                )
                centre = math.sqrt(y0 * y1) if log_scale else (y0 + y1) / 2.0
                candidates: list[str] = [str(row["interval_name"])]
                if row.get("abbr"):
                    candidates.append(str(row["abbr"]))
                self._fit_strip_label(
                    ax, renderer, x0 + _STRIP_UNIT / 2.0, centre, candidates, str(row["font"]), base_fs, y0, y1
                )
        self._strip_ticks(ax, ranks[0], age_lo, age_hi)

    @staticmethod
    def _strip_ticks(ax: mpl.axes.Axes, outer_rank: str, age_lo: float, age_hi: float) -> None:
        """Boundary numerals along the cascade, from the outermost rank."""
        boundaries = sorted(
            {
                v
                for row in get_scale(rank=outer_rank, interval=(age_lo, age_hi))
                for v in (max(float(row["min_ma"]), age_lo), min(float(row["max_ma"]), age_hi))
            }
        )
        if boundaries:
            ax.set_yticks(boundaries)
            ax.set_yticklabels([f"{v:g}" for v in boundaries], fontsize=7)

    def _fit_strip_label(
        self,
        ax: mpl.axes.Axes,
        renderer,
        x_centre: float,
        y_centre: float,
        candidates: list[str],
        colour: str,
        base_fs: float,
        y0: float,
        y1: float,
    ) -> Text:
        """Shrink-to-fit, then abbreviation, then blank (palaeoverse autofit_text)."""
        ax_box = ax.get_window_extent(renderer)
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        cell_h_px = abs(
            float(ax.transData.transform((0.0, y1))[1]) - float(ax.transData.transform((0.0, y0))[1])
        )
        cell_w_px = ax_box.width * (_STRIP_UNIT / x_span) if x_span else ax_box.width
        txt = ax.text(
            x_centre, y_centre, "", ha="center", va="center", color=colour, fontsize=base_fs, zorder=4
        )
        for label in candidates:
            txt.set_text(label)
            fs = base_fs
            fits = False
            while fs >= base_fs * _MIN_LABEL_FRACTION:
                txt.set_fontsize(fs)
                extent = txt.get_window_extent(renderer)
                if extent.height * 1.2 <= cell_h_px and extent.width * 1.1 <= cell_w_px:
                    fits = True
                    break
                fs -= 0.5
            if fits:
                return txt
        txt.set_text("")
        return txt

    @staticmethod
    def _figure_renderer(fig: Figure):
        """Renderer for label fitting; bare Figures need an Agg canvas attached."""
        try:
            return fig.canvas.get_renderer()
        except AttributeError:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            return FigureCanvasAgg(fig).get_renderer()

    @staticmethod
    def _column_header(ax: mpl.axes.Axes, x_centre: float, name: str) -> None:
        """Section name centred above its column."""
        if not name:
            return
        from matplotlib.transforms import blended_transform_factory

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_centre,
            1.015,
            name,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
            zorder=4,
        )
