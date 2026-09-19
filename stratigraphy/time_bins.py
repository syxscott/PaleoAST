# =============================================================================
# FILE: stratigraphy/time_bins.py
# =============================================================================
"""
Geologic time bins, occurrence binning, and stratigraphic ranges.

A Python port of the core of the palaeoverse R package (MIT License),
adapted for PaleoAST:

* :func:`time_bins` — build a bin table from the built-in ICS-2020
  Phanerozoic scale (eon/era/period/epoch ranks) or a user-supplied
  interval table, optionally collapsing intervals into (near-)equal
  duration bins with the dynamic-programming partition of
  ``palaeoverse::time_bins(size=, n_bins=)``.
* :func:`bin_time` — assign fossil occurrences to time bins using the
  five palaeoverse methods ``"mid"``, ``"majority"``, ``"all"``,
  ``"random"`` and ``"point"``.
* :func:`tax_range_time` — first/last appearance datetimes (FAD/LAD)
  from occurrence age ranges.
* :func:`tax_expand_time` — range-through expansion of taxon ranges
  over the bins they overlap, with origination/extinction flags.

Data conventions: ages are in Ma (Myr before 2020 AD), non-negative.
Occurrences/ranges/bins are plain ``list[dict]`` rows so they interoperate
with the spreadsheet model without a pandas dependency.  Bin tables are
ordered OLDEST FIRST and numbered ``bin = 1..N`` from old to young, like
palaeoverse.

Deliberate deviations from palaeoverse (upstream warts):
* ``bin_time(method="mid")`` warns about midpoints that coincide with bin
  BOUNDARIES (upstream compares against bin midpoints, a known bug).
* ``bin_time(method="random")`` assigns NA instead of erroring when an
  occurrence overlaps no bin.
* ``n_bins`` derived from ``size`` is clamped to >= 1 (upstream prints
  "0 time bins" while returning one).

References:
    Gearty, W., et al. (2024). palaeoverse: an R package for
    palaeontological data analysis. Ecography.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

MIN_MA = "min_ma"
MAX_MA = "max_ma"

# =============================================================================
# Built-in scale table (ICS 2020 v0.1 boundaries, Phanerozoic + top of the
# Neoproterozoic).  Stage-level binning is supported by passing a user scale.
# =============================================================================

# (interval_name, rank, max_ma (older bound), min_ma (younger bound),
#  colour (ICS 2020 chart fill hex), abbr (ICS chart abbreviation or None))
# Colours/abbreviations match palaeoverse::GTS2020 (itself from the ICS chart);
# boundaries stay on this table's ICS v0.1 ages, so only fills are merged.
_GTS2020_ROWS: list[tuple[str, str, float, float, str, str | None]] = [
    # Eons
    ("Phanerozoic", "eon", 538.8, 0.0, "#9AD9DD", None),
    # Eras
    ("Cenozoic", "era", 66.0, 0.0, "#F2F91D", None),
    ("Mesozoic", "era", 251.902, 66.0, "#67C5CA", None),
    ("Paleozoic", "era", 538.8, 251.902, "#99C08D", None),
    # Periods
    ("Quaternary", "period", 2.58, 0.0, "#F9F97F", "Q"),
    ("Neogene", "period", 23.03, 2.58, "#FFE619", "Ng"),
    ("Paleogene", "period", 66.0, 23.03, "#FD9A52", "Pg"),
    ("Cretaceous", "period", 145.0, 66.0, "#7FC64E", "K"),
    ("Jurassic", "period", 201.4, 145.0, "#34B2C9", "J"),
    ("Triassic", "period", 251.902, 201.4, "#812B92", "Tr"),
    ("Permian", "period", 298.9, 251.902, "#F04028", "P"),
    ("Carboniferous", "period", 358.9, 298.9, "#67A599", "C"),
    ("Devonian", "period", 419.2, 358.9, "#CB8C37", "D"),
    ("Silurian", "period", 443.8, 419.2, "#B3E1B6", "S"),
    ("Ordovician", "period", 485.4, 443.8, "#009270", "O"),
    ("Cambrian", "period", 538.8, 485.4, "#7FA056", "Cm"),
    # Cenozoic epochs
    ("Holocene", "epoch", 0.0117, 0.0, "#FEEBD2", None),
    ("Pleistocene", "epoch", 2.58, 0.0117, "#FFEFAF", None),
    ("Pliocene", "epoch", 5.333, 2.58, "#FFFF99", None),
    ("Miocene", "epoch", 23.03, 5.333, "#FFFF00", None),
    ("Eocene", "epoch", 56.0, 33.9, "#FDB46C", None),
    ("Oligocene", "epoch", 33.9, 23.03, "#FEC07A", None),
    ("Paleocene", "epoch", 66.0, 56.0, "#FDA75F", None),
]
# NOTE: Paleogene subdivisions are series, not epochs, in the ICS chart; the
# epoch rank intentionally lists the six classic Cenozoic epochs plus the
# (series-level) split, and users wanting exact ICS epochs at finer ranks
# should pass a user scale.

RANKS = ("eon", "era", "period", "epoch")


def _font_for_colour(colour: str | None) -> str:
    """ITU-R BT.601 luminance rule from palaeoverse: white text on dark fills."""
    if not colour:
        return "black"
    c = colour.lstrip("#")
    if len(c) != 6:
        return "black"
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "black" if lum > 0.5 else "white"


def _tidy_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort oldest-first, compute mid/duration, renumber bins, add font."""
    out = sorted(rows, key=lambda r: -float(r[MAX_MA]))
    for i, r in enumerate(out):
        r["max_ma"] = float(r[MAX_MA])
        r["min_ma"] = float(r[MIN_MA])
        if r["max_ma"] < r["min_ma"]:
            raise DataValidationError(
                _("Interval '{0}' has max_ma smaller than min_ma").format(r.get("interval_name", i))
            )
        r["mid_ma"] = (r["max_ma"] + r["min_ma"]) / 2.0
        r["duration_myr"] = r["max_ma"] - r["min_ma"]
        r["bin"] = i + 1
        r.setdefault("rank", "user")
        r.setdefault("abbr", None)
        r.setdefault("colour", None)
        r["font"] = _font_for_colour(r["colour"])
    return out


def get_scale(
    rank: str = "period",
    scale: list[dict[str, Any]] | None = None,
    interval: str | tuple[str, str] | tuple[float, float] | None = "Phanerozoic",
) -> list[dict[str, Any]]:
    """
    Return a bin table (list of dicts, oldest first) for a rank or user scale.

    Parameters:
        rank: one of ``RANKS`` for the built-in scale, ignored for user scales.
        scale: user interval table with keys interval_name, max_ma, min_ma
            (optional abbr, colour).  When given, rank becomes "user".
        interval: restrict the built-in table to a named interval, to a
            two-name window (unique-prefix names, order-free), or to a
            numeric (age_a, age_b) window.  Intervals merely touching a
            numeric window are kept (inclusive, as in palaeoverse).
    """
    if scale is not None:
        for r in scale:
            missing = {"interval_name", MAX_MA, MIN_MA} - set(r)
            if missing:
                raise DataValidationError(
                    _("User scale row is missing columns: {0}").format(", ".join(sorted(missing)))
                )
        rows = [dict(r) for r in scale]
        for r in rows:
            r["rank"] = "user"
        return _tidy_scale(rows)

    if rank not in RANKS:
        raise DataValidationError(_("rank must be one of {0}").format(", ".join(RANKS)))
    table = [
        {"interval_name": name, "rank": rk, MAX_MA: mx, MIN_MA: mn, "colour": colour, "abbr": abbr}
        for (name, rk, mx, mn, colour, abbr) in _GTS2020_ROWS
    ]

    if interval is not None:
        if isinstance(interval, str):
            names = [interval]
        else:
            names = list(interval)
        if len(names) > 2:
            raise DataValidationError(_("interval must be NULL or of length 1 or 2"))
        try:
            bounds = [float(n) for n in names]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # name-based selection: unique-prefix matching, window = union of matched spans
            bounds = []
            for n in names:
                matches = [t for t in table if t["interval_name"] == n] or [
                    t for t in table if t["interval_name"].startswith(n)
                ]
                if not matches:
                    raise DataValidationError(_("Unknown interval: '{0}'").format(n))
                bounds.extend([min(t[MIN_MA] for t in matches), max(t[MAX_MA] for t in matches)])
        lo, hi = min(bounds), max(bounds)
        if hi > 4600.0 or lo < 0.0:
            raise DataValidationError(_("interval ages must lie within [0, 4600] Ma"))
        table = [t for t in table if float(t[MAX_MA]) >= lo and float(t[MIN_MA]) <= hi]

    rows = [t for t in table if t["rank"] == rank]
    if not rows:
        raise DataValidationError(_("No intervals are available for the defined interval range"))
    return _tidy_scale([{k: v for k, v in t.items() if k != "rank"} | {"rank": rank} for t in rows])


# =============================================================================
# Equal-length binning via dynamic programming (palaeoverse time_bins DP)
# =============================================================================


def _equal_length_bins(df: list[dict[str, Any]], n_bins: int) -> list[list[int]]:
    """
    Split the oldest-first interval list ``df`` into ``n_bins`` contiguous
    groups minimising Σ (group duration − target)² (palaeoverse §3.4).

    Recurrence (indices over intervals 1..m, groups k):
        cost[i][k] = min over j<i of  cost[j][k-1] + (C[i]-C[j] - T)²
    Ties resolve to the smallest j (strict ``<``), i.e. the younger block
    swallows as many older intervals as possible.
    """
    m = len(df)
    target = sum(float(r["duration_myr"]) for r in df) / n_bins
    cum = [0.0]
    for r in df:
        cum.append(cum[-1] + float(r["duration_myr"]))

    inf = float("inf")
    cost = [[inf] * (n_bins + 1) for _ in range(m + 1)]
    part = [[0] * (n_bins + 1) for _ in range(m + 1)]
    cost[0][0] = 0.0
    for i in range(1, m + 1):
        cost[i][0] = (cum[i] - cum[0] - target) ** 2
        part[i][0] = 0
    for k in range(1, n_bins):
        for i in range(k + 1, m + 1):
            for j in range(k, i):
                cand = cost[j][k - 1] + (cum[i] - cum[j] - target) ** 2
                if cand < cost[i][k]:  # strict < keeps the smallest j on ties
                    cost[i][k] = cand
                    part[i][k] = j
    # backtrack youngest block first
    groups: list[list[int]] = [[] for _ in range(n_bins)]
    i, k = m, n_bins - 1
    while k >= 0:
        j = part[i][k]  # part[i][0] == 0 by construction
        groups[k] = list(range(j + 1, i + 1))  # 1-based interval positions
        i = j
        k -= 1
    return groups


def time_bins(
    rank: str = "period",
    size: float | None = None,
    n_bins: int | None = None,
    scale: list[dict[str, Any]] | None = None,
    interval: str | tuple[str, str] | tuple[float, float] | None = "Phanerozoic",
) -> list[dict[str, Any]]:
    """
    Generate a time-bin table.

    Without ``size``/``n_bins`` this returns the scale intervals themselves
    (one bin per interval, numbered oldest first).  With ``size`` (target bin
    duration in Myr) or ``n_bins`` the intervals are merged into contiguous,
    near-equal-duration bins with the palaeoverse dynamic program; ``size``
    takes precedence when both are given.

    Returns:
        list of dicts with keys: bin, max_ma, mid_ma, min_ma, duration_myr,
        grouping_rank, intervals (comma-joined member names).
    """
    df = get_scale(rank=rank, scale=scale, interval=interval)

    if size is None and n_bins is None:
        return [
            {
                "bin": r["bin"],
                "max_ma": r["max_ma"],
                "mid_ma": r["mid_ma"],
                "min_ma": r["min_ma"],
                "duration_myr": r["duration_myr"],
                "grouping_rank": r["rank"],
                "intervals": r["interval_name"],
            }
            for r in df
        ]

    if size is not None:
        if size <= 0:
            raise DataValidationError(_("size must be greater than 0"))
        total = max(r["max_ma"] for r in df) - min(r["min_ma"] for r in df)
        k = round(total / size)  # banker's rounding, matching R
        k = max(1, min(k, len(df)))
    else:
        k = int(n_bins)  # type: ignore[call-overload]
        if k < 1:
            raise DataValidationError(_("n_bins must be >= 1"))
        if k > len(df):
            raise DataValidationError(
                _("n_bins ({0}) must not be greater than the number of intervals ({1})").format(k, len(df))
            )

    groups = _equal_length_bins(df, k) if k > 1 else [list(range(1, len(df) + 1))]
    out = []
    for gi, idx in enumerate(groups):
        members = [df[i - 1] for i in idx]
        mx = max(m["max_ma"] for m in members)
        mn = min(m["min_ma"] for m in members)
        out.append(
            {
                "bin": gi + 1,
                "max_ma": mx,
                "mid_ma": (mx + mn) / 2.0,
                "min_ma": mn,
                "duration_myr": float(sum(m["duration_myr"] for m in members)),
                "grouping_rank": members[0]["rank"],
                "intervals": ", ".join(m["interval_name"] for m in members),
            }
        )
    logger.info(
        "time_bins: %d bins, mean %.2f Myr, sd %.2f Myr",
        len(out),
        np.mean([b["duration_myr"] for b in out]),
        np.std([b["duration_myr"] for b in out], ddof=1) if len(out) > 1 else 0.0,
    )
    return out


# =============================================================================
# Occurrence binning
# =============================================================================


def _check_bins(bins: list[dict[str, Any]]) -> None:
    if not bins:
        raise DataValidationError(_("bins must be a non-empty list of time bins"))
    for b in bins:
        if "bin" not in b or MIN_MA not in b or MAX_MA not in b:
            raise DataValidationError(_("bins rows need 'bin', 'min_ma' and 'max_ma' columns"))
        if float(b[MAX_MA]) < float(b[MIN_MA]):
            raise DataValidationError(_("Bin {0}: max_ma must be >= min_ma").format(b["bin"]))


def _check_occurrences(occurrences: list[dict[str, Any]]) -> None:
    if not occurrences:
        raise DataValidationError(_("occurrences must be a non-empty list"))
    for i, occ in enumerate(occurrences):
        if MIN_MA not in occ or MAX_MA not in occ:
            raise DataValidationError(_("occurrences row {0} is missing min_ma/max_ma").format(i))
        try:
            lo, hi = float(occ[MIN_MA]), float(occ[MAX_MA])
        except (TypeError, ValueError) as exc:
            raise DataValidationError(_("occurrences row {0} has non-numeric ages").format(i)) from exc
        if hi < lo:
            raise DataValidationError(
                _("occurrences row {0}: max_ma must be >= min_ma").format(i)
            )


def _overlap_list(occurrences: list[dict[str, Any]], bins: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Strict-inequality overlap: touching only at boundaries does not count
    (palaeoverse bin_list, in bins row order)."""
    out = []
    for occ in occurrences:
        lo, hi = float(occ[MIN_MA]), float(occ[MAX_MA])
        out.append([b for b in bins if hi > float(b[MIN_MA]) and lo < float(b[MAX_MA])])
    return out


def bin_time(
    occurrences: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    method: str = "mid",
    reps: int = 100,
    seed: int | None = None,
    fun: Callable[[npt.NDArray], npt.NDArray] | None = None,
    resolution: float = 0.001,
) -> Any:
    """
    Assign occurrence age ranges to time bins (palaeoverse ``bin_time``).

    Parameters:
        occurrences: rows with numeric min_ma/max_ma (extra keys are kept).
        bins: rows with bin/min_ma/max_ma (row order is significant —
            palaeoverse convention: oldest first).
        method: one of:
            ``"mid"`` — the single bin containing the range midpoint (strict
            bounds; midpoints on a boundary yield NA).
            ``"majority"`` — bin with most overlap, measured by counting
            10,000 points sampled across the range (inclusive); ties go to
            the bin appearing earliest in ``bins``.
            ``"all"`` — one output row per overlapped bin (``n_bins == 0``
            rows are dropped).
            ``"random"`` — a list of ``reps`` datasets, each drawing one
            overlapped bin uniformly at random per occurrence.
            ``"point"`` — a list of ``reps`` datasets, drawing point age
            estimates from the density ``fun`` (x rescaled to [0, 1] across
            the range; default uniform) then binning with CLOSED bounds
            (later bin rows overwrite).
        reps: replicates for ``random``/``point``.
        seed: RNG seed for ``random``/``point``.
        fun: density callable for ``point`` (vectorised on [0, 1] input).
        resolution: grid step (Myr) for the ``point`` method.

    Returns:
        a list of row dicts for mid/majority/all, or a list of such lists
        (length ``reps``) for random/point.  Output rows gain the columns
        ``id``, ``n_bins``, ``bin_assignment``, ``bin_midpoint`` (plus
        ``overlap_percentage`` for majority and ``point_estimates`` for point).
    """
    _check_bins(bins)
    _check_occurrences(occurrences)
    if method not in ("mid", "majority", "all", "random", "point"):
        raise DataValidationError(
            _("method must be one of mid, majority, all, random, point; got '{0}'").format(method)
        )
    if reps < 1:
        raise DataValidationError(_("reps must be >= 1"))

    bin_mid = {b["bin"]: (float(b[MAX_MA]) + float(b[MIN_MA])) / 2.0 for b in bins}
    overlaps = _overlap_list(occurrences, bins)
    n_occ = len(occurrences)

    if method == "mid":
        rows = []
        warned = False
        for i, occ in enumerate(occurrences):
            mid = (float(occ[MIN_MA]) + float(occ[MAX_MA])) / 2.0
            row = dict(occ)
            row["id"] = i + 1
            row["n_bins"] = len(overlaps[i])
            assign = None
            for b in bins:  # later rows overwrite earlier ones
                if float(b[MIN_MA]) < mid < float(b[MAX_MA]):
                    assign = b
            if assign is None and row["n_bins"] > 0:
                warned = True
            row["bin_assignment"] = assign["bin"] if assign else None
            row["bin_midpoint"] = bin_mid[assign["bin"]] if assign else None
            rows.append(row)
        if warned:
            logger.warning(
                "bin_time('mid'): some occurrence midpoints fall on bin boundaries; "
                "those rows have bin_assignment = NA"
            )
        return rows

    if method == "majority":
        rows = []
        for i, occ in enumerate(occurrences):
            lo, hi = float(occ[MIN_MA]), float(occ[MAX_MA])
            row = dict(occ)
            row["id"] = i + 1
            row["n_bins"] = len(overlaps[i])
            best_bin, best_pct = None, -1.0
            if overlaps[i]:
                grid = np.linspace(lo, hi, 10000)
                for b in overlaps[i]:
                    cnt = int(np.sum((grid >= float(b[MIN_MA])) & (grid <= float(b[MAX_MA]))))
                    pct = 100.0 * cnt / 10000
                    if pct > best_pct:  # first max wins ties (bins row order)
                        best_pct, best_bin = pct, b
            row["bin_assignment"] = best_bin["bin"] if best_bin else None
            row["bin_midpoint"] = bin_mid[best_bin["bin"]] if best_bin else None
            row["overlap_percentage"] = best_pct if best_bin else None
            rows.append(row)
        return rows

    if method == "all":
        rows = []
        for i, occ in enumerate(occurrences):
            for b in overlaps[i]:
                row = dict(occ)
                row["id"] = i + 1
                row["n_bins"] = len(overlaps[i])
                row["bin_assignment"] = b["bin"]
                row["bin_midpoint"] = bin_mid[b["bin"]]
                rows.append(row)
        return rows

    if method == "random":
        rng = np.random.default_rng(seed)
        picks = []
        for i in range(n_occ):
            opts = [b["bin"] for b in overlaps[i]]
            if len(opts) == 1:
                picks.append([opts[0]] * reps)
            elif not opts:
                picks.append([None] * reps)  # deviation: NA instead of error
            else:
                idx = rng.integers(0, len(opts), size=reps)
                picks.append([opts[j] for j in idx])
        datasets = []
        for rep in range(reps):
            rows = []
            for i, occ in enumerate(occurrences):
                row = dict(occ)
                row["id"] = i + 1
                row["n_bins"] = len(overlaps[i])
                a = picks[i][rep]
                row["bin_assignment"] = a
                row["bin_midpoint"] = bin_mid.get(a)
                rows.append(row)
            datasets.append(rows)
        return datasets

    # method == "point"
    rng = np.random.default_rng(seed)
    if fun is None:
        def fun(x: npt.NDArray) -> npt.NDArray:  # uniform density
            return np.ones_like(np.asarray(x, dtype=float))
    picks = []
    for occ in occurrences:
        lo, hi = float(occ[MIN_MA]), float(occ[MAX_MA])
        if hi - lo <= 0:
            picks.append(np.full(reps, lo))
            continue
        grid = np.arange(lo, hi + resolution / 2, resolution)
        prob = np.asarray(fun(np.linspace(0.0, 1.0, len(grid))), dtype=float)
        if prob.shape[0] != grid.shape[0] or not np.all(np.isfinite(prob)) or prob.sum() <= 0:
            raise DataValidationError(_("point density must be finite with positive total weight"))
        p = prob / prob.sum()
        picks.append(rng.choice(grid, size=reps, replace=True, p=p))
    datasets = []
    for rep in range(reps):
        rows = []
        for i, occ in enumerate(occurrences):
            est = float(picks[i][rep])
            row = dict(occ)
            row["id"] = i + 1
            row["n_bins"] = len(overlaps[i])
            row["point_estimates"] = est
            assign = None
            for b in bins:  # CLOSED bounds, later rows overwrite
                if float(b[MIN_MA]) <= est <= float(b[MAX_MA]):
                    assign = b["bin"]
            row["bin_assignment"] = assign
            row["bin_midpoint"] = bin_mid.get(assign) if assign is not None else None
            rows.append(row)
        datasets.append(rows)
    return datasets


# =============================================================================
# Ranges: FAD/LAD and range-through expansion
# =============================================================================


def tax_range_time(
    occurrences: list[dict[str, Any]],
    name: str = "taxon",
    by: str = "FAD",
) -> list[dict[str, Any]]:
    """
    Stratigraphic ranges from occurrences (palaeoverse ``tax_range_time``).

    FAD = oldest occurrence max_ma; LAD = youngest occurrence min_ma.
    Taxa are sorted lexicographically first; ``by`` selects the final row
    order (ascending age → youngest-first) among ties by name:
    ``"FAD"``, ``"LAD"`` or ``"name"``.

    Returns:
        rows with keys taxon, taxon_id, max_ma (FAD), min_ma (LAD),
        range_myr, n_occ — ages rounded to 3 decimals, taxon_id = 1..T in
        the returned row order.
    """
    _check_occurrences(occurrences)
    if by not in ("FAD", "LAD", "name"):
        raise DataValidationError(_("by must be 'FAD', 'LAD' or 'name'"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for occ in occurrences:
        if name not in occ or occ[name] is None:
            raise DataValidationError(_("occurrences row is missing the taxon column '{0}'").format(name))
        groups.setdefault(str(occ[name]), []).append(occ)
    taxa = sorted(groups)
    records = []
    for t in taxa:
        rows = groups[t]
        fad = max(float(r[MAX_MA]) for r in rows)
        lad = min(float(r[MIN_MA]) for r in rows)
        records.append(
            {
                "taxon": t,
                "max_ma": round(fad, 3),
                "min_ma": round(lad, 3),
                "range_myr": round(fad - lad, 3),
                "n_occ": len(rows),
            }
        )
    if by == "FAD":
        records.sort(key=lambda r: r["max_ma"])  # stable → ties keep name order
    elif by == "LAD":
        records.sort(key=lambda r: r["min_ma"])
    for i, r in enumerate(records):
        r["taxon_id"] = i + 1
        r_ordered = {k: r[k] for k in ("taxon", "taxon_id", "max_ma", "min_ma", "range_myr", "n_occ")}
        records[i] = r_ordered
    return records


def tax_expand_time(
    ranges: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    ext_orig: bool = True,
) -> list[dict[str, Any]]:
    """
    Range-through expansion of taxon ranges over bins (palaeoverse
    ``tax_expand_time``).

    A taxon occupies every bin whose interval strictly overlaps its
    [LAD, FAD] span (same predicate as the bin_list step of bin_time).
    Rows are grouped by taxon in input order; within a taxon, oldest bin
    first (bins row order).

    Parameters:
        ranges: rows with taxon name (key ``taxon`` or ``name`` key given
            via the row's "taxon" field), max_ma (FAD) and min_ma (LAD).
        bins: bin table rows (bin/min_ma/max_ma), oldest first.
        ext_orig: add boolean ``orig``/``ext`` columns — origination when
            the bin covers the FAD, extinction when it covers the LAD
            (LAD = 0 never flags extinction: extant taxa).

    Returns:
        rows with the taxon keys plus bin, interval/bounds fields (and
        ext/orig when requested).
    """
    _check_bins(bins)
    for r in ranges:
        if MIN_MA not in r or MAX_MA not in r or "taxon" not in r:
            raise DataValidationError(_("ranges rows need 'taxon', 'min_ma' and 'max_ma' keys"))
        if float(r[MIN_MA]) < 0 or float(r[MAX_MA]) < 0:
            raise DataValidationError(_("All range ages must be non-negative"))
    if len(ranges) != len({tuple(sorted((k, str(v)) for k, v in r.items())) for r in ranges}):
        raise DataValidationError(_("ranges must not have duplicated rows"))
    out = []
    for r in ranges:
        lad, fad = float(r[MIN_MA]), float(r[MAX_MA])
        if fad < lad:
            raise DataValidationError(_("Taxon {0}: max_ma must be >= min_ma").format(r["taxon"]))
        for b in bins:
            if lad < float(b[MAX_MA]) and fad > float(b[MIN_MA]):
                row = dict(r)
                row["bin"] = b["bin"]
                row["bin_max_ma"] = float(b[MAX_MA])
                row["bin_min_ma"] = float(b[MIN_MA])
                if "intervals" in b:
                    row["bin_intervals"] = b["intervals"]
                if ext_orig:
                    row["orig"] = fad <= float(b[MAX_MA])
                    row["ext"] = (lad >= float(b[MIN_MA])) and lad > 0
                out.append(row)
    return out
