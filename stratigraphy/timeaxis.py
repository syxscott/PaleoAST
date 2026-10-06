# =============================================================================
# FILE: stratigraphy/timeaxis.py
# =============================================================================
"""
A canonical time axis for time-series work, with unit converters.

WHY THIS EXISTS
~~~~~~~~~~~~~~~
Three different objects in the codebase already carry a time
axis, and they each made their own choice about units and ordering:

1. ``isotope_analysis.IsotopeData(depth, age)`` -- ages in Ma, paired with
   depth, not necessarily sorted.
2. ``correlation.StratigraphicSection.ages`` -- ages in Ma, optional, attached
   to a stratigraphic section whose primary axis is *height*, not age.
3. ``time_bins`` rows -- ``max_ma``/``min_ma``/``mid_ma`` intervals, oldest
   first, not a point axis at all.

A fourth feature (orbital forcing, and any cycle-detection routine that has
to state a period in kyr while the data are in Ma) needs a fourth answer
unless there is a single place that owns the conversion. That place is this
module. It is a thin, *additive* value type: nothing existing is refactored
onto it, because those three representations have their own tests and a wide
blast radius, and because a bin table is genuinely not a point axis. Build a
``TimeAxis`` at the boundary of a new feature and use it there.

CONVENTIONS
~~~~~~~~~~~
* Canonical storage is **ages in Ma, sorted ascending** (oldest
  last, i.e. "time running forwards"), which matches ``time_bins``'
  numeric order and makes ``np.diff`` positive. Convert to the
  "before present" reading with :meth:`TimeAxis.ages_bp_years`.
* Frequencies are **cycles per Ma**; periods are **kyr**. The 1000x
  difference is the single most common unit slip in this domain, so
  :meth:`TimeAxis.frequency_per_ma` and :meth:`TimeAxis.period_kyr` are the
  only two conversions the rest of the code is expected to use.
* Ma is "Myr before 2020 AD" elsewhere in this package
  (``stratigraphy/time_bins.py``), so :meth:`TimeAxis.calendar_years`
  defaults to a present year of 2020 to stay consistent with it rather than
  with the conventional 1950 BP datum.

References:
    1. Gradstein, F.M., Ogg, J.G. & Wiegert, V. 2012. The geologic time
       scale. Journal of Stratigraphy 35: 1110-1130. (Age units and the
       Ma/kyr convention used throughout ``time_bins.py``.)
    2. Lisiecki, L.E. & Raymo, M.E. 2005. LR04 benthic foraminiferal
       oxygen-isotope stack. Paleoceanography 20: PA2001. (The reference
       isotope record whose time scale makes cycle counting worth doing.)
    3. Cohen, A. & Beer, M. 1995. The astronomical time scale. In
       Global Earth History. (Why periods in kyr belong next to ages in Ma
       in the same object.)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import DataValidationError

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, typing only
    from stratigraphy.correlation import StratigraphicSection
    from stratigraphy.isotope_analysis import IsotopeData

logger = logging.getLogger(__name__)

#: Multipliers taking the canonical Ma value to each supported unit.
UNIT_TO_MA: dict[str, float] = {
    "ma": 1.0,
    "myr": 1.0,
    "kyr": 0.001,
    "kya": 0.001,
    "yr": 1.0e-6,
    "a": 1.0e-6,
}


def _as_ages_ma(ages: npt.NDArray | list[float], units: str, name: str) -> npt.NDArray:
    """Validate and convert an age array to Ma, flattened and finite."""
    key = str(units).strip().lower()
    if key not in UNIT_TO_MA:
        raise DataValidationError(_("units must be one of {0}; got '{1}'").format(", ".join(sorted(UNIT_TO_MA)), units))
    arr = np.asarray(ages, dtype=float).flatten()
    if arr.size == 0:
        raise DataValidationError(_("Time axis '{0}' is empty").format(name))
    if not np.all(np.isfinite(arr)):
        raise DataValidationError(
            _("Time axis '{0}' contains non-finite ages").format(name),
            details={"n_non_finite": int(np.sum(~np.isfinite(arr)))},
        )
    return arr * UNIT_TO_MA[key]


class TimeAxis:
    """A validated, sorted age axis in Ma, plus period/frequency converters.

    Build one of these rather than passing a raw age array around whenever a
    routine has to mix a period stated in kyr with ages stated in Ma. It is
    immutable-by-convention: :meth:`ages_ma` is the canonical array and the
    converters are pure functions of it.

    Attributes:
        ages_ma: Ages in Ma, sorted ascending (oldest last).
        source: A short description of where the axis came from, carried
            into :meth:`summary` so a result is traceable to its input.
    """

    __slots__ = ("_ages_ma", "source")

    def __init__(self, ages: npt.NDArray | list[float], units: str = "Ma", source: str = "ages") -> None:
        """Validate, convert and sort an age array.

        Parameters
        ----------
        ages:
            Ages in ``units``. Accepts any 1-D array-like.
        units:
            One of ``"Ma"``, ``"kyr"``, ``"yr"`` (case-insensitive).
        source:
            Label recorded in :meth:`summary`.
        """
        arr = _as_ages_ma(ages, units, source)
        self._ages_ma = np.sort(arr)
        self.source = source

    # -- construction from the three existing representations ---------------

    @classmethod
    def from_ages(cls, ages: npt.NDArray | list[float], units: str = "Ma", source: str = "ages") -> TimeAxis:
        """Build from a bare age array (any of the units above)."""
        return cls(ages, units=units, source=source)

    @classmethod
    def from_isotope_data(cls, data: IsotopeData) -> TimeAxis:
        """Build from ``isotope_analysis.IsotopeData(depth, age)``.

        Uses the ``age`` field, which that class defines in Ma. ``depth`` is
        deliberately ignored: this is a *time* axis, and a depth axis is a
        different quantity that only becomes time through an age model.
        """
        return cls(data.age, units="Ma", source="IsotopeData.age")

    @classmethod
    def from_section(cls, section: StratigraphicSection) -> TimeAxis:
        """Build from ``correlation.StratigraphicSection.ages``.

        Raises if the section carries no ages -- an unsectioned section has
        a height axis, and inventing a time axis for it here would hide the
        absence rather than report it.
        """
        ages = getattr(section, "ages", None)
        if ages is None:
            raise DataValidationError(
                _("Stratigraphic section '{0}' has no age model, so it has no time axis").format(
                    getattr(section, "name", "?")
                )
            )
        return cls(ages, units="Ma", source=f"StratigraphicSection[{getattr(section, 'name', '?')}]")

    @classmethod
    def from_bin_rows(cls, rows: list[dict[str, Any]], key: str = "mid_ma") -> TimeAxis:
        """Build from ``time_bins`` rows (``max_ma``/``min_ma``/``mid_ma``).

        A bin table is an interval table, not a point axis, so a bin has to
        be collapsed to a single age before it can serve as one. ``key``
        selects the collapse: ``"mid_ma"`` uses the stored midpoint,
        ``"max_ma"``/``"min_ma"`` use an interval bound, and ``"auto"``
        prefers ``mid_ma``, falls back to ``(max_ma + min_ma) / 2``, and
        finally to ``max_ma``.
        """
        if not rows:
            raise DataValidationError(_("Cannot build a time axis from an empty bin table"))

        def _pick(row: dict[str, Any]) -> float:
            if key == "auto":
                if "mid_ma" in row:
                    return float(row["mid_ma"])
                if "max_ma" in row and "min_ma" in row:
                    return (float(row["max_ma"]) + float(row["min_ma"])) / 2.0
                if "max_ma" in row:
                    return float(row["max_ma"])
                raise DataValidationError(_("Bin row needs mid_ma, or max_ma and min_ma"))
            if key not in row:
                raise DataValidationError(
                    _("Bin row is missing '{0}'; available: {1}").format(key, ", ".join(sorted(row)))
                )
            return float(row[key])

        ages = np.array([_pick(r) for r in rows], dtype=float)
        return cls(ages, units="Ma", source="time_bins rows")

    # -- accessors ----------------------------------------------------------

    @property
    def ages_ma(self) -> npt.NDArray:
        """The canonical axis, in Ma, ascending."""
        return self._ages_ma

    @property
    def n(self) -> int:
        """Number of ages on the axis."""
        return int(self._ages_ma.size)

    @property
    def span_ma(self) -> float:
        """Total time covered, in Ma."""
        return float(self._ages_ma[-1] - self._ages_ma[0])

    @property
    def dt_ma(self) -> float:
        """Mean sample spacing in Ma.

        The *mean* and not the median, because the point of this property
        is the sampling interval of a record; use the median of
        ``np.diff`` yourself if you need robustness to gaps.
        """
        if self.n < 2:
            return 0.0
        return float(self.span_ma / (self.n - 1))

    @property
    def is_uniform(self) -> bool:
        """Whether the spacing is constant to within 1 % of the mean.

        Uneven sampling is the normal case in cyclostratigraphy (a core is
        sampled more densely in one interval than another), so this is a
        diagnostic to report, not an error to raise.
        """
        if self.n < 3:
            return False
        d = np.diff(self._ages_ma)
        mean = float(np.mean(d))
        return bool(mean > 0 and np.all(np.abs(d - mean) <= 0.01 * mean))

    # -- unit conversion ----------------------------------------------------

    def to_kyr(self) -> npt.NDArray:
        """Ages in kyr before present (Ma x 1000)."""
        return self._ages_ma * 1000.0

    def to_bp_years(self) -> npt.NDArray:
        """Ages in years before present (Ma x 1e6)."""
        return self._ages_ma * 1.0e6

    def calendar_years(self, present_year: int = 2020) -> npt.NDArray:
        """Ages as calendar years AD (negative into deep time).

        Defaults to 2020 to match ``time_bins.py``'s "Myr before 2020 AD"
        convention. Pass ``present_year=1950`` for the conventional BP
        datum; the two differ by 70 years, which is nothing at Ma resolution
        and everything at ka resolution.
        """
        return float(present_year) - self._ages_ma * 1.0e6

    def frequency_per_ma(self, period_kyr: float | npt.NDArray) -> npt.NDArray | float:
        """Convert a period in kyr to a frequency in cycles per Ma.

        f = 1 / P,  with P in kyr and f per Ma  =>  f = 1000 / P_kyr
        """
        p = np.asarray(period_kyr, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = 1000.0 / p
        return float(freq) if freq.ndim == 0 else freq

    def period_kyr(self, frequency_per_ma: float | npt.NDArray) -> npt.NDArray | float:
        """Convert a frequency in cycles per Ma to a period in kyr."""
        f = np.asarray(frequency_per_ma, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            period = 1000.0 / f
        return float(period) if period.ndim == 0 else period

    def frequency_axis(self, n: int = 512, f_min_per_ma: float | None = None) -> npt.NDArray:
        """A linear frequency grid spanning the record, in cycles per Ma.

        The low end is the fundamental ``1 / span_ma`` and the high end is
        the Nyquist limit ``1 / (2 dt)``, the same pair
        ``spectral_analysis.SpectralAnalyzer`` picks by hand when the caller
        supplies no ``frequency_range``.
        """
        if self.n < 3 or self.span_ma <= 0:
            raise DataValidationError(
                _("A frequency axis needs at least 3 ages spanning a positive duration"),
                details={"n": self.n, "span_ma": self.span_ma},
            )
        if n < 2:
            raise DataValidationError(_("n must be >= 2"))
        lo = 1.0 / self.span_ma if f_min_per_ma is None else float(f_min_per_ma)
        hi = 1.0 / (2.0 * self.dt_ma)
        if hi <= lo:
            raise DataValidationError(
                _("Cannot build a frequency axis: Nyquist {0} is not above the fundamental {1}").format(hi, lo)
            )
        return np.linspace(lo, hi, int(n))

    def period_axis(self, n: int = 512) -> npt.NDArray:
        """A log-spaced period grid in kyr, shortest first.

        Log spacing is the right default for cyclostratigraphy: a 405 kyr
        cycle and a 19 kyr cycle differ by more than an order of
        magnitude, and a linear frequency grid resolves the long end far
        better than the short end, which is the opposite of what peak
        picking over a Milankovitch band needs. The grid therefore spans the
        same frequency range as :meth:`frequency_axis`, sampled
        geometrically rather than linearly.
        """
        if self.n < 3 or self.span_ma <= 0:
            raise DataValidationError(
                _("A period axis needs at least 3 ages spanning a positive duration"),
                details={"n": self.n, "span_ma": self.span_ma},
            )
        if n < 2:
            raise DataValidationError(_("n must be >= 2"))
        lo = 1.0 / self.span_ma
        hi = 1.0 / (2.0 * self.dt_ma)
        if hi <= lo:
            raise DataValidationError(
                _("Cannot build a period axis: Nyquist {0} is not above the fundamental {1}").format(hi, lo)
            )
        freqs = np.geomspace(lo, hi, int(n))
        return np.sort(self.period_kyr(freqs))

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Time Axis"),
            "=" * 50,
            _("Source: {0}").format(self.source),
            _("Points: {0}").format(self.n),
            _("Span: {0} Ma ({1} kyr)").format(f"{self.span_ma:.6g}", f"{self.span_ma * 1000.0:.6g}"),
            _("Mean step: {0} Ma").format(f"{self.dt_ma:.6g}"),
            _("Uniform: {0}").format(_("yes") if self.is_uniform else _("no")),
            _("Age range: {0} - {1} Ma").format(f"{self._ages_ma[0]:.6g}", f"{self._ages_ma[-1]:.6g}"),
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        """Number of ages."""
        return self.n

    def __repr__(self) -> str:
        """Unambiguous repr, including the source label."""
        return f"TimeAxis(n={self.n}, span_ma={self.span_ma:.6g}, source={self.source!r})"
