"""Scoped matplotlib styling for the standalone plotters.

Every plotter in this package used to call ``plt.style.use(...)``, which
REPLACES the whole process-global ``rcParams`` dict. That is a real leak in a
long-lived GUI: ``views/ui_plot_canvas.py`` installs the application's look
once at import time (``seaborn-v0_8-whitegrid`` plus the CJK font stack,
font sizes, dpi, spine and grid settings), and running a single
``StratigraphyPlotter`` figure silently replaced all of it for the rest of the
session. Every figure drawn afterwards then came out in the wrong font and at
the wrong size, and the dark-theme pass layered its palette on top of
non-dark defaults.

``scoped_plot_methods`` keeps the visual intent and removes the side effect:
each ``plot_*`` method runs inside ``plt.rc_context``, so the style applies
while the figure and its artists are constructed and is restored when the
method returns.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def resolve_style_params(style: str) -> dict[str, Any]:
    """Return the rcParams for ``style``, with a pre-3.8 name fallback.

    The plotters historically retried ``style.replace("v0_8-", "")`` when
    ``plt.style.use`` raised, because matplotlib renamed its bundled styles in
    3.8. Keep that behaviour.

    The previous implementation silently returned an empty dict when the
    style was unknown, which made ``plt.rc_context({})`` a no-op and let
    the caller's rcParams leak across the scope. We now snapshot the
    active rcParams before resolution so the scoped context always
    restores the caller-visible state, and we still raise ``KeyError``
    on unknown names — the caller (``_wrap``) turns that into a logged
    warning + a neutral context, rather than dropping the scope
    entirely.
    """
    candidates: list[str] = []
    if style:
        candidates.append(style)
        stripped = style.replace("v0_8-", "")
        if stripped != style:
            candidates.append(stripped)

    for candidate in candidates:
        try:
            if candidate in plt.style.library:
                return dict(plt.style.library[candidate])
            return dict(plt.style.get_style(candidate))
        except (OSError, ValueError, KeyError):
            continue

    # Unknown style: snapshot the live rcParams so the rc_context still
    # restores the caller's state (otherwise an unrelated earlier mutation
    # would leak across the scope).
    logger.debug("Could not resolve matplotlib style %r; using current rcParams", style)
    return dict(plt.rcParams)


def scoped_plot_methods(cls: type) -> type:
    """Class decorator: run every ``plot_*`` method in a style context."""
    for name, attr in list(vars(cls).items()):
        if not name.startswith("plot_") or not callable(attr):
            continue
        setattr(cls, name, _wrap(attr))
    return cls


def _wrap(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        style = getattr(self, "_style", "default")
        with plt.rc_context(resolve_style_params(style)):
            return fn(self, *args, **kwargs)

    return wrapper
