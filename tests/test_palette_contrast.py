# =============================================================================
# FILE: tests/test_palette_contrast.py
# =============================================================================
"""
Fail the build when a palette colour pair stops being readable.

WHAT WENT WRONG, AND WHY A TEST IS THE ONLY THING THAT CATCHES IT
-------------------------------------------------------------------
``ColorPaletteDark`` subclasses ``ColorPalette`` and overrides the surfaces
and the text colours. It did not override the *accents*, so the dark theme
inherited ``primary = #1E40AF`` -- a blue picked so that white text would sit
on it -- and then used that same value as a *foreground* on dark surfaces.
The ribbon painted its selected tab in dark blue on dark blue (1.44:1) and
its group titles dark blue on a dark ribbon (2.05:1), against a 4.5:1
requirement. Three related semantic colours failed the same way.

Nothing about that is visible in the source, and the pre-existing QSS test
does not look at colours. The only way to catch it is to compute the ratios.

This pins the pairs the app actually composes. It is deliberately a fixed
list rather than a combinatorial sweep: an exhaustive matrix fails on pairings
the UI never builds, and a test that cries wolf gets ignored. Every entry here
corresponds to a real rule in ``get_modern_stylesheet()`` or a ribbon
stylesheet.
"""

from __future__ import annotations

import re

import pytest

from config.design_system import ColorPalette, ColorPaletteDark

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

# WCAG 2.1 minimum for text of this size.
AA_BODY = 4.5
# Minimum for a non-text boundary: an input border has to be findable.
AA_NON_TEXT = 3.0

# (label, foreground token, background token, minimum)
LIGHT_PAIRS = [
    ("body on primary surface", "text_primary", "bg_primary", AA_BODY),
    ("body on secondary surface", "text_primary", "bg_secondary", AA_BODY),
    ("body on tertiary surface", "text_primary", "bg_tertiary", AA_BODY),
    ("secondary on primary", "text_secondary", "bg_primary", AA_BODY),
    ("secondary on secondary", "text_secondary", "bg_secondary", AA_BODY),
    ("secondary on tertiary", "text_secondary", "bg_tertiary", AA_BODY),
    ("accent on primary", "primary", "bg_primary", AA_BODY),
    ("accent on secondary", "primary", "bg_secondary", AA_BODY),
    ("accent on tertiary", "primary", "bg_tertiary", AA_BODY),
    ("success on primary", "success", "bg_primary", AA_BODY),
    ("warning on primary", "warning", "bg_primary", AA_BODY),
    ("error on primary", "error", "bg_primary", AA_BODY),
    ("info on primary", "info", "bg_primary", AA_BODY),
    ("on_primary on accent", "on_primary", "primary", AA_BODY),
    ("on_primary on success", "on_primary", "success", AA_BODY),
    ("on_primary on error", "on_primary", "error", AA_BODY),
    ("focus ring on primary", "border_focus", "bg_primary", AA_NON_TEXT),
    ("medium border on primary", "border_medium", "bg_primary", 1.3),
]

DARK_PAIRS = [
    ("body on primary surface", "text_primary", "bg_primary", AA_BODY),
    ("body on secondary surface", "text_primary", "bg_secondary", AA_BODY),
    ("body on tertiary surface", "text_primary", "bg_tertiary", AA_BODY),
    ("secondary on primary", "text_secondary", "bg_primary", AA_BODY),
    ("secondary on secondary", "text_secondary", "bg_secondary", AA_BODY),
    ("secondary on tertiary", "text_secondary", "bg_tertiary", AA_BODY),
    ("accent on primary", "primary", "bg_primary", AA_BODY),
    ("accent on secondary", "primary", "bg_secondary", AA_BODY),
    ("accent on tertiary", "primary", "bg_tertiary", AA_BODY),
    ("success on primary", "success", "bg_primary", AA_BODY),
    ("warning on primary", "warning", "bg_primary", AA_BODY),
    ("error on primary", "error", "bg_primary", AA_BODY),
    ("info on primary", "info", "bg_primary", AA_BODY),
    ("on_primary on accent", "on_primary", "primary", AA_BODY),
    ("on_primary on success", "on_primary", "success", AA_BODY),
    ("on_primary on error", "on_primary", "error", AA_BODY),
    ("focus ring on primary", "border_focus", "bg_primary", AA_NON_TEXT),
    ("medium border on primary", "border_medium", "bg_primary", 1.3),
]

# Every accent must be restated for the dark theme. A token missing here is
# inherited, and an accent inherited from the light palette is a colour chosen
# for white backgrounds being used on a near-black one.
ACCENTS = (
    "primary",
    "primary_light",
    "primary_dark",
    "success",
    "warning",
    "error",
    "info",
    "on_primary",
)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    assert HEX.match(hex_color), f"not a #rrggbb colour: {hex_color!r}"
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)


def contrast(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _failures(pal, pairs):
    out = []
    for label, fg_key, bg_key, minimum in pairs:
        fg, bg = getattr(pal, fg_key), getattr(pal, bg_key)
        got = contrast(fg, bg)
        if got < minimum:
            out.append(f"{label}: {fg} on {bg} = {got:.2f}, needs {minimum}")
    return out


def test_light_palette_is_readable():
    bad = _failures(ColorPalette, LIGHT_PAIRS)
    assert bad == [], "unreadable light-theme pairs:\n  " + "\n  ".join(bad)


def test_dark_palette_is_readable():
    bad = _failures(ColorPaletteDark, DARK_PAIRS)
    assert bad == [], "unreadable dark-theme pairs:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("token", ACCENTS)
def test_dark_theme_restates_every_accent(token):
    """The specific regression: an accent silently inherited from light."""
    light = getattr(ColorPalette, token)
    dark = getattr(ColorPaletteDark, token)
    assert dark != light, (
        f"ColorPaletteDark.{token} is {dark}, identical to the light value. "
        "An accent picked for white backgrounds is unreadable on a dark "
        "surface; restate it in ColorPaletteDark."
    )


@pytest.mark.parametrize("token", ACCENTS)
def test_accent_is_an_opaque_colour(token):
    """A gradient or a translucent overlay cannot be contrast-checked.

    ``rgba(...)`` values are used for hover/active overlays, which is correct
    -- but if one of those reaches an accent token, every ratio computed
    against it silently compares the wrong colour.
    """
    value = getattr(ColorPaletteDark, token)
    assert HEX.match(value), (
        f"ColorPaletteDark.{token} = {value!r} is not an opaque #rrggbb "
        "colour, so contrast against it cannot be verified"
    )


def test_contrast_helper_agrees_with_a_known_value():
    """Anchor the maths, so a broken helper cannot make the rest pass.

    Black on white is the maximum ratio the formula can produce (21:1) and
    white on white is the minimum (1:1). If either drifts, every assertion
    above is measuring noise.
    """
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    assert contrast("#FFFFFF", "#FFFFFF") == pytest.approx(1.0, abs=0.001)
    # A mid grey pair with a published ratio.
    assert contrast("#767676", "#FFFFFF") == pytest.approx(4.54, abs=0.02)
