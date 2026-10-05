# =============================================================================
# FILE: visualization/r_export.py
# =============================================================================
"""
Generate editable R scripts that reproduce a PaleoAST figure with ggplot2.

WHY THIS EXISTS
---------------
The audience for PaleoAST figures is paleontologists preparing manuscripts.
A PNG or PDF is a *terminal artefact* -- a reviewer asks for a different
colour scheme or a log time axis and the figure has to be rebuilt. An R script
is a *source artefact*: the data and the grammar are both there, so anyone in
the lab can edit it and re-run it.

So this module produces ``.R`` scripts, not rendered images. Rendering is a
separate concern (see :mod:`visualization.r_render`) that shells out to
``Rscript`` as a SUBPROCESS.

WHY NOT rpy2
-------------
rpy2 links R into the Python process, which brings ABI pinning, a ~150 MB
per-platform runtime to package into the PyInstaller build, and a documented
history of failures in this repository (``pyproject.toml`` pins
``rpy2>=3.5.16,<3.6; platform_system != 'Windows'`` after a nonexistent
5.x pin kept nine CI jobs red). Calling ``Rscript`` as a child process has
none of those constraints, and R being an *optional, user-provided* tool is
exactly the premise of this feature.

DESIGN RULE THAT MATTERS MOST
-----------------------------
This module reads the **result object** (``PCAResult`` and friends), never a
matplotlib ``Figure``. If it derived itself from the existing plotters, the R
output and the on-screen figure would be two implementations of the same
thing -- and two implementations drift. That is not hypothetical: the
``min_variance`` defect fixed in the same session was exactly that shape,
where the dialog collected a value the engine never received.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "RExportResult",
    "RPlotSpec",
    "RScriptExporter",
    "validate_r_script",
]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class RPlotSpec:
    """Everything the user can adjust without touching the generated script.

    Field names deliberately mirror the keys already persisted by
    ``PreferencesDialog`` (views/ui_permutation_dialogs.py) so the two do not
    drift: ``figsize`` and ``dpi`` there become ``ggsave(width=, height=)``
    and ``ggsave(dpi=)`` here.
    """

    theme: str = "classic"           # classic | bw | minimal | grey
    base_size: float = 9.0           # ggplot2 base_size, in points
    figsize: tuple[float, float] = (7.0, 5.5)   # inches
    dpi: int = 300
    point_size: float = 2.0
    alpha: float = 0.85
    show_ellipse: bool = True        # 95% confidence ellipse
    annotate_samples: bool = False
    reverse_time_axis: bool = False  # stratigraphic convention: oldest at the bottom
    point_shape: int = 16            # 16 = filled circle (no stroke, theme-safe)
    color_palette: str = "default"   # name resolved inside the R script
    r_executable: str = ""           # "" => auto-detect
    output_format: str = "pdf"       # pdf | svg | png

    def theme_line(self) -> str:
        """The single line a user edits when retargeting a journal template.

        Deliberately one line: a submission that needs a new journal style
        should not require understanding the rest of the script. The object is
        named THEME because the plot body refers to it by that name, so
        changing this one line changes every panel.
        """
        return f"THEME <- theme_{self.theme}(base_size = {self.base_size:g})"

    def ggsave_line(self, stem: str) -> str:
        stem = re.sub(r"[^\w.-]+", "_", stem)
        fmt = self.output_format
        return (
            f'ggsave("{stem}.{fmt}", plot = p, width = {self.figsize[0]:g}, '
            f"height = {self.figsize[1]:g}, dpi = {self.dpi}, units = \"in\")"
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RExportResult:
    """Where the artefacts landed, and what the script looked like when written.

    ``script_sha256`` is what lets the UI tell "the user edited this" apart
    from "this is still what we generated" -- the regenerate/run split depends
    on it.
    """

    script_path: Path
    data_path: Path | None
    script_sha256: str
    figure_stems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def exists(self) -> bool:
        return self.script_path.is_file()

    def script_is_unmodified(self) -> bool:
        """False once the user has edited the script by hand."""
        if not self.exists():
            return False
        current = hashlib.sha256(self.script_path.read_bytes()).hexdigest()
        return current == self.script_sha256


# ---------------------------------------------------------------------------
# R serialisation
# ---------------------------------------------------------------------------

# Anything the R parser cannot read but Python's repr happily prints.
_PYTHON_LEAKS = (
    "np.float64", "np.int64", "np.float32", "np.bool_", "dtype=",
    "array([", "nan", "inf",
)


def _fmt_float(value: Any) -> str:
    """Format one float for an R numeric literal.

    ``nan``/``inf`` are the traps: Python prints ``nan``, which R reads as an
    undefined symbol, and ``inf`` which it reads as a function call. Both must
    become R's ``NA_real_`` / ``NA_real_`` with an attribute, or the script
    dies at parse time with an error that points nowhere near the cause.
    """
    v = float(value)
    if math.isnan(v):
        return "NA_real_"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    return repr(v)


def _fmt_value(value: Any) -> str:
    if isinstance(value, (bool,)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int,)):
        return str(int(value))
    if isinstance(value, (float,)):
        return _fmt_float(value)
    return _fmt_string(str(value))


def _fmt_string(value: str) -> str:
    """Quote a string for R.

    Backslash, double quote, newline and tab all need escaping -- taxon names
    are exactly the place where these show up ("O'Brien sp.", a backslash in
    a file path, a newline in a malformed label).
    """
    out = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{out}"'


def r_vector(values: Sequence[Any]) -> str:
    """A single R vector literal, e.g. ``c(1, 2.5, NA_real_)``."""
    if len(values) == 0:
        return "numeric(0)"
    return "c(" + ", ".join(_fmt_value(v) for v in values) + ")"


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write data for the R script to read back.

    A companion CSV rather than an embedded literal is deliberate: it keeps one
    serialization path regardless of matrix size (two paths are how the R
    figure and the on-screen figure end up disagreeing), and it hands the user
    a file they can swap for their own data.

    Values go out RAW. Formatting them with the R-literal writer here would put
    quotes inside every CSV field, so a group read back as ``"Group A"`` --
    literally, quotes included -- and silently became a different level name
    than the one the figure was built with.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (bool,)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv):
                return "NA"
            if math.isinf(fv):
                return "Inf" if fv > 0 else "-Inf"
            return repr(fv)
        return str(v)

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([cell(v) for v in row])


# ---------------------------------------------------------------------------
# Script scaffolding
# ---------------------------------------------------------------------------

_HEADER = """\
## ------------------------------------------------------------------
## {title}
##
## Generated by PaleoAST on {stamp}
## Data:   {data_name}
## Render: Rscript {stem}.R      (or open in RStudio and press Run)
##
## Everything below is meant to be edited. The three places you are most
## likely to change, in the order you will hit them:
##   1. THEME   -- one line, for matching a journal's house style
##   2. LABEL   -- one labs() call, for wording and axis titles
##   3. DATA    -- the file name in the read.csv() call above, if you
##                 want to re-plot a different run
## ------------------------------------------------------------------

suppressPackageStartupMessages({{
  library(ggplot2)
}})

## Generated with R {r_version_note}
## To re-use a different colour set, swap the name in scale_colour_manual
## below ("default", "brewer" or "viridis").

data <- read.csv("{data_name}", stringsAsFactors = FALSE,
                 fileEncoding = "UTF-8")
"""


def _footer(spec: RPlotSpec, stem: str) -> str:
    return f"""
## --- output -------------------------------------------------------
{spec.ggsave_line(stem)}
cat("wrote {stem}.{spec.output_format}\\n")
"""


def _assemble(title: str, stamp: str, data_name: str, stem: str,
              body: str, spec: RPlotSpec, r_version_note: str) -> str:
    return (
        _HEADER.format(title=title, stamp=stamp, data_name=data_name,
                       stem=stem, r_version_note=r_version_note)
        + body
        + _footer(spec, stem)
    )


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_r_script(text: str) -> list[str]:
    """Cheap structural checks on a generated R script.

    This cannot replace running R -- it catches the class of defect where the
    *generator* emits something Python-shaped that R will not parse, which is
    the failure mode a hand-written exporter actually has.
    """
    problems: list[str] = []

    if not text.strip():
        return ["script is empty"]

    # Balanced delimiters, ignoring those inside strings and comments.
    stack: list[tuple[str, int]] = []
    in_str = False
    in_comment = False
    i = 0
    line = 1
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            line += 1
            in_comment = False
            i += 1
            continue
        if in_comment:
            i += 1
            continue
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == "#":
            in_comment = True
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack:
                problems.append(f"unmatched closing '{ch}' at line {line}")
                break
            opener, opened = stack.pop()
            if "([{".index(opener) != ")]}".index(ch):
                problems.append(
                    f"mismatched '{opener}' (line {opened}) closed by '{ch}' at line {line}"
                )
                break
        i += 1
    if stack and not problems:
        opener, opened = stack[-1]
        problems.append(f"unclosed '{opener}' opened at line {opened}")
    if in_str:
        problems.append("unterminated string literal")

    # Python leakage: the giveaway is a numpy repr or a bare lowercase nan.
    for leak in _PYTHON_LEAKS:
        if leak in ("nan", "inf"):
            # These are legitimate inside words/comments; only flag standalone
            # numeric positions, which is where a numpy scalar would land.
            for m in re.finditer(rf"(?<![\w.]){leak}(?![\w])", text):
                ctx = text[max(0, m.start() - 20):m.end() + 20]
                if not ctx.lstrip().startswith("#"):
                    problems.append(
                        f"Python-style '{leak}' at offset {m.start()}: {ctx!r}"
                    )
                    break
        elif leak in text:
            m = re.search(re.escape(leak), text)
            problems.append(f"Python artefact '{leak}' at offset {m.start()}")

    # A bare `None` is Python truthiness leaking into a data frame.
    for m in re.finditer(r"(?<![\w.])None(?![\w])", text):
        problems.append(f"Python 'None' at offset {m.start()}")

    return problems


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class RScriptExporter:
    """Writes editable ``.R`` scripts for a set of figure families."""

    def __init__(self, stamp: str = "", r_version_note: str = "4.5.2") -> None:
        self._stamp = stamp
        self._r_version_note = r_version_note

    # -- internal -----------------------------------------------------
    def _write(self, out_dir: Path, stem: str, data_name: str, title: str,
               body: str, spec: RPlotSpec) -> RExportResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        text = _assemble(title, self._stamp, data_name, stem, body, spec,
                         self._r_version_note)
        problems = validate_r_script(text)
        if problems:
            raise ValueError(
                "generated R script failed its own structural check: "
                + "; ".join(problems)
            )
        script_path = out_dir / f"{stem}.R"
        script_path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
        return RExportResult(
            script_path=script_path,
            data_path=out_dir / data_name,
            script_sha256=digest,
            figure_stems=[f"{stem}.{spec.output_format}"],
        )

    # -- PCA scores ---------------------------------------------------
    def export_pca_scores(
        self,
        result: Any,
        out_dir: Path,
        spec: RPlotSpec | None = None,
        pc1: int = 0,
        pc2: int = 1,
        labels: Sequence[str] | None = None,
        groups: Sequence[Any] | None = None,
        stem: str = "pca_scores",
    ) -> RExportResult:
        """Emit a ggplot2 score plot (PC1 vs PC2) with an optional 95% ellipse."""
        spec = spec or RPlotSpec()
        scores = np.asarray(result.get_scores(n_components=max(pc1, pc2) + 1))
        x = scores[:, pc1]
        y = scores[:, pc2]
        n = len(x)

        group_col = None
        if groups is not None and len(groups) == n:
            vals = [str(g) for g in groups]
            levels = sorted(set(vals))
            group_col = ("group", vals, levels)
        label_col = None
        if labels is not None and len(labels) == n:
            label_col = ("label", [str(v) for v in labels], None)

        header = ["sample", f"PC{pc1 + 1}", f"PC{pc2 + 1}"]
        if group_col:
            header.append("group")
        if label_col:
            header.append("label")
        rows = []
        for i in range(n):
            row: list[Any] = [i + 1, x[i], y[i]]
            if group_col:
                row.append(group_col[1][i])
            if label_col:
                row.append(label_col[1][i])
            rows.append(row)

        data_name = f"{stem}_data.csv"
        write_csv(out_dir / data_name, header, rows)

        ev = np.asarray(result.explained_variance, dtype=float)
        var1 = ev[pc1] if pc1 < len(ev) else float("nan")
        var2 = ev[pc2] if pc2 < len(ev) else float("nan")

        # The colour mapping belongs INSIDE ggplot()'s own aes(). Emitting it as
        # a separate expression makes ggplot2 try to add a ggproto, and the
        # script dies with "Cannot add <ggproto> objects together" -- while
        # still parsing cleanly, so no structural check can catch it.
        aes_parts = [f"x = .data$PC{pc1 + 1}", f"y = .data$PC{pc2 + 1}"]
        if group_col:
            aes_parts.append(
                "colour = factor(group, levels = c("
                + ", ".join(_fmt_string(v) for v in group_col[2])
                + "))"
            )
        aes_expr = "aes(" + ", ".join(aes_parts) + ")"

        body = f"""
## --- settings you are most likely to want to change -----------------

{spec.theme_line()}
## e.g. theme_bw() / theme_minimal() / theme_grey(base_size = 9)
{self._palette_block(spec)}
LABELS <- list(
  title    = "PCA Score Plot",
  subtitle = "Centred",
  x = "PC{pc1 + 1} ({var1:.1f}% variance)",
  y = "PC{pc2 + 1} ({var2:.1f}% variance)"
)

p <- ggplot(data, {aes_expr}) +
"""

        # Layers are collected and joined, never hand-spliced with operators.
        # Two ways that goes wrong, both of which still exit 0:
        #   * a leading `+` on its own line is a UNARY PLUS in R, so the
        #     preceding layer becomes a standalone statement;
        #   * omitting a `+` splits the expression, and the trailing layers are
        #     then evaluated (and auto-printed) as stray top-level expressions
        #     while ggsave saves an incomplete plot.
        # Joining with a trailing " +" on every layer removes both. Every
        # layer must live in this list -- none may come from the template
        # above, or it misses its operator.
        layers: list[str] = [
            "geom_hline(yintercept = 0, linewidth = 0.3, colour = \"grey70\")",
            "geom_vline(xintercept = 0, linewidth = 0.3, colour = \"grey70\")",
            f"geom_point(size = {spec.point_size:g}, shape = {spec.point_shape},"
            f"\n             alpha = {spec.alpha:g})",
        ]

        if group_col:
            layers.append(
                "scale_colour_manual(\n"
                "    values = PALETTE[seq_along(levels(factor(data$group)))],\n"
                "    name = NULL\n"
                "  )"
            )

        if spec.show_ellipse and n >= 3:
            if group_col:
                layers.append(
                    "stat_ellipse(aes(group = group, colour = factor(group)),\n"
                    '                 type = "norm", level = 0.95, linewidth = 0.5,\n'
                    "                 show.legend = FALSE)"
                )
            else:
                layers.append(
                    'stat_ellipse(type = "norm", level = 0.95, linewidth = 0.5,\n'
                    '                 colour = "grey40")'
                )

        if spec.annotate_samples and label_col:
            layers.append(
                "geom_text(aes(label = label), size = 3, vjust = -0.8,\n"
                "            show.legend = FALSE)"
            )

        layers.append(
            "labs(\n"
            "    title    = LABELS$title,\n"
            "    subtitle = LABELS$subtitle,\n"
            "    x        = LABELS$x,\n"
            "    y        = LABELS$y\n"
            "  )"
        )
        if group_col:
            layers.append('theme(legend.position = "right")')
        layers.append("THEME")

        # Joined with a trailing " +" on each line. Joining WITHOUT the operator
        # silently produces a script that still exits 0 -- the first two
        # layers become one assignment, the rest are evaluated as stray
        # top-level expressions and auto-printed, and ggsave then saves an
        # incomplete plot. A green exit code is not evidence here; only
        # looking at the figure is.
        body += "\n".join("  " + lay + " +" for lay in layers[:-1])
        body += "\n  " + layers[-1] + "\n"

        return self._write(out_dir, stem, data_name,
                           "PCA score plot (ggplot2)", body, spec)

    # -- PCA scree ----------------------------------------------------
    def export_pca_scree(
        self,
        result: Any,
        out_dir: Path,
        spec: RPlotSpec | None = None,
        stem: str = "pca_scree",
    ) -> RExportResult:
        """Emit a scree plot: variance bars plus a cumulative line on a 2nd axis."""
        spec = spec or RPlotSpec()
        ev = np.asarray(result.explained_variance, dtype=float)
        ev_raw = np.asarray(result.eigenvalues_raw, dtype=float)
        cum = np.asarray(result.cumulative_variance, dtype=float)

        n = min(len(ev), len(cum), len(ev_raw) if ev_raw.size else len(ev))
        ev, cum, ev_raw = ev[:n], cum[:n], ev_raw[:n]

        data_name = f"{stem}_data.csv"
        write_csv(
            out_dir / data_name,
            ["component", "eigenvalue", "variance_pct", "cumulative_pct"],
            [[i + 1, ev_raw[i], ev[i], cum[i]] for i in range(n)],
        )

        # Kaiser criterion: components above the mean eigenvalue.
        kaiser = 0
        if n:
            thr = float(np.mean(ev_raw))
            kaiser = int(np.sum(ev_raw > thr))

        body = f"""
## --- settings you are most likely to want to change -----------------

{spec.theme_line()}

## Height used to map the cumulative percentage onto the bar axis. Computed
## once here: referring to data$ inside the scale's lambda triggers a
## discouraged-data warning and obscures where the scaling comes from.
YMAX <- max(data$variance_pct)

LABELS <- list(
  title    = "PCA Scree Plot",
  subtitle = "Kaiser criterion: {kaiser} of {n} component(s) above the mean eigenvalue",
  x = "Component",
  y = "Variance explained (%)",
  y2 = "Cumulative (%)"
)

p <- ggplot(data, aes(x = factor(component))) +
  geom_col(aes(y = variance_pct), width = 0.62, fill = "steelblue3",
           colour = NA) +
  geom_line(aes(y = cumulative_pct / 100 * YMAX), group = 1,
            colour = "#D55E00", linewidth = 0.6) +
  geom_point(aes(y = cumulative_pct / 100 * YMAX), group = 1,
             colour = "#D55E00", size = 1.2) +
  scale_y_continuous(
    name = LABELS$y,
    expand = expansion(mult = c(0, 0.10)),
    sec.axis = sec_axis(~ . / YMAX * 100, name = LABELS$y2)
  ) +
  labs(title = LABELS$title, subtitle = LABELS$subtitle, x = LABELS$x) +
  THEME +
  theme(axis.title.y.right = element_text(colour = "#D55E00"),
        axis.text.y.right = element_text(colour = "#D55E00"),
        plot.margin = margin(r = 12))
"""
        return self._write(out_dir, stem, data_name,
                           "PCA scree plot (ggplot2)", body, spec)

    # -- shared ------------------------------------------------------
    @staticmethod
    def _palette_block(spec: RPlotSpec) -> str:
        return f"""
## --- colour palette ----------------------------------------------
## Swap PALETTE_NAME below; the rest of the script is unchanged.
## Only "default" is guaranteed to work everywhere -- the alternatives fall
## back to base R (grDevices::hcl.colors) when the package is absent, so the
## script never dies on a missing suggested package.
PALETTE_NAME <- "{spec.color_palette}"
PALETTE <- switch(
  PALETTE_NAME,
  default = c("#2C7FB8", "#D95F0E", "#1A9641", "#7B3294", "#C994C7", "#41B6C4"),
  brewer  = if (requireNamespace("RColorBrewer", quietly = TRUE)) {{
               RColorBrewer::brewer.pal(6, "Dark2")
             }} else {{
               grDevices::hcl.colors(6, palette = "Dark 3")
             }},
  viridis = if (requireNamespace("viridisLite", quietly = TRUE)) {{
               viridisLite::viridis(6)
             }} else {{
               grDevices::hcl.colors(6, palette = "Viridis")
             }}
)
"""
