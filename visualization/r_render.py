# =============================================================================
# FILE: visualization/r_render.py
# =============================================================================
"""
Run a generated R script and bring the figure back into the app.

WHAT THIS IS FOR
----------------
:mod:`visualization.r_export` writes an editable ``.R`` script. This module is
the other half: it executes that script with ``Rscript`` and reports what
appeared on disk.

R is invoked as a CHILD PROCESS, not through rpy2. rpy2 links R into the Python
interpreter, which brings ABI pinning and a per-platform R runtime to package
into the frozen build -- and this repository has three documented rpy2 failures
in ``pyproject.toml`` (a nonexistent 5.x pin that kept nine CI jobs red, the
``R_getVar`` symbol removed in R 4.3.3, and an outright Windows exclusion). A
child process has none of those constraints, and R is expected to be something
the user already has installed.

THE EDITED-SCRIPT PROBLEM
-------------------------
The user is expected to edit the ``.R`` file. So nothing here may assume the
script still looks like the one we generated:
  * the output file may have been renamed, so new files are found by diffing
    the directory rather than by guessing the expected name;
  * a user script may not leave ``p`` behind, so the in-app preview is
    BEST EFFORT and its failure never invalidates a successful render.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "RRunResult",
    "describe_missing_r",
    "find_rscript",
    "run_r_script",
]

# Formats a reader might reasonably produce. PDF/SVG are what the generated
# scripts write; PNG is what the in-app preview needs.
FIGURE_SUFFIXES = (".pdf", ".png", ".svg", ".eps", ".jpeg", ".jpg", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Locating R
# ---------------------------------------------------------------------------

# Windows install roots. The D: drive is included because that is where R
# actually lives on this machine -- a C:-only search finds nothing.
_WINDOWS_ROOTS = (
    r"C:\Program Files\R",
    r"C:\Program Files (x86)\R",
    r"D:\Program Files\R",
    r"D:\R",
)


def _iter_windows_candidates() -> list[Path]:
    out: list[Path] = []
    for root in _WINDOWS_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        # R installs under R-<version>; sort descending so a newer R wins.
        for version_dir in sorted(base.glob("R-*"), reverse=True):
            for rel in (
                Path("bin") / "x64" / "Rscript.exe",
                Path("bin") / "Rscript.exe",
            ):
                cand = version_dir / rel
                if cand.is_file():
                    out.append(cand)
    return out


def _iter_posix_candidates() -> list[Path]:
    out: list[Path] = []
    which = shutil.which("Rscript")
    if which:
        out.append(Path(which))
    for root in (Path("/usr/local/bin"), Path("/usr/bin"), Path.home() / "R" / "bin"):
        cand = root / "Rscript"
        if cand.is_file():
            out.append(cand)
    return out


def find_rscript(configured: str = "") -> Path | None:
    """Locate an ``Rscript`` executable.

    Order: explicit setting, then PATH, then the usual install roots. Returns
    ``None`` rather than raising -- R being absent is a normal, reportable
    state for a feature that is optional by design.
    """
    if configured:
        cand = Path(configured).expanduser()
        if cand.is_file():
            return cand
        # Allow a bare command name in the setting as well.
        which = shutil.which(configured)
        if which:
            return Path(which)
        return None

    which = shutil.which("Rscript")
    if which:
        return Path(which)

    candidates = (_iter_windows_candidates() if os.name == "nt"
                  else _iter_posix_candidates())
    return candidates[0] if candidates else None


def describe_missing_r() -> str:
    """Actionable message, since 'Rscript not found' alone helps nobody."""
    return (
        "R was not found on this machine.\n\n"
        "Install R (https://cran.r-project.org/) and then either:\n"
        "  * make sure 'Rscript' is on PATH, or\n"
        "  * set the full path in Preferences -> R Plotting (e.g.\n"
        "    D:\\Program Files\\R\\R-4.5.2\\bin\\x64\\Rscript.exe)\n\n"
        "The generated .R script does not need PaleoAST at all -- you can open it\n"
        "in RStudio and run it there."
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RRunResult:
    script_path: Path
    ok: bool = False
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    produced: list[Path] = field(default_factory=list)
    preview_png: Path | None = None
    error: str = ""

    def message(self, limit: int = 1200) -> str:
        """The most useful text to show a user who is not an R programmer."""
        if self.timed_out:
            return (
                f"R did not finish within the time limit while running "
                f"{self.script_path.name}.\n"
                "A script that never returns is usually an infinite loop or a "
                "very large grid; check the script and raise the timeout if it "
                "is simply slow."
            )
        if self.ok:
            return f"R finished: {self.script_path.name}"
        detail = (self.stderr or self.stdout or "").strip()
        if len(detail) > limit:
            detail = detail[:limit] + "\n..."
        return f"R reported an error running {self.script_path.name}:\n{detail}"


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

_PREVIEW_TEMPLATE = """\
## Auto-generated preview helper written by PaleoAST.
## It runs YOUR script unchanged, then re-saves the plot it leaves behind as a
## PNG so the application can display it. Delete this file freely; it is
## regenerated on every preview.
options(warn = 1)
source({script!r}, chdir = TRUE)
if (!exists("p")) {{
  stop("the script did not leave an object called 'p'; cannot make a preview")
}}
ggsave({out!r}, plot = p, width = {w}, height = {h}, dpi = {dpi}, units = "in")
"""


def _snapshot(directory: Path) -> dict[str, float]:
    try:
        return {
            p.name: p.stat().st_mtime
            for p in directory.iterdir()
            if p.is_file()
        }
    except OSError:
        return {}


def _make_preview(
    script_path: Path,
    work_dir: Path,
    rscript: Path,
    timeout: float,
) -> Path | None:
    """Best-effort PNG of whatever the script drew. Never raises.

    Deliberately does NOT touch the user's script: it sources it and re-saves
    the result. If the script does not leave ``p`` behind -- which a heavily
    edited script may not -- the preview simply is not available and the caller
    reports the real figures instead.
    """
    preview = work_dir / f"{script_path.stem}.__paleoast_preview.png"
    helper = work_dir / f"{script_path.stem}.__paleoast_preview.R"
    helper.write_text(
        _PREVIEW_TEMPLATE.format(
            script=script_path.name,
            out=preview.name,
            w=7,
            h=5.5,
            dpi=150,
        ),
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            [str(rscript), helper.name],
            capture_output=True, text=True, timeout=timeout, cwd=str(work_dir),
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            helper.unlink()
        except OSError:
            pass
    if proc.returncode != 0 or not preview.is_file():
        return None
    return preview


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_r_script(
    script_path: str | Path,
    *,
    rscript: str = "",
    timeout: float = 300.0,
    make_preview: bool = True,
) -> RRunResult:
    """Execute ``script_path`` with Rscript and report what it produced.

    The script is never modified. ``encoding`` is pinned to UTF-8 because R
    writes UTF-8 while ``text=True`` would otherwise decode with the Windows
    ANSI code page (GBK on this machine) and raise ``UnicodeDecodeError`` on
    any character R emits that GBK cannot represent -- R's own error banner
    contains U+2139, so this is easy to hit.
    """
    script_path = Path(script_path)
    result = RRunResult(script_path=script_path)

    if not script_path.is_file():
        result.error = f"Script not found: {script_path}"
        return result

    exe = find_rscript(rscript)
    if exe is None:
        result.error = describe_missing_r()
        return result

    work_dir = script_path.parent
    before = _snapshot(work_dir)

    try:
        proc = subprocess.run(
            [str(exe), script_path.name],
            capture_output=True, text=True, timeout=timeout, cwd=str(work_dir),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        result.timed_out = True
        result.ok = False
        result.error = "R timed out"
        result.stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        result.stderr = exc.stderr or "" if isinstance(exc.stderr, str) else ""
        result.produced = _new_figures(work_dir, before)
        return result
    except OSError as exc:
        result.error = f"Could not start R: {exc}"
        return result

    result.returncode = proc.returncode
    result.stdout = proc.stdout or ""
    result.stderr = proc.stderr or ""
    result.produced = _new_figures(work_dir, before)
    result.ok = proc.returncode == 0

    if result.ok and not result.produced:
        # R can succeed while writing nothing if the user renamed the output or
        # replaced ggsave with a viewer call. Say so rather than claiming a
        # figure exists.
        result.ok = True
        result.error = (
            "R ran without errors but no figure file appeared. If the script "
            "opens a viewer instead of calling ggsave(), save it to a file."
        )

    if result.ok and make_preview and not any(
        p.suffix.lower() == ".png" for p in result.produced
    ):
        result.preview_png = _make_preview(script_path, work_dir, exe, timeout)

    return result


def _new_figures(directory: Path, before: dict[str, float]) -> list[Path]:
    """Files that appeared or changed since ``before``.

    Diffing beats assuming a filename: the user may have edited the script and
    renamed its output, and returning their file is the whole point of the
    feature.
    """
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        if p.name not in before:
            out.append(p)
            continue
        try:
            if p.stat().st_mtime > before[p.name]:
                out.append(p)
        except OSError:
            continue
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    exe = find_rscript(os.environ.get("RSCRIPT", ""))
    print("Rscript:", exe or "NOT FOUND")
    if exe:
        print(subprocess.run(
            [str(exe), "-e", "cat(R.version.string)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout)
    sys.exit(0 if exe else 1)
