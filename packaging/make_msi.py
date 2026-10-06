"""Build the PaleoAST MSI: harvest, compile, link, validate.

Three steps that are easy to get subtly wrong by hand, so they live in one
place:

    python packaging/harvest_payload.py
    candle -dPaleoAST.wixobj -out obj packaging/PaleoAST.wxs packaging/_payload.wxs
    light  -ext WixUIExtension -ext WixUtilExtension -out dist/...msi obj/*.wixobj

Details that are not obvious and were each a real failure here:

  * candle needs ``-out`` to be a DIRECTORY, with a trailing separator,
    whenever more than one source file is passed. Given a file path it
    treats the extra name as an error.
  * ``-arch`` belongs to candle, not light, and it is the only supported way
    to declare a 64-bit package.
  * light resolves relative Source paths against the process working
    directory, not against the .wxs location, so a relative path that reads
    correctly in the source file still fails to resolve.
  * ICE validation does not run by default for the errors that matter here
    (duplicate files across components, missing keypaths). ``--validate``
    turns on the full set.

Usage:
    python packaging/make_msi.py [--validate]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
PAYLOAD = ROOT / "dist" / "PaleoAST"
MSI = ROOT / "dist" / "PaleoAST-1.1.0-windows-x64.msi"
OBJ = Path(os.environ.get("TEMP", str(ROOT))) / "wixobj"

WIX_CANDIDATES = [
    Path(os.environ.get("WIX", "")) if os.environ.get("WIX") else None,
    Path(os.environ.get("TEMP", "")) / "wix3",
    Path(r"C:\Program Files (x86)\WiX Toolset v3.14\bin"),
]


def find_tool(name: str) -> Path:
    """Locate candle.exe / light.exe.

    WiX v3 is a standalone .NET Framework toolset, so it needs no dotnet SDK,
    but it is not on PATH by default and it is not installed by the MSI either.
    """
    for base in WIX_CANDIDATES:
        if base is None:
            continue
        candidate = base / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise SystemExit(
        f"{name} not found. Looked in: "
        + ", ".join(str(b) for b in WIX_CANDIDATES if b)
        + " and on PATH. Set WIX=<wix bin dir> to point at it."
    )


def run(argv: list[str]) -> None:
    print("  " + " ".join(str(a) for a in argv))
    result = subprocess.run([str(a) for a in argv], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"failed: {argv[0]} exited {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the full ICE validation set (slow, catches keypath and "
        "duplicate-file errors that the default link does not)",
    )
    args = parser.parse_args()

    if not (PAYLOAD / "PaleoAST.exe").is_file():
        raise SystemExit(
            f"no build at {PAYLOAD}. Run build_exe.py first -- the MSI ships "
            "the same tree as the portable zip, on purpose."
        )

    candle = find_tool("candle.exe")
    light = find_tool("light.exe")
    print(f"candle: {candle}\nlight:  {light}")

    print("\n[1/4] harvesting payload")
    run([sys.executable, PACKAGING / "harvest_payload.py"])

    print("\n[2/4] compiling")
    if OBJ.exists():
        shutil.rmtree(OBJ)
    OBJ.mkdir(parents=True)
    # -arch belongs to candle, not light. It is the only supported way to say
    # "this is a 64-bit package" -- Package/@Platform exists but is deprecated
    # -- and it sets the Template Summary line. Without it the line stays
    # "Intel" and ICE80 rejects every 64-bit component in the payload.
    #
    # It also makes Component/@Win64 default to yes, so the harvested
    # components do not carry 228 copies of the same attribute.
    #
    # -out wants a directory with a trailing separator when linking more than
    # one source afterwards.
    run(
        [
            candle,
            "-nologo",
            "-arch",
            "x64",
            "-dPaleoAST.wixobj",
            "-out",
            f"{OBJ}{os.sep}",
            PACKAGING / "PaleoAST.wxs",
            PACKAGING / "_payload.wxs",
        ]
    )

    print("\n[3/4] linking")
    link = [
        light,
        "-nologo",
        "-ext",
        "WixUIExtension",
        "-ext",
        "WixUtilExtension",
        "-out",
        MSI,
    ]
    if args.validate:
        link.append("-sice:ALL")
    link += [OBJ / "PaleoAST.wixobj", OBJ / "_payload.wixobj"]
    run(link)

    print("\n[4/4] done")
    if not MSI.is_file():
        raise SystemExit(f"light reported success but {MSI} does not exist")
    print(f"{MSI}  {MSI.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
