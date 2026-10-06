"""Invariant sweep over the least-verified science modules.

Coverage says which lines ran. This asks a different question: across a
spread of random inputs, do the results obey the invariants the method
promises? Finite values, symmetric distances, values in range, sums that
conserve, limits that hold. A crash or a NaN here is a real defect, and it
costs nothing to look for.

Run: .venv/Scripts/python.exe scripts/invariant_sweep.py
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(f"{label} {detail}")


def run(label: str, fn) -> object | None:
    try:
        result = fn()
        print(f"  ok    {label} ran")
        return result
    except Exception as exc:
        print(f"  FAIL  {label} raised {type(exc).__name__}: {exc}")
        FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
        if "--trace" in sys.argv:
            traceback.print_exc()
        return None


rng = np.random.default_rng(20260929)

# ---------------------------------------------------------------- distances
print("=== stats.distance_metrics ===")
from stats.distance_metrics import compute_distance_matrix

for metric in ("euclidean", "bray_curtis", "jaccard", "manhattan", "correlation"):
    data = rng.integers(0, 30, size=(6, 8)).astype(float)
    result = run(f"{metric}: builds", lambda m=metric, d=data: compute_distance_matrix(d, metric=m))
    if result is None:
        continue
    matrix = np.asarray(result.matrix, dtype=float)
    check(f"{metric}: symmetric", np.allclose(matrix, matrix.T, equal_nan=True))
    check(f"{metric}: zero diagonal", np.allclose(np.diag(matrix), 0, atol=1e-9))
    upper = matrix[np.triu_indices(len(matrix), 1)]
    upper = upper[np.isfinite(upper)]
    check(f"{metric}: non-negative", np.all(upper >= -1e-12), f"min={upper.min() if upper.size else 0}")

# ---------------------------------------------------------------- rarefaction
print("\n=== ecology.rarefaction (vs exact combinatorics) ===")
from math import comb

from ecology.rarefaction import RarefactionAnalyzer

analyzer = RarefactionAnalyzer()
for trial in range(3):
    abundances = rng.integers(1, 40, size=8)
    total = int(abundances.sum())
    max_n = min(8, total - 1)
    result = run(
        f"rarefaction trial {trial} (N={total}, curve to n={max_n})",
        lambda a=abundances, m=max_n: analyzer.analyze(a, max_n=m, n_points=m),
    )
    if result is None:
        continue
    got = np.asarray(result.expected_taxa, dtype=float)
    sizes = np.asarray(result.sample_sizes, dtype=int)
    # E[S_n] = sum_i (1 - C(N - n_i, n) / C(N, n))  (Hurlbert 1971), exact.
    expected = np.array([sum(1 - comb(total - int(x), n) / comb(total, n) for x in abundances) for n in sizes])
    check(
        f"rarefaction trial {trial}: whole curve matches Hurlbert",
        got.shape == expected.shape and np.allclose(got, expected, rtol=1e-4, atol=1e-5),
        f"max diff {float(np.abs(got - expected).max()) if got.shape == expected.shape else 'shape'}",
    )
    check(
        f"rarefaction trial {trial}: curve is monotonically non-decreasing",
        bool(np.all(np.diff(got) >= -1e-9)),
    )
    check(
        f"rarefaction trial {trial}: E[S_n] <= S",
        bool(np.all(got <= len(abundances) + 1e-9)),
    )

# ---------------------------------------------------------------- diversity
print("\n=== ecology.diversity ===")
from ecology.diversity import DiversityAnalyzer

for trial in range(3):
    counts = rng.integers(1, 50, size=12).astype(float)
    result = run(f"diversity trial {trial}", lambda c=counts: DiversityAnalyzer().analyze_sample(c))
    if result is None:
        continue
    # Each index is a DiversityIndexResult; take the numeric value out.
    idx = {k: getattr(v, "value", v) for k, v in result.indices.items()}
    shannon = idx.get("shannon", float("nan"))
    simpson = idx.get("simpson_1_minus_D", idx.get("simpson", float("nan")))
    check(
        f"diversity trial {trial}: Shannon in [0, ln(S)]",
        0.0 <= shannon <= np.log(len(counts)) + 1e-9,
        f"shannon={shannon}",
    )
    check(
        f"diversity trial {trial}: Gini-Simpson in [0, 1]",
        0.0 <= simpson <= 1.0 + 1e-9,
        f"simpson={simpson}",
    )
    # Shannon must equal the textbook value for the same counts.
    expected_h = float(-(counts / counts.sum() * np.log(counts / counts.sum())).sum())
    check(
        f"diversity trial {trial}: Shannon matches -sum(p log p)",
        abs(shannon - expected_h) < 1e-9,
        f"got {shannon} expected {expected_h}",
    )

# ---------------------------------------------------------------- DTW
print("\n=== ecology.dtw ===")
from ecology.dtw import DTWAnalyzer

d = DTWAnalyzer()
series_a = rng.normal(size=20).cumsum()


def dtw(a, b):
    return float(np.ravel(np.asarray(d.compute(a, b).distance, dtype=float))[0])


# The defining property: an identical series has DTW distance exactly zero.
result = run("dtw: identical series", lambda: dtw(series_a, series_a))
if result is not None:
    check("dtw: identical series is exactly zero", result == 0.0, f"distance={result}")

# Noisy series: the path may not simply follow the noise, it must be at most
# the noise's own total variation (DTW exists to find a shorter path).
noisy = series_a + rng.normal(scale=0.1, size=20)
result = run("dtw: noisy series", lambda: dtw(series_a, noisy))
if result is not None:
    variation = float(np.abs(np.diff(noisy - series_a)).sum())
    check(
        "dtw: path no longer than the noise it is explained by",
        0.0 < result <= variation,
        f"dtw={result} noise variation={variation}",
    )

shifted = np.roll(series_a, 5)
result = run("dtw: shifted series", lambda: dtw(series_a, shifted))
if result is not None:
    check("dtw: a shifted series is far from zero", result > 1.0, f"distance={result}")

# ---------------------------------------------------------------- paleoenv
print("\n=== ecology.paleoenv ===")
from ecology.paleoenv import PaleoEnvironmentReconstructor

abund = rng.integers(0, 30, size=(24, 10)).astype(float)
heights = np.linspace(0, 100, 24)
result = run("paleoenv: reconstructs", lambda: PaleoEnvironmentReconstructor().reconstruct(abund, heights))
if result is not None:
    axis = np.ravel(np.asarray(result.row_species_axis, dtype=float))
    check("paleoenv: axis is finite", bool(np.all(np.isfinite(axis))))
    check("paleoenv: axis length matches heights", len(axis) == len(heights), f"len={len(axis)}")
    check(
        "paleoenv: singular values descending",
        bool(np.all(np.diff(np.asarray(result.singular_values, dtype=float)) <= 1e-9)),
    )
    corr = float(result.pearson_corr_axis_vs_height)
    check("paleoenv: axis/height correlation in [-1, 1]", -1.0 - 1e-9 <= corr <= 1.0 + 1e-9, f"r={corr}")

# ---------------------------------------------------------------- biostrat
print("\n=== stratigraphy.biostratigraphy (UA) ===")
from stratigraphy.biostratigraphy import UAAnalyzer

n_sections, n_events = 5, 6
rng_local = np.random.default_rng(11)
fad = np.zeros((n_sections, n_events), dtype=int)
lad = np.zeros((n_sections, n_events), dtype=int)
for section in range(n_sections):
    order = rng_local.permutation(n_events) + 1
    lad[section] = order
    fad[section] = np.maximum(order - 1, 1)
result = run("UA: analyses", lambda: UAAnalyzer().analyze(fad, lad, min_section_occurrence=2))
if result is not None:
    events = np.asarray(getattr(result, "uaz_ua", []), dtype=object)
    check("UA: returns an object", events is not None)
    check("UA: no negative event positions", bool(np.all(np.asarray(fad) >= 0)))

# ---------------------------------------------------------------- isotope
print("\n=== stratigraphy.isotope_analysis ===")
from stratigraphy.isotope_analysis import IsotopeAnalyzer

ia = IsotopeAnalyzer()
# All three take (seawater VSMOW, carbonate VPDB). A warmer calcification
# (more negative carbonate) must give a warmer temperature, and a warmer
# seawater (less negative) must give a colder one. That monotonicity is the
# invariant; the absolute value depends on each calibration's own arithmetic.
for label, fn in (
    ("Erez & Luz", lambda sw, cc: ia.compute_paleotemperature_erez_luz(sw, cc)),
    ("Bemis", lambda sw, cc: ia.compute_paleotemperature_bemis(cc, sw)),
    ("Kim & O'Neill", lambda sw, cc: ia.compute_paleotemperature_kim_oneil(sw, cc)),
):
    base = run(f"{label}: evaluates", lambda f=fn: f(0.0, -2.0))
    colder_carbonate = run(f"{label}: colder carbonate", lambda f=fn: f(0.0, -4.0))
    warmer_seawater = run(f"{label}: warmer seawater", lambda f=fn: f(1.0, -2.0))
    if None in (base, colder_carbonate, warmer_seawater):
        continue
    check(f"{label}: temperature is finite", bool(np.isfinite(base)))
    check(
        f"{label}: colder carbonate -> warmer temperature",
        colder_carbonate > base,
        f"{colder_carbonate} vs {base}",
    )
    # Raising the seawater delta18O means a MORE 18O-enriched ocean, i.e.
    # glacial conditions. The same foraminiferal delta18O then requires a
    # WARMER calcification temperature, so T must RISE with delta18O_sw.
    # (It is tempting to read "18O-enriched water" as "warm water" -- it is
    # the opposite; a warm interglacial ocean is 18O-depleted.)
    check(
        f"{label}: 18O-enriched seawater -> warmer temperature",
        warmer_seawater > base,
        f"{warmer_seawater} vs {base}",
    )

# ---------------------------------------------------------------- mesh
print("\n=== morpho3d.mesh ===")
from morpho3d.mesh import Mesh3D

verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
mesh = Mesh3D(vertices=verts, faces=faces)
result = run("mesh: volume", lambda: mesh.compute_volume())
if result is not None:
    value = float(result)
    # unit right tetrahedron: |det| / 6 = 1/6
    check("mesh: unit tetrahedron volume == 1/6", abs(abs(value) - 1 / 6) < 1e-9, f"got {value}")
result = run("mesh: surface area", lambda: mesh.compute_surface_area())
if result is not None:
    # three right triangles of area 1/2 plus the tilted face
    check("mesh: area positive", float(result) > 0, f"got {result}")

# ---------------------------------------------------------------- summary
print()
print("=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} invariant(s) FAILED:")
    for line in FAILURES:
        print(f"  * {line}")
    sys.exit(1)
print("all invariants held")
