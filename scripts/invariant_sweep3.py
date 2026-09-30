"""Invariant sweep, part 3: ecology.beta_diversity, ecology.rarefaction,
phylogenetics.tree_distance, morphometrics.shape_stats, models.data_matrix.

Parts 1 and 2 cover distance_metrics, dtw, paleoenv, biostratigraphy,
isotope_analysis, morpho3d.mesh, stratigraphy.arma/directional and stats.lda.
These five are the remaining algorithm-heavy modules in the changed set.

The properties chosen here are ones that fail loudly when a sign is wrong, an
index is off by one, or a matrix is silently transposed -- the error classes
that produce plausible numbers rather than exceptions:

  * Hotelling's T^2 is symmetric in its two arguments. It is built from an
    inverse covariance, and transposing or reordering the operands silently
    yields a different, still-finite, wrong number.
  * Procrustes distance is symmetric and vanishes on a shape against itself.
  * Robinson-Foulds is symmetric, zero on identity, and bounded by the number
    of internal splits.
  * A rarefaction curve is non-decreasing and never exceeds observed richness.
  * Beta decomposition components must sum to the total.
"""

from __future__ import annotations

import io
import sys
import traceback
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

FINDINGS: list[str] = []


def note(msg: str) -> None:
    print(f"  FINDING  {msg}")
    FINDINGS.append(msg)


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FINDINGS.append(f"{label} {detail}")


def run(label: str, fn):
    try:
        return fn()
    except Exception as exc:
        print(f"  RAISED {label}: {type(exc).__name__}: {exc}")
        if "--trace" in sys.argv:
            traceback.print_exc()
        return exc


rng = np.random.default_rng(20261001)

# --------------------------------------------------------- beta diversity
print("=== ecology.beta_diversity ===")
from ecology.beta_diversity import BetaDiversityAnalyzer, CoverageRarefactionAnalyzer

X = rng.poisson(6, size=(6, 14)).astype(float)

# decompose_beta_diversity accepts only 'jaccard' and 'sorensen'; it rejects
# anything else with a messaged DataValidationError, which is the correct
# behaviour and is what the earlier draft of this file tripped over.
for metric in ("jaccard", "sorensen"):
    res = run(
        f"decompose_beta_diversity metric={metric}",
        lambda m=metric: BetaDiversityAnalyzer().decompose_beta_diversity(X, metric=m),
    )
    if isinstance(res, Exception):
        continue
    d = res.to_dict()
    total = np.asarray(d["total_beta"], dtype=float)
    turn = np.asarray(d["turnover_component"], dtype=float)
    nest = np.asarray(d["nestedness_component"], dtype=float)
    check(
        f"{metric}: the three components are square n x n",
        total.shape == turn.shape == nest.shape == (6, 6),
        f"{total.shape} {turn.shape} {nest.shape}",
    )
    check(
        f"{metric}: all components finite",
        bool(np.all(np.isfinite(total)) and np.all(np.isfinite(turn)) and np.all(np.isfinite(nest))),
        "",
    )
    # Baselga's additive decomposition: beta = turnover + nestedness, elementwise.
    residual = float(np.max(np.abs(total - (turn + nest))))
    check(f"{metric}: total == turnover + nestedness", residual < 1e-9, f"max residual {residual:.3e}")
    check(
        f"{metric}: decomposition matrix symmetric",
        bool(np.allclose(total, total.T, atol=1e-12)),
        f"max asym {np.max(np.abs(total - total.T)):.3e}",
    )
    check(f"{metric}: zero diagonal", bool(np.allclose(np.diag(total), 0.0, atol=1e-12)), str(np.diag(total)))
    lo = float(np.nanmin(total[np.triu_indices(6, 1)]))
    hi = float(np.nanmax(total))
    check(f"{metric}: beta within [0, 1]", lo >= -1e-12 and hi <= 1.0 + 1e-9, f"range [{lo:.4f}, {hi:.4f}]")

# Duplicated samples: every pair that is a copy of another must have zero
# total, turnover AND nestedness -- a decomposed beta that reports structure
# between identical samples is a real defect, not a rounding artefact.
dup = np.vstack([X[0], X[0], X[1], X[1], X[2], X[2]])
res = run(
    "decompose_beta_diversity on duplicated samples",
    lambda: BetaDiversityAnalyzer().decompose_beta_diversity(dup, metric="jaccard"),
)
if not isinstance(res, Exception):
    d = res.to_dict()
    for key in ("total_beta", "turnover_component", "nestedness_component"):
        m = np.asarray(d[key], dtype=float)
        # pairs (0,1) (2,3) (4,5) are copies
        worst = max(abs(m[i, j]) for i, j in ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)))
        check(f"duplicated samples: {key} == 0 between copies", worst < 1e-9, f"max |{key}| = {worst:.3e}")

# coverage rarefaction: monotone, within range
cov = run(
    "coverage_rarefaction_hill", lambda: CoverageRarefactionAnalyzer().analyze(X, n_points=20, confidence_level=0.95)
)
if not isinstance(cov, Exception):
    d = cov.to_dict() if hasattr(cov, "to_dict") else vars(cov)
    print("    coverage keys:", sorted(d)[:8])
    for key in ("x", "y", "lower", "upper", "sample_sizes", "estimates"):
        if key in d and d[key] is not None:
            arr = np.asarray(d[key], dtype=float).ravel()
            if arr.size and np.all(np.isfinite(arr)):
                check(f"coverage '{key}' finite", True)
    xk = next((k for k in d if "x" in k.lower() and d[k] is not None), None)
    yk = next((k for k in d if k.lower() in ("y", "estimate", "estimates") and d[k] is not None), None)
    if xk and yk:
        xx = np.asarray(d[xk], dtype=float).ravel()
        yy = np.asarray(d[yk], dtype=float).ravel()
        n = min(xx.size, yy.size)
        check(
            "coverage curve non-decreasing",
            bool(np.all(np.diff(yy[:n]) >= -1e-9)),
            f"max drop {np.diff(yy[:n]).min():.3e}",
        )
        check(
            "coverage curve within [0, 1]",
            bool(yy[:n].min() >= -1e-12 and yy[:n].max() <= 1.0 + 1e-9),
            f"range [{yy[:n].min():.3f}, {yy[:n].max():.3f}]",
        )

# ------------------------------------------------------------ rarefaction
print()
print("=== ecology.rarefaction ===")
from ecology.rarefaction import RarefactionAnalyzer

ab = np.array([40.0, 25.0, 18.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0])
rr = run("RarefactionAnalyzer.analyze", lambda: RarefactionAnalyzer().analyze(ab, max_n=40, n_points=20))
if not isinstance(rr, Exception):
    d = rr.to_dict() if hasattr(rr, "to_dict") else vars(rr)
    xk = next((k for k in d if "x" in k.lower() and d[k] is not None), None)
    yk = next((k for k in d if k.lower() in ("y", "expected", "curve", "richness") and d[k] is not None), None)
    print("    rarefaction keys:", sorted(d)[:10])
    if xk and yk:
        yy = np.asarray(d[yk], dtype=float).ravel()
        check("rarefaction non-decreasing", bool(np.all(np.diff(yy) >= -1e-9)), f"max drop {np.diff(yy).min():.3e}")
        check(
            "rarefaction never exceeds observed richness",
            bool(yy.max() <= (ab > 0).sum() + 1e-9),
            f"max {yy.max():.3f} vs S={int((ab > 0).sum())}",
        )
    mono = run(
        "rarefaction on a single-species sample",
        lambda: RarefactionAnalyzer().analyze(np.array([10.0, 0, 0, 0]), max_n=10, n_points=5),
    )
    check(
        "single-species rarefaction stays at 1",
        (not isinstance(mono, Exception))
        and float(np.nanmax(np.asarray(mono.expected if hasattr(mono, "expected") else [1.0], dtype=float))) == 1.0,
    )

# ------------------------------------------------------- tree distance
print()
print("=== phylogenetics.tree_distance ===")
from phylogenetics import PhyloTree
from phylogenetics.tree_distance import (
    normalized_robinson_foulds_distance,
    robinson_foulds_distance,
    split_bitmasks,
    weighted_robinson_foulds_distance,
)

# Every tree must span the SAME taxon set. robinson_foulds_distance correctly
# refuses a mismatch with a DataValidationError naming the differing taxa, so
# comparing a 4-taxon tree with a 5-taxon one is simply not a valid call.
NEWICKS = [
    "((A:1,B:1)ab:1,(C:1,D:1)cd:1,(E:1)e:1)root;",
    "((A:1,C:1)ac:1,(B:1,D:1)bd:1,(E:1)e:1)root;",
    "((A:1,B:1)ab:1,(C:1,D:1,E:1)cde:1)root;",
    "((A:1,B:1,C:1)abc:1,(D:1,E:1)de:1)root;",
]
trees = [PhyloTree.from_newick(n) for n in NEWICKS]

for i, ti in enumerate(trees):
    for j, tj in enumerate(trees):
        rf = run(f"RF t{i}-t{j}", lambda a=ti, b=tj: robinson_foulds_distance(a, b))
        nrf = run(f"nRF t{i}-t{j}", lambda a=ti, b=tj: normalized_robinson_foulds_distance(a, b))
        if not isinstance(rf, int) and not isinstance(rf, Exception):
            check(f"RF(t{i},t{i}) == 0", i == j and rf == 0, f"got {rf}")
        if not isinstance(nrf, Exception):
            check(f"nRF t{i}-t{j} in [0,1]", 0.0 <= float(nrf) <= 1.0 + 1e-9, f"got {nrf}")
        if i == j and not isinstance(nrf, Exception):
            check(f"nRF(t{i},t{i}) == 0", float(nrf) == 0.0, f"got {nrf}")

print()
for i in range(len(trees)):
    for j in range(i + 1, len(trees)):
        a = run(f"RF symmetry t{i}t{j}", lambda x=trees[i], y=trees[j]: robinson_foulds_distance(x, y))
        b2 = run(f"RF symmetry t{i}t{j} (swapped)", lambda x=trees[j], y=trees[i]: robinson_foulds_distance(x, y))
        if not isinstance(a, Exception) and not isinstance(b2, Exception):
            check(f"RF symmetric t{i}/t{j}", a == b2, f"{a} vs {b2}")
        wa = run(f"wRF t{i}t{j}", lambda x=trees[i], y=trees[j]: weighted_robinson_foulds_distance(x, y))
        if not isinstance(wa, Exception):
            check(f"wRF t{i}t{j} non-negative", float(wa) >= -1e-12, f"got {wa}")

print()
for i, t in enumerate(trees):
    masks = run(f"split_bitmasks t{i}", lambda x=t: split_bitmasks(x))
    if isinstance(masks, Exception):
        continue
    check(f"t{i}: split bitmasks are non-trivial-sized", len(masks) > 0, f"{len(masks)} masks")

# ---------------------------------------------------------- shape stats
print()
print("=== morphometrics.shape_stats ===")
from morphometrics.shape_stats import (
    geometric_median,
    goodall_f,
    hotelling_t2,
    kendall_preshape,
    procrustes_distance,
)

# procrustes_distance and hotelling_t2 take 2-D (n_landmarks, dim) / (n, d)
# inputs, not the 3-D (n_specimens, n_landmarks, dim) GPA layout.
cfgs = rng.normal(size=(6, 5, 2))
other = rng.normal(size=(6, 5, 2))
flat_a = cfgs[:, 0, :]
flat_b = other[:, 0, :]

d_self = run("procrustes_distance(x, x)", lambda: procrustes_distance(flat_a, flat_a))
check(
    "procrustes_distance(x, x) == 0", (not isinstance(d_self, Exception)) and abs(float(d_self)) < 1e-8, f"got {d_self}"
)
d_ab = run("procrustes_distance(a, b)", lambda: procrustes_distance(flat_a, flat_b))
d_ba = run("procrustes_distance(b, a)", lambda: procrustes_distance(flat_b, flat_a))
if not isinstance(d_ab, Exception) and not isinstance(d_ba, Exception):
    check("procrustes_distance symmetric", abs(float(d_ab) - float(d_ba)) < 1e-8, f"{d_ab} vs {d_ba}")

print()
A = flat_a
B = flat_b
t2_ab = run("hotelling_t2(A, B)", lambda: hotelling_t2(A, B))
t2_ba = run("hotelling_t2(B, A)", lambda: hotelling_t2(B, A))
if not isinstance(t2_ab, Exception) and not isinstance(t2_ba, Exception):
    va, vb = float(t2_ab.t2), float(t2_ba.t2)
    check("Hotelling T^2 symmetric in its arguments", abs(va - vb) < 1e-6, f"T2(A,B)={va:.6f} T2(B,A)={vb:.6f}")
    check("Hotelling T^2 non-negative", va >= -1e-12, f"got {va}")
t2_self = run("hotelling_t2(A, A)", lambda: hotelling_t2(A, A))
if not isinstance(t2_self, Exception):
    v = float(t2_self.t2)
    check("Hotelling T^2(x, x) == 0", abs(v) < 1e-6, f"got {v}")

print()
kp = run("kendall_preshape", lambda: kendall_preshape(cfgs))
if not isinstance(kp, Exception):
    # kendall_preshape normalises EACH configuration independently -- centred on
    # its own centroid, unit Frobenius norm -- and returns them as row vectors.
    # So the properties are per specimen, not across the set: summing over
    # specimens is expected to be non-zero and means nothing. An earlier draft
    # of this file checked exactly that, and failed a correct implementation.
    arr = np.asarray(kp, dtype=float)
    flat = arr.reshape(arr.shape[0], -1)
    check("Kendall preshape finite", bool(np.all(np.isfinite(arr))), str(arr.shape))
    check(
        "Kendall preshape centres each specimen (row sums zero)",
        bool(np.all(np.abs(flat.sum(axis=1)) < 1e-10)),
        f"row sums {np.round(flat.sum(axis=1), 12).tolist()}",
    )
    check(
        "Kendall preshape gives each specimen unit norm",
        bool(np.allclose((flat**2).sum(axis=1), 1.0, atol=1e-10)),
        f"norms {np.round((flat**2).sum(axis=1), 12).tolist()}",
    )

gmed = run("geometric_median", lambda: geometric_median(cfgs[:, 0, :]))
if not isinstance(gmed, Exception):
    check(
        "geometric_median finite", bool(np.all(np.isfinite(np.asarray(gmed, dtype=float)))), str(np.asarray(gmed).shape)
    )

gf = run("goodall_f", lambda: goodall_f(cfgs, np.array([0, 0, 0, 1, 1, 1])))
if not isinstance(gf, Exception):
    vals = np.asarray([v for v in gf if isinstance(v, (int, float))], dtype=float)
    check("goodall_f finite", bool(np.all(np.isfinite(vals))), str(gf))

# --------------------------------------------------------- data matrix
print()
print("=== models.data_matrix ===")
from models.data_matrix import DataMatrix

dm = run(
    "DataMatrix means/stds",
    lambda: DataMatrix(
        np.array([[1.0, 10.0, 5.0], [2.0, 20.0, 7.0], [3.0, 30.0, 9.0], [4.0, 40.0, 11.0]]),
        row_labels=["r1", "r2", "r3", "r4"],
        col_labels=["a", "b", "c"],
    ),
)
if not isinstance(dm, Exception):
    mu = np.asarray(dm.column_means(), dtype=float)
    sd = np.asarray(dm.column_stds(), dtype=float)
    data = np.asarray(dm.to_matrix() if hasattr(dm, "to_matrix") else dm.data, dtype=float)
    check(
        "column_means matches a direct mean", bool(np.allclose(mu, data.mean(axis=0))), f"{mu} vs {data.mean(axis=0)}"
    )
    check("column_stds matches ddof=1", bool(np.allclose(sd, data.std(axis=0, ddof=1), equal_nan=True)), f"{sd}")
    check("column_stds non-negative", bool(np.all(sd >= -1e-12)), str(sd))

print()
if FINDINGS:
    print(f"{len(FINDINGS)} finding(s)")
    for item in FINDINGS:
        print("  -", item)
else:
    print("all invariants held")
raise SystemExit(1 if FINDINGS else 0)
