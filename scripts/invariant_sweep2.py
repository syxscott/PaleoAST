"""Invariant sweep, part 2: stratigraphy.arma, stratigraphy.directional, stats.lda.

Part 1 (scripts/invariant_sweep.py) covers distance_metrics, dtw, paleoenv,
biostratigraphy (UA), isotope_analysis and morpho3d.mesh. These three are in
the same "algorithm-heavy, least independently verified" category.

Two rules this file follows after the first draft got three things wrong:

* **Read the signature before calling it.** `ARMAAnalyzer.fit` takes
  `times` *and* `values`; passing one array is a caller bug, not a defect.
* **A deliberate, well-messaged exception is correct behaviour.** A
  `ComputationError("LDA requires at least 2 classes")` is the API working. Only
  an *opaque* failure -- a bare IndexError from deep inside numpy, with no
  explanation -- is a finding. Those are checked separately from "ran cleanly".
* **Check the unit in the parameter name.** `DirectionalAnalyzer.analyze` takes
  `angles_deg`; feeding it radians silently produces a very tight arc and a
  mean resultant near 1, which looks like a spectacular false positive and is
  not one.

Run: python scripts/invariant_sweep2.py
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


def note(message: str) -> None:
    print(f"  FINDING  {message}")
    FINDINGS.append(message)


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
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

# ------------------------------------------------------------------- ARMA
print("=== stratigraphy.arma ===")
from stratigraphy.arma import ARMAAnalyzer


def ar1_series(n: int, phi: float, burn: int = 500) -> np.ndarray:
    x = np.zeros(burn + n)
    noise = rng.normal(size=burn + n)
    for t in range(1, burn + n):
        x[t] = phi * x[t - 1] + noise[t]
    return x[burn:]


for n, phi_true in ((100, 0.6), (400, 0.6), (400, 0.0)):
    values = ar1_series(n, phi_true)
    times = np.arange(n, dtype=float)
    res = run(
        f"ARMA(2,2) on AR(1) phi={phi_true} n={n}", lambda v=values, t=times: ARMAAnalyzer().fit(t, v, p=2, q=2, d=0)
    )
    if isinstance(res, Exception):
        continue
    ar = np.atleast_1d(np.asarray(getattr(res, "ar", np.zeros(1)), dtype=float).ravel())
    ma = np.atleast_1d(np.asarray(getattr(res, "ma", np.zeros(1)), dtype=float).ravel())
    check(f"n={n} phi={phi_true}: AR coefficients finite", bool(np.all(np.isfinite(ar))), str(ar))
    check(f"n={n} phi={phi_true}: MA coefficients finite", bool(np.all(np.isfinite(ma))), str(ma))

    fc = run(f"predict n={n} phi={phi_true}", lambda r=res: ARMAAnalyzer().predict(r, 5))
    if isinstance(fc, Exception):
        continue
    v = np.asarray(getattr(fc, "forecasts", np.zeros(1)), dtype=float).ravel()
    check(f"n={n} phi={phi_true}: forecast finite", bool(np.all(np.isfinite(v))), str(v))
    check(f"n={n} phi={phi_true}: forecast length 5", v.size == 5, f"got {v.size}")
    # A stationary AR(1) has unit variance, so a 5-step-ahead forecast should
    # stay in the same ballpark as the data -- not diverge or collapse.
    check(
        f"n={n} phi={phi_true}: forecast scale sane",
        bool(np.all(np.abs(v) < 8 * np.std(values) + 8.0)),
        f"std={np.std(values):.2f} fc={v}",
    )

# constant series
const = np.full(150, 3.25)
res_const = run("ARMA on a constant series", lambda: ARMAAnalyzer().fit(np.arange(150.0), const))
if not isinstance(res_const, Exception):
    fc = run("predict on a constant series", lambda: ARMAAnalyzer().predict(res_const, 3))
    if not isinstance(fc, Exception):
        v = np.asarray(getattr(fc, "forecasts", np.zeros(1)), dtype=float).ravel()
        check("constant: forecast finite", bool(np.all(np.isfinite(v))), str(v))
        check("constant: forecast stays near the constant", bool(np.all(np.abs(v - 3.25) < 2.0)), str(v))
else:
    print(f"  note  constant series raised {type(res_const).__name__} (acceptable if messaged)")

# ------------------------------------------------------------- directional
print()
print("=== stratigraphy.directional ===")
from stratigraphy.directional import DirectionalAnalyzer

# DEGREES, per the parameter name.
tight = rng.normal(loc=0.0, scale=2.0, size=200)  # 2 deg scatter
r_tight = run("axial: tight cluster at 0 deg", lambda: DirectionalAnalyzer().analyze(tight, axial=True))
if not isinstance(r_tight, Exception):
    r_len = float(getattr(r_tight, "resultant_length", 0.0))
    check("tight cluster: resultant length near n", abs(r_len - 200) < 12, f"resultant_length={r_len:.3f}")
    check("tight cluster: significant", bool(getattr(r_tight, "is_significant", False)))

# uniform over the full circle, in degrees
uni = rng.uniform(0.0, 360.0, size=500)
r_uni = run("axial: uniform over 360 deg", lambda: DirectionalAnalyzer().analyze(uni, axial=True))
if not isinstance(r_uni, Exception):
    r_len = float(getattr(r_uni, "resultant_length", 0.0))
    # For n independent uniform directions, |sum of unit vectors| ~ sqrt(n).
    check(
        "uniform: resultant length ~ sqrt(n)",
        abs(r_len - np.sqrt(500)) < 0.5 * np.sqrt(500),
        f"got {r_len:.3f}, expected ~{np.sqrt(500):.1f}",
    )
    check(
        "uniform: not significant",
        not bool(getattr(r_uni, "is_significant", True)),
        "uniform data reported as directional",
    )

# circular (non-axial) data
r_circ = run("circular: uniform over 360 deg", lambda: DirectionalAnalyzer().analyze(uni, axial=False))
if not isinstance(r_circ, Exception):
    r_len = float(getattr(r_circ, "resultant_length", 0.0))
    check("circular uniform: resultant ~ sqrt(n)", abs(r_len - np.sqrt(500)) < 0.5 * np.sqrt(500), f"got {r_len:.3f}")

# a genuine signal: 60 deg, 3 deg scatter
signal = rng.normal(loc=60.0, scale=3.0, size=200)
r_sig = run("circular: cluster at 60 deg", lambda: DirectionalAnalyzer().analyze(signal, axial=False))
if not isinstance(r_sig, Exception):
    mean = float(getattr(r_sig, "mean_direction_deg", 0.0))
    # circular mean is pi-periodic, so accept 60 or 240
    delta = abs(((mean - 60.0 + 90.0) % 180.0) - 90.0)
    check("cluster at 60 deg: mean direction recovered", delta < 3.0, f"mean_direction_deg={mean:.3f}")
    check("cluster: significant", bool(getattr(r_sig, "is_significant", False)))

# ------------------------------------------------------------------- LDA
print()
print("=== stats.lda ===")
from stats.lda import LDAAnalyzer
from utils.exceptions import ComputationError

rng2 = np.random.default_rng(7)
n_per = 40
X = np.vstack(
    [
        rng2.normal(loc=[3.0, 0.0, 0.0], scale=0.4, size=(n_per, 3)),
        rng2.normal(loc=[-3.0, 0.0, 0.0], scale=0.4, size=(n_per, 3)),
    ]
)
y = np.array([0] * n_per + [1] * n_per)

res_lda = run("LDA on two well-separated groups", lambda: LDAAnalyzer().analyze(X, y))
if not isinstance(res_lda, Exception):
    acc = float(getattr(res_lda, "accuracy", 0.0))
    check("separable groups: accuracy high", acc > 0.95, f"accuracy={acc}")
    ev = np.asarray(getattr(res_lda, "eigenvalues", np.zeros(1)), dtype=float).ravel()
    check("eigenvalues descending", bool(np.all(np.diff(ev) <= 1e-9)), str(ev))
    ratio = np.asarray(getattr(res_lda, "eigenvalue_proportions", np.zeros(1)), dtype=float).ravel()
    check("eigenvalue proportions sum to 1", abs(ratio.sum() - 1.0) < 1e-9, f"sum={ratio.sum()}")

# --- deliberate, well-messaged rejections are the API working
one_class = run("LDA with a single class", lambda: LDAAnalyzer().analyze(X, np.zeros(len(y), dtype=int)))
if isinstance(one_class, ComputationError):
    print(f"  ok    single class rejected cleanly: {one_class}")
else:
    note(f"LDA accepted a single class: {one_class!r}")

# --- an opaque failure on degenerate input IS a finding
identical = run(
    "LDA on identical rows (rank-deficient)", lambda: LDAAnalyzer().analyze(np.ones((20, 3)), np.arange(20) % 2)
)
if isinstance(identical, ComputationError):
    print(f"  ok    rank-deficient input rejected cleanly: {identical}")
elif isinstance(identical, Exception):
    note(
        f"rank-deficient input (identical rows) leaks {type(identical).__name__}: {identical} "
        f"-- a bare numpy IndexError with no explanation, where the rest of the "
        f"package raises a messaged ComputationError"
    )
else:
    print("  ok    rank-deficient input returned a result")

single_row = run("LDA with a single sample", lambda: LDAAnalyzer().analyze(X[:1], np.array([0])))
if isinstance(single_row, ComputationError):
    print(f"  ok    single sample rejected cleanly: {single_row}")
elif isinstance(single_row, Exception):
    note(f"single sample leaks {type(single_row).__name__}: {single_row}")
else:
    print("  ok    single sample returned a result")

few = run(
    "LDA with fewer samples than variables",
    lambda: LDAAnalyzer().analyze(rng2.normal(size=(3, 6)), np.array([0, 1, 0])),
)
if isinstance(few, ComputationError):
    print(f"  ok    n<p rejected cleanly: {few}")
elif isinstance(few, Exception):
    note(f"n<p leaks {type(few).__name__}: {few}")
else:
    print("  ok    n<p returned a result")

print()
if FINDINGS:
    print(f"{len(FINDINGS)} finding(s)")
    for item in FINDINGS:
        print("  -", item)
else:
    print("all invariants held")
raise SystemExit(1 if FINDINGS else 0)
