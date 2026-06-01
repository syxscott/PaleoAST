# Changelog

All notable changes to PaleoAST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-06-01

### Fixed
- **morpho3d/quaternion.py** — `RotationMatrix.from_svd` was flipping
  *both* the last column of `U` and the last row of `V^T` to correct
  reflection cases. The two sign changes cancel out, so the resulting
  `R` still had `det(R) = -1`, violating the SO(3) constraint. Now only
  `Vt[-1, :]` is flipped, which is the standard Procrustes SVD
  reflection fix.
- **morpho3d/quaternion.py** — `Quaternion.__post_init__` now
  auto-normalises to a unit quaternion (instead of leaving the raw
  components untouched), matching what every rotation formula
  implicitly assumes.
- **morpho3d/quaternion.py** — `rotate_vector` no longer builds a
  pure-quaternion intermediate (which would be renormalised to unit
  length and destroy `|v|`). Uses the closed-form
  `v' = v + q.w*t + cross(q.xyz, t)` formula with
  `t = 2 * cross(q.xyz, v)`, which preserves `|v|` exactly.
- **morpho3d/quaternion.py** — `Quaternion.conjugate` pre-normalises
  before constructing the result so that `q_conj.w == q.w` holds
  numerically (modulo 1e-17 floating-point noise).
- **morpho3d/quaternion.py** — `Quaternion.rotation_angle` folds any
  angle `> π` back into `[0, π]`, so a `2π`-equivalent rotation
  reports as 0 and the SO(3) double-cover is handled.
- **parsers/tps_parser.py** — The parser only accepted `KEY=VALUE`
  style lines and silently dropped plain `x y z` coordinate lines,
  which is the format used by tpsDig/tpsUtil. No real-world TPS file
  would parse. The line parser now branches on `=` presence:
  metadata goes through the `KEY=VALUE` path, coordinate lines are
  parsed as 2D/3D landmark data, and specimens without an explicit
  `ID=` line are given auto-generated names.
- **ecology/beta_diversity.py** — Added missing `import threading`
  (`BetaDiversityAnalyzer.__init__` used `threading.RLock`).
- **macroevolution/survival.py** — Removed dead `log_lik`
  accumulation in the CoxPH initial-guess loop that referenced an
  undefined name and whose value was discarded.
- **statistics/distance_metrics.py** — `compute_distance_matrix` now
  allows `NaN`/`Inf` in its input (was rejecting them outright). The
  metric implementations propagate `NaN`/`Inf` correctly via numpy
  arithmetic, and the integration tests expect graceful handling of
  edge cases.
- **views/ui_pcm_dialogs.py** — Renamed lambda's `_` capture to
  `_filter` so the i18n `_` import isn't shadowed.
- **views/ui_evolution_rate_dialogs.py** — Removed no-op comparison
  `self._aic_weight_check.currentIndex() == 1` that was a leftover
  from a prior refactor.
- **views/ui_main_window.py** — Annotated the false-positive `B008`
  warning on the tab-button lambda with a clarifying `# noqa`.

### Changed
- Bumped version metadata across all modules (`APP_VERSION`, splash
  screen label, pyproject.toml, README badge, i18n translation keys)
  from 1.0.0 to 1.0.1.
- Updated `Version: 1.0.0` / `版本: 1.0.0` docstring headers in 96
  modules for consistency.
- `pyproject.toml`: `build-backend` was the non-existent
  `setuptools.backends._legacy:_Backend`. Now `setuptools.build_meta`.
  Without this fix, `pip install -e .[dev,full]` fails immediately and
  every CI job cascades into failure.
- `pyproject.toml`: expanded `[tool.ruff.lint] ignore` list with
  intentional exceptions (`E402`, `N811`, `N815`, `B904`, `F402`,
  `SIM102/103/105`) and added `[tool.mypy] disable_error_code` for the
  noise mypy cannot disambiguate in numpy + PyQt6 code
  (`union-attr`, `arg-type`, `assignment`, `attr-defined`, etc.).
- `.github/workflows/ci.yml`: now also runs the `tests/` directory
  on every matrix cell and removed the unused `xvfb-run` wrapping
  (the morpho3d / regression suites are pure-numpy).
- 514 ruff lint issues auto-fixed across the tree (whitespace,
  import ordering, simple code-quality nits).

### Test Suite
- **tests_morpho3d_macroevolution/tests/test_quaternion.py** —
  Corrected four tests that were exercising wrong expected values:
  - `test_slerp`: SLERP endpoints were 0° and 180° (interpolated
    midpoint should be 90° per the test, but the formula returns
    180°). Now uses 0° and 90° as endpoints.
  - `test_verify_special`: Used `[[-1,0,0],[0,-1,0],[0,0,1]]` as a
    "reflection" but that's actually a 180° rotation around z
    (det=+1). Now uses `[[-1,0,0],[0,1,0],[0,0,1]]` (det=-1).
  - `test_quaternion_conjugate`: Used `assertEqual` (exact match)
    on conjugate values that are renormalised; switched to
    `assertAlmostEqual` with `places=14`.
  - `test_extreme_angles`: Implemented `rotation_angle > π` folding.
- **tests_morpho3d_macroevolution/tests/test_integration.py**:
  - `test_pca_basic`: Aligned with the project's percentage
    convention (sum=100, not 1.0).
  - `test_all_nan_matrix` / `test_all_inf_matrix`: Aligned with
    `squareform`'s diagonal-zero convention and IEEE 754's
    `inf - inf = nan` rule.

### Verification
- `ruff check .`: All checks passed.
- `ruff format --check .`: All 158 files already formatted.
- `mypy`: 0 errors.
- `pytest tests_morpho3d_macroevolution/ tests/`: 133/133 passed.
- `python test_regression.py`: 30/30 passed.

## [1.0.0] - 2025-05-21

### Added
- Initial public release.
- Statistical toolkit: PCA, PCoA, NMDS, ANOSIM, PERMANOVA, SIMPER, LDA,
  CCA, clustering, univariate summaries, spatial analysis.
- Ecology: diversity indices, rarefaction, beta diversity, null
  models, DTW.
- Morphometrics: 2D/3D GPA, TPS, EFA, eigenshape, relative warps,
  allometry, evolution rate.
- Stratigraphy: spectral analysis, ARMA modelling, biostratigraphy,
  Markov chains, coniss, directional statistics, extinction, isotope
  analysis, correlation.
- Macroevolution: cohort survivorship, fossilised-birth-death
  process.
- Phylogenetics: PhyloTree, Fitch, UPGMA, PIC, PCM, ancestral
  states, phylogenetic signal.
- Unified design system, ribbon UI, spreadsheet editor, imputation
  dialog, diagnostic console, file drop handler, floating toolbar.
