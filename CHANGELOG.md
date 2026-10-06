# Changelog

All notable changes to PaleoAST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **A machine-readable inventory of PAST 4's features**
  (`docs/parity/past4_features.json`), transcribed from the official PAST 4
  reference manual: 242 entries across twelve menus, each with its page. The
  aim is that "reproduce everything PAST has" becomes a state that can be
  checked rather than one that has to be judged by eye. No test consumes it
  yet.
- **Abundance model fitting** is now reachable by name
  (`run("abundance_models")`). The four models -- log-series, log-normal,
  geometric and broken-stick -- have been in `ecology/advanced.py` with
  cross-validation tests against R, but nothing pointed a user at them. The
  catalog is 79 analyses in 12 categories.
- **A structural test for dead interactive controls**
  (`tests/views/test_dead_controls.py`). A widget that is built, filled and
  laid out but never asked for its value looks like a choice and silently has
  none.

### Fixed
- **Three controls did nothing while appearing to do something.** All three
  were the same defect, and none was caught by a behavioural test, because
  every dialog still ran and still returned a result:
  - the allometry dialog's **"RMA (reduced major axis)"** combo. Constructed,
    filled with OLS and RMA, added to the layout -- and never read. Picking
    RMA ran the OLS regression and labelled the output RMA. Removed rather
    than wired up: RMA fits error in both axes and so needs measurement-error
    estimates the dialog does not collect, making it an engine change rather
    than a wiring change.
  - the PCA dialog's **"Use correlation matrix"** checkbox. A mirror of the
    Similarity Matrix combo, kept in step by `_on_method_changed`, while
    `get_parameters` read only the combo -- untick the box by hand and nothing
    happened. The combo already states the choice and is what reaches the
    engine, so the mirror went rather than being wired in parallel.
  - the macroevolution dialog's **"Starting consensus from:"** combo, offering
    "Consensus (recommended)" and "First specimen". Never read, and the engine
    has no first-specimen mode to read.

  The new test found the second and third on its first run against a tree
  where the first had already been removed, which is the point of having it.
  It counts a control as alive only when a getter is called on it, or when it
  is passed to something that is not a layout method: an earlier version that
  treated any appearance as a read reported all three as healthy.
- **The figure on screen and the figure you export were not the same
  figure.** The interactive plotters hard-coded `get_color_scheme("default")`
  (Paul Tol's colours) while the R export defaulted to Okabe-Ito, so every
  palette a user chose in Preferences changed the exported file and left the
  on-screen figure exactly as it was -- the worst direction for a mismatch,
  because the export moved and the preview did not. Both sides now read one
  registry in `config/colors.py`, under the names the Preferences dialog
  already stored, and the generated R script is handed that list verbatim.
  The default is Okabe-Ito on both paths.
- **The two Okabe-Ito tables disagreed about the eighth colour.** The in-app
  list had grey where the R list had black, so an eight-group figure showed
  grey on screen and black in the export. Okabe & Ito's eighth colour is
  black; it is now black in the one table that exists.
- **`r_palette` was collected by the Preferences dialog and then
  discarded.** Neither `_get_preferences_state` nor `_apply_preferences`
  mentioned the key, so it was never written to QSettings and never read
  back: choosing anything but the default silently did nothing. Both ends
  carry it now, and it is applied at startup so the first figure drawn is
  already in the chosen palette.
- **`get_color_scheme` returned a default for any name it did not
  recognise.** No error, no warning. A typo -- or a name copied from the R
  side, where the palette is `okabeito` rather than `colorblind` -- produced
  a figure in a different palette with nothing to indicate it. An unknown
  name raises now. The magic name `"default"` is gone too: it resolved to one
  palette while the application defaulted to another, which is the same trap
  wearing a different hat.
- **The generated R script asked for packages it did not need.**
  `brewer.pal()` and `viridis(8)` came from RColorBrewer and viridisLite
  behind `requireNamespace`, so the colours could differ between machines --
  and between the preview and the render on any machine missing one. The
  colours are injected instead, which makes the script's claim that it cannot
  die on a missing package true rather than aspirational.

## [1.1.1] - 2026-10-06

Thirteen analyses brought over from PAST3, a way to reach all of them
without a hand-written dialog each, and scripting. Two of the fixes change
numbers that may have been published.

### Added
- **Mantel test and partial Mantel test** (`stats/mantel.py`). The partial
  form follows Guillot & Rousset (2013) and regresses both responses on the
  control before testing the residuals; permuting the raw vectors and
  applying the textbook partial-correlation formula is still the common
  implementation and is anti-conservative. The Monte-Carlo Z of Legendre,
  Fortin & Borcard (2015) is reported alongside the p-value.
- **K-means with an elbow curve** (`stats/clustering.py`). Per-method
  bounds so `n_permutations`-style parameters do not silently become 99.
- **Paired nonparametric tests**: sign test and Wilcoxon signed-rank
  (`stats/univariate.py`). The pairing rule moved into one helper so the
  t-test and the rank tests cannot disagree about which observations form
  a pair.
- **Detrended correspondence analysis** (`stats/detriding.py`), with the CA
  it detrends checked against the textbook SVD of Greenacre (1984).
- **Two-way ANOVA with interaction, repeated-measures ANOVA, intraclass
  correlation and contingency chi-square** (`stats/design_tests.py`).
- **Six growth curves** (`models/growth_models.py`): von Bertalanffy,
  Gompertz, Michaelis-Menten, logistic, Gaussian growth curve and
  sinusoidal, fitted together and ranked by AICc.
- **Cyclostratigraphy** (`stratigraphy/cycles.py`): autocorrelation with
  Bartlett bands, AR(1) prewhitening, REDFIT, multitaper spectral
  estimation, cross-correlation, autoassociation, Mann-Kendall with Sen's
  slope, runs test, and a Milankovitch forcing table.
- **A canonical time axis** (`stratigraphy/timeaxis.py`). Three
  incompatible age representations existed; this converts between them
  without rewriting the three.
- **Moran's I, grid interpolation, nearest-neighbour point-pattern
  statistics and spherical statistics** (`stats/spatial_stats.py`).
  Kriging is deliberately absent and says so -- it needs a fitted
  variogram model, and an IDW formula wearing the name would return a
  surface that looks defensible and is not.
- **A live analysis catalog** (`plugins/catalog.py`): 78 analyses in 12
  categories, addressed by name.
- **A generated parameter form** (`views/analysis_runner.py`). The form is
  built from each analysis's signature, so the fields cannot drift from
  what the code accepts. 96.1% of the 304 parameters across the catalog
  render automatically; the rest name their type and point at the console.
- **Scripting** (`utils/script_session.py`, `views/script_console.py`,
  `utils/script_cli.py`). Python against the live spreadsheet and every
  registered analysis, in a window or headless via `paleoast-run`.

### Fixed
- **PERMANOVA's sums of squares did not sum to the total of the distance
  matrix it was given.** The divisors were `n - 1` and `n_g - 1`; the
  correct ones are `n` and `n_g`. Both terms were inflated, by *different*
  factors, so the error did not cancel: on the cross-validation fixture
  R-squared came out 0.8539 where `vegan::adonis2` gives 0.8636, and F
  81.83 against 88.67. The permutation p-value would never have shown it --
  scaling both terms leaves the permutation distribution's shape alone.
  Now exact against adonis2.
- **The Ricker wavelet scalogram reported periods 3.85x too small.** The
  transform was computed with the Mexican hat wavelet but its frequency
  axis was labelled with the Morlet Fourier factor, so a 385 kyr
  obliquity band was reported as 100 kyr. Any cycle identification read
  off that plot was affected.
- **Six catalogued analyses were never in the executable.** The catalog
  references its targets by name and PyInstaller follows imports, not
  strings, so anything not on a static import path was left out of the
  build while every local check passed. `PaleoAST.spec` now derives its
  hidden imports from the catalog.
- **Two tests asserted the wrong PERMANOVA divisors and cited
  `vegan::adonis2` as doing the same.** Two tests agreeing with each
  other and quoting a reference that says otherwise is how the divisor
  bug survived. The replacement assertion is stated against the distance
  matrix rather than the formula.
- **Six rpy2 accessors in the R cross-validation suite**: a Newick string
  passed as a file path, a character vector passed to a float coercion, a
  tuple index on a data.frame, a component that adonis2 does not have, a
  row labelled `Model` rather than after the term, and `pic()` read as a
  list when it returns a named numeric.
- **The lint gate now passes end to end.** `ruff check` had 96 findings,
  all in `tests/`, and `ruff format --check` had never run successfully --
  CI stops a job at its first failing step, so the format check behind a
  red lint step was never reached. Both are pinned to 0.16.10, since an
  unpinned formatter gate moves whenever ruff publishes a release.

### Changed
- **The MSI's desktop shortcut is now genuinely opt-in.** The component
  carried a comment saying it was opt-in, but it sat in the same component
  group as the Start Menu shortcut under a single `Level="1"` feature, so
  every install -- including a silent `/qn` one -- put an icon on the
  public desktop. It is now its own feature at `Level="101"`, above the
  default `INSTALLLEVEL` of 100: skipped unless named explicitly, and
  reachable from the installer by choosing *Advanced*.

## [1.1.0] - 2026-10-06

First release with the editable-R plotting path and the dark-theme
corrections. The figure is the deliverable, so most of what is fixed here is
what a figure was doing.

### Added
- **R export of ordination plots.** `visualization/r_export.py` writes an
  editable ggplot2 script plus its data as CSV, from the PCA scores or the
  scree plot. The script stands alone: it needs R and ggplot2, not PaleoAST.
- **R rendering.** `visualization/r_render.py` locates `Rscript` (searching
  the D: drive as well as C:), runs the script as a child process, and finds
  the figures it produced by diffing the directory -- so a renamed output is
  still found, and the in-app preview is best effort rather than a
  requirement. Two actions in the File menu, deliberately separate:
  *Export PCA as R Script* regenerates and asks before overwriting a script
  you have edited; *Re-run R Script* only ever executes the file on disk.
- **R settings** in Preferences: `Rscript` path with a Detect button, ggplot2
  theme, base size, output format, timeout, and group colour palette.
- **Scientific colour palettes.** Okabe & Ito (2008) is the default:
  separable under deuteranopia, protanopia and tritanopia, and in greyscale.
  ColorBrewer Dark2, a greyscale set for a photocopied journal, and viridis
  for a magnitude rather than a class. All are base R, so a missing suggested
  package cannot break the generated script.
- **A marker shape channel**, so groups stay separable when colour is not
  enough -- in greyscale, or with more groups than the palette has colours.
- **Windows installer**, as a portable `.exe` and an `.msi`.

### Fixed
- **A score plot silently lost its points.** The scale indexed the palette by
  group count: `PALETTE[seq_along(levels(...))]`. With more groups than
  colours -- nine habitats, or a set of time bins, which is ordinary
  stratigraphic data -- the surplus indices were `NA`, ggplot removed those
  rows, and R still exited 0 with a PDF on disk. The palette is recycled and
  the shortfall reported.
- **The 95% ellipses were drawn in the wrong colour.** `stat_ellipse` does
  not preserve the colour aesthetic, so naming it in the layer's aes gave
  ellipses that did not match their points, with ggplot reporting "the
  following aesthetics were dropped during statistical transformation". The
  region is now computed in base R and drawn with `geom_path`.
- **Small groups no longer emit one warning each.** The minimum sample size
  for a 95% region is applied where the ellipse is built, and the skipped
  groups are named. Four points is the technical minimum, but the region has
  no residual degrees of freedom at that size and the curve expands until it
  spans the panel.
- **Dark theme legibility.** Plot sample labels were invisible: the theme
  repaint covered the axes, ticks, labels, title and spines but never
  `ax.texts`, which is where `annotate()` puts the labels, and an artist's
  colour is fixed when it is created. `ColorPaletteDark` also inherited all
  six accents from the light palette, so a blue chosen for white backgrounds
  was used as a foreground on near-black -- the selected ribbon tab measured
  1.44:1 against a 4.5:1 requirement. A WCAG sweep of the pairs the UI
  composes went from 12 failures to 0.
- **Icons follow the theme.** `VectorIconEngine` hardcoded eleven flat-UI
  colours, including a default pen that disappeared on a dark button, and
  `RibbonButton` never redrew its icon on a theme switch.
- **`.dat` group labels are no longer discarded.** PAST's `{Group}` lines
  were parsed and counted, then dropped when the matrix was built, so the
  file loaded with a correct row count and every grouping-aware method ran as
  if it were a single group. A partial grouping is refused rather than
  padded.
- **Error messages carry their arguments.** Ten of them passed an operation
  name, and sometimes the underlying error, to `str.format` while the English
  catalog entry had no placeholder, so the arguments were silently
  discarded. The Chinese for the same messages read as machine output and no
  longer does.
- **The R code that draws a figure is one keystroke away from being wrong.**
  A leading `+` on its own line is a unary plus in R, and an omitted `+`
  splits the expression; both still exit 0 while `ggsave` writes an
  incomplete plot. The layers are joined with a trailing operator, and
  `validate_r_script` checks it.

### Changed
- **The design system is enforced rather than documented.** `Spacing`
  declared a 4px grid that twelve hardcoded `padding` values ignored, three
  of them off-grid; they are tokens now. The guard reads the stylesheet
  *template*, because the generated sheet cannot tell a token from a literal.
- **The ribbon distinguishes its actions.** The six data transforms shared
  one gear, and the Diversity and Stratigraphy groups were one icon per
  group, so five glyphs covered 29 buttons -- an icon repeating the tab name
  while costing 24px a button. The transforms use their operator; the groups
  with no honest symbol to draw drop the icon.
- **CI jobs have a time limit.** None of the seven had one, so a wedged job
  held a runner until GitHub's 6-hour default.
- **Static analysis is clean.** Mypy went from 11 errors to 0, and Ruff from
  376 findings to 96, none of them in application code.

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

> **Superseded on 2026-10-01 — do not read these numbers as current.**
> Re-running them against `79cbecd` gave:
> - `ruff check .` → **375 errors**, not "all checks passed"
> - `ruff format --check .` → **103 files** would be reformatted (this repo has
>   332 Python files, not 158 — the count below was never right for this tree)
> - `pytest tests tests_morpho3d_macroevolution` → **2365 collected** (5 skipped),
>   not 133
> - `python test_regression.py` → **file does not exist**; it was moved to
>   `scripts/smoke_check.py`
>
> The original claims are kept below for history only.

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
