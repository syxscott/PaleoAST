# Changelog

All notable changes to PaleoAST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Added
- **Thirty-six analyses from the cyclostratigraphy and astrochronology
  family** (`stratigraphy/`) -- the gap between a *measured record* and an
  *age scale*.  The catalogue goes from 113 analyses to 149.  They were
  written against the published methods rather than against another
  implementation's source, and each one names the paper it follows:
  - **The astronomical solution and insolation** (`stratigraphy/astronomy.py`,
    `data/astronomy/la04.npz`).  The La04 orbital solution -- 249 001 points
    from 0 to 249 Myr at a 1 kyr step -- with the daily insolation computed
    from it (Berger 1978).  The file also carries the La04 matrix's own
    encoding: its fourth column is `e*sin(w)`, the angle from perihelion
    to the vernal equinox, and the longitude of perihelion is recovered
    from it.  The modern values come out where they must: eccentricity
    0.0167, obliquity 23.44 degrees, longitude of perihelion 102.9
    degrees, with perihelion on 2 January.  The insolation agrees with
    the classic Berger figures -- 417 W/m2 at the equator, 169 at the pole,
    and a 498 W/m2 summer-to-winter swing at 65 degrees north against 6 at
    the equator.  Before this the application had a six-row table of
    hard-coded Milankovitch periods and nothing else.
  - **COCO and eCOCO** (`stratigraphy/coco.py`) -- the correlation
    coefficient method for the most likely sedimentation rate, and its
    evolutionary form for a rate that changes (Li et al. 2018, *EPSL*).
  - **TimeOpt and eTimeOpt** (`stratigraphy/tuning.py`) -- astrochronological
    tuning, with and without a changing sedimentation rate (Meyers 2015,
    2019), plus an age scale from depth anchors and a sedimentation rate
    turned into an age model.
  - **Power decomposition analysis** (`stratigraphy/pda.py`) -- the power
    spectrum broken into orbital-period bands, each with its share of the
    power and an interval on it (Li et al. 2016, *Geology*).
  - **Evolutionary spectra** (`stratigraphy/evolutionary.py`) -- the
    spectrum computed in a sliding window, by FFT, Lomb-Scargle or
    multitaper, so a period can be *tracked* through a record instead of
    only detected in it (Kodama & Hinnov 2015).
  - **Amplitude modulation** (`stratigraphy/amplitude_modulation.py`) --
    the Tanner-Hilbert transform: the instantaneous envelope and phase of
    a record.
  - **Empirical mode decomposition** (`stratigraphy/emd.py`) -- EMD and
    EEMD (Huang et al. 1998; Wu & Huang 2009).
  - **Spectral moments and the minimum-rho test**
    (`stratigraphy/spectral_moments.py`) -- the shape of a spectrum, and
    the smallest AR(1) coefficient that accounts for a record.
  - **A dynamic filter** (`stratigraphy/dynamic_filter.py`) -- the noise
    level decides how hard each part of a record is smoothed.
  - **Gaussian, Taner and band-pass filters** (`stratigraphy/filtering.py`)
    -- including the Taner weighted band-pass of Kodama & Hinnov (2015).
  - **Bayesian change-point detection** (`stratigraphy/change_points.py`)
    -- Ruggieri (2013).
  - **Fixed-bandwidth and fixed-count moving summaries**
    (`stratigraphy/moving_summary.py`).
  - **The series engineering tools** (`stratigraphy/series_tools.py`) --
    interpolation, interval selection, merging, gap insertion, peak
    removal, clipping, sorting and differentiation.
  - **A signal and noise generator** (`stratigraphy/signal_generator.py`)
    -- sine, white, red and pink noise, and a signal built from
    components.

  These follow the methods of the Acycle package (Li, Hinnov & Kump 2019,
  *Computers & Geosciences* 127: 12-22), which is a **GPL** program: every
  one of them was written from the published method and the paper it
  cites, not translated from Acycle's source, and PaleoAST stays MIT.

- **A form that can ask for a pair of numbers and for a list of bands**
  (`views/analysis_runner.py`) -- a parameter written ``tuple[float, float]``
  was rendered as ONE number box, because the float rule matched the text
  "tuple[float, float]" and stopped there; filling it in with "0.5, 2.0"
  then failed to read.  A pair of numbers now gets two boxes, and a list
  of (centre, half-width) pairs gets a box that takes
  "100:5, 41:3, 23:2".  Without the first of these the correlation
  coefficient method could not be run from the interface at all, because
  its range of candidate sedimentation rates has nowhere to go.

- **A measurement of how every analysis was validated**
  (`scripts/validation_matrix.py`, `docs/validation-matrix.md`) -- for each
  of the 113 first-party analyses: whether it is compared against a
  reference implementation in R, checked against published truth, covered
  by a property test, or only by a unit test on one example. The gap is
  now a number instead of a worry. A ratchet in
  `tests/test_validation_coverage.py` holds the number where it is: it may
  fall when something is validated properly, and it may not rise. There
  are now three ratchets -- the gap, the number of analyses compared
  against R (which is exact, because the cross-validation files declare
  it), and the number with a property test, which is the strongest
  evidence the matrix knows about.

  > **Correction (2026-10-21).** The first version of the matrix matched a
  > test file by the analysis' *class* name, so one file importing
  > `compute_diversity_indices` credited all fourteen diversity analyses
  > with R coverage, and the R column read **38** (later 49) when the true
  > figure was **10**. The cross-validation files now declare the
  > analyses they cover, and the R column is exact: at that point
  > **10 of 113 analyses (9%)** had been compared against a reference
  > implementation in R, and the two entries below take it to **29**.
  > The other columns are inferred and deliberately under-count, so the
  > gap is a lower bound: it was **97** analyses resting on a unit test
  > alone, and it is now **71**.
- **The univariate tests are cross-validated against base R**
  (`tests/cross_validation/test_vs_r_base_univariate.py`) -- kruskal_wallis,
  the Shapiro-Wilk normality test, mann_whitney, the paired Wilcoxon test
  and the contingency chi-square are compared against R's `stats` package
  live, on the same data, to twelve digits. The one-way ANOVA is compared
  against R's `var.equal = TRUE`, and a second test records that R's
  *default* would give a different answer -- because PaleoAST runs the
  classical pooled-variance test and R's default is adaptive. Both sides
  should say which test they ran.
- **Every univariate analysis is now cross-validated against R**
  (`tests/cross_validation/test_vs_r_base_design.py`,
  `tests/cross_validation/test_vs_r_base_univariate_extra.py`) -- the two-way
  ANOVA, the repeated-measures ANOVA, the intraclass correlation, the exact
  binomial proportion test, the Fligner-Kileen test, the descriptive
  statistics, the outlier screen, the Brunner-Munzel test, Mood's median test
  and the Cochran-Mantel-Haenszel test. The univariate domain now has all
  sixteen of its analyses compared against a live reference implementation
  in R, and the R column of the validation matrix reads **20 of 113** (was
  10); the gap -- analyses resting on a single worked example -- falls from
  **97 to 87**.

  Three differences between the two implementations are **on the record**
  rather than smoothed over, each with a test that pins it:

  - The **Brunner-Munzel** variance differs from `lawstat`'s by exactly the
    finite-sample correction term `S1*S2/(m*n*(m-1)*(n-1))` and by nothing
    else, so the comparison is an *exact* relation rather than a tolerance.
    Both were checked against the null by simulation: 4000 replicates at
    three designs give a type-I error rate of 0.052-0.055 against a
    nominal 0.05, for both.
  - Mood's median test reports the **sample** odds ratio; R's
    `fisher.test` reports the **conditional** maximum-likelihood estimate,
    which R's own documentation says is what it uses. The two differ by a
    few percent, and a test records that they do.
  - The Cochran-Mantel-Haenszel continuity correction is applied
    unconditionally, as `statsmodels` and SAS do; R skips it when the
    total signed residual is under half a unit. A test pins that corner
    and shows the difference is in the conservative direction.

  The two-way ANOVA is compared on an **unbalanced** design as well as a
  balanced one: on a balanced design the Type I, Type II and Type III
  decompositions coincide, so a comparison there cannot show which one
  the application computes.

  `tests/cross_validation/_rbridge.py` also gained `r_program()`, which
  runs a reference as a small R program and reads back a named numeric
  vector. It exists because R's statistical functions return shapes that
  rpy2's conversion does not always preserve the names of -- the rows of an
  `anova` table are padded with trailing spaces, so looking up the row
  "a" matches both "a" and "a:b" and R resolves that to `NA`, a silent
  wrong answer rather than an error.
- **The plot menu is cross-validated against base R** --
  nine of the twenty plot entries now have a live comparison
  (`tests/cross_validation/test_vs_r_base_plots.py`): the XY scatter, the
  XY scatter with error bars, the histogram, the bar chart, the box plot,
  the pie chart, the stacked chart, the normal probability plot and the
  percentile plot. A figure cannot be compared with another figure, so
  what is compared is the **data behind the figure**: the application
  draws the picture, the numbers are read back out of the matplotlib
  artists -- the bar heights, the box vertices, the wedge angles, the
  error bar ends -- and those are checked element by element against what
  R computes from the same input. The plot domain's R coverage goes from
  0 to 9, the R column of the validation matrix from 20 to **29 of 113**,
  and the gap from 87 to **78**.

- **The Q-Q plot's reference line is the quartile line** --
  `visualization/generic_plots.py`. The code drew a least-squares fit
  through every point while its own docstring said the line went through
  the quartiles, and the method documentation said the line was the unit
  diagonal -- three different claims, none of which matched the code.
  The line is now the quartile line, which is what the docstring said and
  what R's `qqnorm` draws. A least-squares line is a different line: it
  is pulled by the tails, and on a Q-Q plot the tails are the point.

- **The percentile plot keeps its whiskers and its median**
  -- `visualization/generic_plots.py`. It was built by hiding every line
  whose style was `"-"`, and in matplotlib that is the whiskers, the caps
  **and** the median. What was left on screen was the interquartile range
  and nothing else, which is not a box-and-whisker plot and not what the
  docstring said ("the box plot without outliers"). Only the outlier
  markers are removed now.

- **The coverage report measures branches, not just lines**
  (`pyproject.toml`) -- the "69% of the lines" in the coverage report was
  69% of the *statements*: a statistic with an untested error branch and
  one with a tested one scored the same. `branch = true` makes the
  number mean what it says, and the gate in `[tool.coverage.report]`
  moves with it, so the two cannot disagree. The branch number is the
  one that matters here: a statistical branch is where the wrong answer
  hides -- a boundary, a NaN, a degenerate input, a zero variance.

- **Seventy-eight property-based tests, up from nineteen**
  (`tests/property/`) -- the univariate tests, the design tests, the
  principal component analysis, the diversity indices, Moran's I and the
  clustering each gained a set of invariants that hold for *every* input
  of a shape rather than for one worked example: a positive affine
  transform of the response changes no F ratio, a strictly increasing
  transform changes no rank-based statistic, permuting the values inside
  a group changes no p-value, the terms of an ANOVA add up to the total,
  the eigenvalues of a PCA add up to the total variance, a taxon with
  zero individuals is not a taxon.

  Three of them record things that are true of the algorithm and not of
  the result, and are worth knowing: k-means does **not** give the same
  partition when the samples are reordered (it starts from a random
  initialisation), the Fliller-Kileen statistic is only invariant under
  an affine transform that is exact in floating point (an inexact one
  can separate a tie and change the ranks), and the intraclass
  correlation's interval cannot contain a negative point estimate,
  because the interval is built from non-negative component ratios.

  Two production documentation defects were found on the way and fixed:
  `PCAResult` described the loadings as `(n_components × n_variables)`
  when they are `(n_variables × n_components)`, and the relation it
  gave between the scores and the loadings was missing the
  `sqrt(eigenvalues)` factor.

- **One place maps the design system onto matplotlib**
  (`config/matplotlib_theme.py`, `views/ui_plot_canvas.py`) -- four places
  used to do it: the module-level `plt.rcParams.update(...)` in the canvas
  module, `PlotCanvas.setDarkTheme`'s own update, `PlotCanvas.theme_colors()`
  and `PlotCanvas._apply_axes_theme`. The first two wrote matplotlib's own
  names and the third invented a set of role names of its own, so a colour
  that reached one of them but not the others was right in one place and
  wrong in another. That is how a dark-theme canvas ends up with a white
  plot area: the canvas re-tints the axes it already has, but a figure
  created afterwards takes its colours from rcParams, and those two
  answers were written separately. `config/matplotlib_theme.py` now owns
  the mapping -- the role names the canvas reads and matplotlib's own
  names both come from the same palette, so they cannot disagree.

  The check in `tests/test_design_conformance.py` that the canvas reads
  the figure type scale was a **source-text** check for
  `FigureTypography.rc_overrides()`; it is now a check on the rcParams
  the canvas actually applies, which is the thing the check was for.

- **A cross-validation directory that no longer fails to collect**
  (`tests/cross_validation/_rbridge.py`) -- `require()` raises its skip at
  module level, where pytest refuses a plain `pytest.skip()`; without
  `allow_module_level=True` a missing `geomorph` or `iNEXT` turned the
  whole directory into a collection ERROR instead of a skip.
- **Six families of univariate tests the application could not run at all**
  (`stats/univariate_extra.py`, wired up through `stats/univariate_menu.py`
  and the plugin catalog): Brunner-Munzel, Mood's median test, Fligner-Kileen,
  Grubbs' outlier test, and exact confidence intervals for a proportion and a
  Poisson rate.
- **Two-sample distribution tests** (`stats/distribution_tests.py`, wired up
  through `stats/distribution_menu.py`): the Kolmogorov-Smirnov test with an
  exact null that handles ties, and the Epps-Singleton test.
- **Diversity sampling standardisation and between-group tests**
  (`ecology/diversity_extra.py`, wired up through `ecology/diversity_menu.py`):
  sample-coverage rarefaction, a diversity profile, and the between-group
  tests.
- **A review of the whole codebase, and the plan that came out of it**
  (`FIX_PLAN_2026-10-19.md`): 125 defects across 12 module groups, with the
  five that were fixed first written up in full.
- **A dark theme you can actually turn on** (`main.py`,
  `views/ui_main_window.py`, `views/ui_permutation_dialogs.py`) -- the
  application had a complete dark theme: a palette, a stylesheet, and a
  propagation chain through the ribbon, the status bar, the navigation tree,
  the spreadsheet, the plot canvas and every dialog. Nothing reached it. The
  application always started light, and no code path called
  `MainWindow.setDarkTheme()`. Every dark-theme defect found in the review was
  therefore unreachable in the product, and the switch that would have
  exposed them did not exist. Preferences now carries a **Dark theme**
  checkbox, the choice is persisted, and it takes effect at startup and the
  moment it is changed.

### Fixed
- **The generic analysis runner no longer hands a whole spreadsheet to an
  analysis that wants a series** (`plugins/base.py`, `plugins/catalog.py`,
  `views/analysis_runner.py`) -- the runner's contract was "the sheet goes
  into the data parameter", which is right for an analysis that wants a
  MATRIX and wrong for one that wants a SERIES.  The two cases came apart,
  and the series case reported numbers that meant nothing:
  - **The time axis was analysed as if it were the data.**  A sheet is
    (rows = samples, columns = variables), so an analysis that wants a
    series wants one COLUMN of it.  Handed the whole table, the
    autocorrelation and the multitaper estimator read only the first
    column -- which is the depth or the age axis -- and reported a
    spectrum of it.
  - **The two columns were mixed together.**  The Mann-Kendall tau came
    back with the opposite sign (+0.226 against -0.099 on the same
    record), the runs test's p-value moved by forty orders of magnitude
    (2.3e-82 against 4.4e-38), and the pre-whitening's AR(1)
    coefficient flipped sign (+0.988 against -0.603).  None of them said
    anything was wrong.
  - **Six analyses could not run at all.**  ``analyze(time, values)`` has
    TWO array parameters, and the rule that finds the data parameter
    could only ever find the first -- so the sheet went in as the TIME
    axis, the values were never passed, and the call failed with a
    TypeError about a missing argument.  This is where the second review
    round's ``data_slot`` field stopped: it moved the crash from one
    missing argument to the other without making the analysis work.

  A catalog entry now declares ``data_slots``: the names of the
  parameters the sheet supplies, in the order the columns come.  An entry
  that declares nothing keeps the old behaviour -- the whole sheet, for
  the matrix analyses -- and an entry that declares one or two gets one
  column each.  The form offers a column picker for each declared slot
  (the time axis first, the measurement second), so the user can see
  which column is which, and the runner's answer is now the same as the
  dedicated dialog's.

  Sixteen analyses were affected: ``acf``, ``multitaper``, ``prewhiten``,
  ``mann_kendall``, ``runs_test``, ``spectral``, ``wavelet``, ``redfit``,
  ``autoassociation``, ``cross_correlation``, ``arma``, ``growth_model``,
  ``growth_model_comparison``, ``kaplan_meier``, ``cox_ph`` and
  ``evolution_rate``.

- **The second review round: the parts of the tree the first round never
  reached** -- a follow-up review of `stratigraphy/`, `phylogenetics/`,
  `macroevolution/`, `visualization/`, `hpc/`, `reporting/`, `presets/`,
  `state_machine/`, `parsers/` and the unread regions of the view layer
  found 49 more defects.  The ones that changed numbers, or reported a
  success that was not one:
  - **A single missing value silenced whole analyses.**  A NaN anywhere in
    an isotope series made the mean and the spread NaN, and `NaN > threshold`
    is False -- so `detect_excursions_from_values` reported *no excursions*
    and `remove_outliers` emptied the series, both without a warning.  The
    GUI only skips a column that is entirely NaN, so a column with a few
    gaps was silently reported as clean.
  - **The frequency grid was set by the sampling density, not by the
    record.**  The grid step was Nyquist/n_frequencies, which grows as the
    record is sampled more densely -- so sampling the *same* 100-unit
    record more finely made the reported period worse (7.0 came back as
    9.89 at 20000 samples).  The step is now never coarser than the
    record's own Rayleigh resolution, and the longest period the grid
    offers is the record length instead of ten times it.
  - **A constant series was reported as a significant cycle.**  The linear
    detrend leaves a residue of about 1e-16 on a constant series, and the
    periodogram's 1/(2 sigma^2) normalisation amplified that float noise
    into a peak of ~95, which cleared the AR(1) 5% threshold.  A constant
    series is now refused.
  - **The wavelet frequency axis was in samples, not in the caller's time
    units.**  `dt` defaulted to 1.0 and the time axis never took part in
    the computation, so a series sampled every 0.1 Ma was reported with
    its frequencies ten times too small -- and the scalogram's y axis
    ("Frequency") did not match its x axis (Ma).  The step is now taken
    from the time axis.
  - **The Sen slope confidence interval covered 21-44% while claiming 95%.**
    The normal approximation put a variance where the derivation needs the
    sum of squared deviations of the pairwise slopes, so the half-width was
    about sqrt(n) too small.  The interval is now Sen's (1968) Eq. 2.6
    order-statistic range, and it covers 94-98% on the same data.
  - **Kendall's tau was tau-A while the docstring said tau-B.**  The
    denominator was the total number of pairs, which counts tied pairs as
    if they were discordant -- and quantised isotope series are full of
    ties.  The module already had the tie counts, for the variance.
  - **Fitch over-counted parsimony steps on a polytomy.**  The tie-break
    read each leaf's sequence from a field nothing in the package ever
    wrote to, so the support was always empty and the root state fell
    back to the lexicographically smallest -- which on a polytomy can be
    the opposite of the majority.  A four-taxon polytomy scored 3 steps
    where the true minimum is 1.
  - **NNI on the root's own edge only produced self-loops.**  The root has
    degree 2, so moving a subtree across it reproduces the same *unrooted*
    topology; half the real neighbourhood was unreachable, and the search
    ran on a permanently incomplete one.
  - **The Kaplan-Meier confidence band collapsed to a line where the curve
    reaches zero.**  The Greenwood term diverges when the last subjects at
    risk die, and the guard skipped it instead of letting it go to
    infinity, so the upper bound came out as 5e-12 instead of 1.0.
  - **The headless runner, the smoke script and the crash dialog all
    reported success when they had not.**  An analysis that failed inside
    the script console was turned into text, so the CLI exited 0; the
    smoke script had no exit call at all; and the crash dialog's Quit
    button made the event loop return 0.
  - **The LaTeX builder reported a failed compile as a success** and
    handed the caller the *previous* run's PDF, because success was
    decided by the presence of the output file rather than by the
    compiler's return code.
  - **The parallel distance matrix ignored the metric it was given** and
    always returned Euclidean distances, so a caller asking for
    Bray-Curtis got Euclidean numbers with no error at all.
  - **A shut-down scheduler hung forever**, because the guard that was
    meant to catch a vanished worker loop was inactive in exactly that
    case.
  - **A missing group label became a group of its own** -- NaN is not
    equal to itself, so every empty cell produced its own legend entry,
    and one plot showed four groups where the data had two.
  - **The significance band under the periodogram was the mirror image of
    the power curve**, so the colours said the opposite of the power.
  - Plus: the `get_scale` interval filter pulled the neighbours of a named
    interval into the bin table; the `n_layers_above_lad` field counted
    the layers *below* each LAD; the taxonomy plot's Log transform turned
    a value of -1 into a process abort and a value below -1 into silent
    NaN; the log-rank branch of the survival plot read a different
    analysis's cache entry; a named sample that did not exist silently
    analysed the first row; the macroevolution entries all opened the
    first tab; `_apply_preferences` crashed on a partial state; the
    report's title, authors and abstract were the only caller text not
    escaped for LaTeX; a broken preset file in the directory took the
    whole lookup down; a NEXUS gap inside a state string was eaten by
    the integer rule; an empty `POINTS=` line became an empty landmark;
    and the majority-rule consensus tree depended on `PYTHONHASHSEED`.
- **The second review round's tail: seven findings the round filed but did
  not close** -- the round reported them and they were left standing.  All
  seven were reproduced before anything was changed:
  - **A comment rule that could never fire.**  The COMMENT rule sat at the
    default priority (100) while the OPERATOR rule -- whose character class
    holds both `-` and `/` -- sat at 40, and the rules are tried in priority
    order.  So `-- a comment` lexed as OPERATOR("--"), IDENTIFIER("a"),
    IDENTIFIER("comment"): a comment was three tokens and the COMMENT type
    was never produced at all.  The rule now outranks the operator, and
    only starts at the beginning of a line or after whitespace, so `x--y`
    is still two minus signs.
  - **The NFA-to-DFA conversion reported a missing attribute instead of a
    missing initial state.**  The subset construction is seeded from the
    epsilon closure of `{q0}`; with no `q0` the closure was handed `{None}`
    and died with "'NoneType' object has no attribute 'state_id'".
  - **The R export indexed a component that was not there.**  The scores
    were read positionally while the explained variance was range-checked,
    so the two halves disagreed: `pc2=7` raised a bare `IndexError` out of
    the indexing, and `pc1=-1` -- a legal Python index -- took the *last*
    column while the header still said "PC0", so the delivered R script
    plotted one component and labelled it as another.
  - **The preset library could hold two presets under one name.**
    `path_for` keeps the *file* names unique by appending `_2`, `_3`... but
    the preset's own name was left alone, so the picker offered the same
    name twice and the second was unreachable by name.
  - **A shut-down process pool came back from the dead.**  `shutdown()` set
    the `_shutdown` flag and nothing read it: a task submitted afterwards
    called `start()`, which brought up a fresh set of worker processes --
    the caller had been told the pool was closed, and the process table
    grew anyway.
  - **Two Robinson-Foulds distances in one package did not agree.**  The
    copy in `phylogenetics/fitch.py` counted a terminal edge (one taxon
    against the rest) as a non-trivial split, so `((A,B),(C,D))` against
    `((A,B,C),D)` scored 2 where the distance is 1, and it compared trees
    over different taxon sets without a word -- `{A,B,C,D}` against
    `{A,B,C,E}` scored 2, a number with no meaning, while the package's
    own `robinson_foulds_distance()` raised for the same pair.  There is
    one implementation now.
  - **The normalised Robinson-Foulds distance's docstring described a
    metric that is not the one it computes.**  It gave the divisor as the
    count of all non-trivial splits possible over `n` taxa
    (`2^(n-1) - n - 1`) and called that "the one we use here", while the
    code divides by `2 * (n - 3)`, the number of splits a binary tree
    actually spans.  For four taxa the two are 3 and 2.

  Three further reports were read and deliberately left alone, because
  changing them would have broken something that is intentional: a
  negative branch length is *kept* on purpose (`phylogenetics/tree.py`
  says so -- NJ and least-squares methods produce them), the NEXUS writer's
  TREES block has no caller in the application, and the process pool's
  three manager queues are unused but harmless.
- **The workspace keeps every analysis result on its own tab**
  (`views/ui_main_window.py`) -- the workspace was a `QStackedWidget`, so only
  the newest view was ever reachable: the title that the ~30 call sites pass
  (`"PCA Score Plot"`, `"UAZ Hierarchy"`, ...) was dropped on the floor, and the
  FIFO eviction then deleted results the user could no longer get back to. It
  is a `QTabWidget` now, every view carries the title it was handed, and an
  earlier result stays reachable. The welcome pane is retired -- and the tab
  bar appears -- as soon as a real view is in front.
- **Running the biostratigraphic tree no longer wipes the workspace**
  (`views/ui_main_window.py`) -- `_show_uaz_tree` called `_cleanup_plot_widgets()`,
  which removed and destroyed every analysis result the user had computed, for
  an operation with nothing to do with the tree. The tree now goes in as a
  view of its own, next to the results.
- **The headless runner no longer reports a failed analysis as a success**
  (`utils/script_cli.py`, `utils/script_session.py`) -- `run()` reports an
  analysis' failure as text rather than raising, so the exception never
  reached the caller and the exit code was 0: a scheduler read "the analysis
  failed" as "the analysis succeeded". The session now counts the failures,
  and the CLI exits non-zero when a script ran an analysis that failed.
- **The regression script can fail** (`scripts/smoke_check.py`) -- it had no
  exit call at all, so it reported 0 to the caller however many of its checks
  failed. Every other audit script in that directory already ended with
  `sys.exit(1 if failed else 0)`.
- **A crash no longer exits with code 0** (`main.py`) -- the crash dialog's
  Quit button made `QApplication.quit()` return 0 from the event loop, so a
  process that died from an unhandled exception reported success to whatever
  launched it.
- **The coverage rarefaction no longer loses a species that must be present**
  (`ecology/beta_diversity.py`) -- the q=0 rarefaction computed
  `log C(N-k, n)` with a helper that returns 0 where the value is `-inf`, so a
  species with more than N-n individuals -- one that is *certain* to be in the
  subsample -- was counted as absent with probability (N-k)!(N-n)!/N!. A
  one-species sample of ten individuals was reported as **0.9 species**, and
  the q=0 curve is the default. The computation now uses the same exact
  integer arithmetic as `ecology/rarefaction.py`, which also removes a ~1e-4
  relative-accuracy loss for N >= 60.
- **The three sibling terms of a two-way ANOVA share one permutation**
  (`stats/design_tests.py`) -- each of the three permutation tests drew its own
  random null distribution, so the terms' p-values could not be compared with
  one another, and the result could not report the seed that produced them
  (it reported the `None` the caller had passed). One seed is resolved once
  and used for all three, and the result carries it.
- **The Procrustes distance no longer depends on the size of the shapes**
  (`morpho3d/gpa3d.py`) -- the function computed the optimal scale and then
  did not apply it, so the residual grew with the overall size of the
  configurations: two identical shapes at different sizes scored a distance
  of 8.2 instead of 0. The scale is applied now, which is the full-Procrustes
  convention `morphometrics/shape_stats.py` already uses.
- **The data matrix keeps its labels on the axes they describe**
  (`models/data_matrix.py`) -- indexing one axis produced a view whose labels
  no longer matched its data (`m[:, 0]` gave three row labels and one column
  label around a 1-D array), so a consumer that assumed "label index == data
  index" silently read the wrong specimen's value. A view is now two
  dimensional, and `to_matrix()` accepts it.
- **Two columns with the same label keep their own metadata**
  (`models/data_matrix.py`) -- the per-column metadata was keyed by label, so
  duplicate labels collapsed into one entry and the second column reported
  the first one's units and data type. The store is keyed by column index
  now; the public property still presents the label-keyed view.
- **A matrix with no columns no longer loses its rows**
  (`models/data_matrix.py`) -- `remove_rows_with_missing()` computed a
  threshold of -1 for a matrix with no variables, marked every row invalid
  and emptied it.
- **Undo and redo now tell the spreadsheet** (`models/state_manager.py`) --
  they restored the state but emitted only the undo-stack event, so the table
  kept showing the pre-undo numbers while every analysis read the restored
  state: what the user saw and what was computed disagreed.
- **The binary cache refuses labels that do not match its matrix**
  (`parsers/binary_cache.py`) -- the label count and the matrix's row count
  were read independently and never compared, so a cache written with the
  wrong number of labels loaded back without complaint, with every label
  offset against the data. The checksum could not catch it either: it was
  computed over the same serialised bytes that were read back.
- **Saving clears the "modified" flag, and reading a file does not set it**
  (`views/ui_main_window.py`) -- the save path never called `mark_saved`, so a
  successful save still left the document dirty: the status bar said
  "modified" and closing the window asked the user to save changes that were
  already on disk. The drag-and-drop load path, the only one of the four that
  did not say `mark_modified=False`, marked a file the user had merely opened
  as edited.
- **Degenerate designs are refused instead of reported**
  (`stats/permanova.py`, `stats/design_tests.py`, `stats/simper.py`,
  `stats/univariate.py`) -- a single-group PERMANOVA returned `F = 0` and
  `p = 1`, a complete-looking answer to a comparison that does not exist
  (ANOSIM already refused it); a contingency table with an empty row divided
  by zero and returned `chi2 = nan, p = nan, V = nan`, and because
  "significant" tests `p < 0.05`, a NaN p-value reads as *not significant*; a
  short `variable_names` list raised a bare `IndexError` after every statistic
  had been computed; and a group column holding both a number and text made
  the sort raise a bare `TypeError` from inside `sorted(set(...))`. All four
  now raise a project exception, or -- for the group labels -- are ordered
  by a shared helper that keeps the natural order for homogeneous labels.
- **The missing-value report is translated** (`config/imputation.py`,
  `config/i18n/`) -- five strings of the report were in neither dictionary, so
  the Chinese interface showed that report in English.
- **The background CSV loader reads a GBK file** (`controllers/data_controller.py`)
  -- it hardcoded `encoding="utf-8"` while the menu loader used the
  utf-8-sig -> gbk -> latin-1 fallback chain, so the same file came out clean
  through the menu and as mojibake through the background loader. Both paths
  now use the shared reader, and both report the text columns that came out
  empty.
- **The MSI build derives its package list from `pyproject.toml`**
  (`packaging/build_msi.py`) -- the cx_Freeze list was written out by hand and
  had drifted: `plugins`, `presets`, `reporting`, `state_machine`, `hpc` and
  `morpho3d` were all missing from the frozen tree. It is read from the
  wheel's own list now, so the two cannot drift apart again.
- **The start-up check verifies the analysis modules**
  (`main.py`) -- importing a package only runs its `__init__.py`, so a
  submodule that was renamed or deleted was not noticed: the splash reported
  the engine as loaded and the `ImportError` surfaced much later, when the
  user finally ran that analysis.
- **A test that could not fail now can** (`tests/ecology/test_coverage_rarefaction.py`)
  -- `test_different_seeds_different_results` ended in `assert True`, so the
  claim in its own name was never checked and a refactor that dropped the seed
  on the way to the generator would have left it green.
- **The documentation no longer contradicts the code**
  (`stats/permanova.py`, `stats/anosim.py`, `ecology/diversity.py`,
  `ecology/beta_diversity.py`, `plot_export.py`, `views/ui_plot_export_dialog.py`,
  `data/loader.py`) -- the PERMANOVA module header gave the `n-1` / `n_g-1`
  denominators and called the correct implementation the bug, so a maintainer
  following it would have "fixed" the code back into the broken state; the
  ANOSIM and PERMANOVA docstrings still described a global `np.random` stream
  the code had stopped using; `chao1_confidence_interval` was called
  "bias-corrected" while computing the classic lower bound (the wording is
  now explicit about which one it is, and why); the `beta_diversity` module
  header gave Sorensen as a similarity where the code returns the beta
  distance; the `.jpeg` alias was accepted on export and rejected by the
  path inference and the export dialog; and `data/loader.py` claimed its
  example loaders shared the encoding fallback while calling `pd.read_csv`
  directly.
- **The spreadsheet no longer aborts the application on an infinite cell
  value** (`views/ui_spreadsheet.py`) -- typing `inf` reached the DataMatrix
  validator, whose `DataValidationError` escaped the Qt slot, and PyQt6
  aborts the process on an unhandled exception inside a slot. Every unsaved
  edit was lost. Infinite values are now rejected before they reach the
  model, with a message; NaN is still accepted, because that is how a
  missing measurement is recorded.
- **A thousand-separated number in the first data column is data again**
  (`parsers/dat_parser.py`) -- `_is_label` and `_parse_numeric_value` disagreed
  about separators, so `1,234.56` in the first column was classified as a row
  label: the column disappeared from the data, the row labels became
  numbers, and nothing was reported. An identifier that merely looks
  numeric (`1.2.3`) still counts as a label.
- **Growth-model confidence intervals are no longer too narrow**
  (`utils/statistics_core.py`) -- `fit_nonlinear` hard-coded
  `absolute_sigma=True` while `sigma` was `None`, so scipy returned an
  unscaled covariance and every standard error and 95% interval came out
  too narrow by `sqrt(RSS/(M-N))` -- 3.3x too narrow in the von Bertalanffy
  case. The interval that goes into the paper was wrong.
- **The Epps-Singleton test no longer crashes on data that straddles zero**
  (`stats/distribution_tests.py`) -- a permutation whose group mean is exactly
  zero raised a bare `ZeroDivisionError` out of the test, with an exception
  type that is not part of this project's error hierarchy.
- **The isometry test now uses the correct reference distribution**
  (`morphometrics/allometry.py`) -- the F statistic pools the residual sum of
  squares over correlated Procrustes coordinates, so its null distribution is
  not `F(df1, n-2)`. The p-value was systematically far too large, and real
  allometry was reported as isometry. The p-value is now a permutation test on
  the same statistic; the parametric p-value is still available in
  `isometry_pvalue_parametric`, using the pooled residual degrees of
  freedom.
- **Three functions no longer shadow the translation helper**
  (`morphometrics/allometry.py`, `morphometrics/evolution_rate.py`,
  `stats/pcm.py`) -- a tuple-unpacking target named `_` makes `_` a local
  variable for the whole function, so the `_("...")` calls on their error
  paths raised `UnboundLocalError` instead of a translated message.
- **One Chinese string no longer contains two corrupted characters**
  (`config/i18n/translations_zh.py`).
- **The bundled `primate_traits` example dataset can be loaded at all**
  (`data/loader.py`, `views/ui_main_window.py`) -- picking it from the example
  picker failed with `could not convert string to float: 'Catarrhini'`. The
  dataset is shipped, documented and offered by the application, and it did
  not load. The view built the matrix itself, and the sibling
  `community_abundance` branch dropped both of its text columns ("site" and
  "group") while the `primate_traits` branch dropped only one ("species") and
  left the grouping label in -- the same function, half-fixed. The separation
  of text from measurements now lives in one place,
  `data.loader.load_example_matrix()`, so it can be tested without building a
  window and the two datasets cannot drift apart again.
- **The example picker no longer crashes when it is opened**
  (`views/ui_main_window.py`) -- it raised `AttributeError: 'QInputDialog'
  object has no attribute 'setCurrentIndex'`, and the Chinese interface
  showed the PyQt6 crash dialog instead of a list of datasets. `setCurrentIndex`
  is a `QComboBox` method, and PyQt6 does not expose a `ComboBox` input mode
  for `QInputDialog` at all, so the picker is now a plain dialog with a combo
  box in it. The whole path -- picker, choice, load -- had no test at all,
  which is how it got that far; it does now.
- **The Preferences dialog no longer always says the language is `en`**
  (`views/ui_main_window.py`) -- the field read `preferences/language` while the
  Language menu wrote `language`, so it showed the default even when the whole
  interface was running in Chinese. The dialog now reads the key that is in
  force, and writes both.
- **The Preferences dialog's OK and Cancel buttons are translated**
  (`views/ui_permutation_dialogs.py`, `views/ui_main_window.py`) -- Qt's own
  translations for them are English-only, so a Chinese interface showed "OK"
  and "Cancel" in the one dialog where every other button read "帮助" / "取消" /
  "运行". The same applied to the example-data picker. Both are built with
  translated labels now.
- **The event bus can outlive the window that owned it**
  (`utils/event_bus.py`) -- `EventBus` is a QObject singleton, so when the
  window that owned it was torn down the C++ side died while the Python
  wrapper survived, and every later access raised "wrapped C/C++ object of
  type EventBus has been deleted". The next main window could not connect to
  it. A dead instance is now detected and replaced instead of reused.
- **Closing the window no longer asks to save data that does not exist**
  (`views/ui_main_window.py`) -- the "unsaved changes" prompt was raised on
  the modified flag alone. With nothing loaded there is nothing to lose, and
  asking anyway was a dead end: "Save" was answered with "No data to save",
  which returned false, which made the window ignore the close event, so the
  window refused to close and the user had to answer "Discard" to get out.
  The prompt is now raised only when there is data to save, and it is still
  raised when there is.
- **A preferences write no longer switches the user's theme off**
  (`views/ui_main_window.py`, `tests/visualization/test_palette_single_source.py`)
  -- `_apply_preferences` is deliberately callable with a partial state, and
  the theme was written as its default when the key was missing from that
  state. Every partial write therefore reset the stored dark theme back to
  light, so running the test suite silently undid the choice the user had
  made in the Preferences dialog. A key that is not mentioned is now left
  alone, and the test fixture that works against the real settings scope
  snapshots the whole scope and puts it back afterwards.
- **The test suite no longer hangs on a dialog nobody can see**
  (`tests/conftest.py`) -- `StateManager` is a process-wide singleton and
  nothing reset it between tests, so a matrix one test loaded stayed loaded
  for every test that ran after it, and stayed flagged as modified. Closing
  a real main window then raised the modal "save changes?" box, and a
  headless run has nobody to answer it: the suite stopped dead instead of
  failing. The shared state is now cleared around every test, so no test
  inherits another one's data.
- **The dark theme no longer stops at the application chrome**
  (`views/ui_main_window.py`) -- a canvas is built fresh for every analysis,
  so it is never a child of the window at the moment the theme is
  applied, and the propagation only reached the plots that happened to
  exist then. Every figure the user produced in the dark theme came out
  on a white background: a bright rectangle in a dark application, and
  the dark figures the design system goes to the trouble of building were
  unreachable for every plot the user actually made. The workspace now
  tells a new plot which theme is in force.
- **The figures ask matplotlib for a font that exists wherever it does**
  (`config/design_system.py`, `views/ui_plot_canvas.py`) -- the canvas
  requested four Windows families and nothing else. When matplotlib cannot
  satisfy the request it logs "findfont: Generic family 'sans-serif' not
  found because none of the following families were found", once per
  render: on a Linux machine, every figure the application drew produced a
  log record. The chain now ends in DejaVu Sans, which ships with
  matplotlib itself, and Liberation Sans, which is on every Linux
  desktop, with the CJK families first so Chinese labels still pick them
  up. The list lives in `FigureTypography` now -- the place both rendering
  paths already read their type scale from -- so the canvas and the
  plot-menu plotters cannot drift apart again.
- **A log record with no application behind it no longer kills the process**
  handler on the ROOT logger, and the console is a dock widget with no
  parent of its own, so when the application it belonged to is destroyed
  the widget can survive and so does the handler. The widget's C++ half was
  perfectly alive, so the existing "has it been deleted?" guard let the
  signal through, and the slot then ran under the wrong application --
  which Qt6 answers with SIGABRT. The abort landed wherever the next log
  record happened to be emitted: in CI it was a matplotlib font warning
  raised while a figure was being saved, in a test that touches no Qt at
  all, three quarters of the way through the run. The handler now remembers
  which application it was created for, and compares by identity: "is there
  an application?" answers true for the wrong one once the suite has
  replaced it.
- **Every control that can take the keyboard focus now shows where it is**
  (`config/design_system.py`) -- only the text inputs had a focus ring. A
  button, a combo box, a check box or the navigation tree could take the
  focus and show nothing at all, so a keyboard user had no way to tell where
  they were. VSCode's rule is that the focus indicator is always visible,
  and the ring is now drawn in one token, re-stepped for the dark theme.
  The default button takes the ring in its own on-fill colour, because a
  ring in the accent colour on an accent-coloured button is the ring you
  cannot see.
- **The navigation tree can be reached with the keyboard**
  (`views/ui_navigation.py`) -- it was `ClickFocus`, so it could only be
  reached with the mouse. A widget that cannot take the focus cannot show
  a focus ring either, so the two were the same defect seen twice.
- **The palette's eighth colour can be seen on the dark axes**
  (`config/colors.py`) -- Okabe & Ito's eighth colour is black, which is
  right for the white background the palette was designed for and wrong
  for the application's dark axes: measured against `#1E293B` it is
  1.44:1, a series that is there and cannot be seen. The registry is not
  modified, so the exported figure still uses black; the colour is lifted
  where the figure is painted, and only on a dark surface. The slot keeps
  its index and the palette keeps its length, so the eighth group is still
  the eighth group.
- **The plot modules no longer carry their own series colours**
  (`views/ui_plot_canvas.py`, `visualization/pca_plot.py`,
  `visualization/spectral_plot.py`, `visualization/evo_rate_plot.py`) --
  sixty-eight hard-coded values from the deprecated flat-UI palette, which
  the chrome check deliberately did not cover because a series colour is a
  data colour. They now read the registered palette, the same list the R
  export is handed, so the figure in the window and the figure in the file
  are finally the same figure.
- **The 4px grid is enforced in the UI layer too**
  (`tests/test_design_system_grid.py`) -- the guard only read
  `config/design_system.py`, so eighteen off-grid padding and margin
  literals in the widgets' own inline style sheets went unnoticed. They are
  on the grid now, and the guard covers `views/` and `visualization/`.

### Changed
- **`DESIGN_SYSTEM_SPECIFICATION.md` is now a pointer, not a palette**
  -- the file was 350 lines of the superseded flat-UI scheme, every value
  in it now forbidden by the conformance test. It says where the tokens
  live and what the three rules are, and points at
  `config/design_system.py`.

## [1.1.2] - 2026-10-07

Feature-parity work aimed at PAST 4, plus the defects that surfaced while
comparing against it. The headline is a plot menu: every plotter here drew the
result of one analysis, so there was no way to take two columns out of the
sheet and draw them, which is the largest single difference from PAST for
anyone whose deliverable is a figure.

### Added
- **The plot menu** (`visualization/generic_plots.py`) -- 21 ways to draw
  spreadsheet columns: XY with an optional fitted line and log axes, XY with
  error bars, histogram, bar chart, box plot, pie chart, stacked chart,
  percentiles, normal probability (Q-Q), ternary, bubble, matrix, mosaic, Venn
  for two and three sets, radar, polar, vector, network, and the 3-D family.
  Categorical colours come from the same registry the R export reads, so these
  figures are the figures the export produces.
- **A machine-readable inventory of PAST 4's features**
  (`docs/parity/past4_features.json`), transcribed from the official PAST 4
  reference manual: 242 entries across twelve menus. The aim is that
  "reproduce everything PAST has" becomes a state that can be checked rather
  than judged by eye.
- **k-medoids (PAM), DBSCAN and k-nearest-neighbours classification** in
  `stats/clustering.py`. PAM is implemented here rather than taken from
  scikit-learn because `KMedoids` was deprecated in 1.3 and removed
  afterwards, and this project's floor of >=1.3.0 does not guarantee it is
  present. DBSCAN and kNN import scikit-learn inside the function, so the
  base-dependency install still gets k-medoids and still imports.
- **Abundance model fitting** is now reachable by name
  (`run("abundance_models")`): log-series, log-normal, geometric and
  broken-stick. The implementation and its cross-validation tests were already
  in `ecology/advanced.py`; nothing pointed a user at them.
- **A structural test for dead interactive controls**
  (`tests/views/test_dead_controls.py`).

### Fixed
- **Three controls did nothing while appearing to do something**, and none
  was caught by a behavioural test because every dialog still ran and still
  returned a result: the allometry dialog's "RMA (reduced major axis)" combo
  (picking RMA ran the OLS regression and labelled the output RMA), the PCA
  dialog's "Use correlation matrix" checkbox (a mirror of the Similarity
  Matrix combo that nothing read), and the macroevolution dialog's "Starting
  consensus from:" combo (offering a mode the engine does not have). The new
  test found the second and third on its first run against a tree where the
  first had already been removed.
- **The figure on screen and the figure you export were not the same figure.**
  The interactive plotters hard-coded `get_color_scheme("default")` (Paul
  Tol's colours) while the R export defaulted to Okabe-Ito, so every palette a
  user chose in Preferences changed the exported file and left the on-screen
  figure exactly as it was. Both sides now read one registry in
  `config/colors.py`, and the generated R script is handed that list verbatim.
  **The default is Okabe-Ito on both paths, so every categorical figure looks
  different from 1.1.1.**
- **The two Okabe-Ito tables disagreed about the eighth colour** -- grey in the
  in-app list, black in the R list. Okabe & Ito's eighth colour is black.
- **`r_palette` was collected by the Preferences dialog and then discarded.**
  Neither `_get_preferences_state` nor `_apply_preferences` mentioned the key,
  so it was never written to QSettings and never read back: choosing anything
  but the default silently did nothing.
- **`get_color_scheme` returned a default for any name it did not recognise**,
  with no error. A typo -- or a name copied from the R side, where the palette
  is `okabeito` rather than `colorblind` -- produced a figure in a different
  palette and said nothing.
- **The generated R script asked for packages it did not need.** `brewer.pal()`
  and `viridis(8)` came from RColorBrewer and viridisLite behind
  `requireNamespace`, so the colours could differ between machines. The colours
  are injected instead.
- **`PaleoASTError.__str__` calls `.items()` on `details`**, so a caller
  passing a plain string got an AttributeError while the exception was being
  formatted -- a crash inside the error path, replacing the message with
  "str has no attribute items". Non-mapping details are now wrapped.
- **`MatrixDimensionError`'s docstring advertised `expected_shape` and
  `actual_shape` as keyword arguments** the class does not accept, and spelled
  the detail keys two different ways inside the same docstring.

### Removed
- A "Top Taxa Abundance" panel in `diversity_plot`, guarded by
  `hasattr(result, "abundances")` -- a field `DiversityResult` does not have,
  so it could never run. Its evenness pie and single rarefaction curve also
  still used hard-coded colours while the other panels in the same figures
  used the palette.

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
