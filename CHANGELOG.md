# Changelog

All notable changes to PaleoAST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-06-01

### Fixed
- **morpho3d/quaternion.py**: `RotationMatrix.from_svd` was flipping *both* the
  last column of `U` and the last row of `V^T` to fix reflection cases. The two
  sign changes cancel out, so the resulting `R` still had `det(R) = -1`,
  violating the SO(3) constraint. Now only `Vt[-1, :]` is flipped, which is
  the standard Procrustes SVD fix.
- **parsers/tps_parser.py**: The parser only handled `KEY=VALUE` style lines
  and silently dropped plain `x y z` coordinate lines, which is the format
  used by tpsDig/tpsUtil. No real-world TPS file would parse. The line parser
  now branches on `=` presence: metadata goes through the `KEY=VALUE` path,
  coordinate lines are parsed as 2D or 3D landmark data with auto-generated
  specimen IDs when the `ID=` line is missing.

### Changed
- Bumped version metadata across all modules (`APP_VERSION`, splash screen
  label, pyproject.toml, README badge, i18n translation keys).
- All 30 regression tests pass; 21/21 3D GPA tests pass (was 9/21 before fix).

## [1.0.0] - 2025-05-21

### Added
- Initial public release.
- Statistical toolkit: PCA, PCoA, NMDS, ANOSIM, PERMANOVA, SIMPER, LDA,
  CCA, clustering, univariate summaries, spatial analysis.
- Ecology: diversity indices, rarefaction, beta diversity, null models, DTW.
- Morphometrics: 2D/3D GPA, TPS, EFA, eigenshape, relative warps, allometry,
  evolution rate.
- Stratigraphy: spectral analysis, ARMA modelling, biostratigraphy,
  Markov chains, coniss, directional statistics, extinction, isotope
  analysis, correlation.
- Macroevolution: cohort survivorship, fossilised-birth-death process.
- Phylogenetics: PhyloTree, Fitch, UPGMA, PIC, PCM, ancestral states,
  phylogenetic signal.
- Unified design system, ribbon UI, spreadsheet editor, imputation dialog,
  diagnostic console, file drop handler, floating toolbar.
