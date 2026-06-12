# =============================================================================
# FILE: ecology/paleoenv.py
# =============================================================================
"""
Quantitative Paleo-Environment Reconstruction for PaleoAST

This module implements a pure-Python replacement for the classical
correspondence analysis (CA) workflow used in R FactoMineR.
The objective is to extract a single latent environmental gradient
(often interpreted as a paleobathymetric / productivity axis) from a
multivariate fossil-abundance matrix using a singular value
decomposition (SVD) of the standardised residual matrix.

The module exposes a single class, PaleoEnvironmentReconstructor,
which combines:

    1. Sum-normalised contingency table construction.
    2. Standardised-residual matrix computation.
    3. Truncated SVD via scipy.linalg.svd.
    4. Row-score extraction along the first principal axis.
    5. Direction calibration by Pearson correlation with the
       stratigraphic height array.

The implementation is thread-safe (guarded by a threading.RLock)
and uses _ from config.i18n for all user-facing strings.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg as scipy_linalg

from config.i18n import _
from utils.exceptions import ComputationError, DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class PaleoEnvironmentResult:
    """Result container for the paleo-environmental reconstruction.

    Attributes:
        heights: The stratigraphic height array used as the calibration target.
        taxon_names: Names of the taxa (matrix columns).
        row_species_axis: Row scores on the first correspondence axis.
        column_species_axis: Column scores on the first correspondence axis.
        singular_values: All non-zero singular values of the standardised residual matrix.
        explained_inertia: Proportion of total inertia explained by the first axis.
        pearson_corr_axis_vs_height: Pearson correlation coefficient.
        was_flipped: True if the gradient was automatically flipped.
        n_samples: Number of stratigraphic samples (rows).
        n_taxa: Number of taxa (columns).
        summary_text: Human-readable multi-line summary.
    """

    heights: npt.NDArray
    taxon_names: list[str]
    row_species_axis: npt.NDArray
    column_species_axis: npt.NDArray
    singular_values: npt.NDArray
    explained_inertia: float
    pearson_corr_axis_vs_height: float
    was_flipped: bool
    n_samples: int
    n_taxa: int
    summary_text: str = ""

    def summary(self) -> str:
        """Return a localised summary of the reconstruction.

        Built lazily on every call so the labels re-translate if the
        user switches the application language after the analysis.
        Numeric diagnostics (inertia, correlation, sample/taxon
        counts) are still correct because they are stored on the
        result.
        """
        lines = [
            _("Paleo-environmental reconstruction (Correspondence Analysis)"),
            "=" * 60,
            _("Samples: {0}").format(self.n_samples),
            _("Taxa: {0}").format(self.n_taxa),
            _("Axis 1 inertia explained: {0:.4f} ({1:.2%})").format(
                self.explained_inertia, self.explained_inertia
            ),
            _("Pearson r(axis 1, height): {0:.4f}").format(
                self.pearson_corr_axis_vs_height
            ),
            _("Polarity auto-calibrated: {0}").format(
                _("yes") if self.was_flipped else _("no")
            ),
            "",
        ]
        return "\n".join(lines)


class PaleoEnvironmentReconstructor:
    """Quantitative paleo-environmental reconstructor based on
    Correspondence Analysis (CA). The class receives a stratigraphic
    abundance (or 0/1 presence-absence) matrix and a vector of
    stratigraphic heights. It runs the classical Benzecri / Greenacre
    correspondence-analysis pipeline on a pure-Python foundation (no R
    dependencies) and returns the reconstructed environmental axis
    together with diagnostic indicators.

    Mathematical Summary
    --------------------
    Let F be the input abundance matrix of shape (n_samples, n_taxa).
    Denote r = F @ 1 (row masses) and c = F.T @ 1 (column masses)
    and the grand total N = sum(F).

    The probability table is P = F / N. The matrix of standardised
    residuals is::

        S_ij = (P_ij - r_i * c_j) / sqrt(r_i * c_j)

    Performing SVD S = U * Sigma * V.T yields the row scores
    F_r = U * Sigma and column scores F_c = V * Sigma for the first
    principal inertia axis.

    Direction Calibration
    ---------------------
    Stratigraphic convention dictates that older layers lie at higher
    stratigraphic heights. The reconstructed axis is therefore
    automatically sign-corrected by enforcing a positive Pearson
    correlation with the height vector (when calibrate_direction=True).

    Thread-Safety
    -------------
    All public methods are guarded by a re-entrant lock so that the
    reconstructor can be shared between analysis threads safely.
    """

    def __init__(self) -> None:
        """Initialise the reconstructor with a thread-safe lock and logger."""
        self._logger = logging.getLogger(f"{__name__}.PaleoEnvironmentReconstructor")
        self._lock = threading.RLock()
        self._last_result: PaleoEnvironmentResult | None = None
        self._logger.info(_("PaleoEnvironmentReconstructor initialised"))

    def reconstruct(
        self,
        abundance_matrix: npt.NDArray,
        heights: npt.NDArray,
        taxon_names: list[str] | None = None,
        sample_names: list[str] | None = None,
        calibrate_direction: bool = True,
    ) -> PaleoEnvironmentResult:
        """Reconstruct the dominant paleo-environmental gradient.

        Parameters:
            abundance_matrix: 2D array of shape (n_samples, n_taxa) with
                non-negative entries (counts, percentages, or 0/1).
            heights: 1D array of length n_samples with the stratigraphic
                height of each sample. Heights must be strictly monotonic.
            taxon_names: Optional list of taxon names of length n_taxa.
            sample_names: Optional list of sample names of length n_samples.
            calibrate_direction: If True (default) the first axis is
                sign-corrected to positively correlate with the height.

        Returns:
            PaleoEnvironmentResult with the inferred row scores and
            diagnostics.

        Raises:
            DataValidationError: If the matrix contains negatives, all-zero
                rows/columns, mismatched dimensions, or non-monotonic heights.
            ComputationError: If the SVD fails to extract a non-trivial axis.
        """
        with self._lock:
            mat = np.asarray(abundance_matrix, dtype=np.float64)
            h_arr = np.asarray(heights, dtype=np.float64)

            if mat.ndim != 2:
                raise DataValidationError(
                    _("abundance_matrix must be 2-dimensional, got {0}-D").format(mat.ndim),
                    details={"shape": mat.shape},
                )
            if h_arr.ndim != 1:
                raise DataValidationError(
                    _("heights must be 1-dimensional, got {0}-D").format(h_arr.ndim),
                    details={"shape": h_arr.shape},
                )
            if mat.shape[0] != h_arr.shape[0]:
                raise DataValidationError(
                    _("Number of samples in abundance_matrix ({0}) does not match heights ({1})").format(
                        mat.shape[0], h_arr.shape[0]
                    ),
                    details={"n_samples": mat.shape[0], "n_heights": h_arr.shape[0]},
                )
            if np.any(mat < 0.0):
                raise DataValidationError(
                    _("abundance_matrix contains negative values; CA requires non-negative input"),
                    details={"min_value": float(np.min(mat))},
                )

            n_samples, n_taxa = mat.shape
            if n_samples < 2 or n_taxa < 2:
                raise DataValidationError(
                    _("abundance_matrix must be at least 2x2; got {0}x{1}").format(n_samples, n_taxa),
                    details={"shape": (n_samples, n_taxa)},
                )

            row_sums = mat.sum(axis=1)
            col_sums = mat.sum(axis=0)
            zero_rows = np.where(row_sums <= 0.0)[0]
            zero_cols = np.where(col_sums <= 0.0)[0]
            if zero_rows.size > 0:
                raise DataValidationError(
                    _("abundance_matrix contains all-zero rows at indices {0}").format(zero_rows.tolist()),
                    details={"zero_row_indices": zero_rows.tolist()},
                )
            if zero_cols.size > 0:
                raise DataValidationError(
                    _("abundance_matrix contains all-zero columns at indices {0}").format(zero_cols.tolist()),
                    details={"zero_col_indices": zero_cols.tolist()},
                )

            diffs = np.diff(h_arr)
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                raise DataValidationError(
                    _("heights must be strictly monotonic (ascending or descending)"),
                    details={"heights": h_arr.tolist()},
                )

            if taxon_names is None:
                taxon_names = [f"Taxon_{j + 1}" for j in range(n_taxa)]
            if len(taxon_names) != n_taxa:
                raise DataValidationError(
                    _("taxon_names length ({0}) does not match n_taxa ({1})").format(
                        len(taxon_names), n_taxa
                    ),
                )
            if sample_names is not None and len(sample_names) != n_samples:
                raise DataValidationError(
                    _("sample_names length ({0}) does not match n_samples ({1})").format(
                        len(sample_names), n_samples
                    )
                )

            self._logger.info(
                _("CA reconstruction: {0} samples x {1} taxa").format(n_samples, n_taxa)
            )
            total = float(mat.sum())
            if total <= 0.0:
                raise ComputationError(_("Total abundance is zero; cannot normalise"))

            prob = mat / total
            r_masses = prob.sum(axis=1)
            c_masses = prob.sum(axis=0)

            denom = np.sqrt(np.outer(r_masses, c_masses))
            if np.any(denom == 0.0):
                raise ComputationError(
                    _("Zero mass detected after normalisation; CA cannot proceed")
                )
            residuals = (prob - np.outer(r_masses, c_masses)) / denom

            try:
                u_mat, s_vals, vt_mat = scipy_linalg.svd(residuals, full_matrices=False)
            except Exception as exc:
                raise ComputationError(
                    _("SVD on the standardised residual matrix failed"),
                    original_exception=exc,
                ) from exc

            if s_vals.size == 0 or float(s_vals[0]) < 1e-12:
                raise ComputationError(
                    _("Leading singular value is non-positive; axis extraction failed")
                )

            row_axis_raw = u_mat[:, 0] * float(s_vals[0])
            col_axis_raw = vt_mat[0, :] * float(s_vals[0])

            with np.errstate(invalid="ignore"):
                corr_matrix = np.corrcoef(row_axis_raw, h_arr)
            corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0
            if not np.isfinite(corr):
                corr = 0.0

            was_flipped = False
            row_axis = row_axis_raw.copy()
            col_axis = col_axis_raw.copy()
            if calibrate_direction and corr < 0.0:
                # Flip both row and column scores by the same factor so that
                # the biplot interpretation remains internally consistent.
                row_axis = -row_axis_raw
                col_axis = -col_axis_raw
                was_flipped = True
                self._logger.info(
                    _("Polarity auto-calibrated: sign of axis 1 flipped to match height monotonicity")
                )

            total_inertia = float(np.sum(s_vals ** 2))
            axis_inertia = float(s_vals[0] ** 2)
            explained = axis_inertia / total_inertia if total_inertia > 0 else 0.0

            summary_text = self._build_summary(
                n_samples=n_samples,
                n_taxa=n_taxa,
                corr=corr,
                was_flipped=was_flipped,
                explained=explained,
                sample_names=sample_names,
            )

            result = PaleoEnvironmentResult(
                heights=h_arr,
                taxon_names=taxon_names,
                row_species_axis=row_axis,
                column_species_axis=col_axis,
                singular_values=s_vals,
                explained_inertia=explained,
                pearson_corr_axis_vs_height=corr,
                was_flipped=was_flipped,
                n_samples=n_samples,
                n_taxa=n_taxa,
                summary_text=summary_text,
            )
            self._last_result = result
            self._logger.info(
                _(
                    "CA axis 1 extracted: inertia={0:.4f}, r(height)={1:.4f}, flipped={2}"
                ).format(explained, corr, was_flipped)
            )
            return result

    @property
    def last_result(self) -> PaleoEnvironmentResult | None:
        """Return the last PaleoEnvironmentResult produced."""
        with self._lock:
            return self._last_result

    def reconstruct_from_dataframe(
        self,
        df,
        height_column: str = "height",
        taxon_columns: list[str] | None = None,
        calibrate_direction: bool = True,
    ) -> PaleoEnvironmentResult:
        """Reconstruct the paleo-environmental axis from a pandas DataFrame.

        Parameters:
            df: Input DataFrame with sample rows.
            height_column: Name of the column containing stratigraphic heights.
            taxon_columns: List of column names to use as taxa. If None, all
                numeric columns except height_column are used.
            calibrate_direction: If True the axis is polarity-calibrated.

        Returns:
            PaleoEnvironmentResult
        """
        try:
            import pandas as pd  # noqa: F401
        except ImportError as exc:
            raise ComputationError(
                _("pandas is required for reconstruct_from_dataframe"),
                original_exception=exc,
            ) from exc

        if height_column not in df.columns:
            raise DataValidationError(
                _("height_column '{0}' not found in DataFrame").format(height_column),
                details={"available": list(df.columns)},
            )

        if taxon_columns is None:
            taxon_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != height_column
            ]
        if not taxon_columns:
            raise DataValidationError(
                _("No numeric taxon columns found in DataFrame"),
                details={"columns": list(df.columns)},
            )

        mat = df[taxon_columns].to_numpy(dtype=np.float64)
        heights = df[height_column].to_numpy(dtype=np.float64)
        return self.reconstruct(
            abundance_matrix=mat,
            heights=heights,
            taxon_names=list(taxon_columns),
            calibrate_direction=calibrate_direction,
        )

    def _build_summary(
        self,
        n_samples: int,
        n_taxa: int,
        corr: float,
        was_flipped: bool,
        explained: float,
        sample_names: list[str] | None,
    ) -> str:
        """Build a localised human-readable summary of the CA result."""
        lines = [
            _("Paleo-environmental reconstruction (Correspondence Analysis)"),
            "============================================================",
            _("Samples: {0}").format(n_samples),
            _("Taxa: {0}").format(n_taxa),
            _("Axis 1 inertia explained: {0:.4f} ({1:.2%})").format(explained, explained),
            _("Pearson r(axis 1, height): {0:.4f}").format(corr),
            _("Polarity auto-calibrated: {0}").format(_("yes") if was_flipped else _("no")),
            "",
        ]
        return "\n".join(lines)