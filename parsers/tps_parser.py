# parsers/tps_parser.py
"""
TPS (Thin Plate Spline) File Parser for PaleoAST

Parses .tps format files commonly used in morphometrics for landmark data.
TPS format is widely used in geometric morphometrics for storing:
- 2D/3D landmark coordinates
- Sliding semilandmarks
- Curve points
- Associated specimen data

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class TPSParseError(Exception):
    """Exception raised when TPS file parsing fails.

    Attributes:
        file_path: Path to the file that failed to parse
        line_number: Line number where the error occurred (1-indexed)
        line_content: The original line content that caused the error
        message: Detailed error message
    """

    def __init__(self, message: str, file_path: str | None = None, line_number: int = 0, line_content: str = ""):
        self.file_path = file_path
        self.line_number = line_number
        self.line_content = line_content
        full_message = f"TPS Parse Error"
        if file_path:
            full_message += f" in {os.path.basename(file_path)}"
        if line_number > 0:
            full_message += f" at line {line_number}"
        full_message += f": {message}"
        if line_content:
            full_message += f" (line: {line_content[:50]}{'...' if len(line_content) > 50 else ''})"
        super().__init__(full_message)


@dataclass
class TPSParseErrorSummary:
    """Container for aggregated TPS parse errors."""

    errors: list[TPSParseError] = field(default_factory=list)
    file_path: str | None = None

    def add_error(self, error: TPSParseError) -> None:
        self.errors.append(error)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def summary(self) -> str:
        if not self.errors:
            return "No parse errors"
        lines = [f"TPS Parse Errors ({len(self.errors)} total):"]
        for err in self.errors:
            lines.append(f"  Line {err.line_number}: {err.message}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class TPSSpecimen:
    """Container for a single specimen's TPS data."""

    id: str
    landmarks: np.ndarray  # Shape: (n_landmarks, 2) or (n_landmarks, 3)
    scale: float | None = None
    curve_points: dict | None = None
    raw_data: dict | None = None


@dataclass
class TPSFile:
    """Container for a complete TPS file."""

    specimens: list[TPSSpecimen]
    n_landmarks: int
    n_dimensions: int  # 2 or 3
    comments: list[str]
    file_path: str | None = None

    def to_matrix(self) -> np.ndarray:
        """Convert landmarks to a 2D matrix (n_specimens, n_landmarks * n_dimensions)."""
        if len(self.specimens) == 0:
            return np.array([])

        flattened = []
        for spec in self.specimens:
            flattened.append(spec.landmarks.flatten())
        return np.array(flattened)

    def summary(self) -> str:
        lines = [
            f"TPS File: {len(self.specimens)} specimens",
            f"Landmarks per specimen: {self.n_landmarks}",
            f"Dimensions: {self.n_dimensions}D",
        ]
        return "\n".join(lines)


class TPSParser:
    """Parser for TPS (Thin Plate Spline) format files."""

    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize TPS parser.

        Parameters:
            strict_mode: If True, raise TPSParseError on parse failures.
                        If False, collect errors and continue (legacy behavior).
        """
        self._logger = logging.getLogger(f"{__name__}.TPSParser")
        self.specimens: list[TPSSpecimen] = []
        self.n_landmarks: int = 0
        self.n_dimensions: int = 0
        self.comments: list[str] = []
        self._current_spec: TPSSpecimen | None = None
        self._current_landmarks: list = []
        self._in_curve: bool = False
        self._strict_mode = strict_mode
        self._parse_errors: TPSParseErrorSummary = TPSParseErrorSummary()

    def parse(self, file_path: str) -> TPSFile:
        """
        Parse a TPS file and return a TPSFile object.

        Parameters:
            file_path: Path to the .tps file

        Returns:
            TPSFile object containing all specimen data

        Raises:
            FileNotFoundError: If file doesn't exist
            TPSParseError: If file format is invalid (in strict mode)
            ValueError: If file format is invalid (no specimens found)
        """
        self._logger.info(f"Parsing TPS file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TPS file not found: {file_path}")

        self.specimens = []
        self.comments = []
        self._current_spec = None
        self._current_landmarks = []
        # Reset instance state variables for re-use
        self.n_landmarks = 0
        self.n_dimensions = 0
        self._parse_errors = TPSParseErrorSummary(file_path=file_path)

        # Detect and handle BOM
        with open(file_path, encoding="utf-8-sig", errors="replace") as f:
            content = f.read()

        for line_num, line in enumerate(content.splitlines(), 1):
            original_line = line
            line = line.strip()
            if not line or line.startswith("!"):
                # Comment line (comment char is '!')
                if line.startswith("!"):
                    self.comments.append(line[1:].strip())
                continue

            try:
                self._parse_line(line, line_num)
            except TPSParseError as e:
                if self._strict_mode:
                    raise
                self._parse_errors.add_error(e)
                self._logger.warning(f"Error parsing line {line_num}: {e}")

        # Handle last specimen if not added
        if self._current_spec is not None and self._current_landmarks:
            self._finalize_specimen()

        if len(self.specimens) == 0:
            if self._parse_errors.has_errors():
                raise TPSParseError(
                    f"No valid specimens found. {self._parse_errors.summary()}",
                    file_path=file_path,
                )
            raise ValueError("No valid specimens found in TPS file")

        self._logger.info(f"Parsed {len(self.specimens)} specimens")

        if self._parse_errors.has_errors():
            self._logger.warning(f"Parse completed with {len(self._parse_errors.errors)} errors:\n{self._parse_errors.summary()}")

        return TPSFile(
            specimens=self.specimens,
            n_landmarks=self.n_landmarks,
            n_dimensions=self.n_dimensions,
            comments=self.comments,
            file_path=file_path,
        )

    def _parse_line(self, line: str, line_num: int = 0) -> None:
        """Parse a single line of TPS format.

        Standard TPS files (as written by tpsDig/tpsUtil) use two line
        styles:

        * ``KEY=VALUE`` for metadata (LM, ID, DIM, SCALE, ...).
        * Plain ``x y z`` (or ``x y``) for landmark coordinates.

        The previous implementation only handled the ``KEY=VALUE`` form
        and dropped plain coordinate lines, which meant no real-world
        TPS file would parse. This version accepts both styles.

        Raises:
            TPSParseError: If a coordinate line cannot be parsed
        """
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip().upper()
            value = parts[1].strip()

            if key == "LM":
                # Number of landmarks
                try:
                    self.n_landmarks = int(value)
                except ValueError as e:
                    raise TPSParseError(
                        f"Invalid LM value '{value}': {e}",
                        file_path=self._parse_errors.file_path,
                        line_number=line_num,
                        line_content=line,
                    )
            elif key == "DIM":
                # Dimensions (2 or 3)
                try:
                    self.n_dimensions = int(value)
                except ValueError as e:
                    raise TPSParseError(
                        f"Invalid DIM value '{value}': {e}",
                        file_path=self._parse_errors.file_path,
                        line_number=line_num,
                        line_content=line,
                    )
                if self.n_dimensions not in (2, 3):
                    raise TPSParseError(
                        f"Invalid DIM value '{value}': must be 2 or 3",
                        file_path=self._parse_errors.file_path,
                        line_number=line_num,
                        line_content=line,
                    )
            elif key == "SCALE":
                # Scale factor
                try:
                    scale = float(value)
                except ValueError as e:
                    raise TPSParseError(
                        f"Invalid SCALE value '{value}': {e}",
                        file_path=self._parse_errors.file_path,
                        line_number=line_num,
                        line_content=line,
                    )
                if self._current_spec is not None:
                    self._current_spec.scale = scale
            elif key == "ID":
                # Specimen ID - start new specimen
                if self._current_spec is not None and self._current_landmarks:
                    self._finalize_specimen()

                self._current_spec = TPSSpecimen(
                    id=value, landmarks=np.array([]), scale=None, curve_points=None, raw_data={"id": value}
                )
                self._current_landmarks = []
            elif key == "CO":
                # Curve order
                if self._current_spec is not None:
                    if self._current_spec.curve_points is None:
                        self._current_spec.curve_points = {}
                    self._in_curve = True
                    try:
                        self._current_spec.curve_points["order"] = [int(x) for x in value.split()]
                    except ValueError as e:
                        raise TPSParseError(
                            f"Invalid CO (curve order) value '{value}': {e}",
                            file_path=self._parse_errors.file_path,
                            line_number=line_num,
                            line_content=line,
                        )
            elif key == "POINTS":
                # Curve points for current curve
                if self._current_spec is not None and self._in_curve:
                    if "points" not in self._current_spec.curve_points:
                        self._current_spec.curve_points["points"] = []
                    try:
                        coords = [float(x) for x in value.split()]
                        self._current_spec.curve_points["points"].append(coords)
                    except ValueError as e:
                        raise TPSParseError(
                            f"Invalid POINTS value '{value}': {e}",
                            file_path=self._parse_errors.file_path,
                            line_number=line_num,
                            line_content=line,
                        )
            elif key == "END":
                # End of specimen
                if self._current_spec is not None and self._current_landmarks:
                    self._finalize_specimen()
            return

        # No '=' sign: this is a landmark coordinate line ``x y [z]``.
        tokens = line.split()
        if not tokens:
            return
        try:
            coords = [float(t) for t in tokens]
        except ValueError as e:
            # This is likely not a coordinate line; check if it looks like one
            # If the tokens look like they should be numbers but aren't, report error
            raise TPSParseError(
                f"Cannot parse coordinate line '{line}': {e}",
                file_path=self._parse_errors.file_path,
                line_number=line_num,
                line_content=line,
            )
        if len(coords) not in (2, 3):
            raise TPSParseError(
                f"Invalid coordinate dimension: expected 2 or 3, got {len(coords)}",
                file_path=self._parse_errors.file_path,
                line_number=line_num,
                line_content=line,
            )

        # If we don't yet have a specimen context, create one with an
        # auto-generated ID. Some TPS files don't include an ID line.
        if self._current_spec is None:
            self._current_spec = TPSSpecimen(
                id=f"Specimen_{len(self.specimens) + 1}",
                landmarks=np.array([]),
                scale=None,
                curve_points=None,
                raw_data={},
            )
            self._current_landmarks = []

        # Infer dimensions if not yet set
        if self.n_dimensions == 0:
            self.n_dimensions = len(coords)
        elif len(coords) != self.n_dimensions:
            raise TPSParseError(
                f"Invalid coordinate dimension: expected {self.n_dimensions}D, got {len(coords)}D",
                file_path=self._parse_errors.file_path,
                line_number=line_num,
                line_content=line,
            )
        self._current_landmarks.append(coords)

        # If we have all landmarks, finalize
        if self.n_landmarks > 0 and len(self._current_landmarks) == self.n_landmarks:
            self._finalize_specimen()

    def _finalize_specimen(self) -> None:
        """Add the current specimen to the list."""
        if self._current_spec is None or not self._current_landmarks:
            return

        self._current_spec.landmarks = np.array(self._current_landmarks)

        if self.n_dimensions == 0:
            self.n_dimensions = self._current_spec.landmarks.shape[1]
        if self.n_landmarks == 0:
            self.n_landmarks = self._current_spec.landmarks.shape[0]

        self.specimens.append(self._current_spec)
        self._current_spec = None
        self._current_landmarks = []


def parse_tps_file(file_path: str) -> TPSFile:
    """
    Convenience function to parse a TPS file.

    Parameters:
        file_path: Path to the .tps file

    Returns:
        TPSFile object
    """
    parser = TPSParser()
    return parser.parse(file_path)
