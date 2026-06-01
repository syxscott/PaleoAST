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
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TPSSpecimen:
    """Container for a single specimen's TPS data."""
    id: str
    landmarks: np.ndarray  # Shape: (n_landmarks, 2) or (n_landmarks, 3)
    scale: Optional[float] = None
    curve_points: Optional[dict] = None
    raw_data: Optional[dict] = None


@dataclass
class TPSFile:
    """Container for a complete TPS file."""
    specimens: list[TPSSpecimen]
    n_landmarks: int
    n_dimensions: int  # 2 or 3
    comments: list[str]
    file_path: Optional[str] = None

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

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.TPSParser")
        self.specimens: list[TPSSpecimen] = []
        self.n_landmarks: int = 0
        self.n_dimensions: int = 0
        self.comments: list[str] = []
        self._current_spec: Optional[TPSSpecimen] = None
        self._current_landmarks: list = []
        self._in_curve: bool = False

    def parse(self, file_path: str) -> TPSFile:
        """
        Parse a TPS file and return a TPSFile object.

        Parameters:
            file_path: Path to the .tps file

        Returns:
            TPSFile object containing all specimen data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
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

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('!'):
                    # Comment line
                    if line.startswith('!'):
                        self.comments.append(line[1:].strip())
                    continue

                try:
                    self._parse_line(line)
                except Exception as e:
                    self._logger.warning(f"Error parsing line {line_num}: {e}")

        # Handle last specimen if not added
        if self._current_spec is not None and self._current_landmarks:
            self._finalize_specimen()

        if len(self.specimens) == 0:
            raise ValueError("No valid specimens found in TPS file")

        self._logger.info(f"Parsed {len(self.specimens)} specimens")

        return TPSFile(
            specimens=self.specimens,
            n_landmarks=self.n_landmarks,
            n_dimensions=self.n_dimensions,
            comments=self.comments,
            file_path=file_path
        )

    def _parse_line(self, line: str) -> None:
        """Parse a single line of TPS format.

        Standard TPS files (as written by tpsDig/tpsUtil) use two line
        styles:

        * ``KEY=VALUE`` for metadata (LM, ID, DIM, SCALE, ...).
        * Plain ``x y z`` (or ``x y``) for landmark coordinates.

        The previous implementation only handled the ``KEY=VALUE`` form
        and dropped plain coordinate lines, which meant no real-world
        TPS file would parse. This version accepts both styles.
        """
        if '=' in line:
            parts = line.split('=', 1)
            key = parts[0].strip().upper()
            value = parts[1].strip()

            if key == 'LM':
                # Number of landmarks
                self.n_landmarks = int(value)
            elif key == 'DIM':
                # Dimensions (2 or 3)
                self.n_dimensions = int(value)
            elif key == 'SCALE':
                # Scale factor
                scale = float(value)
                if self._current_spec is not None:
                    self._current_spec.scale = scale
            elif key == 'ID':
                # Specimen ID - start new specimen
                if self._current_spec is not None and self._current_landmarks:
                    self._finalize_specimen()

                self._current_spec = TPSSpecimen(
                    id=value,
                    landmarks=np.array([]),
                    scale=None,
                    curve_points=None,
                    raw_data={'id': value}
                )
                self._current_landmarks = []
            elif key == 'CO':
                # Curve order
                if self._current_spec is not None:
                    if self._current_spec.curve_points is None:
                        self._current_spec.curve_points = {}
                    self._in_curve = True
                    self._current_spec.curve_points['order'] = [int(x) for x in value.split()]
            elif key == 'POINTS':
                # Curve points for current curve
                if self._current_spec is not None and self._in_curve:
                    if 'points' not in self._current_spec.curve_points:
                        self._current_spec.curve_points['points'] = []
                    coords = [float(x) for x in value.split()]
                    self._current_spec.curve_points['points'].append(coords)
            elif key == 'END':
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
        except ValueError:
            # Not a coordinate line; ignore silently.
            return
        if len(coords) not in (2, 3):
            return

        # If we don't yet have a specimen context, create one with an
        # auto-generated ID. Some TPS files don't include an ID line.
        if self._current_spec is None:
            self._current_spec = TPSSpecimen(
                id=f"Specimen_{len(self.specimens) + 1}",
                landmarks=np.array([]),
                scale=None,
                curve_points=None,
                raw_data={}
            )
            self._current_landmarks = []

        # Infer dimensions if not yet set
        if self.n_dimensions == 0:
            self.n_dimensions = len(coords)
        elif len(coords) != self.n_dimensions:
            raise ValueError(
                f"Invalid coordinate dimension: expected {self.n_dimensions}, got {len(coords)}"
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
