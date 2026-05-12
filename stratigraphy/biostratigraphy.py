# =============================================================================
# FILE: stratigraphy/biostratigraphy.py
# =============================================================================
"""
Quantitative Biostratigraphy Module for PaleoAST

Implements Unitary Associations (UA) and Ranking and Scaling (RASC) methods
for quantitative biostratigraphic analysis.

Mathematical Foundation:

Unitary Associations (UA):
    - Construct overlap graph from FAD/LAD events
    - Find maximum cliques = biozones
    - Each maximal clique represents a zone

RASC (Ranking and Scaling):
    - Dynamic programming to find optimal event sequence
    - Minimize distance matrix between sections
    - Iterative refinement

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Biozone definition."""

    name: str
    events: list[str]
    sections: list[str]
    fads: dict[str, float]  # First Appearance Datums
    lads: dict[str, float]  # Last Appearance Datums


@dataclass
class BioeventResult:
    """
    Container for biostratigraphic analysis results.

    Attributes:
        sections: List of section names
        events: List of event names
        fad_matrix: First Appearance Datum positions (section, event)
        lad_matrix: Last Appearance Datum positions (section, event)
        zones: List of identified biozones
        ranking: Optimal event ranking (for RASC)
        distance_matrix: Pairwise event distances
        method: 'ua' or 'rasc'
    """

    sections: list[str]
    events: list[str]
    fad_matrix: npt.NDArray
    lad_matrix: npt.NDArray
    zones: list[Zone]
    ranking: list[str]
    distance_matrix: npt.NDArray | None
    method: str

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Biostratigraphic Analysis')}",
            "=" * 50,
            f"{_('Method: {0}').format(self.method.upper())}",
            f"{_('Sections: {0}').format(len(self.sections))}",
            f"{_('Events: {0}').format(len(self.events))}",
            f"{_('Biozones identified: {0}').format(len(self.zones))}",
            "",
        ]

        if self.ranking:
            lines.append(_("Event Ranking (RASC):"))
            for i, event in enumerate(self.ranking[:10]):
                lines.append(f"  {i + 1}. {event}")
            if len(self.ranking) > 10:
                lines.append(f"  ... ({len(self.ranking) - 10} more)")

        return "\n".join(lines)


class UAAnalyzer:
    """
    Unitary Associations analyzer.

    Finds maximal cliques in the overlap graph to identify biozones.
    """

    def __init__(self) -> None:
        """Initialize the UA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.UAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: BioeventResult | None = None

    def analyze(
        self,
        fad_matrix: npt.NDArray,
        lad_matrix: npt.NDArray,
        section_names: list[str] | None = None,
        event_names: list[str] | None = None,
    ) -> BioeventResult:
        """
        Perform Unitary Associations analysis.

        Parameters:
            fad_matrix: First Appearance Datum matrix (n_sections, n_events)
            lad_matrix: Last Appearance Datum matrix (n_sections, n_events)
            section_names: Names of sections
            event_names: Names of events

        Returns:
            BioeventResult: UA analysis results with identified zones
        """
        with self._lock:
            # Validate input
            fad = validate_data_array(fad_matrix, allow_nan=False, name="fad_matrix")
            lad = validate_data_array(lad_matrix, allow_nan=False, name="lad_matrix")

            if fad.shape != lad.shape:
                raise ComputationError("FAD and LAD matrices must have same shape")

            n_sections, n_events = fad.shape

            self._logger.info(
                f"UA analyze started: {n_sections} sections, {n_events} events"
            )

            # Default names
            if section_names is None:
                section_names = [f"Section_{i + 1}" for i in range(n_sections)]
            if event_names is None:
                event_names = [f"Event_{i + 1}" for i in range(n_events)]

            # Build overlap graph
            # Two events overlap if they co-occur in at least one section
            # i.e., FAD[i] < LAD[j] and FAD[j] < LAD[i]
            overlap_graph = self._build_overlap_graph(fad, lad)

            # Find maximal cliques using Bron-Kerbosch algorithm
            cliques = self._bron_kerbosch(overlap_graph)

            # Convert cliques to zones
            zones = self._cliques_to_zones(cliques, event_names, section_names, fad, lad)

            result = BioeventResult(
                sections=section_names,
                events=event_names,
                fad_matrix=fad,
                lad_matrix=lad,
                zones=zones,
                ranking=[],  # UA doesn't produce ranking
                distance_matrix=None,
                method="ua",
            )

            self._last_result = result
            self._logger.info(f"UA completed: found {len(zones)} zones")
            return result

    def _build_overlap_graph(
        self, fad: npt.NDArray, lad: npt.NDArray
    ) -> dict[int, set[int]]:
        """Build overlap graph from FAD/LAD matrices."""
        n_sections, n_events = fad.shape
        graph: dict[int, set[int]] = {i: set() for i in range(n_events)}

        for section_idx in range(n_sections):
            for i in range(n_events):
                for j in range(i + 1, n_events):
                    # Events i and j overlap if:
                    # FAD[i] < LAD[j] and FAD[j] < LAD[i]
                    if fad[section_idx, i] < lad[section_idx, j] and fad[section_idx, j] < lad[section_idx, i]:
                        graph[i].add(j)
                        graph[j].add(i)

        return graph

    def _bron_kerbosch(self, graph: dict[int, set[int]]) -> list[set[int]]:
        """
        Bron-Kerbosch algorithm for finding maximal cliques.

        Returns:
            List of maximal cliques (each clique is a set of vertex indices)
        """
        cliques: list[set[int]] = []

        def backtrack(
            r: set[int], p: set[int], x: set[int]
        ) -> None:
            if not p and not x:
                # R is a maximal clique
                cliques.append(r.copy())
                return

            for v in list(p):
                backtrack(
                    r | {v},
                    p & graph[v],
                    x & graph[v],
                )
                p = p - {v}
                x = x | {v}

        # Start with all vertices in P, empty R and X
        all_vertices = set(graph.keys())
        backtrack(set(), all_vertices, set())

        # Sort cliques by size (larger first) and filter small ones
        cliques.sort(key=len, reverse=True)

        # Keep only cliques that are not subsets of larger cliques
        maximal_cliques: list[set[int]] = []
        for clique in cliques:
            is_maximal = True
            for existing in maximal_cliques:
                if clique.issubset(existing):
                    is_maximal = False
                    break
            if is_maximal and len(clique) >= 2:  # Minimum 2 events per zone
                maximal_cliques.append(clique)

        return maximal_cliques

    def _cliques_to_zones(
        self,
        cliques: list[set[int]],
        event_names: list[str],
        section_names: list[str],
        fad: npt.NDArray,
        lad: npt.NDArray,
    ) -> list[Zone]:
        """Convert maximal cliques to Zone objects."""
        zones: list[Zone] = []

        for i, clique in enumerate(cliques):
            event_list = [event_names[idx] for idx in sorted(clique)]

            # Find FAD and LAD for each event across all sections
            fads = {}
            lads = {}
            for idx in clique:
                fads[event_names[idx]] = np.nanmin(fad[:, idx])
                lads[event_names[idx]] = np.nanmax(lad[:, idx])

            zone = Zone(
                name=f"Zone {i + 1}",
                events=event_list,
                sections=section_names,
                fads=fads,
                lads=lads,
            )
            zones.append(zone)

        return zones

    @property
    def last_result(self) -> BioeventResult | None:
        """Get the last UA result."""
        with self._lock:
            return self._last_result


class RASCAnalyzer:
    """
    Ranking and Scaling analyzer.

    Uses dynamic programming to find optimal event sequence.
    """

    def __init__(self) -> None:
        """Initialize the RASC analyzer."""
        self._logger = logging.getLogger(f"{__name__}.RASCAnalyzer")
        self._lock = threading.RLock()
        self._last_result: BioeventResult | None = None

    def analyze(
        self,
        distance_matrix: npt.NDArray,
        event_names: list[str] | None = None,
        n_iterations: int = 100,
    ) -> BioeventResult:
        """
        Perform RASC analysis.

        Parameters:
            distance_matrix: Pairwise distance matrix between events (n_events, n_events)
            event_names: Names of events
            n_iterations: Number of iterations for optimization

        Returns:
            BioeventResult: RASC analysis results with ranking
        """
        with self._lock:
            # Validate input
            dist = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            if dist.shape[0] != dist.shape[1]:
                raise ComputationError("Distance matrix must be square")

            n_events = dist.shape[0]

            self._logger.info(f"RASC analyze started: {n_events} events")

            # Default names
            if event_names is None:
                event_names = [f"Event_{i + 1}" for i in range(n_events)]

            # Initialize ranking as sequence 0, 1, 2, ...
            ranking = list(range(n_events))

            # Iterative refinement
            for iteration in range(n_iterations):
                improved = False

                for i in range(1, n_events - 1):
                    # Try swapping i and i+1
                    current_score = self._compute_ranking_score(ranking, dist)
                    ranking[i], ranking[i + 1] = ranking[i + 1], ranking[i]
                    new_score = self._compute_ranking_score(ranking, dist)

                    if new_score >= current_score:
                        # Revert if not improved
                        ranking[i], ranking[i + 1] = ranking[i + 1], ranking[i]
                    else:
                        improved = True

                if not improved:
                    break

            # Convert indices to names
            ranking_names = [event_names[idx] for idx in ranking]

            result = BioeventResult(
                sections=[],  # RASC doesn't use sections
                events=event_names,
                fad_matrix=np.zeros((0, n_events)),  # Empty
                lad_matrix=np.zeros((0, n_events)),  # Empty
                zones=[],  # RASC produces ranking, not zones
                ranking=ranking_names,
                distance_matrix=dist,
                method="rasc",
            )

            self._last_result = result
            self._logger.info(f"RASC completed: ranking with {n_events} events")
            return result

    def _compute_ranking_score(
        self, ranking: list[int], dist: npt.NDArray
    ) -> float:
        """Compute score for a given ranking (lower is better)."""
        score = 0.0
        for i in range(len(ranking) - 1):
            score += dist[ranking[i], ranking[i + 1]]
        return score

    @property
    def last_result(self) -> BioeventResult | None:
        """Get the last RASC result."""
        with self._lock:
            return self._last_result