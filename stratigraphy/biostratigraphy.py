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
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

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
    # Optional hierarchical metadata for Unitary Association Zones (UAZ)
    uaz_id: int | None = None
    uaz_name: str | None = None
    parent_ua_indices: list[int] | None = None
    dissimilarity_to_predecessor: float | None = None


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
        uaz_groups: Optional list of merged Unitary Association Zones (UAZ).
            Each entry is a dict with keys ``uaz_id``, ``uaz_name``,
            ``zone_indices`` and ``event_union`` describing the merging
            hierarchy. ``None`` if the merging step was not executed.
    """

    sections: list[str]
    events: list[str]
    fad_matrix: npt.NDArray
    lad_matrix: npt.NDArray
    zones: list[Zone]
    ranking: list[str]
    distance_matrix: npt.NDArray | None
    method: str
    uaz_groups: list[dict[str, object]] | None = None
    endemic_filtered_events: list[str] | None = None
    cyclic_contradictions: list[dict[str, object]] | None = None

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

        if self.uaz_groups is not None:
            lines.append(_("Unitary Association Zones (UAZ): {0}").format(len(self.uaz_groups)))
            for grp in self.uaz_groups[:5]:
                lines.append(
                    _("  - {0}: {1} UA(s), {2} events").format(
                        grp.get("uaz_name", "?"),
                        len(grp.get("zone_indices", [])),  # type: ignore[arg-type]
                        len(grp.get("event_union", [])),  # type: ignore[arg-type]
                    )
                )
            if len(self.uaz_groups) > 5:
                lines.append(
                    _("  ... ({0} more UAZ)").format(len(self.uaz_groups) - 5)
                )
            lines.append("")

        if self.endemic_filtered_events:
            lines.append(
                _("Endemic species filtered: {0}").format(len(self.endemic_filtered_events))
            )

        if self.cyclic_contradictions:
            lines.append(
                _("Cyclic contradictions detected: {0}").format(len(self.cyclic_contradictions))
            )

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
        min_section_occurrence: int = 2,
        uaz_similarity_threshold: float = 0.8,
        enable_cyclic_check: bool = True,
    ) -> BioeventResult:
        """
        Perform Unitary Associations analysis.

        This orchestrator method chains together three advanced industrial-grade
        preprocessing & post-processing steps around the classical Bron-Kerbosch
        maximum-clique finder:

        1. ``filter_endemic_species`` removes endemic taxa whose cross-section
           occurrence is below ``min_section_occurrence``.
        2. ``_detect_cyclic_contradictions`` flags FAD/LAD inversions between
           sections (only when ``enable_cyclic_check=True``).
        3. ``_merge_to_uaz`` aggregates highly similar maximal cliques into
           Unitary Association Zones (UAZ) according to the
           ``uaz_similarity_threshold`` parameter.

        Parameters:
            fad_matrix: First Appearance Datum matrix (n_sections, n_events)
            lad_matrix: Last Appearance Datum matrix (n_sections, n_events)
            section_names: Names of sections
            event_names: Names of events
            min_section_occurrence: Minimum number of distinct sections in which
                a taxon must appear to be retained. Endemic taxa occurring in
                fewer sections are filtered out. Must be ``>= 1``; default 2.
            uaz_similarity_threshold: Sørensen-style similarity threshold in
                ``[0, 1]`` used by ``_merge_to_uaz`` to merge highly
                overlapping Unitary Associations into Unitary Association
                Zones. ``0.8`` is the empirical Guex default.
            enable_cyclic_check: When False, skip the
                :meth:`_detect_cyclic_contradictions` pass. This is
                useful for trusted datasets where the user explicitly
                wants to suppress the (relatively expensive) O(N^2)
                pairwise scan.

        Returns:
            BioeventResult: UA analysis results with identified zones, optional
                UAZ grouping information, the list of endemic-filtered events
                and any cyclic contradictions detected.

        Raises:
            ComputationError: If the FAD and LAD matrices do not share the
                same shape.
            DataValidationError: If the input matrices fail validation.
        """
        with self._lock:
            # Validate input
            fad = validate_data_array(fad_matrix, allow_nan=False, name="fad_matrix")
            lad = validate_data_array(lad_matrix, allow_nan=False, name="lad_matrix")

            if fad.shape != lad.shape:
                raise ComputationError("FAD and LAD matrices must have same shape")

            n_sections, n_events = fad.shape

            self._logger.info(f"UA analyze started: {n_sections} sections, {n_events} events")

            # Default names
            if section_names is None:
                section_names = [f"Section_{i + 1}" for i in range(n_sections)]
            if event_names is None:
                event_names = [f"Event_{i + 1}" for i in range(n_events)]

            # ------------------------------------------------------------------
            # Step 1: Endemic-species filtering
            # ------------------------------------------------------------------
            fad, lad, event_names, endemic_filtered = self.filter_endemic_species(
                fad=fad,
                lad=lad,
                event_names=event_names,
                min_section_occurrence=min_section_occurrence,
            )

            # ------------------------------------------------------------------
            # Step 2: Cyclic contradiction detection (opt-in via UI / API)
            # ------------------------------------------------------------------
            if enable_cyclic_check:
                cyclic_contradictions = self._detect_cyclic_contradictions(
                    fad=fad, lad=lad, event_names=event_names
                )
            else:
                self._logger.info(
                    _("Cyclic contradiction detection skipped by user request")
                )
                cyclic_contradictions = []

            # Build overlap graph
            # Two events overlap if they co-occur in at least one section
            # i.e., FAD[i] < LAD[j] and FAD[j] < LAD[i]
            overlap_graph = self._build_overlap_graph(fad, lad)

            # Find maximal cliques using Bron-Kerbosch algorithm
            cliques = self._bron_kerbosch(overlap_graph)

            # Convert cliques to zones
            zones = self._cliques_to_zones(cliques, event_names, section_names, fad, lad)

            # ------------------------------------------------------------------
            # Step 3: Merge highly similar cliques into UAZ
            # ------------------------------------------------------------------
            uaz_groups = self._merge_to_uaz(
                cliques=cliques,
                event_names=event_names,
                similarity_threshold=uaz_similarity_threshold,
            )

            # Annotate zones with UAZ hierarchy metadata (in place)
            if uaz_groups:
                # Build a single-pass index instead of recomputing
                # ``next(...)`` for every zone-to-UAZ link.
                uaz_by_id: dict[int, dict[str, object]] = {
                    int(u["uaz_id"]): u for u in uaz_groups  # type: ignore[arg-type]
                }
                zone_lookup: dict[int, list[int]] = {idx: [] for idx in range(len(zones))}
                for uaz in uaz_groups:
                    for z_idx in uaz["zone_indices"]:  # type: ignore[index]
                        zone_lookup[int(z_idx)].append(int(uaz["uaz_id"]))  # type: ignore[arg-type]

                for z_idx, uaz_ids in zone_lookup.items():
                    if not uaz_ids:
                        continue
                    uaz_id = uaz_ids[0]
                    uaz = uaz_by_id.get(uaz_id)
                    if uaz is None:
                        continue
                    zones[z_idx].uaz_id = uaz_id
                    zones[z_idx].uaz_name = str(uaz.get("uaz_name", f"UAZ {uaz_id}"))
                    zones[z_idx].parent_ua_indices = list(uaz.get("zone_indices", []))  # type: ignore[arg-type]
                    # Propagate the chain-level dissimilarity-to-predecessor
                    # onto every zone that belongs to this UAZ so that
                    # downstream consumers (tree view, reports) can render
                    # a per-zone "distance from previous UAZ" column.
                    pred_d = uaz.get("dissimilarity_to_predecessor", 0.0)
                    try:
                        zones[z_idx].dissimilarity_to_predecessor = float(pred_d)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        zones[z_idx].dissimilarity_to_predecessor = None

            result = BioeventResult(
                sections=section_names,
                events=event_names,
                fad_matrix=fad,
                lad_matrix=lad,
                zones=zones,
                ranking=[],  # UA doesn't produce ranking
                distance_matrix=None,
                method="ua",
                uaz_groups=uaz_groups,
                endemic_filtered_events=endemic_filtered,
                cyclic_contradictions=cyclic_contradictions,
            )

            self._last_result = result
            self._logger.info(
                _("UA completed: {0} zones, {1} UAZ groups, {2} endemic filtered, {3} cyclic conflicts").format(
                    len(zones),
                    len(uaz_groups) if uaz_groups else 0,
                    len(endemic_filtered),
                    len(cyclic_contradictions),
                )
            )
            return result

    def _build_overlap_graph(self, fad: npt.NDArray, lad: npt.NDArray) -> dict[int, set[int]]:
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

        def backtrack(r: set[int], p: set[int], x: set[int]) -> None:
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

    # =====================================================================
    # Industrial-Grade Extension Methods
    # =====================================================================

    def filter_endemic_species(
        self,
        fad: npt.NDArray,
        lad: npt.NDArray,
        event_names: list[str],
        min_section_occurrence: int = 2,
    ) -> tuple[npt.NDArray, npt.NDArray, list[str], list[str]]:
        """
        Filter out endemic ('singular' or 'local') species from the FAD/LAD matrices.

        A species is considered endemic if it occurs (i.e. has a finite FAD
        smaller than its LAD) in strictly fewer than ``min_section_occurrence``
        sections. The filtering logic follows the classic UAgraph approach
        where locally-restricted taxa introduce spurious contradictions and
        inflate the overlap graph.

        Parameters:
            fad: First Appearance Datum matrix of shape (n_sections, n_events).
            lad: Last Appearance Datum matrix of shape (n_sections, n_events).
            event_names: List of event (species) names of length ``n_events``.
            min_section_occurrence: Minimum number of distinct sections in
                which a species must be present to be retained. Values
                ``< 1`` are clamped to ``1``.

        Returns:
            A tuple ``(fad_filtered, lad_filtered, event_names_filtered,
            endemic_filtered_names)`` where:

            * ``fad_filtered`` and ``lad_filtered`` are the FAD/LAD matrices
              with endemic columns removed.
            * ``event_names_filtered`` is the pruned list of event names.
            * ``endemic_filtered_names`` is the list of removed species,
              kept for downstream reporting.

        Raises:
            DataValidationError: If the input matrices are dimensionally
                inconsistent.
        """
        with self._lock:
            if min_section_occurrence < 1:
                self._logger.warning(
                    _("min_section_occurrence={0} is below 1; clamping to 1").format(min_section_occurrence)
                )
                min_section_occurrence = 1

            fad_arr = np.asarray(fad, dtype=np.float64)
            lad_arr = np.asarray(lad, dtype=np.float64)
            if fad_arr.shape != lad_arr.shape:
                raise ComputationError("FAD and LAD matrices must have same shape")
            if len(event_names) != fad_arr.shape[1]:
                raise ComputationError("event_names length does not match matrix width")

            n_sections, n_events = fad_arr.shape
            endemic_mask: list[bool] = []

            for col in range(n_events):
                fad_col = fad_arr[:, col]
                lad_col = lad_arr[:, col]
                # A species is considered present in a section whenever it
                # has a finite, well-defined stratigraphic range, *including*
                # a single-layer occurrence where FAD == LAD. Using a strict
                # ``<`` would wrongly mark every single-point taxon as
                # "absent" and silently filter it out, contradicting the
                # classical UAgraph convention where any recorded
                # occurrence (point or interval) counts.
                finite_mask = np.isfinite(fad_col) & np.isfinite(lad_col)
                present_mask = finite_mask & (fad_col <= lad_col)
                n_present = int(np.sum(present_mask))
                if n_present < min_section_occurrence:
                    endemic_mask.append(True)
                else:
                    endemic_mask.append(False)

            endemic_indices = [i for i, m in enumerate(endemic_mask) if m]
            endemic_filtered_names = [event_names[i] for i in endemic_indices]
            kept_indices = [i for i, m in enumerate(endemic_mask) if not m]

            if not kept_indices:
                self._logger.warning(
                    _("filter_endemic_species removed every species; keeping at least one to avoid empty matrix")
                )
                # Defensive fallback: keep the most widespread species
                counts = [
                    int(np.sum(np.isfinite(fad_arr[:, c]) & np.isfinite(lad_arr[:, c]) & (fad_arr[:, c] <= lad_arr[:, c])))
                    for c in range(n_events)
                ]
                best = int(np.argmax(counts)) if counts else 0
                kept_indices = [best]
                endemic_filtered_names = [event_names[i] for i in range(n_events) if i != best]

            fad_filtered = fad_arr[:, kept_indices]
            lad_filtered = lad_arr[:, kept_indices]
            event_names_filtered = [event_names[i] for i in kept_indices]

            self._logger.info(
                _("filter_endemic_species: removed {0} endemic species (kept {1} / {2})").format(
                    len(endemic_filtered_names), len(event_names_filtered), n_events
                )
            )
            return fad_filtered, lad_filtered, event_names_filtered, endemic_filtered_names

    def _detect_cyclic_contradictions(
        self,
        fad: npt.NDArray,
        lad: npt.NDArray,
        event_names: list[str],
    ) -> list[dict[str, object]]:
        """
        Detect cyclic FAD/LAD ordering contradictions across sections.

        For every pair of events ``(i, j)`` the algorithm builds a directed
        edge ``i -> j`` in section ``s`` when the FAD of ``i`` is older than
        that of ``j`` (FAD_i[s] < FAD_j[s]). If, in another section ``t``,
        the order is reversed (FAD_j[t] < FAD_i[t]), this produces a
        logical inversion. A pair of events whose mutual ordering
        *oscillates* across sections is reported as a strong contradiction.

        Parameters:
            fad: First Appearance Datum matrix of shape (n_sections, n_events).
            lad: Last Appearance Datum matrix of shape (n_sections, n_events).
            event_names: List of event (species) names of length ``n_events``.

        Returns:
            A list of dictionaries, each describing a detected cyclic
            contradiction. The dictionary contains the event names, the
            number of sections supporting each ordering, and the actual
            section indices for diagnostic purposes.
        """
        with self._lock:
            fad_arr = np.asarray(fad, dtype=np.float64)
            n_sections, n_events = fad_arr.shape
            if n_events < 2:
                return []

            contradictions: list[dict[str, object]] = []

            # Build a pairwise voting table: for each unordered pair (i, j)
            # count how many sections place i before j and j before i.
            for i in range(n_events):
                for j in range(i + 1, n_events):
                    # Ignore pairs where any FAD value is not strictly defined
                    fi = fad_arr[:, i]
                    fj = fad_arr[:, j]
                    valid_mask = np.isfinite(fi) & np.isfinite(fj)
                    if not np.any(valid_mask):
                        continue

                    order_i_before_j = (fi < fj) & valid_mask
                    order_j_before_i = (fj < fi) & valid_mask
                    n_i_before_j = int(np.sum(order_i_before_j))
                    n_j_before_i = int(np.sum(order_j_before_i))

                    # A cyclic contradiction exists when the two species
                    # show mutually exclusive orderings in different sections.
                    if n_i_before_j > 0 and n_j_before_i > 0:
                        sections_i = np.where(order_i_before_j)[0].tolist()
                        sections_j = np.where(order_j_before_i)[0].tolist()
                        entry = {
                            "event_a": event_names[i],
                            "event_b": event_names[j],
                            "n_sections_a_before_b": n_i_before_j,
                            "n_sections_b_before_a": n_j_before_i,
                            "sections_a_before_b": sections_i,
                            "sections_b_before_a": sections_j,
                        }
                        contradictions.append(entry)
                        # NOTE: emit per-pair details at DEBUG level only.
                        # An O(N^2) WARNING per contradiction would flood
                        # the log and the GUI status bar for any non-trivial
                        # dataset; the aggregated WARNING below is enough
                        # for the user, and the structured entries are still
                        # available on ``BioeventResult.cyclic_contradictions``.
                        self._logger.debug(
                            (
                                "Cyclic FAD contradiction: '%s' precedes '%s' in "
                                "%d section(s), but is preceded by it in %d section(s) "
                                "-> sections=%s/%s"
                            ),
                            event_names[i],
                            event_names[j],
                            n_i_before_j,
                            n_j_before_i,
                            sections_i,
                            sections_j,
                        )

            if not contradictions:
                self._logger.info(
                    _("No cyclic FAD contradictions detected across {0} sections").format(n_sections)
                )
            else:
                self._logger.warning(
                    _("Detected {0} cyclic FAD contradiction(s) across {1} sections").format(
                        len(contradictions), n_sections
                    )
                )

            return contradictions

    def _merge_to_uaz(
        self,
        cliques: list[set[int]],
        event_names: list[str],
        similarity_threshold: float = 0.8,
    ) -> list[dict[str, object]]:
        """
        Merge highly similar maximal cliques into Unitary Association Zones (UAZ).

        This implements the classical Guex (1991) biozone-merging heuristic:
        the dissimilarity index between two adjacent Unitary Associations is
        computed as the symmetric difference divided by the intersection. A
        normalised *similarity* is then derived as
        ``sim = 1 - 0.5 * (|A xor B| / |A ∩ B|)`` (clamped to ``[0, 1]``) and
        cliques whose similarity is at least ``similarity_threshold`` are
        merged into a single UAZ.

        Mathematical formulation::

            d(A, B) = |A xor B| / |A ∩ B|       (dissimilarity index)
            sim(A, B) = max(0.0, 1 - 0.5 * d(A, B))
            merge iff sim(A, B) >= threshold

        Note:
            Identical cliques have ``sim = 1`` (dissimilarity 0); cliques
            sharing no species have ``sim = 0``.

        Parameters:
            cliques: List of maximal cliques (each a set of event indices)
                produced by the Bron-Kerbosch routine.
            event_names: List of event names of length ``n_events``.
            similarity_threshold: Threshold in ``[0, 1]`` above which two
                adjacent UAs are merged. Default 0.8 follows Guex (1991).

        Returns:
            A list of dictionaries describing each merged UAZ group. Each
            entry has the keys ``uaz_id``, ``uaz_name``, ``zone_indices``,
            ``event_union``, ``mean_similarity`` and
            ``dissimilarity_to_predecessor``.
        """
        with self._lock:
            if not cliques:
                return []

            similarity_threshold = float(np.clip(similarity_threshold, 0.0, 1.0))

            # Convert cliques to frozensets for hashing
            clique_sets: list[frozenset[int]] = [frozenset(c) for c in cliques]

            # Working copies - we will iteratively merge
            merged_indices: list[list[int]] = [[i] for i in range(len(clique_sets))]
            merged_sets: list[set[int]] = [set(s) for s in clique_sets]
            # ``dissimilarity_to_prev[k]`` is the d(merged_sets[k-1], merged_sets[k])
            # i.e. the dissimilarity *to the predecessor in the current chain*.
            # The first element has no predecessor, so its slot is 0.0.
            dissimilarity_to_prev: list[float] = [0.0] * len(merged_indices)

            def _pair_dissimilarity(a_set: set[int], b_set: set[int]) -> float:
                inter = a_set & b_set
                if not inter:
                    return float("inf")
                union = a_set | b_set
                return (len(union) - len(inter)) / len(inter)

            def _pair_similarity(a_set: set[int], b_set: set[int]) -> float:
                d = _pair_dissimilarity(a_set, b_set)
                if not np.isfinite(d):
                    return 0.0
                return float(max(0.0, 1.0 - 0.5 * d))

            # Initialise the predecessor chain consistently with the
            # current adjacency.
            for k in range(1, len(merged_sets)):
                dissimilarity_to_prev[k] = _pair_dissimilarity(
                    merged_sets[k - 1], merged_sets[k]
                )

            # Iterative single-link-style merging: pick the best adjacent
            # similarity, merge it, recompute, repeat until threshold unmet.
            progress = True
            # Counter for assigning sequential UAZ identifiers.
            uaz_counter = 0
            # Final result will be the order in which the chain is finalised
            chain: list[dict[str, object]] = []

            # Use a simple priority approach: at each iteration, find the
            # best similarity among adjacent (chronologically-ordered) groups
            # and merge the best pair.
            while progress:
                progress = False
                best_sim = similarity_threshold
                best_pair: tuple[int, int] | None = None

                # We consider pairs (a, b) such that b = a + 1 in the
                # current merge chain (post-sort). This is the simple
                # "adjacent UA" semantics used by Guex.
                for a in range(len(merged_sets) - 1):
                    b = a + 1
                    sim = _pair_similarity(merged_sets[a], merged_sets[b])
                    if sim >= best_sim:
                        best_sim = sim
                        best_pair = (a, b)

                if best_pair is not None:
                    a, b = best_pair
                    new_set = merged_sets[a] | merged_sets[b]
                    new_indices = merged_indices[a] + merged_indices[b]
                    # Replace position ``a`` with the merged entry and
                    # drop position ``b``.
                    merged_sets = [merged_sets[k] for k in range(len(merged_sets)) if k != b]
                    merged_indices = [merged_indices[k] for k in range(len(merged_indices)) if k != b]
                    dissimilarity_to_prev = [
                        dissimilarity_to_prev[k] for k in range(len(dissimilarity_to_prev)) if k != b
                    ]
                    merged_sets[a] = new_set
                    merged_indices[a] = new_indices

                    # Re-evaluate the dissimilarity-to-predecessor *chain*
                    # links around the modified position. ``a`` now holds
                    # the merged group; its predecessor link uses
                    # merged_sets[a-1], and the next position (a+1) is the
                    # former (b+1) entry whose predecessor is now the new
                    # merged group.
                    if a > 0:
                        dissimilarity_to_prev[a] = _pair_dissimilarity(
                            merged_sets[a - 1], merged_sets[a]
                        )
                    else:
                        dissimilarity_to_prev[a] = 0.0
                    if a + 1 < len(merged_sets):
                        dissimilarity_to_prev[a + 1] = _pair_dissimilarity(
                            merged_sets[a], merged_sets[a + 1]
                        )
                    progress = True

            # Build UAZ groups
            for chain_idx, idx_list in enumerate(merged_indices):
                union_set = set().union(*[set(clique_sets[i]) for i in idx_list])
                event_union = [event_names[i] for i in sorted(union_set)]
                # Mean intra-UAZ similarity from consecutive UA pairs.
                diss_values: list[float] = []
                for k, j in enumerate(idx_list):
                    if k == 0:
                        continue
                    d = _pair_dissimilarity(set(clique_sets[idx_list[k - 1]]), set(clique_sets[j]))
                    diss_values.append(d)

                if diss_values and all(np.isfinite(diss_values)):
                    mean_sim = float(max(0.0, 1.0 - 0.5 * float(np.mean(diss_values))))
                else:
                    mean_sim = 1.0 if len(idx_list) == 1 else 0.0

                uaz_counter += 1
                # ``dissimilarity_to_predecessor`` is the *chain-level*
                # dissimilarity to the previous UAZ, NOT the within-merge
                # dissimilarity of the last merge step. The previous
                # implementation conflated the two and reported the
                # internal-merge value, which is unrelated to the
                # neighbouring UAZ.
                pred_d = (
                    float(dissimilarity_to_prev[chain_idx])
                    if chain_idx < len(dissimilarity_to_prev)
                    else 0.0
                )
                if not np.isfinite(pred_d):
                    pred_d = float("inf")
                chain.append(
                    {
                        "uaz_id": uaz_counter,
                        "uaz_name": f"UAZ {uaz_counter}",
                        "zone_indices": list(idx_list),
                        "event_union": event_union,
                        "mean_similarity": mean_sim,
                        "dissimilarity_to_predecessor": pred_d,
                    }
                )

            self._logger.info(
                _("_merge_to_uaz: built {0} UAZ groups from {1} UAs (threshold={2})").format(
                    len(chain), len(clique_sets), similarity_threshold
                )
            )
            return chain

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

                    # The score is a *cost* (lower is better), so we
                    # accept a swap only if it *decreases* the score.
                    # The previous code used ``new_score >= current_score``
                    # to revert, which would never accept any swap
                    # (since the initial ordering was already chosen
                    # and the score could only grow or stay the same).
                    if new_score < current_score:
                        improved = True
                    else:
                        # Revert the swap
                        ranking[i], ranking[i + 1] = ranking[i + 1], ranking[i]

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

    def _compute_ranking_score(self, ranking: list[int], dist: npt.NDArray) -> float:
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
