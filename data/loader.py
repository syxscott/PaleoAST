"""
================================================================================
PaleoAST Data Loading Utilities
================================================================================

This module provides convenient loading functions for example datasets
included with the PaleoAST package.

Datasets:
---------
- moth_wings: 2D landmark data for morphometric analysis
- community_abundance: Species abundance matrix for ecological analysis
- primate_tree: Phylogenetic tree in NEWICK format
- primate_traits: Trait data for phylogenetic comparative methods

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from models.data_matrix import DataMatrix
    from phylogenetics.tree import PhyloTree

# Get the data directory path
_DATA_DIR = files("data") / "examples"

logger = logging.getLogger(__name__)

# Encodings tried in order when reading CSVs live in ``utils.csv_io``,
# shared with ``controllers/data_controller.py`` so the interactive import path
# and the example-data loaders cannot drift apart.
from utils.csv_io import read_csv_with_fallback


def _get_data_path(filename: str) -> str:
    """
    Get the full path to an example data file.

    Parameters:
        filename: Name of the data file

    Returns:
        Full path to the data file
    """
    data_path = _DATA_DIR / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Example data file not found: {filename}\n"
            f"Expected path: {data_path}\n"
            "Please ensure the data package is properly installed."
        )
    return str(data_path)


def load_moth_wings() -> tuple[np.ndarray, list[str]]:
    """
    Load the moth wing landmark dataset.

    This dataset contains 2D landmark coordinates for 30 mosquito wing
    specimens (Culex pipiens), with 8 landmarks per specimen.

    The data is based on the classic morphometric dataset from:
    Rohlf, F.J. (1990). Morphometrics. Annual Review of Ecology
    and Systematics, 21, 299-316.

    Returns:
        Tuple of:
            - landmarks: np.ndarray of shape (30, 16) containing
              flattened landmark coordinates (8 landmarks x 2 coords)
            - specimen_ids: List of specimen identifiers

    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If the TPS file cannot be parsed

    Example:
        >>> landmarks, ids = load_moth_wings()
        >>> landmarks.shape
        (30, 16)
        >>> len(ids)
        30
    """
    from parsers.tps_parser import TPSParser

    data_path = _get_data_path("moth_wings.tps")
    parser = TPSParser()
    tps_file = parser.parse(data_path)

    # Extract landmarks and specimen IDs
    landmarks = tps_file.to_matrix()
    specimen_ids = [spec.id for spec in tps_file.specimens]

    return landmarks, specimen_ids


def load_community() -> pd.DataFrame:
    """
    Load the community abundance dataset.

    This dataset contains species abundance data for 20 sites across
    5 habitat types, with 28 plant species. Based on the structure
    of classic vegan package datasets (varespec, dune).

    Data source: Simulated data based on Jongman et al. (1995)
    Data Analysis in Community and Landscape Ecology.

    Returns:
        DataFrame with columns:
            - site: Site identifier
            - group: Habitat type
            - 28 species abundance columns

    Raises:
        FileNotFoundError: If the data file is not found

    Example:
        >>> df = load_community()
        >>> df.shape
        (20, 30)
        >>> df['group'].unique()
        array(['Grassland', 'Woodland', 'Meadow', 'Wetland', 'Disturbed'])
    """
    data_path = _get_data_path("community_abundance.csv")
    df, _encoding = read_csv_with_fallback(str(data_path))

    return df


def load_primate_tree() -> PhyloTree:
    """
    Load the primate phylogenetic tree.

    This dataset contains a composite primate phylogeny with 13 tips
    representing major primate groups. Branch lengths are in millions
    of years.

    Based on: Springer et al. (2012). Macroevolutionary dynamics and
    historical biogeography of primate diversification. PLoS ONE 7(11): e49521.

    Returns:
        PhyloTree object with 13 leaf nodes

    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If the NEWICK file cannot be parsed

    Example:
        >>> tree = load_primate_tree()
        >>> tree.leaf_count
        13
        >>> tree.leaf_names[:3]
        ['Homo_sapiens', 'Pan_troglodytes', 'Gorilla_gorilla']
    """
    from phylogenetics.tree import PhyloTree

    data_path = _get_data_path("primate_tree.nwk")

    with open(data_path, encoding="utf-8") as f:
        content = f.read()

    # Remove comment lines for parsing
    lines = [line for line in content.split("\n") if not line.strip().startswith("!")]
    newick = "\n".join(lines)

    tree = PhyloTree.from_newick(newick)

    return tree


def load_primate_traits() -> pd.DataFrame:
    """
    Load the primate trait dataset.

    This dataset contains morphological, behavioral, and ecological
    trait data for 13 primate species matching the phylogenetic tree tips.

    Compiled from:
    - Smithsonian National Museum of Natural History
    - Springer et al. (2012). PLoS ONE 7(11): e49521.

    Returns:
        DataFrame with columns:
            - species: Species name
            - group: Higher taxonomic group
            - body_mass_g: Body mass in grams
            - home_range_km2: Home range in km²
            - litter_size: Number of offspring
            - social_group_size: Mean group size
            - day_range_km: Daily travel distance
            - brain_vol_cm3: Brain volume
            - pelage_darkness: Pelage darkness score (0-1)
            - activity_cycle: Diurnal (1.0) or nocturnal (0.0)

    Raises:
        FileNotFoundError: If the data file is not found

    Example:
        >>> traits = load_primate_traits()
        >>> traits.shape
        (13, 10)
        >>> traits['species'].tolist() == load_primate_tree().leaf_names
        True
    """
    data_path = _get_data_path("primate_traits.csv")
    df, _encoding = read_csv_with_fallback(str(data_path))

    return df


# The example CSVs carry TEXT columns that are not measurements: a row key
# ("site", "species") and a grouping label ("group" -- a habitat type in
# one dataset, a taxonomic name such as "Catarrhini" in the other).  The
# loaders above return them as a DataFrame, text columns included, which
# is the right contract for a loader.
#
# Turning that into the numeric matrix the application analyses is a
# SEPARATE step, and it is where the text has to be separated from the
# numbers.  Leaving a text column in makes ``to_numpy(dtype=float)`` raise
# "could not convert string to float: 'Catarrhini'", which is exactly
# what happened to the bundled primate_traits example: the file was
# shipped, documented, listed in the example picker, and could not be
# loaded.
#
# It lives here rather than in the view so that it can be tested without
# building a main window.  The grouping label is kept as specimen
# metadata, because it is the variable these datasets exist to be
# compared on.

_EXAMPLE_ROW_KEY = {
    "community_abundance": "site",
    "primate_traits": "species",
}
_EXAMPLE_GROUP_COLUMN = "group"


def load_example_matrix(name: str) -> DataMatrix:
    """Build the numeric :class:`DataMatrix` for a bundled example dataset.

    Parameters:
        name: One of ``"moth_wings"``, ``"community_abundance"`` or
            ``"primate_traits"``.  ``"primate_tree"`` is a phylogeny, not
            a matrix, and is not handled here.

    Returns:
        DataMatrix with the numeric columns only, the row key as row
        labels, and the grouping label as specimen metadata.

    Raises:
        ValueError: If ``name`` is not a matrix-shaped example dataset.
        FileNotFoundError: If the data file is not found.
    """
    from models.data_matrix import DataMatrix

    if name == "moth_wings":
        landmarks, specimen_ids = load_moth_wings()
        data = np.asarray(landmarks, dtype=float)
        n_points = data.shape[1] // 2
        col_labels = [f"{c}{i + 1}" for i in range(n_points) for c in ("x", "y")]
        return DataMatrix(data, row_labels=list(specimen_ids), col_labels=col_labels)

    if name == "community_abundance":
        frame = load_community()
    elif name == "primate_traits":
        frame = load_primate_traits()
    else:
        raise ValueError(f"Not a matrix-shaped example dataset: {name!r}")

    row_key = _EXAMPLE_ROW_KEY.get(name)
    row_labels = frame[row_key].astype(str).tolist() if row_key and row_key in frame.columns else None

    group_column = _EXAMPLE_GROUP_COLUMN
    group_values = frame[group_column].astype(str).tolist() if group_column in frame.columns else None

    measurements = frame.drop(columns=[c for c in (row_key, group_column) if c and c in frame.columns])
    # Only the numeric columns are measurements.  Selecting them explicitly
    # means an unexpected text column in a future dataset is ignored
    # rather than crashing the load with a raw ValueError from numpy.
    numeric = measurements.select_dtypes(include=[np.number])

    # The specimen metadata always has one entry per data row.  When the
    # frame carries a group column the entry names the group; when it does
    # not, the entry is empty.  This used to be one expression,
    # ``group_values or [None] * len(numeric)``, which mixed a list of
    # strings with a list of None and left a type checker guessing which
    # of the two the iteration produced.
    specimen_metadata: list[dict[str, Any]] = (
        [{"group": group} for group in group_values] if group_values is not None else [{} for _ in range(len(numeric))]
    )

    return DataMatrix(
        numeric.to_numpy(dtype=float),
        row_labels=row_labels,
        col_labels=list(numeric.columns),
        specimen_metadata=specimen_metadata,
    )


def list_example_datasets() -> list[dict[str, str]]:
    """
    List all available example datasets.

    Returns:
        List of dictionaries with dataset information:
            - name: Dataset name
            - file: Data file name
            - description: Brief description

    Example:
        >>> datasets = list_example_datasets()
        >>> len(datasets)
        4
    """
    return [
        {
            "name": "moth_wings",
            "file": "moth_wings.tps",
            "description": "2D landmark data (30 specimens x 8 landmarks)",
        },
        {
            "name": "community_abundance",
            "file": "community_abundance.csv",
            "description": "Species abundance matrix (20 sites x 28 species)",
        },
        {
            "name": "primate_tree",
            "file": "primate_tree.nwk",
            "description": "Primate phylogeny (13 tips)",
        },
        {
            "name": "primate_traits",
            "file": "primate_traits.csv",
            "description": "Primate trait data (13 species x 8 traits)",
        },
    ]
