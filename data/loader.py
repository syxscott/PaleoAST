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
version: 1.0.1
"""

from __future__ import annotations

import os
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from phylogenetics.tree import PhyloTree

# Get the data directory path
_DATA_DIR = files("data") / "examples"


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
    df = pd.read_csv(data_path)

    return df


def load_primate_tree() -> "PhyloTree":
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
    df = pd.read_csv(data_path)

    return df


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
