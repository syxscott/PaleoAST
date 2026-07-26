"""
================================================================================
PaleoAST Data Package
================================================================================

This package provides example datasets for testing, demonstration,
and teaching purposes in the PaleoAST paleontological analysis platform.

Available Datasets
------------------
- moth_wings: 2D landmark data for morphometric analysis (GPA, EFA, TPS)
- community_abundance: Species abundance matrix for ecological analysis
- primate_tree: Phylogenetic tree in NEWICK format
- primate_traits: Trait data for phylogenetic comparative methods

Quick Start
-----------
>>> from data import load_moth_wings, load_community, load_primate_tree
>>> landmarks, ids = load_moth_wings()
>>> community_df = load_community()
>>> tree = load_primate_tree()

Author: PaleoAST Development Team
version: 1.0.1
"""

from data.loader import (
    load_community,
    load_moth_wings,
    load_primate_traits,
    load_primate_tree,
    list_example_datasets,
)

__all__ = [
    "load_moth_wings",
    "load_community",
    "load_primate_tree",
    "load_primate_traits",
    "list_example_datasets",
]
