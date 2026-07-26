"""
================================================================================
Tests for PaleoAST Example Datasets
================================================================================

This test module verifies that example datasets can be loaded correctly
and provides golden-value comparisons for unit testing.

Tests:
------
- TPS parser can load moth_wings.tps
- Community CSV loads correctly with expected dimensions
- Primate tree parses as valid NEWICK
- Primate traits match tree tip labels
- Data loader functions return expected types

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMothWingsDataset:
    """Tests for the moth wing landmark dataset."""

    def test_load_moth_wings_shape(self):
        """Test that moth wings dataset has expected shape."""
        from data import load_moth_wings

        landmarks, specimen_ids = load_moth_wings()

        # 30 specimens, each with 8 landmarks * 2 coordinates = 16 values
        assert landmarks.shape == (30, 16), f"Expected (30, 16), got {landmarks.shape}"
        assert len(specimen_ids) == 30, f"Expected 30 specimen IDs, got {len(specimen_ids)}"

    def test_load_moth_wings_ids(self):
        """Test that specimen IDs follow expected pattern."""
        from data import load_moth_wings

        _, specimen_ids = load_moth_wings()

        # IDs should match pattern Mosquito_XXX
        for spec_id in specimen_ids:
            assert spec_id.startswith("Mosquito_"), f"Invalid ID format: {spec_id}"

    def test_load_moth_wings_no_nans(self):
        """Test that landmark data contains no NaN values."""
        from data import load_moth_wings

        landmarks, _ = load_moth_wings()

        assert not np.any(np.isnan(landmarks)), "Landmark data contains NaN values"
        assert not np.any(np.isinf(landmarks)), "Landmark data contains infinite values"

    def test_load_moth_wings_positive_coords(self):
        """Test that landmark coordinates are in expected positive range."""
        from data import load_moth_wings

        landmarks, _ = load_moth_wings()

        # X coordinates (even indices) should be roughly 80-140
        x_coords = landmarks[:, 0::2]
        assert np.all(x_coords > 50), "X coordinates unexpectedly small"
        assert np.all(x_coords < 200), "X coordinates unexpectedly large"

        # Y coordinates (odd indices) should be roughly 90-165
        y_coords = landmarks[:, 1::2]
        assert np.all(y_coords > 80), "Y coordinates unexpectedly small"
        assert np.all(y_coords < 200), "Y coordinates unexpectedly large"


class TestCommunityDataset:
    """Tests for the community abundance dataset."""

    def test_load_community_shape(self):
        """Test that community dataset has expected dimensions."""
        from data import load_community

        df = load_community()

        # 20 sites, 30 columns (site, group, 28 species)
        assert df.shape == (20, 30), f"Expected (20, 30), got {df.shape}"

    def test_load_community_columns(self):
        """Test that community dataset has expected columns."""
        from data import load_community

        df = load_community()

        assert "site" in df.columns, "Missing 'site' column"
        assert "group" in df.columns, "Missing 'group' column"

        # Check that species columns exist (at least some)
        species_cols = [c for c in df.columns if c not in ("site", "group")]
        assert len(species_cols) >= 20, f"Expected at least 20 species columns, got {len(species_cols)}"

    def test_load_community_groups(self):
        """Test that community dataset has expected habitat groups."""
        from data import load_community

        df = load_community()

        expected_groups = {"Grassland", "Woodland", "Meadow", "Wetland", "Disturbed"}
        actual_groups = set(df["group"].unique())

        assert actual_groups == expected_groups, f"Expected groups {expected_groups}, got {actual_groups}"

    def test_load_community_no_negative(self):
        """Test that abundance values are non-negative."""
        from data import load_community

        df = load_community()

        species_data = df.iloc[:, 2:]  # Exclude site and group columns
        assert (species_data >= 0).all().all(), "Abundance data contains negative values"

    def test_load_community_has_zeros(self):
        """Test that abundance data contains zeros (realistic sparse data)."""
        from data import load_community

        df = load_community()

        species_data = df.iloc[:, 2:]
        assert (species_data == 0).any().any(), "Abundance data has no zeros (unrealistic)"


class TestPrimateTreeDataset:
    """Tests for the primate phylogenetic tree dataset."""

    def test_load_primate_tree_leaf_count(self):
        """Test that primate tree has expected number of leaves."""
        from data import load_primate_tree

        tree = load_primate_tree()

        assert tree.leaf_count == 13, f"Expected 13 leaves, got {tree.leaf_count}"

    def test_load_primate_tree_leaf_names(self):
        """Test that primate tree has expected leaf names."""
        from data import load_primate_tree

        tree = load_primate_tree()

        expected_names = {
            "Homo_sapiens",
            "Pan_troglodytes",
            "Gorilla_gorilla",
            "Hylobates_lar",
            "Colobus_guereza",
            "Cercocebus_galeritus",
            "Papio_anubis",
            "Macaca_mulatta",
            "Aotus_trivirgatus",
            "Saimiri_sciureus",
            "Lemur_catta",
            "Propithecus_verreauxi",
            "Eulemur_fulvus",
        }

        assert set(tree.leaf_names) == expected_names, f"Unexpected leaf names: {tree.leaf_names}"

    def test_load_primate_tree_has_branch_lengths(self):
        """Test that primate tree has branch lengths."""
        from data import load_primate_tree

        tree = load_primate_tree()

        # Check that tree can be converted to Newick with lengths
        newick = tree.to_newick(include_lengths=True)
        assert ":" in newick, "Tree Newick representation lacks branch lengths"


class TestPrimateTraitsDataset:
    """Tests for the primate trait dataset."""

    def test_load_primate_traits_shape(self):
        """Test that primate traits dataset has expected shape."""
        from data import load_primate_traits

        df = load_primate_traits()

        # 13 species, 10 columns (species + 9 traits)
        assert df.shape == (13, 10), f"Expected (13, 10), got {df.shape}"

    def test_load_primate_traits_columns(self):
        """Test that primate traits dataset has expected columns."""
        from data import load_primate_traits

        df = load_primate_traits()

        expected_cols = {
            "species",
            "group",
            "body_mass_g",
            "home_range_km2",
            "litter_size",
            "social_group_size",
            "day_range_km",
            "brain_vol_cm3",
            "pelage_darkness",
            "activity_cycle",
        }

        assert set(df.columns) == expected_cols, f"Unexpected columns: {df.columns}"

    def test_load_primate_traits_species_match_tree(self):
        """Test that trait species match tree tip labels."""
        from data import load_primate_tree, load_primate_traits

        tree = load_primate_tree()
        traits = load_primate_traits()

        tree_species = set(tree.leaf_names)
        trait_species = set(traits["species"].values)

        assert tree_species == trait_species, (
            f"Tree species {tree_species} do not match trait species {trait_species}"
        )

    def test_load_primate_traits_no_nans(self):
        """Test that trait data contains no NaN values in numeric columns."""
        from data import load_primate_traits

        df = load_primate_traits()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not df[numeric_cols].isna().any().any(), "Trait data contains NaN values"


class TestDataLoaderIntegration:
    """Integration tests for data loader utilities."""

    def test_list_example_datasets(self):
        """Test that list_example_datasets returns expected information."""
        from data import list_example_datasets

        datasets = list_example_datasets()

        assert len(datasets) == 4, f"Expected 4 datasets, got {len(datasets)}"

        dataset_names = {d["name"] for d in datasets}
        expected_names = {"moth_wings", "community_abundance", "primate_tree", "primate_traits"}
        assert dataset_names == expected_names, f"Unexpected dataset names: {dataset_names}"

    def test_all_loaders_return_expected_types(self):
        """Test that all loader functions return expected types."""
        from data import load_community, load_moth_wings, load_primate_tree, load_primate_traits

        landmarks, ids = load_moth_wings()
        assert isinstance(landmarks, np.ndarray), "load_moth_wings landmarks not ndarray"
        assert isinstance(ids, list), "load_moth_wings ids not list"

        community_df = load_community()
        import pandas as pd

        assert isinstance(community_df, pd.DataFrame), "load_community not DataFrame"

        tree = load_primate_tree()
        assert tree.leaf_count > 0, "load_primate_tree has no leaves"

        traits_df = load_primate_traits()
        assert isinstance(traits_df, pd.DataFrame), "load_primate_traits not DataFrame"
