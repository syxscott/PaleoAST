# =============================================================================
# Test: Newick parser - BOM, NHX metadata, quoted labels, comments
# =============================================================================
"""
Tests for enhanced Newick parser features.

Covers:
- UTF-8 BOM stripping in parse_file
- NHX [&&NHX ...] metadata extraction (canonical ':' and comma separators,
  leaf and internal nodes, before and after branch length)
- Quoted labels ('Homo sapiens' or "Homo sapiens")
- [comment] skipping (including same-line comments)
- Unbalanced parentheses and stray ')' must raise ValueError
- to_newick round-trip for labels with spaces/special characters
"""

from __future__ import annotations

import os
import tempfile

import pytest

from parsers.newick_parser import NewickParser, parse_newick


def _walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


class TestNewickParserEnhancements:
    """Verify Newick parser handles BOM, NHX, quoted labels, comments."""

    def test_quoted_label_single_quote(self):
        """Quoted labels with single quotes should work."""
        newick = "('Homo sapiens':0.1,'Pan troglodytes':0.2);"
        tree = parse_newick(newick)
        leaves = tree.leaf_names
        assert "Homo sapiens" in leaves
        assert "Pan troglodytes" in leaves

    def test_quoted_label_double_quote(self):
        """Quoted labels with double quotes should work."""
        newick = '("Homo sapiens":0.1,"Pan troglodytes":0.2);'
        tree = parse_newick(newick)
        leaves = tree.leaf_names
        assert "Homo sapiens" in leaves
        assert "Pan troglodytes" in leaves

    def test_nhx_metadata_internal_node_colon_separator(self):
        """Canonical NHX (':'-separated attrs) must land on the internal node."""
        newick = "((A:0.1,B:0.2)[&&NHX:B=95:D=N],C:0.3);"
        tree = parse_newick(newick)
        nhx_nodes = [n for n in _walk(tree.root) if n.metadata]
        assert len(nhx_nodes) == 1
        assert not nhx_nodes[0].is_leaf
        assert nhx_nodes[0].metadata == {"B": "95", "D": "N"}
        assert tree.leaf_count == 3

    def test_nhx_metadata_comma_separator(self):
        """Comma-separated NHX attrs must also parse."""
        newick = "((A:0.1,B:0.2)[&&NHX B=95,D=N],C:0.3);"
        tree = parse_newick(newick)
        nhx_nodes = [n for n in _walk(tree.root) if n.metadata]
        assert nhx_nodes[0].metadata == {"B": "95", "D": "N"}

    def test_nhx_metadata_before_branch_length(self):
        """NHX between the closing paren and the branch length."""
        newick = "((A:0.1,B:0.2)[&&NHX:B=95]:1.5,C:0.3);"
        tree = parse_newick(newick)
        nhx_nodes = [n for n in _walk(tree.root) if n.metadata]
        assert nhx_nodes[0].metadata == {"B": "95"}
        assert nhx_nodes[0].branch_length == pytest.approx(1.5)
        assert nhx_nodes[0].name == ""

    def test_nhx_metadata_after_branch_length(self):
        """NHX after the branch length: (...):1.5[&&NHX:B=95]."""
        newick = "((A:0.1,B:0.2):1.5[&&NHX:B=95],C:0.3);"
        tree = parse_newick(newick)
        nhx_nodes = [n for n in _walk(tree.root) if n.metadata]
        assert nhx_nodes[0].metadata == {"B": "95"}
        assert nhx_nodes[0].branch_length == pytest.approx(1.5)

    def test_nhx_metadata_on_leaf(self):
        """Leaf-level NHX (the most common placement, e.g. ETE3 output)."""
        newick = "(A[&&NHX:S=human]:0.1,B:0.2);"
        tree = parse_newick(newick)
        leaves = {leaf.name: leaf for leaf in tree.root.get_leaves()}
        assert set(leaves) == {"A", "B"}
        assert leaves["A"].metadata == {"S": "human"}
        assert leaves["A"].branch_length == pytest.approx(0.1)
        assert leaves["B"].metadata == {}

    def test_nhx_metadata_on_leaf_after_length(self):
        """Leaf NHX after the branch length."""
        newick = "(A:0.1[&&NHX:S=human],B:0.2);"
        tree = parse_newick(newick)
        leaves = {leaf.name: leaf for leaf in tree.root.get_leaves()}
        assert leaves["A"].metadata == {"S": "human"}
        assert leaves["A"].branch_length == pytest.approx(0.1)

    def test_bom_stripping(self):
        """UTF-8 BOM in file should be stripped on parse_file."""
        parser = NewickParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nwk", delete=False, encoding="utf-8-sig"
        ) as f:
            # Write content (the utf-8-sig codec itself adds the BOM)
            f.write("(A:0.1,B:0.2);")
            tmp_path = f.name

        try:
            result = parser.parse_file(tmp_path)
            assert len(result) >= 1
            assert result[0].leaf_count == 2
            assert "A" in result[0].leaf_names
        finally:
            os.unlink(tmp_path)

    def test_comment_before_tree(self):
        """A comment on its own line must be skipped."""
        newick = "[This is a comment]\n(A:0.1,B:0.2);"
        tree = parse_newick(newick)
        assert tree.leaf_count == 2

    def test_comment_same_line_as_tree(self):
        """BEAST/PAUP-style comment on the same line must not swallow the tree."""
        newick = "[&R] (A:0.1,B:0.2);"
        tree = parse_newick(newick)
        assert tree.leaf_count == 2
        assert set(tree.leaf_names) == {"A", "B"}

    def test_comment_inside_clade(self):
        """A comment between children must be skipped."""
        newick = "(A:0.1,[bootstrap=89]B:0.2);"
        tree = parse_newick(newick)
        assert set(tree.leaf_names) == {"A", "B"}
        assert tree.leaf_count == 2

    def test_unmatched_open_paren_raises(self):
        """Unmatched opening parenthesis must raise ValueError."""
        parser = NewickParser()
        with pytest.raises(ValueError):
            parser.parse("((A:0.1,B:0.2);")

    def test_stray_close_paren_raises(self):
        """A stray ')' must raise ValueError instead of looping forever."""
        parser = NewickParser()
        with pytest.raises(ValueError):
            parser.parse("(A:0.1,B:0.2)extra);")

    def test_unterminated_comment_raises(self):
        """An unterminated [comment must raise ValueError."""
        parser = NewickParser()
        with pytest.raises(ValueError):
            parser.parse("[unclosed comment (A:0.1,B:0.2);")

    def test_empty_tree_raises(self):
        """Empty Newick string should raise ValueError."""
        parser = NewickParser()
        with pytest.raises(ValueError):
            parser.parse("")

    def test_to_newick_roundtrip_special_labels(self):
        """Labels with spaces/special characters must survive a round-trip."""
        tree = parse_newick("('Homo sapiens':0.1,'Pan troglodytes':0.2)Hominidae;")
        out = tree.to_newick()
        reparsed = parse_newick(out + ";")
        assert set(reparsed.leaf_names) == {"Homo sapiens", "Pan troglodytes"}
        assert reparsed.root.name == "Hominidae"
