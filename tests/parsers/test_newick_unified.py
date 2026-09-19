# =============================================================================
# Test: unified Newick parsing core (W7 borrow from DendroPy)
# =============================================================================
"""
Covers the merged grammar shared by parsers.NewickParser and
phylogenetics._NewickParser: line/column diagnostics, quote superset,
depth guard, strict branch lengths, lenient multi-tree recovery.
"""

from __future__ import annotations

import pytest

from parsers.newick_parser import NewickParser, TreeComparator, parse_newick
from phylogenetics.tree import PhyloTree
from utils.exceptions import NewickParseError
from utils.newick_core import parse_newick_trees


class TestParseErrorDiagnostics:
    def test_error_is_valueerror_subclass(self):
        with pytest.raises(ValueError):
            parse_newick("(A:0.1,B:0.2)extra);")

    def test_line_and_column_attributes(self):
        text = "(\nA:0.1,B:0.2);  )"
        with pytest.raises(NewickParseError) as excinfo:
            parse_newick(text)
        err = excinfo.value
        assert err.line == 2
        assert err.column == 16
        assert err.position == text.index("  )") + 2
        assert "line 2" in str(err)

    def test_caret_context_in_message(self):
        with pytest.raises(NewickParseError) as excinfo:
            parse_newick("(A:0.1,B:0.2)C;  )")
        err = excinfo.value
        assert "^" in str(err)
        assert err.line == 1

    def test_phylogenetics_parser_raises_same_class(self):
        with pytest.raises(NewickParseError):
            PhyloTree.from_newick("(A:0.1,B:0.2)extra);")

    def test_empty_input_messages_preserved(self):
        with pytest.raises(ValueError, match="no valid tree found"):
            NewickParser().parse("")
        with pytest.raises(ValueError, match="Empty Newick input"):
            PhyloTree.from_newick("")


class TestQuoteAndCommentSuperset:
    def test_double_quotes_in_phylogenetics(self):
        tree = PhyloTree.from_newick('("Homo sapiens":0.1,"Pan":0.2);')
        assert set(tree.leaf_names) == {"Homo sapiens", "Pan"}

    def test_doubled_single_quote_escape_in_parsers(self):
        tree = parse_newick("('it''s':0.1,B:0.2);")
        assert "it's" in tree.leaf_names

    def test_backslash_escape_in_single_quotes(self):
        tree = PhyloTree.from_newick(r"('it\'s':0.1,B:0.2);")
        assert "it's" in tree.leaf_names

    def test_nested_bracket_comment_both_parsers(self):
        text = "(A:0.1[&&NHX:S=human],B[outer[inner]x]:0.2);"
        tree = parse_newick(text)
        leaves = {leaf.name: leaf for leaf in tree.root.get_leaves()}
        assert leaves["A"].metadata == {"S": "human"}
        assert leaves["B"].metadata == {}
        tree2 = PhyloTree.from_newick(text)
        assert {n.name for n in tree2.root.get_leaves()} == {"A", "B"}

    def test_bang_line_comments(self):
        text = "!citation: someone 1999\n(A:0.1,B:0.2)C;"
        assert parse_newick(text).leaf_names == ["A", "B"]
        assert PhyloTree.from_newick(text).leaf_names == ["A", "B"]


class TestDepthAndStrictness:
    def test_depth_guard_parsers(self):
        deep = "(" * 1200 + "A" + ")" * 1200 + ";"
        with pytest.raises(ValueError, match="depth"):
            parse_newick(deep)

    def test_depth_guard_phylogenetics(self):
        deep = "(" * 1200 + "A" + ")" * 1200 + ";"
        with pytest.raises(ValueError, match="depth"):
            PhyloTree.from_newick(deep)

    def test_garbage_branch_length_raises(self):
        with pytest.raises(NewickParseError, match="Invalid branch length"):
            parse_newick("(A:x,B:0.2);")
        with pytest.raises(NewickParseError, match="Invalid branch length"):
            PhyloTree.from_newick("(A:x,B:0.2);")

    def test_empty_parens_raises(self):
        with pytest.raises(NewickParseError, match="Expected a node name"):
            parse_newick("(,A:0.1,B:0.2)C;")
        with pytest.raises(NewickParseError, match="Empty"):
            parse_newick("()C;")

    def test_scientific_notation(self):
        tree = parse_newick("(A:1e-3,B:2.5E+1)C;")
        assert tree.root.children[0].branch_length == pytest.approx(0.001)
        assert tree.root.children[1].branch_length == pytest.approx(25.0)


class TestMultiTreeRecovery:
    def test_lenient_multi_keeps_good_trees(self):
        content = "(A:0.1,B:0.2)C;\n(broken tree;\n(D:0.3,E:0.4)F;\n"
        trees = NewickParser().parse_multi(content)
        assert [t.root.name for t in trees] == ["C", "F"]

    def test_semicolon_inside_quoted_label(self):
        trees = NewickParser().parse_multi("(A:0.1,'x;y',B:0.2)C;")
        assert len(trees) == 1
        assert "x;y" in trees[0].leaf_names

    def test_phylogenetics_multi(self):
        trees = PhyloTree.from_newick_multi("(A:0.1,B:0.2)C; (D:0.3,E:0.4)F;")
        assert [t.root.name for t in trees] == ["C", "F"]

    def test_single_rejects_multi(self):
        with pytest.raises(ValueError, match="multiple trees"):
            PhyloTree.from_newick("(A:0.1,B:0.2)C; (D:0.3,E:0.4)F;")

    def test_spec_output_accumulator(self):
        specs = parse_newick_trees("(A,B)C;", out=[])
        assert len(specs) == 1
        assert specs[0].name == "C"
        assert [child.name for child in specs[0].children] == ["A", "B"]


class TestSplitCanonicalization:
    def test_split_order_invariance(self):
        tree1 = parse_newick("(A,B,C)D;")
        tree2 = parse_newick("(C,A,B)D;")
        splits1 = TreeComparator._get_splits(tree1.root)
        splits2 = TreeComparator._get_splits(tree2.root)
        assert splits1 == splits2
        assert len(splits1) == 3
        assert TreeComparator.rf_distance(tree1, tree2) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
