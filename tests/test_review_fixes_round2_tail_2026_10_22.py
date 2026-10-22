# =============================================================================
# FILE: tests/test_review_fixes_round2_tail_2026_10_22.py
# =============================================================================
"""
Regression tests for the tail of the 2026-10-22 second-round review.

These are the findings the second round filed but did not close: a comment
rule that could never fire, an automaton that reported a missing attribute
instead of a missing initial state, an R export that indexed a component
that was not there, a preset library that could hold two presets of the same
name, a process pool that came back from the dead, and two Robinson-Foulds
distances in one package that did not agree with each other.

Every test here pins a behaviour that used to be wrong.  Each one names the
defect in its docstring, because a test that only says "it works" does not
say what it is keeping working.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


def _double(x):
    """A module-level function: a lambda cannot be pickled by multiprocessing."""
    return x * 2


# =============================================================================
# the tokenizer
# =============================================================================
class TestTokenizerComments:
    def test_a_line_comment_is_one_comment_token(self):
        """``--`` and ``//`` were unreachable.

        The COMMENT rule sat at the default priority (100) while the
        OPERATOR rule -- whose character class contains both ``-`` and
        ``/`` -- sat at 40, and the rules are tried in priority order.  So
        ``-- a comment`` lexed as OPERATOR("--") IDENTIFIER("a")
        IDENTIFIER("comment"): the comment was three tokens, and the
        COMMENT type was never produced at all.
        """
        from state_machine.tokenizer import TokenType, create_basic_lexer

        lexer = create_basic_lexer()
        for source, expected in (("-- a comment\n", "-- a comment"), ("// a comment\n", "// a comment")):
            tokens = lexer.tokenize(source)
            assert tokens[0].type is TokenType.COMMENT
            assert tokens[0].value == expected
            # The rest of the line is not a run of identifiers.
            assert all(t.type is not TokenType.IDENTIFIER for t in tokens[1:])
            assert tokens[-1].type is TokenType.EOF

    def test_two_minus_signs_are_still_two_minus_signs(self):
        """Making ``--`` a comment must not eat ``x--y``.

        The comment now only starts at the beginning of a line or after
        whitespace, so an operator run that happens to contain two dashes
        is still an operator.
        """
        from state_machine.tokenizer import TokenType, create_basic_lexer

        tokens = create_basic_lexer().tokenize("x--y\n")
        kinds = [t.type for t in tokens]
        assert kinds[0] is TokenType.IDENTIFIER
        assert kinds[1] is TokenType.OPERATOR
        assert tokens[1].value == "--"

    def test_a_hash_comment_still_works(self):
        """The reachable comment form must not have been the one broken."""
        from state_machine.tokenizer import TokenType, create_basic_lexer

        tokens = create_basic_lexer().tokenize("# a comment\n")
        assert tokens[0].type is TokenType.COMMENT


# =============================================================================
# the automaton
# =============================================================================
class TestAutomatonConversion:
    def test_to_dfa_without_an_initial_state_says_so(self):
        """The conversion used to die on a missing attribute.

        The subset construction is seeded from the epsilon closure of
        ``{q0}``.  With no ``q0`` the closure was handed ``{None}`` and
        died with "'NoneType' object has no attribute 'state_id'" -- a
        message about an attribute, naming neither the automaton nor the
        thing that was actually missing.
        """
        from state_machine.automaton import NFA

        nfa = NFA(name="no-initial")
        nfa.add_state("q0")

        with pytest.raises(RuntimeError, match="initial state"):
            nfa.to_dfa()

    def test_a_working_conversion_still_works(self):
        """The guard must not block the ordinary case."""
        from state_machine.automaton import NFA

        nfa = NFA(name="works")
        start = nfa.add_state("q0", is_initial=True)
        end = nfa.add_state("q1", is_accepting=True)
        nfa.add_transition(start, "a", end)

        dfa = nfa.to_dfa()
        assert dfa.accepts_string("a")
        assert not dfa.accepts_string("b")


# =============================================================================
# the R export
# =============================================================================
class TestPCAExportBounds:
    @pytest.fixture(scope="class")
    def result(self):
        from stats.pca import PCAResult

        explained = np.array([40.0, 30.0, 20.0, 10.0])
        return PCAResult(
            scores=np.arange(20.0).reshape(5, 4),
            loadings=np.zeros((4, 4)),
            eigenvalues=np.sqrt(explained),
            explained_variance=explained,
            cumulative_variance=np.cumsum(explained),
            eigenvalues_raw=explained,
            mean_vector=np.zeros(4),
            std_vector=None,
            n_components=4,
            method="covariance",
            singular_values=np.sqrt(explained),
        )

    def test_a_component_that_is_not_there_is_rejected(self, result, tmp_path):
        """``pc2=7`` used to raise a bare IndexError out of the indexing.

        The scores were indexed positionally while the explained variance
        was range-checked, so the two halves of the function disagreed
        about what an out-of-range component means.
        """
        from visualization.r_export import RScriptExporter

        exporter = RScriptExporter()
        with pytest.raises(ValueError, match="outside"):
            exporter.export_pca_scores(result, tmp_path, pc2=7)

    def test_a_negative_component_is_rejected(self, result, tmp_path):
        """``pc1=-1`` used to produce a figure whose label lied.

        A negative index is a legal Python index, so the scores were read
        from the LAST column while the header still said "PC0": the
        delivered R script plotted one component and labelled it as
        another.
        """
        from visualization.r_export import RScriptExporter

        exporter = RScriptExporter()
        with pytest.raises(ValueError, match="outside"):
            exporter.export_pca_scores(result, tmp_path, pc1=-1)

    def test_the_ordinary_export_is_unchanged(self, result, tmp_path):
        """The bound check must not break the call the UI makes."""
        from visualization.r_export import RScriptExporter

        export = RScriptExporter().export_pca_scores(result, tmp_path)
        header = export.data_path.read_text(encoding="utf-8").splitlines()[0]
        assert header == "sample,PC1,PC2"


# =============================================================================
# the preset library
# =============================================================================
class TestPresetNames:
    def test_two_presets_of_the_same_name_are_both_reachable(self, tmp_path):
        """The library could hold two presets under one name.

        ``path_for`` keeps the *file* names unique by appending ``_2``,
        ``_3``... but the preset's own name was left alone, so saving
        "My Preset" twice produced two files both named "My Preset".  The
        picker then offered the same name twice and
        ``load_by_name("My Preset")`` could only ever reach the first --
        the second was unreachable by name.
        """
        from presets.manager import PresetManager
        from presets.schema import Preset

        manager = PresetManager(directory=str(tmp_path))
        preset = Preset(name="My Preset", analysis_id="pca", params={"n_components": 2})
        first = manager.save(preset)
        second = manager.save(preset)

        assert os.path.basename(first) != os.path.basename(second)

        names = manager.list_names()
        assert len(names) == 2
        assert len(set(names)) == 2, f"the same name is offered twice: {names}"

        # Both are reachable by their own name.
        for name in names:
            assert manager.load_by_name(name).name == name

    def test_a_preset_saved_under_an_explicit_path_keeps_its_name(self, tmp_path):
        """An explicit path is the caller's own naming decision."""
        from presets.manager import PresetManager
        from presets.schema import Preset

        manager = PresetManager(directory=str(tmp_path))
        target = tmp_path / "chosen.json"
        manager.save(Preset(name="My Preset", analysis_id="pca"), path=str(target))

        assert manager.list_names() == ["My Preset"]


# =============================================================================
# the process pool
# =============================================================================
class TestProcessPoolLifecycle:
    def test_a_shut_down_pool_does_not_come_back(self):
        """A closed pool used to reopen under the caller's feet.

        ``shutdown()`` set the ``_shutdown`` flag, and nothing read it:
        ``submit_task`` found no pool, called ``start()``, and ``start()``
        brought up a fresh set of worker processes.  The caller had been
        told the pool was closed, and the process table grew anyway.
        """
        from hpc.process_pool import ProcessPool
        from utils.exceptions import ComputationError

        pool = ProcessPool(n_workers=1)
        pool.start()
        pool.shutdown()
        assert pool._pool is None

        with pytest.raises(ComputationError):
            pool.submit_task("t1", _double, 21)

        assert pool._pool is None, "the pool was revived"

    def test_starting_a_shut_down_pool_is_refused(self):
        """``start()`` must not be a way around the same guard."""
        from hpc.process_pool import ProcessPool
        from utils.exceptions import ComputationError

        pool = ProcessPool(n_workers=1)
        pool.shutdown()

        with pytest.raises(ComputationError):
            pool.start()

    def test_a_live_pool_still_runs_tasks(self):
        """The guard must not block the ordinary case."""
        from hpc.process_pool import ProcessPool

        with ProcessPool(n_workers=1) as pool:
            task = pool.submit_task("t1", _double, 21)
            assert pool.get_result(task, timeout=60) == 42


# =============================================================================
# the two Robinson-Foulds distances
# =============================================================================
class TestRobinsonFouldsAgreement:
    def test_the_package_has_one_robinson_foulds_distance(self):
        """The Fitch copy counted terminal edges as non-trivial splits.

        Its local split extractor took a node's whole descendant set, so
        a split of one taxon against the rest counted.  Comparing
        ``((A,B),(C,D))`` with ``((A,B,C),D)`` therefore scored 2 where
        the Robinson-Foulds distance is 1 -- and the package's own
        ``robinson_foulds_distance()`` said 1 for the same pair.
        """
        from phylogenetics.fitch import FitchAlgorithm
        from phylogenetics.tree import PhyloTree
        from phylogenetics.tree_distance import robinson_foulds_distance

        first = PhyloTree.from_newick("((A,B),(C,D));")
        second = PhyloTree.from_newick("((A,B,C),D);")

        fitch = FitchAlgorithm()
        assert fitch.compute_distance_from_parsimony(first, second) == 1
        assert fitch.compute_distance_from_parsimony(first, second) == robinson_foulds_distance(first, second)

    def test_trees_over_different_taxa_are_refused(self):
        """A comparison across different taxon sets has no answer.

        The Fitch copy returned 2 for a tree over {A,B,C,D} against one over
        {A,B,C,E}: the two split sets are defined over different leaf
        universes, so the symmetric difference is a number without a
        meaning.  The package's other implementation raised.
        """
        from phylogenetics.fitch import FitchAlgorithm
        from phylogenetics.tree import PhyloTree

        first = PhyloTree.from_newick("((A,B),(C,D));")
        second = PhyloTree.from_newick("((A,B),(C,E));")

        with pytest.raises(Exception, match="taxon"):
            FitchAlgorithm().compute_distance_from_parsimony(first, second)

    def test_the_docstring_states_the_divisor_the_code_uses(self):
        """The normalised distance's docstring contradicted its own code.

        It described the divisor as the count of all non-trivial splits
        possible over ``n`` taxa -- ``2^(n-1) - n - 1`` -- and called that
        "the one we use here", while the code divided by ``2 * (n - 3)``,
        the number of splits a binary tree actually spans.  For four taxa
        the two are 3 and 2, so the documented metric is not the metric.
        """
        from phylogenetics import tree_distance

        doc = inspect.getdoc(tree_distance.normalized_robinson_foulds_distance) or ""
        assert "2 * (n - 3)" in doc
        assert "2^(n-1)" not in doc
