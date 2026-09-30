"""Regression tests for DFA minimisation in ``state_machine/automaton.py``.

Minimisation must return an *equivalent* DFA. That is the whole contract, and
it is checkable without any reference implementation: for every string in a
test set, ``accepts`` must agree before and after.

The previous refinement was guarded by ``len(partitions) > 1`` and stopped at
the first symbol that produced a split. A DFA that starts from a single
partition therefore skipped refinement altogether, and the result was
over-minimised: for ``/a?/`` it merged the two accepting states and the
minimised DFA accepted ``aa``, ``aaa`` and ``aaaa``, none of which the original
accepted. A workflow state machine that minimises this way accepts sequences
the author never permitted.
"""

from __future__ import annotations

from itertools import product

import pytest

from state_machine.automaton import RegexCompiler

ALPHABET = "ab"
DEPTH = 4


def _strings(depth: int = DEPTH) -> list[str]:
    out = []
    for n in range(depth + 1):
        out.extend("".join(t) for t in product(ALPHABET, repeat=n))
    return out


def _language(fa, cases: list[str]) -> set[str]:
    return {s for s in cases if fa.accepts_string(s)}


def _minimized(pattern: str):
    dfa = RegexCompiler().compile(pattern, to_dfa=True)
    return dfa, dfa.minimize_hopcroft()


class TestMinimisationPreservesTheLanguage:
    @pytest.mark.parametrize("pattern", ["a?", "a*", "(a|b)*", "ab?", "a?b"])
    def test_language_is_unchanged(self, pattern):
        dfa, minimized = _minimized(pattern)
        cases = _strings()
        before, after = _language(dfa, cases), _language(minimized, cases)

        assert sorted(after - before) == [], (
            f"minimising /{pattern}/ invented {sorted(after - before)} -- "
            "the minimised DFA accepts strings the original rejects"
        )
        assert sorted(before - after) == [], (
            f"minimising /{pattern}/ dropped {sorted(before - after)}"
        )

    def test_optional_single_character_does_not_become_star(self):
        """The concrete over-acceptance: /a?/ must stay /a?/.

        Both states of this DFA are accepting, so a minimiser that skips
        refinement from a single starting partition merges them and the result
        accepts any number of a's.
        """
        dfa, minimized = _minimized("a?")
        assert dfa.accepts_string("a")
        for rejected in ("aa", "aaa", "aaaa"):
            assert not dfa.accepts_string(rejected)
            assert not minimized.accepts_string(rejected), (
                f"the minimised /a?/ accepts {rejected!r}"
            )

    def test_minimisation_actually_reduces_states(self):
        """A fix that stopped splitting anything would pass the tests above.

        Checked separately so "language preserved" cannot be satisfied by simply
        doing nothing.
        """
        for pattern in ("a*", "(a|b)*"):
            dfa, minimized = _minimized(pattern)
            assert len(minimized._states) < len(dfa._states), (
                f"/{pattern}/ should have fewer states after minimisation"
            )

    def test_equivalence_under_repetition(self):
        """Re-check beyond the sampled depth: no divergence at longer lengths."""
        dfa, minimized = _minimized("a?")
        for n in range(1, 25):
            s = "a" * n
            assert dfa.accepts_string(s) == minimized.accepts_string(s), (
                f"disagreement at {s!r}"
            )

    def test_alternation_is_not_merged_into_something_larger(self):
        """A state machine's practical failure mode: accepting a superset."""
        dfa, minimized = _minimized("ab?")
        for s in _strings():
            assert dfa.accepts_string(s) == minimized.accepts_string(s), (
                f"/ab?/ diverges on {s!r}"
            )
