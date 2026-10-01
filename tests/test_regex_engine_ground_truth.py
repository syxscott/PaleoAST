"""
Ground truth for the regex engine in ``state_machine/automaton.py``.

Why this file exists
--------------------
The engine had three defects that all produced *confidently wrong* answers
rather than errors, and none of them were caught because the module had no
tests at all:

1. ``minimize_hopcroft`` merged states that were not equivalent whenever the
   DFA's initial partition had a single block -- which happens exactly when
   every state is accepting, as for ``/a?/``. The result accepted ``"aa"`` and
   ``"ab"``. A correct minimisation must preserve the language exactly.
2. Character classes had no range support: ``[a-z]`` became the three-element
   set ``{'a', '-', 'z'}``, so the *negated* class ``[^a-z]`` accepted every
   letter it was supposed to exclude -- ``[^a-z]+`` matched ``"color"``.
3. ``.`` was parsed as a literal character, so ``/a.b/`` matched the string
   ``"a.b"`` and nothing else.

A fourth and fifth issue were not silent-wrong but unstated: anchors (``^``,
``$``) and counted repetition (``{n,m}``) were unimplemented and were also
treated as literals. They now raise ``NotImplementedError`` with a message
that says what to do instead.

The oracle
----------
Python's own ``re`` module, which is a correct, heavily-exercised regex
engine. ``accepts_string`` is a full-string match, so ``re.fullmatch`` is the
right comparison -- verified by reading its implementation, not assumed.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from state_machine.automaton import RegexCompiler, regex_to_dfa, regex_to_nfa

# Patterns the engine claims to support.
SUPPORTED = [
    "a", "ab", "ba", "a|b", "ab|cd", "a*", "a+", "a?", "ab*", "(ab)*", "(ab)+",
    "a(b|c)d", "a(bc|d)*e", "(a|b)*abb", "colou?r", "(a+)+b", "[abc]+", "[abc]",
    "[a-z]+", "[a-z]", "[^a-z]+", "[^0-9]", "[a-zA-Z0-9]+", "[0-9]+", "a.b",
    "abc|def|ghi", "a(b|c)(d|e)", "(a|b)(c|d)(e|f)", "x*y*z*", "(|a)",
]

# Inputs wide enough that a wrong DFA is very unlikely to agree by accident.
PROBES = [
    "", "a", "b", "c", "d", "e", "x", "y", "z", "A", "Z", "0", "9", "5",
    "aa", "ab", "ba", "bb", "abc", "abd", "acb", "cd", "aab", "abb", "abcd",
    "abab", "abba", "aabb", "aaaa", "aaaaa", "colou", "color", "colour", "colr",
    "aXb", "a b", "a.b", "xyz", "ccd", "acc", "42", "7", "abcde", "dcba",
    "e", "acbd", "cba", "bb", "aabbcc", "ababab", "cc", "dd",
]


class TestAgreesWithPythonRe:
    @pytest.mark.parametrize("pattern", SUPPORTED)
    def test_every_probe_agrees_with_re_fullmatch(self, pattern):
        dfa = regex_to_dfa(pattern)
        expected = re.compile(pattern)
        for text in PROBES:
            got = bool(dfa.accepts_string(text))
            want = expected.fullmatch(text) is not None
            assert got == want, (
                f"/{pattern}/ vs {text!r}: engine says {got}, re says {want}"
            )

    @pytest.mark.parametrize(
        "pattern",
        ["a?", "a*", "a+", "ab?", "(ab)?", "colou?r", "a(b|c)?d", "(a|b)*abb"],
    )
    def test_optional_does_not_behave_like_star(self, pattern):
        """The regression that exposed the minimisation bug.

        ``/a?/`` must not accept ``"aa"``. Its subset construction yields two
        states that are BOTH accepting, so a minimiser that starts from the
        accepting/non-accepting split alone has nothing to refine on and merges
        them.
        """
        dfa = regex_to_dfa(pattern)
        for text in PROBES:
            assert bool(dfa.accepts_string(text)) == (
                re.fullmatch(pattern, text) is not None
            ), f"/{pattern}/ vs {text!r}"


class TestMinimisationPreservesTheLanguage:
    """Minimisation is an optimisation: it must never change what is accepted."""

    @pytest.mark.parametrize("pattern", SUPPORTED)
    def test_language_unchanged_by_minimisation(self, pattern):
        nfa = RegexCompiler().compile(pattern, to_dfa=False)
        before = nfa.to_dfa()
        after = before.minimize_hopcroft()
        for text in PROBES:
            assert bool(before.accepts_string(text)) == bool(after.accepts_string(text)), (
                f"/{pattern}/ vs {text!r}: minimisation changed the language"
            )

    def test_all_accepting_dfa_is_still_refined(self):
        """The specific shape that used to slip through: every state accepting."""
        nfa = RegexCompiler().compile("a?", to_dfa=False)
        dfa = nfa.to_dfa()
        assert all(state.is_accepting for state in dfa.states), (
            "fixture no longer exercises the all-accepting case"
        )
        minimal = dfa.minimize_hopcroft()
        assert minimal.accepts_string("") is True
        assert minimal.accepts_string("a") is True
        assert minimal.accepts_string("aa") is False


class TestCharacterClassRanges:
    @pytest.mark.parametrize(
        "cls,members",
        [("a-z", 26), ("a-zA-Z0-9", 62), ("0-9", 10), ("abc", 3), ("a-", 2)],
    )
    def test_class_membership(self, cls, members):
        pattern = f"[{cls}]+"
        dfa = regex_to_dfa(pattern)
        for char in "abcxyz09AZ-!":
            want = re.fullmatch(pattern, char) is not None
            assert bool(dfa.accepts_string(char)) == want, f"[{cls}] vs {char!r}"

    def test_negated_range_excludes_the_range(self):
        """``[^a-z]`` must not match a lowercase letter.

        With ranges unexpanded, ``forbidden`` was ``{'a','-','z'}`` and every
        other letter -- b, c, d, ... -- was accepted.
        """
        dfa = regex_to_dfa("[^a-z]+")
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert dfa.accepts_string(char) is False, f"[^a-z] wrongly accepted {char!r}"
        for char in "AZ09-!":
            assert dfa.accepts_string(char) is True, f"[^a-z] wrongly rejected {char!r}"

    def test_positive_range_includes_the_range(self):
        dfa = regex_to_dfa("[a-z]+")
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert dfa.accepts_string(char) is True
        for char in "AZ09":
            assert dfa.accepts_string(char) is False


class TestDotIsNotALiteral:
    @pytest.mark.parametrize("text,expected", [
        ("axb", True), ("a b", True), ("a1b", True), ("ab", False),
        ("axxb", False),
    ])
    def test_dot_matches_exactly_one_character(self, text, expected):
        dfa = regex_to_dfa("a.b")
        assert dfa.accepts_string(text) is expected
        assert dfa.accepts_string(text) == (re.fullmatch("a.b", text) is not None)

    def test_dot_still_matches_a_literal_period_character(self):
        """/a.b/ must match "a.b" -- via the dot, not by treating it literally.

        The old bug made the dot a literal, so "axb" was rejected. Checking
        the reverse direction too catches a "fix" that just made the dot
        match nothing.
        """
        dfa = regex_to_dfa("a.b")
        assert dfa.accepts_string("axb") is True
        assert dfa.accepts_string("a.b") is True


class TestUnsupportedFeaturesRefuseLoudly:
    """Unimplemented syntax must raise, not quietly match its own text."""

    @pytest.mark.parametrize("pattern", ["a{2,3}", "[0-9]{2}", "a{2}", "x{1,}"])
    def test_counted_repetition_raises(self, pattern):
        with pytest.raises(NotImplementedError, match="Counted repetition"):
            regex_to_dfa(pattern)

    @pytest.mark.parametrize("pattern", ["a*?", "a+?", "a??", "a**"])
    def test_lazy_quantifiers_raise(self, pattern):
        """A lazy quantifier used to build a DFA that accepted NOTHING.

        ``a*?`` accepted neither "a" nor "" nor anything else -- a total
        mismatch with ``re``, and silent about it.
        """
        with pytest.raises(NotImplementedError, match="quantifier"):
            regex_to_dfa(pattern)

    @pytest.mark.parametrize("pattern", ["^a", "a$", "^abc$"])
    def test_anchors_raise(self, pattern):
        with pytest.raises(NotImplementedError, match="Anchor"):
            regex_to_dfa(pattern)

    def test_the_message_says_what_to_do_instead(self):
        with pytest.raises(NotImplementedError) as excinfo:
            regex_to_dfa("a{2,3}")
        message = str(excinfo.value)
        assert "*" in message and "+" in message, message


class TestNfaAndDfaAgree:
    @pytest.mark.parametrize("pattern", SUPPORTED)
    def test_nfa_and_dfa_accept_the_same_strings(self, pattern):
        nfa = regex_to_nfa(pattern)
        dfa = regex_to_dfa(pattern)
        for text in PROBES:
            assert nfa.accepts_string(text) == dfa.accepts_string(text), (
                f"/{pattern}/ vs {text!r}: NFA and DFA disagree"
            )

    @pytest.mark.parametrize("pattern", ["a", "a?", "a*", "[a-z]+", "a(b|c)d"])
    def test_repeated_calls_are_deterministic(self, pattern):
        first = [regex_to_dfa(pattern).accepts_string(t) for t in PROBES]
        second = [regex_to_dfa(pattern).accepts_string(t) for t in PROBES]
        assert first == second
