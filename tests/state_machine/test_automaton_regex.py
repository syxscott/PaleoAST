"""Regression tests for the regex front end of ``state_machine/automaton.py``.

The pattern of every defect here is the same: a construct the engine did not
implement was turned into a *literal*, so the compiled automaton quietly
matched something the author never wrote. ``/a.b/`` matched the three-character
string "a.b"; ``/^abc/`` matched "^abc"; ``[^a-z]`` excluded three characters
instead of a range. Each of those returns a plausible answer, which is what
makes them dangerous.

Where the engine genuinely cannot express a construct, the tests require a
refusal instead, because a silent wrong answer is the worse outcome.
"""

from __future__ import annotations

import string

import pytest

from state_machine.automaton import RegexCompiler


def _match(pattern: str, text: str) -> bool:
    return RegexCompiler().compile(pattern).accepts_string(text)


class TestDotIsAWildcard:
    def test_dot_matches_one_arbitrary_character(self):
        assert _match("a.b", "axb")
        assert _match("a.b", "a1b")

    def test_dot_accepts_many_strings_not_just_the_literal_one(self):
        # The bug: '.' was a CHAR node, so /a.b/ accepted exactly one string,
        # "a.b". A wildcard accepts a whole family.
        accepted = [t for t in ("axb", "a1b", "a.b", "a b", "a-b") if _match("a.b", t)]
        assert len(accepted) == 5, f"/a.b/ only accepted {accepted}"

    def test_dot_matches_exactly_one_character(self):
        assert not _match("a.b", "ab")
        assert not _match("a.b", "axxb")

    def test_dot_does_not_match_a_newline(self):
        # Documented in both RegexNodeType.ANY and _build_any; the code used to
        # add chr(10) anyway.
        assert not _match("a.b", "a\nb")


class TestUnsupportedConstructsAreRefused:
    """A construct that cannot be expressed must raise, not match literally."""

    @pytest.mark.parametrize("pattern", ["^abc", "abc$", "^abc$"])
    def test_anchors_raise(self, pattern):
        with pytest.raises(NotImplementedError, match="Anchor"):
            RegexCompiler().compile(pattern)

    def test_counted_repetition_raises(self):
        with pytest.raises(NotImplementedError, match="Counted repetition"):
            RegexCompiler().compile("a{2,3}")

    @pytest.mark.parametrize("pattern", ["a*?", "a+?", "a??", "a**"])
    def test_second_quantifier_raises(self, pattern):
        # The bug: the first quantifier was consumed and the function returned,
        # so the second became a literal character and /a*?/ accepted nothing.
        with pytest.raises(NotImplementedError):
            RegexCompiler().compile(pattern)

    def test_anchor_anywhere_is_refused_not_treated_as_a_literal(self):
        """A mid-pattern '^' is refused too, not compiled to a literal '^'."""
        with pytest.raises(NotImplementedError, match="Anchor"):
            RegexCompiler().compile("a^")


class TestCharacterClassRanges:
    def test_range_includes_both_endpoints(self):
        assert _match("[a-z]", "m")
        assert _match("[a-z]", "a")
        assert _match("[a-z]", "z")

    def test_range_excludes_outside_characters(self):
        assert not _match("[a-z]", "A")
        assert not _match("[a-z]", "0")

    def test_negated_range_excludes_the_whole_range(self):
        # The bug: [a-z] expanded to {'a', '-', 'z'}, so the negated class
        # accepted every letter of the alphabet it was meant to exclude.
        for ch in string.ascii_lowercase:
            assert not _match("[^a-z]+", ch), f"[^a-z] accepted {ch!r}"

    def test_negated_range_still_accepts_outside_characters(self):
        assert _match("[^a-z]+", "A")
        assert _match("[^a-z]+", "7")

    def test_digit_range(self):
        for ch in "0123456789":
            assert _match("[0-9]", ch)
        assert not _match("[0-9]", "a")

    def test_trailing_hyphen_is_literal(self):
        assert _match("[a-]", "-")
        assert _match("[a-]", "a")

    def test_backslash_escapes_still_beat_a_range(self):
        """\\D is the pre-defined negated class the other test file covers."""
        assert not _match("[\\D]", "5")
        assert _match("[\\D]", "x")


class TestOrdinaryPatternsStillWork:
    """The refusals above must not have broken the supported syntax."""

    @pytest.mark.parametrize(
        "pattern,accept,reject",
        [
            ("abc", "abc", "ab"),
            ("a|b", "a", "c"),
            ("a*", "", "b"),
            ("a+", "aa", ""),
            ("a?", "", "b"),
            ("(ab)+c", "ababc", "abab"),
            ("[xyz]", "y", "q"),
        ],
    )
    def test_supported_constructs(self, pattern, accept, reject):
        assert _match(pattern, accept), f"/{pattern}/ should accept {accept!r}"
        assert not _match(pattern, reject), f"/{pattern}/ should reject {reject!r}"
