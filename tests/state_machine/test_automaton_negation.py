# =============================================================================
# Test: automaton.py negation handling for \D, \W, \S
# =============================================================================
"""
Verify that pre-defined negated character classes (\\D, \\W, \\S) are
correctly interpreted by the automaton.

The escape_map in _parse_escape represents these as '^0123456789',
'^A-Za-z0-9_', and '^ \\t\\n\\r'. The _build_charclass must then
interpret the leading '^' as negation and accept everything BUT the
listed characters.
"""

from __future__ import annotations

import string

from state_machine.automaton import RegexCompiler


def _compile_and_match(pattern_chars: str, test_char: str) -> bool:
    """Compile a single char class and test if test_char is accepted."""
    compiler = RegexCompiler()
    fa = compiler.compile(f"[{pattern_chars}]")
    return fa.accepts_string(test_char)


class TestAutomatonNegation:
    """Verify negated character classes (\\D, \\W, \\S) work correctly."""

    def test_digit_negation_accepts_letters(self):
        """\\D (non-digit) should accept letters."""
        for ch in "abcXYZ":
            assert _compile_and_match("^0123456789", ch), (
                f"Letter '{ch}' should be accepted by \\D"
            )

    def test_digit_negation_rejects_digits(self):
        """\\D should reject digits."""
        for ch in "0123456789":
            assert not _compile_and_match("^0123456789", ch), (
                f"Digit '{ch}' should be rejected by \\D"
            )

    def test_word_negation_accepts_special(self):
        """\\W (non-word) should accept spaces and punctuation."""
        word_chars = string.ascii_letters + string.digits + "_"
        # Test space
        assert _compile_and_match("^" + word_chars, " "), (
            "Space should be accepted by \\W"
        )
        assert _compile_and_match("^" + word_chars, "!"), (
            "Punctuation should be accepted by \\W"
        )

    def test_word_negation_rejects_alphanumeric(self):
        """\\W should reject letters and digits."""
        word_chars = string.ascii_letters + string.digits + "_"
        for ch in "abcXYZ0123456789":
            assert not _compile_and_match("^" + word_chars, ch), (
                f"'{ch}' should be rejected by \\W"
            )

    def test_space_negation_accepts_letters(self):
        """\\S (non-space) should accept letters, reject whitespace."""
        assert _compile_and_match("^ \\t\\n\\r", "a"), (
            "Letter should be accepted by \\S"
        )

    def test_space_negation_rejects_whitespace(self):
        """\\S should reject whitespace."""
        for ch in [" ", "\t", "\n", "\r"]:
            assert not _compile_and_match("^ \\t\\n\\r", ch), (
                f"Whitespace '{ch}' should be rejected by \\S"
            )

    def test_positive_class_still_works(self):
        """Regular positive char classes should still work after fix."""
        for ch in "abc":
            assert _compile_and_match("abc", ch), (
                f"'{ch}' should be accepted by [abc]"
            )
        for ch in "xyz012":
            assert not _compile_and_match("abc", ch), (
                f"'{ch}' should be rejected by [abc]"
            )