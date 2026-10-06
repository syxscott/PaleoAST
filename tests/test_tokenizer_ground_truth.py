"""
Ground truth for ``state_machine/tokenizer.py`` (26.5% covered).

A lexer's characteristic failure is not a wrong number but a stream that lies:
characters invented or lost, positions that drift, or two tokenisation paths
that disagree. All three are decidable from the token stream alone.

Two defects were found this way, and both were invisible from the outside
because the values, line numbers and columns were all correct:

1. ``LexerTokenizer.__init__`` compiled the combined pattern -- which
   populates the ``r0..rN`` -> rule mapping -- and then re-initialised that
   mapping to ``{}`` on the next line. Every rule match therefore found no
   group name, ``token_type`` stayed ``None``, and every token fell through to
   UNKNOWN. ``create_basic_lexer()`` could not tell an identifier from a
   number from an operator; it just happened to get the text right.

2. The same factory registered its 19 NEXUS keywords upper-cased
   (``IF``, ``END``, ``TAXLABELS`` ...) while the lexer defaulted to
   case-sensitive matching, so the lookup compared ``'if'`` against ``'IF'``
   and none of them could ever fire.
"""

from __future__ import annotations

import re

import pytest

from state_machine.tokenizer import (
    LexerRule,
    LexerTokenizer,
    TokenType,
    create_basic_lexer,
)

SAMPLES = [
    "x = 1 + 2 * 3;",
    "a = 1\nb = 2",
    "a = 1\n\nb = 2",
    'name = "hello world"',
    "name = 'hi'",
    "pi = 3.14159",
    "if x > 1 and y < 2:",
    "a\t=\t1",
    "a = 1\r\nb = 2",
    "only   whitespace   ",
    "",
    "a = " + " + ".join(str(i) for i in range(60)),
]


@pytest.fixture(scope="module")
def lexer() -> LexerTokenizer:
    return create_basic_lexer()


def _significant(tokens):
    return [t for t in tokens if t.type not in (TokenType.WHITESPACE, TokenType.EOF)]


class TestNoCharactersLostOrInvented:
    @pytest.mark.parametrize("source", SAMPLES, ids=range(len(SAMPLES)))
    def test_token_values_reconstruct_the_source(self, lexer, source):
        """Joining the token values must give the source back.

        Whitespace and newlines are emitted as skipped tokens, so compare
        against the source with all whitespace removed.
        """
        joined = "".join(t.value for t in _significant(lexer.tokenize(source)))
        # Compare whitespace-insensitively on BOTH sides: a string literal
        # such as "hello world" legitimately contains a space, so stripping
        # the source but not the token stream would report a false loss.
        assert "".join(joined.split()) == "".join(source.split()), f"got {joined!r}"

    @pytest.mark.parametrize("source", SAMPLES, ids=range(len(SAMPLES)))
    def test_batch_and_incremental_agree(self, lexer, source):
        """Two tokenisation paths over the same input must not diverge."""
        batch = [(t.type, t.value, t.line, t.column) for t in lexer.tokenize(source)]
        incremental = [(t.type, t.value, t.line, t.column) for t in lexer.tokenize_incremental(source)]
        assert batch == incremental


class TestClassification:
    """The bug: every token used to come back UNKNOWN."""

    def test_a_real_statement_is_classified(self, lexer):
        kinds = {t.type for t in _significant(lexer.tokenize('x = 1 + 2.5; name = "hi"'))}
        assert TokenType.IDENTIFIER in kinds
        assert TokenType.OPERATOR in kinds
        assert TokenType.INTEGER in kinds
        assert TokenType.FLOAT in kinds
        assert TokenType.PUNCTUATION in kinds
        assert TokenType.STRING in kinds
        assert TokenType.UNKNOWN not in kinds, "a token fell through to UNKNOWN"

    def test_string_beats_identifier(self, lexer):
        """A quoted word must lex as one STRING, not an IDENTIFIER + quote."""
        literals = [t for t in _significant(lexer.tokenize('"abc"')) if t.type == TokenType.STRING]
        assert [t.value for t in literals] == ['"abc"']

    def test_float_beats_integer(self, lexer):
        kinds = [t.type for t in _significant(lexer.tokenize("2.5"))]
        assert kinds == [TokenType.FLOAT]

    def test_number_then_underscore(self, lexer):
        """`2.5` is one float, not `2`, `.`, `5`."""
        values = [t.value for t in _significant(lexer.tokenize("2.5"))]
        assert values == ["2.5"]


class TestPositions:
    def test_line_numbers_track_the_source(self, lexer):
        source = "a = 1\nbb = 2\nccc = 3"
        by_line: dict[int, set[str]] = {}
        for token in _significant(lexer.tokenize(source)):
            by_line.setdefault(token.line, set()).add(token.value)
        assert by_line[1] == {"a", "=", "1"}
        assert by_line[2] == {"bb", "=", "2"}
        assert by_line[3] == {"ccc", "=", "3"}

    def test_columns_increase_within_a_line(self, lexer):
        columns = [t.column for t in _significant(lexer.tokenize("alpha = beta + 1"))]
        assert columns == sorted(columns)
        assert columns[0] == 1

    def test_get_tokens_with_lines_agrees_with_tokenize(self, lexer):
        source = "a = 1\nb = 2"
        by_line = lexer.get_tokens_with_lines(source)
        flat = [t for line in sorted(by_line) for t in _significant(by_line[line])]
        assert [t.value for t in flat] == [t.value for t in _significant(lexer.tokenize(source))]


class TestKeywords:
    def test_nexus_keywords_are_recognised(self, lexer):
        """Upper-cased registration with a case-sensitive lexer meant the
        lookup compared 'if' against 'IF' and never matched."""
        found = {t.value: t.type for t in _significant(lexer.tokenize("if x else"))}
        assert found["if"] == TokenType.KEYWORD
        assert found["else"] == TokenType.KEYWORD

    def test_a_non_keyword_stays_an_identifier(self, lexer):
        found = {t.value: t.type for t in _significant(lexer.tokenize("then"))}
        assert found["then"] == TokenType.IDENTIFIER

    def test_is_keyword_agrees_with_the_type(self, lexer):
        for token in _significant(lexer.tokenize("if x else then")):
            assert token.is_keyword() == (token.type == TokenType.KEYWORD)

    def test_case_sensitivity_is_honoured_when_requested(self):
        strict = LexerTokenizer(
            [LexerRule(TokenType.IDENTIFIER, re.compile(r"[A-Za-z_]+"), priority=1)],
            case_sensitive=True,
        )
        strict.set_keywords({"if": TokenType.KEYWORD})
        kinds = {t.value: t.type for t in strict.tokenize("if IF") if t.type != TokenType.EOF}
        assert kinds["if"] == TokenType.KEYWORD
        assert kinds["IF"] == TokenType.IDENTIFIER


class TestValueAccessors:
    def test_integer_accessor_round_trips(self, lexer):
        integers = [t for t in _significant(lexer.tokenize("1 42 7")) if t.type == TokenType.INTEGER]
        assert [t.get_int_value() for t in integers] == [1, 42, 7]

    def test_float_accessor_round_trips(self, lexer):
        floats = [t for t in _significant(lexer.tokenize("1.5 2.25")) if t.type == TokenType.FLOAT]
        assert [t.get_float_value() for t in floats] == [1.5, 2.25]

    def test_string_accessor_strips_the_delimiters(self, lexer):
        """Documented behaviour: the accessor yields the content, not the
        literal. The raw text is still on ``token.value``."""
        literal = next(t for t in _significant(lexer.tokenize('"hi"')) if t.type == TokenType.STRING)
        assert literal.value == '"hi"'
        assert literal.get_string_value() == "hi"

    def test_the_wrong_accessor_raises_rather_than_guessing(self, lexer):
        """Asking a string for a number must not quietly return 0.0."""
        literal = next(t for t in _significant(lexer.tokenize('"hi"')) if t.type == TokenType.STRING)
        with pytest.raises(ValueError):
            literal.get_float_value()
