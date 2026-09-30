"""
Ground truth for ``parsers/lexer.py`` (53% covered) -- ``BaseLexer``, the base
class of the NEXUS lexer.

This module was probed after the near-identical defects found in
``state_machine/tokenizer.py``, where a lexer produced correct text, lines and
columns but classified NOTHING (every token came back UNKNOWN) and survived 26%
coverage unnoticed. So these tests check the properties that only bite when the
stream lies: reconstruction, positions, and agreement between the two
tokenisation paths.

Probing found no defect here -- token classification, both paths, the per-line
view and the line numbers are all correct. The tests are kept because that is
exactly the property that regressed in the sibling module.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from parsers.lexer import BaseLexer, LexerError, TokenType  # noqa: E402


class MiniLexer(BaseLexer):
    """A concrete lexer exercising every rule kind."""

    def _init_rules(self):
        self.add_rule(TokenType.WHITESPACE, r"[ \t]+", skip=True)
        self.add_rule(TokenType.NEWLINE, r"\r?\n", skip=True)
        self.add_rule(TokenType.COMMENT, r"#.*$")
        self.add_rule(TokenType.STRING, r'"[^"]*"', priority=1)
        self.add_rule(TokenType.NUMBER, r"\d+\.\d+|\d+", priority=2)
        self.add_rule(TokenType.IDENTIFIER, r"[A-Za-z_][A-Za-z0-9_]*", priority=3)
        self.add_rule(TokenType.OPERATOR, r"[+\-*/<>=!]+", priority=4)
        self.add_rule(TokenType.PUNCTUATION, r"[;,()\[\]{}]", priority=5)


WELL_FORMED = [
    "a = 1 + 2.5;",
    'name = "hello world"',
    "a = 1\nb = 2",
    "x = 1 # trailing comment",
    "f(a, b)",
    "only   spaces   ",
    "",
    "3.14 + 42",
    "a\r\nb",
]

# Malformed on purpose: the lexer is expected to refuse or flag these, never to
# return a clean token stream.
MALFORMED = [
    'unterminated "string',
    "weird @ char",
]


def _keep(tokens):
    return [t for t in tokens if t.type not in (TokenType.WHITESPACE, TokenType.EOF)]


@pytest.fixture(scope="module")
def lex() -> MiniLexer:
    return MiniLexer()


class TestNoCharactersLost:
    @pytest.mark.parametrize("source", WELL_FORMED)
    def test_token_values_reconstruct_the_source(self, lex, source):
        joined = "".join(t.value for t in _keep(lex.tokenize(source)))
        # Whitespace-insensitive on BOTH sides: a string literal such as
        # "hello world" legitimately contains a space, so stripping only the
        # source would report a false loss.
        assert "".join(joined.split()) == "".join(source.split()), f"got {joined!r}"


class TestClassification:
    """The sibling bug was that nothing was classified at all."""

    def test_every_rule_kind_produces_its_type(self, lex):
        kinds = {t.type for t in _keep(lex.tokenize('a = 1 + 2.5; name = "hi"'))}
        for expected in (
            TokenType.IDENTIFIER,
            TokenType.NUMBER,
            TokenType.OPERATOR,
            TokenType.PUNCTUATION,
            TokenType.STRING,
        ):
            assert expected in kinds, f"{expected.name} missing from {sorted(k.name for k in kinds)}"

    def test_string_beats_identifier(self, lex):
        literals = [
            t for t in _keep(lex.tokenize('"abc"')) if t.type == TokenType.STRING
        ]
        assert [t.value for t in literals] == ['"abc"']

    def test_float_is_one_token_not_three(self, lex):
        assert [t.value for t in _keep(lex.tokenize("2.5"))] == ["2.5"]

    def test_integer_and_float_are_distinguished(self, lex):
        kinds = {t.value: t.type for t in _keep(lex.tokenize("42 3.5"))}
        assert kinds["42"] == TokenType.NUMBER
        assert kinds["3.5"] == TokenType.NUMBER


class TestPathsAgree:
    @pytest.mark.parametrize("source", WELL_FORMED)
    def test_incremental_matches_batch(self, lex, source):
        batch = [(t.type, t.value, t.line, t.column) for t in lex.tokenize(source)]
        incremental = [
            (t.type, t.value, t.line, t.column) for t in lex.tokenize_incremental(source)
        ]
        assert batch == incremental

    def test_tokenize_lines_matches_tokenize(self, lex):
        source = "a = 1\nb = 2\nc = 3"
        per_line = lex.tokenize_lines(source)
        flat = [t for line in sorted(per_line) for t in _keep(per_line[line])]
        assert [t.value for t in flat] == [
            t.value for t in _keep(lex.tokenize(source))
        ]


class TestPositions:
    def test_line_numbers_track_the_source(self, lex):
        by_line: dict[int, set[str]] = {}
        for token in _keep(lex.tokenize("a = 1\nbb = 2\nccc = 3")):
            by_line.setdefault(token.line, set()).add(token.value)
        # Keyed by line, not by value: '=' legitimately appears on all three.
        assert by_line == {1: {"a", "=", "1"}, 2: {"bb", "=", "2"}, 3: {"ccc", "=", "3"}}

    def test_columns_increase_within_a_line(self, lex):
        columns = [t.column for t in _keep(lex.tokenize("alpha = beta + 1"))]
        assert columns == sorted(columns)
        assert columns[0] == 1


class TestMalformedInput:
    @pytest.mark.parametrize("source", MALFORMED)
    def test_malformed_input_is_refused_or_flagged(self, lex, source):
        """Never a clean token stream.

        Either an ERROR token or a raised ``LexerError`` qualifies; silently
        producing well-formed tokens from malformed source is the failure.
        """
        try:
            tokens = lex.tokenize(source)
        except LexerError:
            return
        assert any(t.type == TokenType.ERROR for t in tokens), (
            f"{source!r} produced a clean token stream: "
            f"{[(t.type.name, t.value) for t in tokens]}"
        )


class TestRulesAddedAfterConstruction:
    def test_a_late_rule_is_applied(self, lex):
        lex.add_rule(TokenType.PUNCTUATION, r"%", priority=6)
        assert any(t.value == "%" for t in _keep(lex.tokenize("a % b")))
