"""Regression tests for ``state_machine/tokenizer.py`` token classification.

The whole point of a tokeniser is what *kind* of token it emits, and that was
the one property nothing tested. `tests/state_machine/` contained a single file
about automaton negation, which never calls `tokenize`, never names a
`TokenType`, and never looks at `_group_rules` -- so the classification bug
below was invisible to the suite.

The bug: `LexerTokenizer.__init__` called `_compile_pattern()` and only then
assigned `self._group_rules = {}`. `_compile_pattern()` clears and rebuilds
that map, so the assignment wiped all nine entries the method had just filled
in. `name in self._group_rules` was therefore always False, `token_type` stayed
None, and every token fell through to UNKNOWN. The regex still matched and the
text, line and column were all correct, so nothing looked broken.
"""

from __future__ import annotations

import pytest

from state_machine.tokenizer import (
    LexerRule,
    LexerTokenizer,
    TokenType,
    create_basic_lexer,
)


@pytest.fixture
def lexer() -> LexerTokenizer:
    return create_basic_lexer()


class TestTokensAreClassified:
    def test_group_rules_survive_compilation(self, lexer):
        """The invariant the init order broke: the map is populated.

        `_compile_pattern()` clears and rebuilds this map, so it has to exist
        *before* that call. Checking its size is the direct guard.
        """
        assert len(lexer._group_rules) > 0, (
            "_group_rules is empty, so no token can ever be classified and "
            "every token falls through to UNKNOWN"
        )

    def test_not_everything_is_unknown(self, lexer):
        tokens = [t for t in lexer.tokenize("count = 42") if t.type is not TokenType.EOF]
        kinds = {t.type for t in tokens}
        assert kinds != {TokenType.UNKNOWN}, "the lexer classified nothing"

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("count", TokenType.IDENTIFIER),
            ("42", TokenType.INTEGER),
            ("=", TokenType.OPERATOR),
            (";", TokenType.PUNCTUATION),
        ],
    )
    def test_each_category_is_recognised(self, lexer, source, expected):
        tokens = [t for t in lexer.tokenize(source) if t.type is not TokenType.EOF]
        assert tokens, f"{source!r} produced no tokens"
        assert tokens[0].type is expected, f"{source!r} -> {tokens[0].type.name}"

    def test_classification_survives_a_long_mixed_source(self, lexer):
        source = "BEGIN TAXLABELS; nchar = 12 if flag > 1 ;"
        kinds = {t.type for t in lexer.tokenize(source)}
        assert TokenType.KEYWORD in kinds
        assert TokenType.IDENTIFIER in kinds
        assert TokenType.INTEGER in kinds
        assert TokenType.OPERATOR in kinds
        assert TokenType.PUNCTUATION in kinds
        assert TokenType.UNKNOWN not in kinds


class TestNexusKeywordsAreReachable:
    """create_basic_lexer registers keywords upper-cased; the lexer must be
    case-insensitive for any of them to ever match."""

    @pytest.mark.parametrize("word", ["begin", "BEGIN", "BeGiN", "end", "END", "taxlabels", "TAXLABELS", "if", "IF"])
    def test_keyword_matches_in_any_case(self, lexer, word):
        tokens = [t for t in lexer.tokenize(word) if t.type is not TokenType.EOF]
        assert tokens[0].type is TokenType.KEYWORD, f"{word!r} -> {tokens[0].type.name}"

    def test_a_non_keyword_is_still_an_identifier(self, lexer):
        tokens = [t for t in lexer.tokenize("my_taxon") if t.type is not TokenType.EOF]
        assert tokens[0].type is TokenType.IDENTIFIER


class TestBothTokenizeApisAgree:
    """Two entry points on one lexer must not disagree about token types."""

    SOURCES = [
        "BEGIN TAXLABELS; nchar = 12 if flag > 1 ;",
        "  leading and trailing whitespace  ",
        "no tokens needing skip flags here",
    ]

    @pytest.mark.parametrize("source", SOURCES)
    def test_incremental_matches_batch(self, lexer, source):
        batch = [
            (t.type, t.value) for t in lexer.tokenize(source) if t.type is not TokenType.EOF
        ]
        incremental = [
            (t.type, t.value) for t in lexer.tokenize_incremental(source) if t.type is not TokenType.EOF
        ]
        assert incremental == batch, "the two tokenisation APIs disagree"

    def test_incremental_classifies_types(self, lexer):
        tokens = [
            t
            for t in lexer.tokenize_incremental("count = 42")
            if t.type is not TokenType.EOF
        ]
        assert {t.type for t in tokens} != {TokenType.UNKNOWN}


class TestSkipFlagsAreHonouredByBothApis:
    def test_whitespace_is_skipped_by_both(self, lexer):
        source = "a   b"
        batch = [t.value for t in lexer.tokenize(source) if t.type is not TokenType.EOF]
        incremental = [t.value for t in lexer.tokenize_incremental(source) if t.type is not TokenType.EOF]
        assert batch == incremental
        assert all(v.strip() for v in batch), f"whitespace leaked through: {batch}"


class TestCustomLexerStillCompiles:
    """A hand-built rule set must classify too, not just the shared factory."""

    def test_minimal_rule_set(self):
        import re

        rules = [
            LexerRule(TokenType.NUMBER, re.compile(r"\d+"), priority=10),
            LexerRule(TokenType.IDENTIFIER, re.compile(r"[a-z]+"), priority=20),
            # Whitespace needs a rule of its own: a lexer with no whitespace
            # rule treats a space as an unexpected character, which is correct
            # behaviour rather than a bug.
            LexerRule(TokenType.WHITESPACE, re.compile(r"\s+"), priority=99, skip=True),
        ]
        custom = LexerTokenizer(rules)
        assert len(custom._group_rules) == 3
        tokens = [t for t in custom.tokenize("ab 12") if t.type is not TokenType.EOF]
        assert [t.type for t in tokens] == [TokenType.IDENTIFIER, TokenType.NUMBER]

    def test_rule_set_without_a_whitespace_rule_reports_it_as_error(self):
        """Pins the behaviour the previous test tripped over, deliberately."""
        import re

        rules = [
            LexerRule(TokenType.NUMBER, re.compile(r"\d+"), priority=10),
            LexerRule(TokenType.IDENTIFIER, re.compile(r"[a-z]+"), priority=20),
        ]
        custom = LexerTokenizer(rules)
        tokens = [t for t in custom.tokenize("ab 12") if t.type is not TokenType.EOF]
        assert [t.type for t in tokens] == [
            TokenType.IDENTIFIER,
            TokenType.ERROR,
            TokenType.NUMBER,
        ]
