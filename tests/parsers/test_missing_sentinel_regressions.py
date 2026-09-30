"""Regression tests for the shared missing-value sentinel batch.

Each test here pins a defect that the batch actually shipped, and each was
checked against the pre-fix code to confirm it goes red. The comments say
which defect it locks down; a test that passes both before and after a fix is
decoration and is not worth keeping.

Defects covered:
  1. TPS: a coordinate line mixing a sentinel with a non-numeric token escaped
     as a bare ``ValueError`` with no file/line context, because the sentinel
     substitution lived in a second, unguarded branch.
  2. NEXUS: ``-`` was promoted to MISSING unconditionally, which silently split
     hyphenated words (``begin-trees``, ``Hsapiens-1``) into three tokens with
     a plausible-looking MISSING wedged in the middle.
  3. NEXUS: the ``*`` ASTERISK rule was shadowed by the MISSING rule, leaving
     a public enum member that no input could ever produce.
  4. ``MISSING_SENTINELS`` stored uppercase spellings but documented itself as
     case-insensitive, so ``"na" in MISSING_SENTINELS`` was ``False``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.nexus_lexer import NexusLexer, NexusTokenType
from parsers.sentinels import MISSING_SENTINELS, is_missing_token
from parsers.tps_parser import TPSParseError, parse_tps_file

GLYPHS = ("?", "*", "-")


def _write_tps(tmp_path: Path, content: str) -> str:
    p = tmp_path / "probe.tps"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestTPSSentinelErrorContext:
    """Defect 1: the sentinel branch skipped the ValueError guard."""

    def test_sentinel_plus_junk_raises_tps_error_with_context(self, tmp_path):
        # Before the fix this escaped as a bare ValueError from the
        # `float("nan") if is_missing_token(t) else float(t)` comprehension,
        # carrying no file name and no line number.
        path = _write_tps(
            tmp_path,
            "LM=2\nID=Specimen1\n10.0 ? banana\n30.0 40.0\n",
        )
        with pytest.raises(TPSParseError) as excinfo:
            parse_tps_file(path)
        # The error must be the *structured* one, not a leaked ValueError.
        assert excinfo.value.file_path
        assert excinfo.value.line_number == 3

    def test_junk_without_sentinel_still_raises_tps_error(self, tmp_path):
        """Control: the pre-existing path must keep behaving identically."""
        path = _write_tps(
            tmp_path,
            "LM=2\nID=Specimen1\n10.0 20.0 banana\n30.0 40.0\n",
        )
        with pytest.raises(TPSParseError) as excinfo:
            parse_tps_file(path)
        assert excinfo.value.line_number == 3

    def test_sentinel_alone_still_parses(self, tmp_path):
        """The capability the batch was added for must survive the fix."""
        path = _write_tps(
            tmp_path,
            "LM=2\nID=Specimen1\n10.0 ?\n30.0 40.0\n",
        )
        result = parse_tps_file(path)
        assert result.n_landmarks == 2
        assert len(result.specimens) == 1


class TestNexusHyphenIsNotMissing:
    """Defect 2: ``-`` split hyphenated words into a fake MISSING token."""

    @pytest.mark.parametrize("src", ["begin-trees", "Hsapiens-1", "T-rex-01"])
    def test_hyphenated_word_has_no_missing_token(self, src):
        tokens = NexusLexer().tokenize(src)
        assert not any(t.type is NexusTokenType.MISSING for t in tokens), (
            f"{src!r} was silently split into a MISSING token: {tokens}"
        )

    def test_hyphenated_keyword_splits_into_begin_and_trees(self):
        """The hyphen must land on the pre-existing UNKNOWN path, not MISSING.

        UNKNOWN is the lexer's explicit 'I do not recognise this' signal (it
        also logs a warning). MISSING looks like a legitimate data value, so a
        consumer has no way to notice the mis-tokenisation. This is the
        difference between failing loudly and failing silently.
        """
        tokens = NexusLexer().tokenize("begin-trees;")
        kinds = [t.type for t in tokens if t.type is not NexusTokenType.EOF]
        assert NexusTokenType.BEGIN_BLOCK in kinds
        assert NexusTokenType.TREES in kinds
        assert NexusTokenType.UNKNOWN in kinds
        assert NexusTokenType.MISSING not in kinds

    @pytest.mark.parametrize("glyph", GLYPHS)
    def test_bare_glyph_is_still_missing(self, glyph):
        """Positive control: the batch's actual capability must not regress."""
        tokens = NexusLexer().tokenize(glyph)
        missing = [t for t in tokens if t.type is NexusTokenType.MISSING]
        assert len(missing) == 1
        assert missing[0].value == glyph

    def test_format_declaration_line_still_lexes(self):
        """`FORMAT MISSING=? GAP=-;` is the standard NEXUS declaration."""
        tokens = NexusLexer().tokenize("FORMAT MISSING=? GAP=-;")
        values = [t.value for t in tokens if t.type is NexusTokenType.MISSING]
        assert values == ["?", "-"]


class TestNexusNoUnreachableGlyphRule:
    """Defect 3: a public enum member no input could ever produce."""

    def test_asterisk_enum_member_is_gone(self):
        # The ASTERISK member advertised a token that the MISSING rule made
        # impossible. If it ever comes back, the two must not overlap.
        assert not hasattr(NexusTokenType, "ASTERISK")

    @pytest.mark.parametrize("glyph", GLYPHS)
    def test_only_the_missing_rule_claims_each_glyph(self, glyph):
        """A rule shadowed by a higher-priority rule is dead code.

        This is the general form of the defect: any second rule matching the
        same glyph can never win, so it is a trap for the next reader.
        """
        lexer = NexusLexer()
        claimants = [
            rule
            for rule in lexer._rules
            if rule.pattern.match(glyph) is not None
            and rule.pattern.match(glyph).group() == glyph
        ]
        assert len(claimants) == 1, (
            f"{glyph!r} is claimed by {[r.token_type.name for r in claimants]}; "
            "all but one are unreachable"
        )
        assert claimants[0].token_type is NexusTokenType.MISSING


class TestSentinelSetIsActuallyCaseInsensitive:
    """Defect 4: documented as case-insensitive, behaved as uppercase-only."""

    @pytest.mark.parametrize(
        "token", ["na", "NA", "nan", "NaN", "NAN", "none", "None", "null", "NULL"]
    )
    def test_membership_folds_case(self, token):
        # Before the fix only the uppercase spellings were members, so
        # `"na" in MISSING_SENTINELS` was False -- a missed missing value.
        assert token in MISSING_SENTINELS
        assert is_missing_token(token)

    def test_non_sentinels_are_still_rejected(self):
        """Case folding must not over-accept: these are real values."""
        for token in ("", "0", "1.5", "n", "nullish", "-1", "none_of", "n/a/"):
            assert not is_missing_token(token), f"{token!r} must not read as missing"
            assert token not in MISSING_SENTINELS

    def test_stored_spellings_are_uppercase(self):
        """Guards the fast path MISSING_SENTINELS_UPPER is derived from."""
        for member in MISSING_SENTINELS:
            assert member == member.upper()
