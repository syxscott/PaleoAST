"""
================================================================================
Tests for NEXUS Lexer - Nested Comment Bug Fixes
================================================================================

本测试文件验证 NEXUS lexer 的嵌套注释解析是否正确。

Bug 修复历史:
    - 2026-07-27: 修复 _scan_comment 递归方向错误
      当遇到 [ 时应增加嵌套深度, ] 时应减少嵌套深度

测试覆盖:
    1. 简单注释: [ comment ]
    2. 嵌套注释: [[ nested ]]
    3. 多层嵌套: [[ outer [ inner [ deepest ] ] outer_end ]]
    4. 混合注释: [ outer [[ nested ]] still_outer ]
    5. 未关闭注释: [ unclosed
    6. 空注释: []
    7. 多行注释
    8. 注释中的特殊字符

作者: PaleoAST Development Team
"""

from __future__ import annotations

import pytest

from parsers.nexus_lexer import NexusLexer, NexusTokenType


class TestNexusLexerSimpleComments:
    """简单注释测试"""

    def setup_method(self):
        self.lexer = NexusLexer()

    def test_simple_comment(self):
        """测试简单注释 [ comment ]"""
        source = "[ this is a comment ]"
        tokens = self.lexer.tokenize(source)

        # 应该有一个 COMMENT token
        comment_tokens = [t for t in tokens if t.type == NexusTokenType.COMMENT]
        assert len(comment_tokens) == 1
        assert comment_tokens[0].value == "[ this is a comment ]"

    def test_empty_comment(self):
        """测试空注释 []"""
        source = "[]"
        tokens = self.lexer.tokenize(source)

        comment_tokens = [t for t in tokens if t.type == NexusTokenType.COMMENT]
        assert len(comment_tokens) == 1
        assert comment_tokens[0].value == "[]"

    def test_unclosed_comment(self):
        """测试未关闭的注释 [ unclosed"""
        source = "[ unclosed comment without closing"
        tokens = self.lexer.tokenize(source)

        comment_tokens = [t for t in tokens if t.type == NexusTokenType.COMMENT]
        assert len(comment_tokens) == 1
        # 未关闭的注释也应该被捕获
        assert comment_tokens[0].value == source

    def test_multiline_comment(self):
        """测试多行注释"""
        source = "[ line1\nline2\nline3 ]"
        tokens = self.lexer.tokenize(source)

        comment_tokens = [t for t in tokens if t.type == NexusTokenType.COMMENT]
        assert len(comment_tokens) == 1
        assert "\n" in comment_tokens[0].value


class TestNexusLexerNestedComments:
    """嵌套注释测试 - 核心测试"""

    def setup_method(self):
        self.lexer = NexusLexer()

    def test_simple_nested_comment(self):
        """测试简单嵌套注释 [[ nested ]]"""
        source = "[[ nested comment ]]"
        tokens = self.lexer.tokenize(source)

        # 嵌套注释应该被识别为 NESTED_COMMENT
        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == "[[ nested comment ]]"

    def test_double_nested_comment(self):
        """测试双层嵌套注释 [[ outer [ inner ] ]]"""
        source = "[[ outer [ inner ] ]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == "[[ outer [ inner ] ]]"

    def test_triple_nested_comment(self):
        """测试三层嵌套注释 [[ outer [ inner [ deepest ] ] ]]"""
        source = "[[ outer [ inner [ deepest ] ] ]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == "[[ outer [ inner [ deepest ] ] ]]"

    def test_mixed_nesting(self):
        """测试混合嵌套 [ outer [[ nested ]] still_outer ]"""
        source = "[ outer [[ nested ]] still_outer ]"
        tokens = self.lexer.tokenize(source)

        # 外层是普通注释, 内层 [[ nested ]] 是嵌套注释
        comment_tokens = [t for t in tokens if t.type == NexusTokenType.COMMENT]
        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]

        # 外层整体作为一个注释
        assert len(comment_tokens) == 1
        assert comment_tokens[0].value == source

    def test_nested_comment_with_special_chars(self):
        """测试嵌套注释中的特殊字符"""
        source = "[[ comment with [ and ] inside ]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == source


class TestNexusLexerIntegration:
    """集成测试 - 与其他 NEXUS 元素的交互"""

    def setup_method(self):
        self.lexer = NexusLexer()

    def test_comment_before_taxlabels(self):
        """测试 TAXLABELS 前的注释"""
        source = """#NEXUS
[[ metadata comment ]]
BEGIN TAXA;
END;
"""
        tokens = self.lexer.tokenize(source)

        # 应该正确识别嵌套注释和关键字
        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        keyword_tokens = [t for t in tokens if t.type == NexusTokenType.KEYWORD]

        assert len(nested_tokens) == 1
        assert nested_tokens[0].value.strip() == "[[ metadata comment ]]"

    def test_nested_comment_in_matrix(self):
        """测试 MATRIX 中的嵌套注释"""
        source = """#NEXUS
BEGIN CHARACTERS;
[[ character data comment ]]
DIMENSIONS NCHAR=10;
MATRIX
[[ taxon1 sequence ]]
taxon1 ACGTACGTAC
;
END;
"""
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        # 应该有 2 个嵌套注释
        assert len(nested_tokens) == 2

    def test_multiple_nested_comments(self):
        """测试多个嵌套注释"""
        source = "[[ comment 1 ]] text [[ comment 2 ]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 2
        assert nested_tokens[0].value == "[[ comment 1 ]]"
        assert nested_tokens[1].value == "[[ comment 2 ]]"


class TestNexusLexerEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.lexer = NexusLexer()

    def test_consecutive_nested_comments(self):
        """测试连续嵌套注释 [[a]][[b]]"""
        source = "[[a]][[b]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 2

    def test_nested_comment_adjacent_to_text(self):
        """测试嵌套注释紧邻文本"""
        source = "taxon[[comment]]name"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        identifier_tokens = [t for t in tokens if t.type == NexusTokenType.IDENTIFIER]

        # 不应该有合并的标识符
        assert len(nested_tokens) == 1
        assert len(identifier_tokens) == 2

    def test_deeply_nested(self):
        """测试深层嵌套 (5层)"""
        source = "[[[[[ deep ]]]]]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == source

    def test_empty_nested_comment(self):
        """测试空嵌套注释 [[]]"""
        source = "[[]]"
        tokens = self.lexer.tokenize(source)

        nested_tokens = [t for t in tokens if t.type == NexusTokenType.NESTED_COMMENT]
        assert len(nested_tokens) == 1
        assert nested_tokens[0].value == "[[]]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
