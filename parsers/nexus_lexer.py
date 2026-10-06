"""
================================================================================
PaleoAST Parsers - NEXUS Lexer Module
================================================================================

本模块实现NEXUS格式文件的词法分析器。

NEXUS格式是系统发育学领域最常用的数据交换格式，
由Maddison, Swofford和Maddison (1997)设计。

NEXUS文件结构:
==============================================================================

1. 文件头
-----------
    #NEXUS

    必须以#NEXUS标记开始（大小写不敏感）。

2. 数据块 (Blocks)
-----------
    BEGIN TAXA;
        ...
    END;

    BEGIN CHARACTERS;
        ...
    END;

    BEGIN TREES;
        ...
    END;

3. 注释格式
-----------
    - 单行注释: [...]
    - 嵌套注释: [[...]

4. 关键字
-----------
    TAXLABELS, CHARSTATELABELS, DIMENSIONS, MATRIX,
    FORMAT, TREES, TREE, UTAXAMATRIX, CHARGROUPS等

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import re
from enum import Enum, auto

from .lexer import BaseLexer, LexerError, Token

logger = logging.getLogger(__name__)


class NexusTokenType(Enum):
    """
    NEXUS词法分析Token类型

    这些类型覆盖了NEXUS文件的所有语法元素。
    """

    # 特殊标记
    NEXUS_HEADER = auto()  # #NEXUS
    BEGIN_BLOCK = auto()  # BEGIN
    END_BLOCK = auto()  # END
    END_STATEMENT = auto()  # ;

    # 标识符和关键字
    IDENTIFIER = auto()  # 标识符
    KEYWORD = auto()  # NEXUS关键字
    TAXLABELS = auto()  # TAXLABELS
    CHARSTATELABELS = auto()  # CHARSTATELABELS
    DIMENSIONS = auto()  # DIMENSIONS
    MATRIX = auto()  # MATRIX
    FORMAT = auto()  # FORMAT
    TREES = auto()  # TREES
    TREE = auto()  # TREE
    NEWICK = auto()  # NEWICK树格式

    # 数值和字符串
    INTEGER = auto()  # 整数
    FLOAT = auto()  # 浮点数
    STRING = auto()  # 字符串 (带引号)

    # 符号
    EQUALS = auto()  # =
    COLON = auto()  # :
    COMMA = auto()  # ,
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LPAREN = auto()  # (
    RPAREN = auto()  # )

    # NOTE: there is deliberately no ASTERISK member. ``*`` used to have its own
    # rule, but the MISSING rule also matches it and outranks it (priority 6
    # vs 53, same 1-char match), so the ASTERISK branch became unreachable dead
    # code -- every ``*`` became MISSING while the enum still advertised an
    # ASTERISK token that nothing could ever emit. ``*`` is MorphoJ/geomorph's
    # missing-landmark convention, so MISSING is the intended meaning; the
    # enum member was removed rather than left as a trap for the next reader
    # who writes ``if tok.type is NexusTokenType.ASTERISK``.

    # 空白和注释
    WHITESPACE = auto()  # 空白
    NEWLINE = auto()  # 换行
    COMMENT = auto()  # 注释
    NESTED_COMMENT = auto()  # 嵌套注释

    # 特殊
    SEQUENCE = auto()  # 序列数据
    TAXON_NAME = auto()  # 分类单元名称
    UNKNOWN = auto()  # 未知字符
    MISSING = auto()  # missing-value sentinel (? or - per NEXUS standard; * also accepted)

    # 终止符
    EOF = auto()
    ERROR = auto()


class NexusLexer(BaseLexer):
    """
    NEXUS文件词法分析器

    从零实现的状态机词法分析器，处理NEXUS格式的所有语法元素。

    状态转换图:
    -----------
    START ─┬─ '#NEXUS' ──> NEXUS_HEADER
           ├─ 'BEGIN' ────> BEGIN_BLOCK
           ├─ 'END' ──────> END_BLOCK
           ├─ ';' ────────> END_STATEMENT
           ├─ '[' ────────> COMMENT/NESTED
           ├─ '"' ────────> STRING
           ├─ '0-9' ──────> NUMBER
           ├─ 'A-Za-z_' ─> IDENTIFIER/KEYWORD
           └─ 其他 ───────> SYMBOL

    嵌套注释处理:
    -----------
    使用栈追踪注释嵌套深度:
        - 遇到 '[': depth += 1
        - 遇到 ']': depth -= 1
        - depth == 0: 注释结束

    示例:
        >>> lexer = NexusLexer()
        >>> with open('data.nex') as f:
        ...     tokens = lexer.tokenize(f.read())
    """

    def __init__(self):
        super().__init__(skip_whitespace=True, skip_newlines=False, case_sensitive=False)
        self._logger = logging.getLogger(f"{__name__}.NexusLexer")

        # 嵌套注释状态 (用于tokenize循环优化)
        self._in_nested_comment = False

        # (line, column) of an unterminated '[' seen during the current
        # tokenize() call, or None. Populated by _scan_comment and turned into
        # a LexerError at the end of tokenize().
        self._unterminated_comment: tuple[int, int] | None = None

        # NEXUS关键字集合 (大小写不敏感)
        self._keywords = {
            "BEGIN",
            "END",
            "TAXLABELS",
            "CHARSTATELABELS",
            "DIMENSIONS",
            "MATRIX",
            "FORMAT",
            "TREES",
            "TREE",
            "LINK",
            "SET",
            "OPTIONS",
            "UTAXAMATRIX",
            "CHARGROUPS",
            "TITLE",
            "NTAX",
            "NCHAR",
            "INTERLEAVE",
            "DATATYPE",
            "MISSING",
            "GAP",
            "SYMBOLS",
            "EQUATE",
            "MATCHCHAR",
            "NEWTAXA",
            "NOLABELS",
            "STATUS",
            "TAXNAME",
        }

    def _init_rules(self) -> None:
        """
        初始化NEXUS词法规则

        按优先级从高到低排列:
        1. #NEXUS头部 (最高优先级)
        2. 字符串 (引号保护)
        3. 嵌套注释
        4. 单行注释
        5. 浮点数
        6. 整数
        7. 标识符和关键字
        8. 空白 (跳过)
        9. 换行 (跳过)
        """
        # NEXUS missing-value tokens. The lexer only emits MISSING for the
        # single-character glyphs ``?``, ``*`` and ``-``; the textual
        # sentinels like ``NA``/``NaN`` live in ``MISSING_SENTINELS`` for the
        # matrix parsers to consume but must NOT be promoted here, since the
        # NEXUS lexer also feeds identifiers (``NA`` is a valid taxon name).
        #
        # ``-`` is the one glyph that is NOT safe to match unconditionally: it
        # is a legal NEXUS gap character but it is also the word separator in
        # hyphenated block keywords (``begin-trees``) and in taxon labels
        # (``Hsapiens-1``). Matching it bare used to split those into three
        # tokens with a bogus MISSING wedged in the middle -- a silent
        # mis-tokenisation of the identifier that a downstream consumer has no
        # way to detect. The lookarounds keep ``-`` a MISSING token only when
        # it stands alone (``"-"``), which is the gap use case, and leave
        # hyphenated words to fail loudly via LexerError instead of quietly
        # becoming the wrong label. ``?`` and ``*`` are never valid identifier
        # characters, so they need no such guard.
        self.add_rule(NexusTokenType.MISSING, r"\?|\*|(?<![A-Za-z0-9_])-(?![A-Za-z0-9_])", priority=6)

        # #NEXUS 头部
        self.add_rule(NexusTokenType.NEXUS_HEADER, r"^#NEXUS", priority=1)

        # 浮点数
        # The exponent must be allowed INDEPENDENTLY of the fractional part:
        # `1E-5` is a perfectly ordinary CONTINUOUS value, but a rule that
        # requires `\.\d+` split it into INTEGER(1) + IDENTIFIER(e) +
        # INTEGER(-5), i.e. three tokens and a shredded number.
        self.add_rule(
            NexusTokenType.FLOAT,
            r"-?\d+(?:\.\d+)?[eE][+-]?\d+|-?\d+\.\d+(?:[eE][+-]?\d+)?",
            priority=10,
        )

        # 整数
        self.add_rule(NexusTokenType.INTEGER, r"-?\d+", priority=20)

        # 标识符 (包括关键字)
        # NEXUS taxon labels are frequently non-ASCII (`Félix cánidé`), so the
        # class must be Unicode-aware. An ASCII-only class turned every such
        # character into an UNKNOWN token, shredding the name with no error.
        # The leading class deliberately stays ASCII-ish so a name cannot START
        # with a digit (which is INTEGER's job).
        self.add_rule(
            NexusTokenType.IDENTIFIER,
            r"[^\W\d][\w.]*",
            priority=30,
        )

        # 字符串 (双引号)
        self.add_rule(NexusTokenType.STRING, r'"[^"]*"', priority=5)

        # 字符串 (单引号)
        # NEXUS escapes an embedded apostrophe by doubling it (''), which is
        # exactly what `NexusWriter._quote_taxon` emits. Without accepting the
        # doubling here, the writer's own output for `O'Brien sp.` round-tripped
        # into two tokens ('O' + 'Brien sp.'), silently truncating the name.
        self.add_rule(NexusTokenType.STRING, r"'(?:[^']|'')*'", priority=5)

        # 符号
        self.add_rule(NexusTokenType.LBRACKET, r"\[", priority=40)

        self.add_rule(NexusTokenType.RBRACKET, r"\]", priority=41)

        self.add_rule(NexusTokenType.LPAREN, r"\(", priority=42)

        self.add_rule(NexusTokenType.RPAREN, r"\)", priority=43)

        self.add_rule(NexusTokenType.COLON, r":", priority=50)

        self.add_rule(NexusTokenType.COMMA, r",", priority=51)

        self.add_rule(NexusTokenType.EQUALS, r"=", priority=52)

        # No ``*`` rule: ``*`` is claimed by the MISSING rule above (priority 6).
        # Adding a priority-53 ``*`` rule here would be shadowed on every input
        # and would only re-introduce the unreachable ASTERISK branch.

        self.add_rule(NexusTokenType.END_STATEMENT, r";", priority=60)

        # 空白 (跳过)
        self.add_rule(NexusTokenType.WHITESPACE, r"[ \t]+", priority=100, skip=True)

        # 换行 (保留)
        self.add_rule(NexusTokenType.NEWLINE, r"\r?\n", priority=101, skip=False)

    def tokenize(self, source: str) -> list[Token]:
        """
        对NEXUS文件进行词法分析

        重写基类方法以处理嵌套注释和特殊NEXUS语法。

        参数:
            source: NEXUS文件内容

        返回:
            Token列表
        """
        tokens: list[Token] = []
        position = 0
        length = len(source)

        # 重置状态
        self._line = 1
        self._column = 1
        self._in_nested_comment = False
        self._unterminated_comment = None

        self._logger.debug(f"Tokenizing NEXUS source of length {length}")

        while position < length:
            # 检查注释开始 (包括嵌套注释 [[ )
            if position < length and source[position : position + 1] == "[":
                # 检查是否是嵌套注释开始 [[ (需要看当前位置和下一位置)
                if position + 2 <= length and source[position : position + 2] == "[[":
                    # 嵌套注释: 调用_scan_comment处理完整的嵌套结构
                    result = self._scan_comment(source, position)
                    if result:
                        comment_token, new_pos = result
                        tokens.append(comment_token)
                        position = new_pos
                        self._update_position(source, position - len(comment_token.value), position)
                    continue
                elif self._is_comment_start(source, position):
                    # 普通注释
                    result = self._scan_comment(source, position)
                    if result:
                        comment_token, new_pos = result
                        tokens.append(comment_token)
                        position = new_pos
                        self._update_position(source, position - len(comment_token.value), position)
                    continue

            # 尝试匹配其他规则
            match = self._try_match(source, position)

            if match is not None:
                token = self._create_token(match, position)

                # 更新位置
                new_position = match.end()
                position = new_position
                self._update_position(source, match.start(), new_position)

                # 关键字检测
                if token.type == NexusTokenType.IDENTIFIER:
                    upper_val = token.value.upper()
                    if upper_val in self._keywords:
                        # 转换为对应的关键字类型
                        token = self._convert_keyword_token(token)

                # 跳过空白
                if token.type == NexusTokenType.WHITESPACE:
                    continue

                tokens.append(token)
                self._logger.debug(f"Token: {token}")
            else:
                # 无匹配
                char = source[position]

                # 跳过孤立换行
                if char == "\n" or char == "\r":
                    position += 1
                    if char == "\r" and position < length and source[position] == "\n":
                        position += 1
                    self._line += 1
                    self._column = 1
                    continue

                # 跳过多余空白
                if char in " \t":
                    position += 1
                    self._column += 1
                    continue

                # 未知字符
                token = Token(type=NexusTokenType.UNKNOWN, value=char, line=self._line, column=self._column)
                tokens.append(token)
                self._logger.warning(f"Unknown character at line {self._line}, column {self._column}: '{char}'")
                position += 1
                self._column += 1

        # 添加EOF token
        tokens.append(Token(type=NexusTokenType.EOF, value="", line=self._line, column=self._column))

        # An unterminated '[' consumed the remainder of the file (MATRIX, rows,
        # END;), so the token stream looks structurally valid while carrying
        # none of the data. Returning it silently produced an empty/partial
        # parse with NO diagnostic at all. Raise only when there is real data
        # downstream to have lost: a comment that runs to end-of-input is
        # harmless on its own and is pinned by
        # tests/parsers/test_nexus_lexer.py::test_unclosed_comment, so an
        # input that is nothing but a comment must still tokenize.
        if self._unterminated_comment is not None:
            u_line, u_col = self._unterminated_comment
            self._unterminated_comment = None
            content_tokens = [
                t
                for t in tokens
                if t.type
                not in (NexusTokenType.EOF, NexusTokenType.NEWLINE, NexusTokenType.WHITESPACE, NexusTokenType.COMMENT)
            ]
            if content_tokens:
                raise LexerError(
                    "Unterminated comment: no matching ']' for the '[' at "
                    f"line {u_line}, column {u_col}; the rest of the file was "
                    "consumed as comment text",
                    line=u_line,
                    column=u_col,
                    char="[",
                )

        self._logger.info(f"NEXUS Tokenization complete: {len(tokens)} tokens")
        return tokens

    def _scan_comment(self, source: str, position: int) -> tuple[Token, int] | None:
        """
        扫描注释内容

        处理NEXUS格式的嵌套注释。

        参数:
            source: 源代码
            position: 起始位置 (指向注释开始的 '[')

        返回:
            (注释Token, 新位置) 或 None

        终止语义 (与 tests/parsers/test_nexus_lexer.py 一致):
            开启形式 ("[[ " 或 "[") 先行消耗并计深度 1; 此后 "[[" / "]]"
            作为单元各使深度 +1 / -1, 单个 '[' / ']' 同样各 +1 / -1。
            注释在其配对深度归零处结束。不得只认 "]]" 而忽略单个 ']'
            ——否则普通注释永不终止, 会吞掉文件剩余全部内容
            (2026-09 复审发现的回归)。
        """
        start_line = self._line
        start_column = self._column
        comment_start = position
        length = len(source)

        nested_opener = source[position : position + 2] == "[["
        pos = position + (2 if nested_opener else 1)
        depth = 1

        while pos < length and depth > 0:
            two = source[pos : pos + 2]
            if two == "[[":
                depth += 1
                pos += 2
                continue
            if two == "]]":
                depth -= 1
                pos += 2
                continue
            if source[pos] == "[":
                depth += 1
            elif source[pos] == "]":
                depth -= 1
            # 行号/列号统一由 tokenize 里的 _update_position 维护,
            # 这里不再累计, 否则多行注释会导致行号双重计数。
            pos += 1

        # 构建注释token
        end_pos = pos if depth == 0 else length
        value = source[comment_start:end_pos]

        # 检查是否未关闭
        if depth > 0:
            self._in_nested_comment = False
            # An unterminated '[' swallows everything that follows (the MATRIX,
            # the rows, even END;) into this one COMMENT token, and tokenize()
            # used to return SUCCESS with no diagnostic — so a single missing
            # ']' produced an empty/partial structure and no error. `tokenize()`
            # records `_unterminated_comment` for exactly this, so signal the
            # caller while still returning the token (tests/parsers/
            # test_nexus_lexer.py::test_unclosed_comment pins that shape).
            self._unterminated_comment = (start_line, start_column)
            self._logger.warning(
                "Unterminated comment opened at line %d, column %d; consumed the rest of the input",
                start_line,
                start_column,
            )
            return (
                Token(type=NexusTokenType.COMMENT, value=value, line=start_line, column=start_column),
                end_pos,
            )

        # 类型取决于开启形式: "[[" 开启 → NESTED_COMMENT;
        # 普通 '[' 开启的注释即使内容含 "[[" 也是 COMMENT,
        # 否则 "[ outer [[ nested ]] still_outer ]" 会被误判类型。
        if nested_opener:
            token_type = NexusTokenType.NESTED_COMMENT
        else:
            token_type = NexusTokenType.COMMENT

        self._in_nested_comment = False
        return (Token(type=token_type, value=value, line=start_line, column=start_column), pos)

    def _is_comment_start(self, source: str, position: int) -> bool:
        """
        检查是否开始注释

        NEXUS注释以 '[' 开始，但 [[ 是嵌套注释开始。

        参数:
            source: 源代码
            position: 位置

        返回:
            是否开始注释
        """
        if source[position : position + 1] != "[":
            return False
        if position + 1 < len(source) and source[position + 1 : position + 2] == "[":
            return False  # 嵌套注释
        return True

    def _convert_keyword_token(self, token: Token) -> Token:
        """
        将标识符Token转换为关键字Token

        参数:
            token: 标识符Token

        返回:
            关键字Token
        """
        upper_value = token.value.upper()

        keyword_map = {
            "BEGIN": NexusTokenType.BEGIN_BLOCK,
            "END": NexusTokenType.END_BLOCK,
            "TAXLABELS": NexusTokenType.TAXLABELS,
            "CHARSTATELABELS": NexusTokenType.CHARSTATELABELS,
            "DIMENSIONS": NexusTokenType.DIMENSIONS,
            "MATRIX": NexusTokenType.MATRIX,
            "FORMAT": NexusTokenType.FORMAT,
            "TREES": NexusTokenType.TREES,
            "TREE": NexusTokenType.TREE,
        }

        if upper_value in keyword_map:
            return Token(
                type=keyword_map[upper_value],
                value=token.value,
                line=token.line,
                column=token.column,
                end_line=token.end_line,
                end_column=token.end_column,
            )

        return token

    def _try_match(self, source: str, position: int) -> re.Match | None:
        """
        尝试在当前位置匹配规则

        跳过注释和空白字符。

        参数:
            source: 源代码
            position: 当前位置

        返回:
            匹配结果或None
        """
        if self._combined_pattern is None:
            return None

        # 跳过空白和注释
        while position < len(source):
            char = source[position]

            # 跳过空白
            if char in " \t":
                position += 1
                continue

            # 检查注释开始 - 返回 None 让 tokenize 处理
            if char == "[":
                return None

            # 跳过换行
            if char == "\n" or char == "\r":
                break

            # 找到非空白字符
            break

        # 尝试匹配
        match = self._combined_pattern.match(source, position)
        return match


def create_nexus_lexer() -> NexusLexer:
    """
    创建NEXUS专用词法分析器的便捷函数

    返回:
        配置好的NexusLexer实例
    """
    return NexusLexer()
