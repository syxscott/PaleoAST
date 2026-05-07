"""
================================================================================
PaleoAST State Machine Framework - Tokenizer Module
================================================================================

本模块提供词法分析器的基础设施，包括：
- Token类型定义
- Token类（不可变数据结构）
- LexerTokenizer基类（基于状态机的词法分析器）

词法分析器数学原理:
    词法分析器本质上是一个有限状态自动机(DFA)，
    将输入字符流转换为Token流。

    输入: Σ* (字符序列)
    输出: Τ* (Token序列)
    
    其中Τ是Token类型的有限集合。

作者: PaleoAST Development Team
版本: 3.0.0
"""

from __future__ import annotations
from typing import (
    Dict, Set, List, Optional, Callable, Any, 
    Iterator, Tuple, NamedTuple, Match, Pattern
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import re
import logging
from collections import deque

# 配置日志
logger = logging.getLogger(__name__)

# ================================================================================
# Token类型定义
# ================================================================================

class TokenType(Enum):
    """
    Token类型枚举
    
    词法分析器识别的所有Token类型。
    每个类型对应词法分析器中的一个正则表达式模式。
    
    通用Token类型:
        - EOF: 文件结束标记
        - ERROR: 词法错误
        - WHITESPACE: 空白字符（可选跳过）
        - NEWLINE: 换行符
        - COMMENT: 注释内容
        - IDENTIFIER: 标识符
        - KEYWORD: 关键字
        - NUMBER: 数字常量
        - STRING: 字符串常量
        - OPERATOR: 运算符
        - PUNCTUATION: 标点符号
        - UNKNOWN: 未知Token
    """
    # 特殊Token
    EOF = auto()
    ERROR = auto()
    UNKNOWN = auto()
    
    # 空白和注释
    WHITESPACE = auto()
    NEWLINE = auto()
    COMMENT = auto()
    
    # 标识符和关键字
    IDENTIFIER = auto()
    KEYWORD = auto()
    
    # 字面量
    NUMBER = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    CHARACTER = auto()
    
    # 符号
    OPERATOR = auto()
    PUNCTUATION = auto()
    BRACKET = auto()
    
    # 特定格式Token (NEXUS)
    NEXUS_BLOCK = auto()
    TAXLABELS = auto()
    CHARSTATELABELS = auto()
    TREES = auto()
    NEWICK = auto()


class Token(NamedTuple):
    """
    Token数据结构 (Immutable)
    
    表示词法分析器输出的最小语义单元。
    
    属性:
        type: Token类型
        value: Token的文本值
        line: 起始行号 (1-based)
        column: 起始列号 (1-based)
        end_line: 结束行号
        end_column: 结束列号
        metadata: 额外的元数据字典
    
    数学表示:
        T = (τ, v, l, c) ∈ Τ × Σ* × ℕ × ℕ
    
    示例:
        >>> Token(TokenType.IDENTIFIER, "TAXLABELS", 1, 1, 1, 10)
        >>> Token(TokenType.NUMBER, "42", 5, 10, 5, 12)
    """
    type: TokenType
    value: str
    line: int
    column: int
    end_line: int = 0
    end_column: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __new__(
        cls,
        type: TokenType,
        value: str,
        line: int,
        column: int,
        end_line: int = 0,
        end_column: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Token:
        """创建Token，确保位置信息正确"""
        if end_line == 0:
            end_line = line
        if end_column == 0:
            end_column = column + len(value) - 1
        return super().__new__(
            cls, type, value, line, column, 
            end_line, end_column, 
            metadata or {}
        )
    
    @property
    def span(self) -> Tuple[int, int]:
        """获取Token的(起始位置, 结束位置)"""
        return (self.column, self.end_column)
    
    @property
    def length(self) -> int:
        """获取Token的长度"""
        return self.end_column - self.column + 1
    
    def is_whitespace(self) -> bool:
        """检查是否为空白Token"""
        return self.type in (TokenType.WHITESPACE, TokenType.NEWLINE)
    
    def is_keyword(self) -> bool:
        """检查是否为关键字Token"""
        return self.type == TokenType.KEYWORD
    
    def is_literal(self) -> bool:
        """检查是否为字面量Token"""
        return self.type in (
            TokenType.NUMBER, TokenType.INTEGER,
            TokenType.FLOAT, TokenType.STRING, TokenType.CHARACTER
        )
    
    def is_operator(self) -> bool:
        """检查是否为运算符Token"""
        return self.type == TokenType.OPERATOR
    
    def get_int_value(self) -> int:
        """获取整数数值"""
        if self.type in (TokenType.INTEGER, TokenType.NUMBER):
            return int(self.value)
        raise ValueError(f"Token '{self.value}' is not an integer")
    
    def get_float_value(self) -> float:
        """获取浮点数值"""
        if self.type in (TokenType.FLOAT, TokenType.NUMBER, TokenType.INTEGER):
            return float(self.value)
        raise ValueError(f"Token '{self.value}' is not a number")
    
    def get_string_value(self) -> str:
        """获取字符串值（去除引号）"""
        if self.type == TokenType.STRING:
            # 移除引号
            if self.value.startswith('"') and self.value.endswith('"'):
                return self.value[1:-1]
            if self.value.startswith("'") and self.value.endswith("'"):
                return self.value[1:-1]
        return self.value
    
    def __repr__(self) -> str:
        """Token的调试字符串表示"""
        return (
            f"Token({self.type.name}, '{self.value}', "
            f"line={self.line}, col={self.column})"
        )
    
    def __str__(self) -> str:
        """Token的用户友好字符串表示"""
        return self.value


# ================================================================================
# 词法分析器基类
# ================================================================================

@dataclass
class LexerRule:
    """
    词法规则定义
    
    将TokenType与正则表达式模式关联。
    
    属性:
        token_type: 对应的Token类型
        pattern: 匹配正则表达式
        priority: 优先级（数字越小优先级越高）
        skip: 是否跳过此Token
        is_literal: 是否是字面量（精确匹配）
    """
    token_type: TokenType
    pattern: Pattern[str]
    priority: int = 100
    skip: bool = False
    is_literal: bool = False


class LexerTokenizer:
    """
    基于正则表达式的词法分析器
    
    核心算法:
        1. 使用多个正则表达式并行匹配输入
        2. 选择最长匹配（Maximal Munch原则）
        3. 相同长度时按优先级选择
        
    性能优化:
        - 使用re.compile预编译正则表达式
        - 使用re.LOCALE处理多字节字符
        - 支持增量式处理大文件
    
    数学原理:
        对于输入字符串 s = c1c2...cn，
        词法分析器找到最大前缀 w = c1...cm (m ≤ n)，
        使得存在规则 r 使得 w ∈ L(r)，
        其中 L(r) 是正则表达式 r.pattern 定义的语言。
    
    示例:
        >>> rules = [
        ...     LexerRule(TokenType.KEYWORD, re.compile(r'\\b(IF|ELSE|FOR)\\b'), priority=10),
        ...     LexerRule(TokenType.IDENTIFIER, re.compile(r'[A-Za-z_][A-Za-z0-9_]*'), priority=20),
        ...     LexerRule(TokenType.NUMBER, re.compile(r'\\d+(\\.\\d+)?'), priority=30),
        ... ]
        >>> lexer = LexerTokenizer(rules)
        >>> tokens = lexer.tokenize("IF x = 42 THEN y = 1.5")
    """
    
    def __init__(
        self,
        rules: Optional[List[LexerRule]] = None,
        skip_whitespace: bool = True,
        skip_newlines: bool = False,
        case_sensitive: bool = True
    ):
        """
        初始化词法分析器
        
        参数:
            rules: 词法规则列表
            skip_whitespace: 是否跳过空白字符
            skip_newlines: 是否跳过换行符
            case_sensitive: 是否区分大小写
        """
        self._rules: List[LexerRule] = rules or []
        self._skip_whitespace = skip_whitespace
        self._skip_newlines = skip_newlines
        self._case_sensitive = case_sensitive
        
        # 编译合并后的正则表达式
        self._combined_pattern: Optional[Pattern[str]] = None
        self._compile_pattern()
        
        # 位置跟踪
        self._line = 1
        self._column = 1
        
        # 日志
        self._logger = logging.getLogger(f"{__name__}.Lexer")
        
        # 关键字映射
        self._keywords: Dict[str, TokenType] = {}
    
    def _compile_pattern(self) -> None:
        """
        编译合并的正则表达式
        
        使用 (?P<name>pattern) 命名捕获组来区分不同的规则。
        编译后的模式允许一次性匹配所有可能的Token。
        """
        if not self._rules:
            return
        
        # 按优先级排序
        sorted_rules = sorted(self._rules, key=lambda r: r.priority)
        
        # 构建分支模式
        branches = []
        for rule in sorted_rules:
            flags = 0 if self._case_sensitive else re.IGNORECASE
            pattern_str = rule.pattern.pattern
            branches.append(f"(?P<{rule.token_type.name}>{pattern_str})")
        
        combined = "|".join(branches)
        self._combined_pattern = re.compile(combined, flags=re.MULTILINE)
    
    def add_rule(
        self,
        token_type: TokenType,
        pattern: str,
        priority: int = 100,
        skip: bool = False
    ) -> LexerRule:
        """
        添加一条词法规则
        
        参数:
            token_type: Token类型
            pattern: 正则表达式模式字符串
            priority: 优先级
            skip: 是否跳过
        
        返回:
            创建的LexRule对象
        """
        compiled_pattern = re.compile(pattern)
        rule = LexerRule(
            token_type=token_type,
            pattern=compiled_pattern,
            priority=priority,
            skip=skip
        )
        self._rules.append(rule)
        self._compile_pattern()
        return rule
    
    def add_keyword(self, keyword: str, token_type: TokenType) -> None:
        """
        添加关键字映射
        
        参数:
            keyword: 关键字字符串
            token_type: 对应的Token类型
        """
        self._keywords[keyword.upper() if not self._case_sensitive else keyword] = token_type
    
    def set_keywords(self, keywords: Dict[str, TokenType]) -> None:
        """
        批量设置关键字映射
        
        参数:
            keywords: {关键字: Token类型} 字典
        """
        for kw, ttype in keywords.items():
            self.add_keyword(kw, ttype)
    
    def tokenize(self, source: str) -> List[Token]:
        """
        对输入字符串进行词法分析
        
        核心算法:
            1. 从当前位置开始，使用合并正则匹配
            2. 如果匹配成功，生成Token并移动位置
            3. 如果匹配失败，生成ERROR Token并移动一个字符
            4. 重复直到到达字符串末尾
        
        参数:
            source: 源代码字符串
        
        返回:
            Token列表
        
        时间复杂度: O(n * m)
            其中 n 是输入长度，m 是规则数量
        """
        tokens: List[Token] = []
        position = 0
        length = len(source)
        
        self._line = 1
        self._column = 1
        
        self._logger.debug(f"Tokenizing source of length {length}")
        
        while position < length:
            match = self._combined_pattern.match(source, position) if self._combined_pattern else None
            
            if match:
                # 确定匹配的Token类型
                token_type = None
                for name, value in match.groupdict().items():
                    if value is not None:
                        try:
                            token_type = TokenType[name]
                        except KeyError:
                            # 尝试从规则中查找
                            for rule in self._rules:
                                if rule.token_type.name == name:
                                    token_type = rule.token_type
                                    break
                        break
                
                if token_type is None:
                    # 回退：使用第一个非None组
                    for gname, gvalue in zip(match.group().split(), [match.group()]):
                        if gvalue is not None:
                            token_type = TokenType.UNKNOWN
                            break
                
                token_value = match.group()
                start_pos = position
                start_line = self._line
                start_col = self._column
                
                # 更新位置
                position = match.end()
                position = self._update_position(
                    source, start_pos, position
                )
                
                # 创建Token
                token = Token(
                    type=token_type or TokenType.UNKNOWN,
                    value=token_value,
                    line=start_line,
                    column=start_col,
                    end_line=self._line,
                    end_column=self._column
                )
                
                # 检查是否是关键字
                if token.type == TokenType.IDENTIFIER:
                    check_key = token.value.upper() if not self._case_sensitive else token.value
                    if check_key in self._keywords:
                        token = Token(
                            type=self._keywords[check_key],
                            value=token.value,
                            line=token.line,
                            column=token.column,
                            end_line=token.end_line,
                            end_column=token.end_column,
                            metadata=token.metadata
                        )
                
                # 决定是否添加Token
                should_add = True
                if token.is_whitespace() and self._skip_whitespace:
                    should_add = False
                if token.type == TokenType.NEWLINE and self._skip_newlines:
                    should_add = False
                
                # 检查是否标记为跳过
                for rule in self._rules:
                    if rule.token_type == token.type and rule.skip:
                        should_add = False
                        break
                
                if should_add:
                    tokens.append(token)
                    self._logger.debug(f"Generated token: {token}")
            else:
                # 没有匹配：生成ERROR token并前移一个字符
                char = source[position]
                token = Token(
                    type=TokenType.ERROR,
                    value=char,
                    line=self._line,
                    column=self._column
                )
                tokens.append(token)
                
                self._logger.warning(
                    f"Lexical error at line {self._line}, column {self._column}: "
                    f"unexpected character '{char}'"
                )
                
                position += 1
                position = self._update_position(source, position - 1, position)
        
        # 添加EOF token
        tokens.append(Token(
            type=TokenType.EOF,
            value="",
            line=self._line,
            column=self._column
        ))
        
        self._logger.info(f"Tokenization complete: {len(tokens)} tokens generated")
        return tokens
    
    def _update_position(self, source: str, start: int, end: int) -> int:
        """
        更新行列位置
        
        参数:
            source: 源代码
            start: 起始位置
            end: 结束位置
        
        返回:
            新的列位置
        """
        for i in range(start, end):
            if source[i] == '\n':
                self._line += 1
                self._column = 1
            else:
                self._column += 1
        return end
    
    def tokenize_incremental(self, source: str) -> Iterator[Token]:
        """
        增量式词法分析（生成器版本）
        
        适用于处理超大文件，避免一次性加载所有Token。
        
        参数:
            source: 源代码字符串
        
        生成:
            Token对象
        """
        position = 0
        length = len(source)
        
        self._line = 1
        self._column = 1
        
        while position < length:
            match = self._combined_pattern.match(source, position) if self._combined_pattern else None
            
            if match:
                token_value = match.group()
                start_line = self._line
                start_col = self._column
                
                position = match.end()
                position = self._update_position(source, position - len(token_value), position)
                
                token = Token(
                    type=TokenType.UNKNOWN,
                    value=token_value,
                    line=start_line,
                    column=start_col,
                    end_line=self._line,
                    end_column=self._column
                )
                
                if not token.is_whitespace():
                    yield token
            else:
                char = source[position]
                yield Token(
                    type=TokenType.ERROR,
                    value=char,
                    line=self._line,
                    column=self._column
                )
                position += 1
                position = self._update_position(source, position - 1, position)
        
        yield Token(type=TokenType.EOF, value="", line=self._line, column=self._column)
    
    def get_tokens_with_lines(self, source: str) -> Dict[int, List[Token]]:
        """
        按行分组获取Token
        
        参数:
            source: 源代码
        
        返回:
            {行号: [Token列表]} 字典
        """
        tokens = self.tokenize(source)
        lines: Dict[int, List[Token]] = {}
        
        for token in tokens:
            if token.line not in lines:
                lines[token.line] = []
            lines[token.line].append(token)
        
        return lines


# ================================================================================
# 便捷函数
# ================================================================================

def create_basic_lexer() -> LexerTokenizer:
    """
    创建基本的通用词法分析器
    
    返回:
        配置好的LexerTokenizer实例
    """
    rules = [
        # 空白和注释
        LexerRule(TokenType.WHITESPACE, re.compile(r'[ \t]+'), skip=True),
        LexerRule(TokenType.NEWLINE, re.compile(r'\r?\n')),
        LexerRule(TokenType.COMMENT, re.compile(r'--.*$|//.*$|#.*$', re.MULTILINE)),
        
        # 标识符和关键字
        LexerRule(TokenType.IDENTIFIER, re.compile(r'[A-Za-z_][A-Za-z0-9_]*'), priority=10),
        
        # 数字
        LexerRule(TokenType.FLOAT, re.compile(r'\d+\.\d+([eE][+-]?\d+)?'), priority=20),
        LexerRule(TokenType.INTEGER, re.compile(r'\d+'), priority=30),
        
        # 字符串
        LexerRule(TokenType.STRING, re.compile(r'"[^"]*"|\'[^\']*\''), priority=5),
        
        # 运算符
        LexerRule(TokenType.OPERATOR, re.compile(r'[+\-*/%=<>!&|^~]+'), priority=40),
        
        # 标点
        LexerRule(TokenType.PUNCTUATION, re.compile(r'[;,:\[\]{}().]'), priority=50),
    ]
    
    lexer = LexerTokenizer(rules)
    
    # 添加常见关键字
    keywords = {
        'BEGIN', 'END', 'IF', 'ELSE', 'WHILE', 'FOR', 'RETURN',
        'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'NULL',
        'TAXLABELS', 'CHARACTERS', 'TREES', 'MATRIX', 'DIMENSIONS',
        'NEXUS', 'FORMAT', 'OPTIONS', 'SET'
    }
    
    for kw in keywords:
        lexer.add_keyword(kw, TokenType.KEYWORD)
    
    return lexer
