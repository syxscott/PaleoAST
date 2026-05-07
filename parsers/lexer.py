"""
================================================================================
PaleoAST Parsers - Base Lexer Module
================================================================================

本模块提供词法分析器(Lexer)的基类实现。

词法分析器核心概念:
==============================================================================

1. 词法分析器定义
------------------
词法分析器(Scanner/Lexer)是将输入字符流转换为Token流的程序。
它是编译器/解释器的前端组件。

数学表示:
    Lexer: Σ* → Τ*
    其中 Σ 是输入字符集，Τ 是Token类型集合

2. 状态机实现
------------------
词法分析器内部使用有限状态自动机(DFA)实现。
每个正则表达式模式对应一个状态机。

3. 最大 Munch 原则
------------------
当有多个可能的Token匹配时，选择最长的匹配。
当长度相同时，选择优先级最高的匹配。

4. Token结构
------------------
Token = (Type, Value, Line, Column)
- Type: Token类型枚举
- Value: Token的文本值
- Line: 起始行号 (1-based)
- Column: 起始列号 (1-based)

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import (
    Dict, Set, List, Optional, Callable, Any,
    Iterator, Tuple, Pattern, Match
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import re

logger = logging.getLogger(__name__)


class LexerError(Exception):
    """
    词法分析错误异常
    
    当输入包含无法识别的字符序列时抛出。
    
    属性:
        message: 错误消息
        line: 错误发生行号
        column: 错误发生列号
        char: 导致错误的字符
    """
    
    def __init__(
        self,
        message: str,
        line: int = 0,
        column: int = 0,
        char: str = ''
    ):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.char = char
    
    def __str__(self) -> str:
        return (
            f"LexerError at line {self.line}, column {self.column}: "
            f"{self.message} (character: '{self.char}')"
        )


@dataclass
class Token:
    """
    Token数据结构
    
    表示词法分析器输出的最小语义单元。
    
    属性:
        type: Token类型
        value: Token的文本值
        line: 起始行号 (1-based)
        column: 起始列号 (1-based)
        end_line: 结束行号
        end_column: 结束列号
    """
    type: 'TokenType'
    value: str
    line: int
    column: int
    end_line: int = 0
    end_column: int = 0
    
    def __post_init__(self):
        if self.end_line == 0:
            self.end_line = self.line
        if self.end_column == 0:
            self.end_column = self.column + len(self.value) - 1
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
    
    @property
    def span(self) -> Tuple[int, int]:
        """获取Token的(起始列, 结束列)"""
        return (self.column, self.end_column)


class TokenType(Enum):
    """Token类型基类"""
    EOF = auto()
    ERROR = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    WHITESPACE = auto()
    NEWLINE = auto()
    COMMENT = auto()


@dataclass
class LexerRule:
    """
    词法规则定义
    
    将TokenType与正则表达式模式关联。
    
    属性:
        token_type: 对应的Token类型
        pattern: 匹配正则表达式 (预编译)
        priority: 优先级 (数字越小优先级越高)
        skip: 是否跳过此Token (不生成)
    """
    token_type: TokenType
    pattern: Pattern[str]
    priority: int = 100
    skip: bool = False


class BaseLexer(ABC):
    """
    词法分析器基类
    
    提供词法分析器的通用框架，子类通过添加规则来定义词法。
    
    核心算法:
        1. 从头开始逐字符扫描
        2. 对每个位置，尝试所有规则找最长匹配
        3. 生成Token并更新位置
        4. 重复直到文件结束
    
    性能优化:
        - 使用re.compile预编译正则表达式
        - 合并规则为单一正则表达式
        - 支持增量式处理
    
    示例:
        >>> class MyLexer(BaseLexer):
        ...     def _init_rules(self):
        ...         self.add_rule(TokenType.NUMBER, r'\\d+')
        ...         self.add_rule(TokenType.IDENTIFIER, r'[a-zA-Z_]\\w*')
        >>> lexer = MyLexer()
        >>> tokens = lexer.tokenize("count = 42")
    """
    
    def __init__(
        self,
        skip_whitespace: bool = True,
        skip_newlines: bool = False,
        case_sensitive: bool = True
    ):
        """
        初始化词法分析器
        
        参数:
            skip_whitespace: 是否跳过空白字符
            skip_newlines: 是否跳过换行符
            case_sensitive: 是否区分大小写
        """
        self._rules: List[LexerRule] = []
        self._skip_whitespace = skip_whitespace
        self._skip_newlines = skip_newlines
        self._case_sensitive = case_sensitive
        self._combined_pattern: Optional[Pattern[str]] = None
        
        # 位置跟踪
        self._line = 1
        self._column = 1
        
        # 日志
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化规则
        self._init_rules()
        
        # 编译规则
        self._compile_rules()
    
    @abstractmethod
    def _init_rules(self) -> None:
        """
        初始化词法规则
        
        子类必须实现此方法以添加特定的词法规则。
        
        示例:
            def _init_rules(self):
                self.add_rule(TokenType.NUMBER, r'\\d+', priority=10)
                self.add_rule(TokenType.IDENTIFIER, r'[a-zA-Z_]\\w*', priority=20)
        """
        pass
    
    def add_rule(
        self,
        token_type: TokenType,
        pattern: str,
        priority: int = 100,
        skip: bool = False
    ) -> LexerRule:
        """
        添加词法规则
        
        参数:
            token_type: Token类型
            pattern: 正则表达式模式
            priority: 优先级
            skip: 是否跳过
        
        返回:
            创建的规则对象
        """
        compiled = re.compile(pattern)
        rule = LexerRule(
            token_type=token_type,
            pattern=compiled,
            priority=priority,
            skip=skip
        )
        self._rules.append(rule)
        self._compile_rules()
        return rule
    
    def _compile_rules(self) -> None:
        """编译合并所有规则为单一正则表达式"""
        if not self._rules:
            return
        
        # 按优先级排序
        sorted_rules = sorted(self._rules, key=lambda r: r.priority)
        
        # 构建命名捕获组
        branches = []
        for rule in sorted_rules:
            # 使用命名字段来标识规则
            pattern_str = rule.pattern.pattern
            branches.append(f"(?P<_{id(rule)}>{pattern_str})")
        
        combined = "|".join(branches)
        
        flags = re.MULTILINE
        if not self._case_sensitive:
            flags |= re.IGNORECASE
        
        self._combined_pattern = re.compile(combined, flags=flags)
        
        # 构建id到规则的映射
        self._rule_map: Dict[int, LexerRule] = {id(r): r for r in sorted_rules}
    
    def tokenize(self, source: str) -> List[Token]:
        """
        对输入进行词法分析
        
        参数:
            source: 源代码字符串
        
        返回:
            Token列表
        
        异常:
            LexerError: 当遇到无法识别的字符
        """
        tokens: List[Token] = []
        position = 0
        length = len(source)
        
        # 重置位置
        self._line = 1
        self._column = 1
        
        self._logger.debug(f"Tokenizing source of length {length}")
        
        while position < length:
            # 尝试匹配
            match = self._try_match(source, position)
            
            if match is not None:
                token = self._create_token(match, position)
                
                # 更新位置
                new_position = match.end()
                self._update_position(source, position, new_position)
                position = new_position
                
                # 决定是否添加Token
                if not self._should_skip(token):
                    tokens.append(token)
                    self._logger.debug(f"Token: {token}")
            else:
                # 无匹配: 生成ERROR token
                char = source[position]
                token = Token(
                    type=TokenType.ERROR,
                    value=char,
                    line=self._line,
                    column=self._column
                )
                tokens.append(token)
                
                self._logger.error(
                    f"LexerError at line {self._line}, column {self._column}: "
                    f"unexpected character '{char}'"
                )
                
                raise LexerError(
                    f"Unexpected character: '{char}'",
                    line=self._line,
                    column=self._column,
                    char=char
                )
        
        # 添加EOF token
        tokens.append(Token(
            type=TokenType.EOF,
            value="",
            line=self._line,
            column=self._column
        ))
        
        self._logger.info(f"Tokenization complete: {len(tokens)} tokens")
        return tokens
    
    def _try_match(self, source: str, position: int) -> Optional[Match[str]]:
        """
        尝试在当前位置匹配所有规则
        
        使用最大Munch原则选择最长匹配。
        
        参数:
            source: 源代码
            position: 当前位置
        
        返回:
            最佳匹配结果
        """
        if self._combined_pattern is None:
            return None
        
        best_match: Optional[Match[str]] = None
        best_rule: Optional[LexerRule] = None
        
        for rule in self._rules:
            pattern = rule.pattern
            match = pattern.match(source, position)
            
            if match is not None:
                # 优先选择更长的匹配
                if best_match is None or len(match.group()) > len(best_match.group()):
                    best_match = match
                    best_rule = rule
                # 相同长度按优先级
                elif len(match.group()) == len(best_match.group()):
                    if best_rule is None or rule.priority < best_rule.priority:
                        best_match = match
                        best_rule = rule
        
        return best_match
    
    def _create_token(self, match: Match[str], position: int) -> Token:
        """
        从匹配结果创建Token
        
        参数:
            match: 正则匹配结果
            position: 匹配起始位置
        
        返回:
            Token对象
        """
        # 确定匹配的规则
        matched_rule = None
        for rule_id, rule in self._rule_map.items():
            group_name = f"_{rule_id}"
            if match.group(group_name) is not None:
                matched_rule = rule
                break
        
        # 备用方法: 遍历规则检查
        if matched_rule is None:
            for rule in self._rules:
                if rule.pattern.match(match.string, position) == match:
                    matched_rule = rule
                    break
        
        token_type = matched_rule.token_type if matched_rule else TokenType.ERROR
        value = match.group()
        
        return Token(
            type=token_type,
            value=value,
            line=self._line,
            column=self._column,
            end_line=self._line,
            end_column=self._column + len(value) - 1
        )
    
    def _should_skip(self, token: Token) -> bool:
        """
        检查Token是否应该被跳过
        
        参数:
            token: 要检查的Token
        
        返回:
            如果应跳过返回True
        """
        # 检查空白
        if token.type == TokenType.WHITESPACE and self._skip_whitespace:
            return True
        
        # 检查换行
        if token.type == TokenType.NEWLINE and self._skip_newlines:
            return True
        
        # 检查规则标记
        for rule in self._rules:
            if rule.token_type == token.type and rule.skip:
                return True
        
        return False
    
    def _update_position(self, source: str, start: int, end: int) -> None:
        """
        更新行列位置
        
        参数:
            source: 源代码
            start: 起始位置
            end: 结束位置
        """
        for i in range(start, end):
            if source[i] == '\n':
                self._line += 1
                self._column = 1
            else:
                self._column += 1
    
    def tokenize_lines(self, source: str) -> Dict[int, List[Token]]:
        """
        按行分组Token
        
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
    
    def tokenize_incremental(self, source: str) -> Iterator[Token]:
        """
        增量式词法分析 (生成器)
        
        适用于处理超大文件。
        
        参数:
            source: 源代码
        
        生成:
            Token对象
        """
        position = 0
        length = len(source)
        
        self._line = 1
        self._column = 1
        
        while position < length:
            match = self._try_match(source, position)
            
            if match is not None:
                token = self._create_token(match, position)
                new_position = match.end()
                self._update_position(source, position, new_position)
                position = new_position
                
                if not self._should_skip(token):
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
                self._update_position(source, position - 1, position)
        
        yield Token(type=TokenType.EOF, value="", line=self._line, column=self._column)
