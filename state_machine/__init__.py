"""
================================================================================
PaleoAST Phase 3 - State Machine Framework
================================================================================

本模块提供通用的有限状态机(FSM)框架，用于构建词法分析器和语法解析器。

有限状态机数学定义:
    FSM = (Q, Σ, δ, q0, F)

其中:
    Q: 状态集合 (States)
    Σ: 输入字母表 (Alphabet)
    δ: 转移函数 (Transition Function) δ: Q × Σ → Q
    q0: 初始状态 (Initial State) q0 ∈ Q
    F: 终止状态集合 (Final States) F ⊆ Q

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .automaton import DFA, NFA, FiniteAutomaton, RegexCompiler
from .base import State, StateMachine, Transition
from .tokenizer import LexerTokenizer, Token, TokenType

__all__ = [
    "DFA",
    "NFA",
    "FiniteAutomaton",
    "LexerTokenizer",
    "RegexCompiler",
    "State",
    "StateMachine",
    "Token",
    "TokenType",
    "Transition",
]

# ================================================================================
# 模块级配置常量
# ================================================================================

MAX_STATES = 65536  # 最大状态数量限制
MAX_TRANSITIONS = 1048576  # 最大转移数量限制
DEFAULT_BUFFER_SIZE = 8192  # 默认输入缓冲区大小
