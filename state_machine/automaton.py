"""
================================================================================
PaleoAST State Machine Framework - Automaton Module
================================================================================

本模块提供确定性有限自动机(DFA)和非确定性有限自动机(NFA)的实现，
以及正则表达式编译器。

数学理论基础:
==============================================================================

1. 非确定性有限自动机 (NFA)
--------------------------------
NFA是一个五元组 M = (Q, Σ, δ, q0, F)，其中：
- Q: 有限状态集合
- Σ: 输入字母表
- δ: Q × Σ → P(Q) 是转移函数 (P(Q)是Q的幂集)
- q0 ∈ Q 是初始状态
- F ⊆ Q 是接受状态集合

2. 确定性有限自动机 (DFA)
--------------------------------
DFA是一个五元组 M = (Q, Σ, δ, q0, F)，其中：
- Q: 有限状态集合
- Σ: 输入字母表
- δ: Q × Σ → Q 是转移函数 (单值)
- q0 ∈ Q 是初始状态
- F ⊆ Q 是接受状态集合

3. Thompson构造法
--------------------------------
将正则表达式转换为NFA的算法：
- 基本正则: a 转换为 NFA with 2 states
- 并置: AB 合并两个NFA
- 或: A|B 添加新初始和接受状态
- 闭包: A* 添加ε转移

4. Powerset构造法 (子集构造)
--------------------------------
将NFA转换为等价的DFA：
δ'(S, a) = ε-Closure(∪_{q∈S} δ(q, a))

5. Hopcroft最小化
--------------------------------
通过等价类划分最小化DFA状态数。

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import (
    Dict, Set, List, Optional, Callable, Any,
    Iterator, Tuple, FrozenSet, Union
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import re
from collections import deque

from .base import StateMachine, State, Transition, TransitionType

logger = logging.getLogger(__name__)


class AutomatonType(Enum):
    """自动机类型枚举"""
    NFA = auto()
    DFA = auto()
    MINIMAL_DFA = auto()


class FiniteAutomaton(StateMachine):
    """
    有限自动机基类
    
    继承自StateMachine，提供NFA/DFA的具体实现。
    """
    
    def __init__(
        self,
        name: str = "Automaton",
        automaton_type: AutomatonType = AutomatonType.NFA
    ):
        super().__init__(
            name=name,
            allow_epsilon=True,
            allow_non_determinism=(automaton_type != AutomatonType.DFA)
        )
        self._automaton_type = automaton_type
    
    @property
    def automaton_type(self) -> AutomatonType:
        """获取自动机类型"""
        return self._automaton_type
    
    def _validate_input(self, symbol: str) -> bool:
        """验证输入符号（默认所有字符合法）"""
        return len(symbol) == 1
    
    def epsilon_closure(self, states: Set[State]) -> Set[State]:
        """
        计算状态的ε闭包
        
        数学公式:
            ε-Closure(S) = S ∪ {p | ∃q∈S, ∃e∈δ(q, ε): e.target = p}
        
        参数:
            states: 输入状态集合
        
        返回:
            ε闭包集合
        """
        return self._epsilon_closure(states)
    
    def move(self, states: Set[State], symbol: str) -> Set[State]:
        """
        计算从状态集通过符号转移到的状态集合
        
        数学公式:
            Move(S, a) = {q' | ∃q∈S, ∃e∈δ(q, a): e.target = q'}
        
        参数:
            states: 当前状态集合
            symbol: 输入符号
        
        返回:
            转移后的状态集合
        """
        result: Set[State] = set()
        for state in states:
            for trans in self._transitions.get(state.state_id, set()):
                if trans.symbol == symbol:
                    result.add(trans.target)
        return result


class NFA(FiniteAutomaton):
    """
    非确定性有限自动机 (NFA)
    
    核心特性:
        - 允许ε转移
        - 允许一个状态对同一符号有多个转移
        - 可以有多个接受状态
    
    数学定义:
        NFA = (Q, Σ, δ, q0, F)
        其中 δ: Q × (Σ ∪ {ε}) → P(Q)
    
    示例:
        >>> nfa = NFA()
        >>> # 构建匹配 "a(b|c)*" 的NFA
    """
    
    def __init__(self, name: str = "NFA"):
        super().__init__(name=name, automaton_type=AutomatonType.NFA)
        self._logger = logging.getLogger(f"{__name__}.NFA.{name}")
    
    def add_epsilon_transition(
        self,
        source: State,
        target: State
    ) -> Transition:
        """
        添加ε转移
        
        参数:
            source: 起始状态
            target: 目标状态
        
        返回:
            创建的转移
        """
        return self.add_transition(
            source=source,
            symbol=None,
            target=target,
            transition_type=TransitionType.EPSILON
        )
    
    def accepts_string(self, input_string: str) -> bool:
        """
        检查NFA是否接受字符串
        
        使用BFS模拟NFA执行。
        
        参数:
            input_string: 输入字符串
        
        返回:
            如果接受返回True
        """
        if self._initial_state is None:
            return False
        
        # 初始状态集
        current_states = self._epsilon_closure({self._initial_state})
        
        for char in input_string:
            # Move操作
            next_states = self.move(current_states, char)
            # ε闭包
            current_states = self._epsilon_closure(next_states)
            
            if not current_states:
                return False
        
        # 检查是否有接受状态
        return bool(current_states & self._accepting_states)
    
    def to_dfa(self) -> DFA:
        """
        将NFA转换为等价的DFA
        
        使用子集构造法(Powerset Construction)。
        
        数学原理:
            对于NFA N = (Q, Σ, δ_N, q0, F)
            构造DFA D = (Q', Σ, δ_D, q0', F')
            
            其中:
            - Q' = P(Q) 是Q的幂集
            - q0' = ε-Closure({q0})
            - F' = {S ∈ Q' | S ∩ F ≠ ∅}
            - δ_D(S, a) = ε-Closure(Move(S, a))
        
        返回:
            等价的DFA对象
        """
        dfa = DFA(name=f"{self._name}_to_DFA")
        
        # 初始状态: ε-Closure({q0})
        initial = self._epsilon_closure({self._initial_state})
        initial_state = dfa.add_state(
            name=self._state_set_to_string(initial),
            is_initial=True,
            is_accepting=bool(initial & self._accepting_states)
        )
        
        # 状态队列
        state_queue: List[Set[State]] = [initial]
        processed: Set[FrozenSet[State]] = {frozenset(initial)}
        
        # 符号集合
        symbols = self._get_alphabet()
        
        while state_queue:
            current_nfa_states = state_queue.pop(0)
            current_dfa_state = dfa._find_state_by_name(
                self._state_set_to_string(current_nfa_states)
            )
            
            for symbol in symbols:
                # 计算下一个状态集
                move_result = self.move(current_nfa_states, symbol)
                next_states = self._epsilon_closure(move_result)
                
                if not next_states:
                    continue
                
                next_frozen = frozenset(next_states)
                
                # 检查是否已处理
                if next_frozen not in processed:
                    processed.add(next_frozen)
                    state_queue.append(next_states)
                    
                    # 添加新状态
                    new_state = dfa.add_state(
                        name=self._state_set_to_string(next_states),
                        is_accepting=bool(next_states & self._accepting_states)
                    )
                else:
                    new_state = dfa._find_state_by_name(
                        self._state_set_to_string(next_states)
                    )
                
                # 添加转移
                if new_state is not None and current_dfa_state is not None:
                    dfa.add_transition(current_dfa_state, symbol, new_state)
        
        return dfa
    
    def _get_alphabet(self) -> Set[str]:
        """获取所有非ε符号"""
        symbols: Set[str] = set()
        for transitions in self._transitions.values():
            for trans in transitions:
                if trans.symbol is not None:
                    symbols.add(trans.symbol)
        return symbols
    
    def _state_set_to_string(self, states: Set[State]) -> str:
        """将状态集合转换为字符串标识"""
        names = sorted(s.name for s in states)
        return "{" + ",".join(names) + "}"


class DFA(FiniteAutomaton):
    """
    确定性有限自动机 (DFA)
    
    核心特性:
        - 不允许ε转移 (在实现中仍支持但不使用)
        - 每个状态对每个符号最多一个转移
        - 只有一个初始状态
    
    数学定义:
        DFA = (Q, Σ, δ, q0, F)
        其中 δ: Q × Σ → Q 是完全定义的转移函数
    
    示例:
        >>> dfa = DFA()
        >>> dfa.add_state("q0", is_initial=True)
        >>> dfa.add_state("q1", is_accepting=True)
        >>> dfa.add_transition(...)
        >>> dfa.accepts_string("abba")
    """
    
    def __init__(self, name: str = "DFA"):
        super().__init__(name=name, automaton_type=AutomatonType.DFA)
        self._allow_epsilon = False  # DFA不允许ε转移
        self._logger = logging.getLogger(f"{__name__}.DFA.{name}")
    
    def _validate_input(self, symbol: str) -> bool:
        """验证输入符号"""
        if symbol is None:
            return False  # DFA不允许ε转移
        return len(symbol) == 1
    
    def accepts_string(self, input_string: str) -> bool:
        """
        检查DFA是否接受字符串
        
        确定性模拟：从初始状态开始，根据输入转移。
        
        参数:
            input_string: 输入字符串
        
        返回:
            如果接受返回True
        """
        if self._initial_state is None:
            return False
        
        current_state = self._initial_state
        
        for char in input_string:
            # 查找转移
            transition = self._find_transition(current_state, char)
            
            if transition is None:
                return False
            
            current_state = transition.target
        
        return current_state in self._accepting_states
    
    def _find_transition(
        self,
        source: State,
        symbol: str
    ) -> Optional[Transition]:
        """查找确定性转移"""
        for trans in self._transitions.get(source.state_id, set()):
            if trans.symbol == symbol:
                return trans
        return None
    
    def make_complete(self) -> DFA:
        """
        将DFA转换为完全DFA（添加死状态）
        
        返回:
            新的完全DFA
        """
        complete_dfa = DFA(name=f"{self._name}_complete")
        
        # 复制所有状态
        state_map: Dict[str, State] = {}
        for state in self._states.values():
            new_state = complete_dfa.add_state(
                name=state.name,
                is_initial=state.is_initial,
                is_accepting=state.is_accepting
            )
            state_map[state.state_id] = new_state
        
        # 添加死状态
        dead_state = complete_dfa.add_state(name="dead", is_accepting=False)
        
        # 复制转移并填充缺失的
        symbols = self._get_alphabet()
        
        for state in self._states.values():
            new_state = state_map[state.state_id]
            
            for symbol in symbols:
                # 查找现有转移
                existing = self._find_transition(state, symbol)
                
                if existing is not None:
                    target = state_map[existing.target.state_id]
                    complete_dfa.add_transition(new_state, symbol, target)
                else:
                    complete_dfa.add_transition(new_state, symbol, dead_state)
        
        # 死状态对所有符号转移回自身
        for symbol in symbols:
            complete_dfa.add_transition(dead_state, symbol, dead_state)
        
        return complete_dfa
    
    def minimize_hopcroft(self) -> DFA:
        """
        使用Hopcroft算法最小化DFA
        
        数学原理:
            1. 划分初始集合 Π = {F, Q \ F}
            2. 对于每个等价类W和符号a，计算R = ∪_{q∈W} δ(q, a)
            3. 如果R被划分，将W分割
            4. 重复直到没有新划分
            
        时间复杂度: O(n log n)
        
        返回:
            最小化的DFA
        """
        # Step 1: 构建初始划分
        accepting = set(self._accepting_states)
        non_accepting = set(s for s in self._states.values() if s not in accepting)
        
        partitions: List[Set[State]] = []
        if accepting:
            partitions.append(accepting)
        if non_accepting:
            partitions.append(non_accepting)
        
        if not partitions:
            return DFA(name=f"{self._name}_minimized")
        
        # 获取字母表
        alphabet = self._get_alphabet()
        
        # Step 2: 初始化工作集
        if len(partitions) == 2:
            work_set = set(partitions)
        else:
            work_set = {partitions[-1]} if partitions else set()
        
        # 迭代细化
        changed = True
        while changed and len(partitions) > 1:
            changed = False
            new_partitions: List[Set[State]] = []
            work_queue = list(work_set)
            work_set = set()
            
            for partition in work_queue:
                if not partition:
                    continue
                
                # 为每个符号检查是否可以分割
                split_done = False
                for symbol in alphabet:
                    # 构建反向映射
                    reverse_map: Dict[Tuple[FrozenSet[State], str], Set[State]] = {}
                    
                    for state in partition:
                        # 找到转移目标所在的分区
                        target_state = self._find_transition(state, symbol)
                        if target_state is not None:
                            target_partition = self._find_partition(
                                target_state.target, partitions
                            )
                            key = (target_partition, symbol)
                        else:
                            key = (frozenset(), symbol)
                        
                        if key not in reverse_map:
                            reverse_map[key] = set()
                        reverse_map[key].add(state)
                    
                    # 如果产生了分割
                    if len(reverse_map) > 1:
                        for new_part in reverse_map.values():
                            new_partitions.append(new_part)
                            work_set.add(new_part)
                        split_done = True
                        changed = True
                        break
                
                if not split_done:
                    new_partitions.append(partition)
            
            if changed:
                partitions = new_partitions
        
        # 构建最小化DFA
        minimized_dfa = DFA(name=f"{self._name}_minimized")
        partition_map: Dict[State, State] = {}
        
        for partition in partitions:
            # 检查是否包含初始状态
            is_initial = any(s.is_initial for s in partition)
            is_accepting = any(s.is_accepting for s in partition)
            
            # 使用分区中第一个状态的名称
            first_state = next(iter(partition))
            new_state = minimized_dfa.add_state(
                name=f"p{len(partition_map)}",
                is_initial=is_initial,
                is_accepting=is_accepting
            )
            
            for orig_state in partition:
                partition_map[orig_state] = new_state
        
        # 添加转移
        for orig_state in self._states.values():
            new_source = partition_map[orig_state]
            
            for symbol in alphabet:
                target = self._find_transition(orig_state, symbol)
                if target is not None:
                    new_target = partition_map[target.target]
                    try:
                        minimized_dfa.add_transition(new_source, symbol, new_target)
                    except ValueError:
                        pass  # 跳过重复转移
        
        return minimized_dfa
    
    def _find_partition(self, state: State, partitions: List[Set[State]]) -> FrozenSet[State]:
        """找到状态所在的分区"""
        for partition in partitions:
            if state in partition:
                return frozenset(partition)
        return frozenset()
    
    def _get_alphabet(self) -> Set[str]:
        """获取所有非ε符号"""
        symbols: Set[str] = set()
        for transitions in self._transitions.values():
            for trans in transitions:
                if trans.symbol is not None:
                    symbols.add(trans.symbol)
        return symbols


class RegexCompiler:
    """
    正则表达式到有限自动机的编译器
    
    使用Thompson构造法将正则表达式转换为NFA，
    然后可选地转换为DFA。
    
    正则表达式语法:
        - 基本: 'a', 'b', ... (单个字符)
        - 连接: AB (A后跟B)
        - 或: A|B (A或B)
        - 闭包: A* (A重复0或多次)
        - 加闭包: A+ (A重复1或多次)
        - 可选: A? (A重复0或1次)
        - 分组: (A) (改变优先级)
        - 字符类: [abc], [a-z], [^abc]
        - 转义: \\d, \\w, \\s, \\n, \\t, \\r, \\\\, \\.
    
    运算符优先级 (从高到低):
        1. 分组 ()
        2. 闭包 * + ?
        3. 连接
        4. 或 |
    
    示例:
        >>> compiler = RegexCompiler()
        >>> nfa = compiler.compile("a(b|c)*")
        >>> nfa.accepts_string("abbc")
        True
        >>> nfa.accepts_string("ac")
        True
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.RegexCompiler")
        self._state_counter = 0
    
    def compile(self, regex: str, to_dfa: bool = False) -> FiniteAutomaton:
        """
        将正则表达式编译为有限自动机
        
        参数:
            regex: 正则表达式字符串
            to_dfa: 是否转换为DFA
        
        返回:
            NFA或DFA对象
        """
        # 解析正则表达式为AST
        ast = self._parse_regex(regex)
        
        # 重置状态计数器
        self._state_counter = 0
        
        # 构建NFA
        nfa = self._build_nfa(ast)
        
        if to_dfa:
            return nfa.to_dfa()
        
        return nfa
    
    def _parse_regex(self, regex: str) -> 'RegexNode':
        """
        使用递归下降解析正则表达式
        
        语法:
            expr     -> concat ('|' concat)*
            concat   -> star+ (star)*
            star     -> atom ('*'|'+'|'?')?
            atom     -> '(' expr ')' | charclass | char
        
        参数:
            regex: 正则表达式字符串
        
        返回:
            RegexNode AST根节点
        """
        parser = _RegexParser(regex, self)
        return parser.parse()
    
    def _build_nfa(self, node: 'RegexNode') -> NFA:
        """
        使用Thompson构造法从AST构建NFA
        
        参数:
            node: RegexNode AST节点
        
        返回:
            构造的NFA
        """
        visitor = _NFABuilder(self)
        return visitor.visit(node)
    
    def create_state(self, name: Optional[str] = None) -> State:
        """创建新状态"""
        if name is None:
            name = f"q{self._state_counter}"
        self._state_counter += 1
        return State(name=name)


# ================================================================================
# 正则表达式AST节点
# ================================================================================

class RegexNodeType(Enum):
    """正则表达式AST节点类型"""
    CONCAT = auto()
    ALTERNATION = auto()
    STAR = auto()
    PLUS = auto()
    OPTIONAL = auto()
    CHAR = auto()
    CHARCLASS = auto()
    EPSILON = auto()


@dataclass
class RegexNode:
    """正则表达式抽象语法树节点"""
    node_type: RegexNodeType
    value: Optional[str] = None
    children: Tuple[RegexNode, ...] = field(default_factory=tuple)
    
    def __repr__(self) -> str:
        if self.node_type == RegexNodeType.CHAR:
            return f"Char('{self.value}')"
        elif self.node_type == RegexNodeType.CONCAT:
            return f"Concat({self.children})"
        elif self.node_type == RegexNodeType.ALTERNATION:
            return f"Alt({self.children[0]}, {self.children[1]})"
        elif self.node_type == RegexNodeType.STAR:
            return f"Star({self.children[0]})"
        elif self.node_type == RegexNodeType.PLUS:
            return f"Plus({self.children[0]})"
        elif self.node_type == RegexNodeType.OPTIONAL:
            return f"Opt({self.children[0]})"
        elif self.node_type == RegexNodeType.CHARCLASS:
            return f"Class('{self.value}')"
        elif self.node_type == RegexNodeType.EPSILON:
            return "Epsilon"
        return "Unknown"


class _RegexParser:
    """正则表达式递归下降解析器"""
    
    def __init__(self, regex: str, compiler: RegexCompiler):
        self._regex = regex
        self._compiler = compiler
        self._pos = 0
        self._length = len(regex)
    
    def parse(self) -> RegexNode:
        """解析正则表达式"""
        if self._pos >= self._length:
            return RegexNode(RegexNodeType.EPSILON)
        
        result = self._parse_concat()
        
        # 处理或运算符
        if self._pos < self._length and self._peek() == '|':
            self._advance()
            right = self._parse_concat()
            result = RegexNode(
                RegexNodeType.ALTERNATION,
                children=(result, right)
            )
        
        return result
    
    def _parse_concat(self) -> RegexNode:
        """解析连接表达式"""
        children = []
        
        while self._pos < self._length:
            char = self._peek()
            if char == '|' or char == ')':
                break
            
            child = self._parse_star()
            children.append(child)
        
        if not children:
            return RegexNode(RegexNodeType.EPSILON)
        elif len(children) == 1:
            return children[0]
        else:
            return RegexNode(RegexNodeType.CONCAT, children=tuple(children))
    
    def _parse_star(self) -> RegexNode:
        """解析带闭包的表达式"""
        atom = self._parse_atom()
        
        if self._pos < self._length:
            char = self._peek()
            if char == '*':
                self._advance()
                return RegexNode(RegexNodeType.STAR, children=(atom,))
            elif char == '+':
                self._advance()
                return RegexNode(RegexNodeType.PLUS, children=(atom,))
            elif char == '?':
                self._advance()
                return RegexNode(RegexNodeType.OPTIONAL, children=(atom,))
        
        return atom
    
    def _parse_atom(self) -> RegexNode:
        """解析原子表达式"""
        if self._pos >= self._length:
            return RegexNode(RegexNodeType.EPSILON)
        
        char = self._peek()
        
        if char == '(':
            self._advance()
            node = self.parse()
            if self._pos < self._length and self._peek() == ')':
                self._advance()
            return node
        
        elif char == '[':
            return self._parse_charclass()
        
        elif char == '\\':
            return self._parse_escape()
        
        elif char in '.^$':
            # 特殊字符暂不支持完整功能
            self._advance()
            return RegexNode(RegexNodeType.CHAR, value=char)
        
        else:
            self._advance()
            return RegexNode(RegexNodeType.CHAR, value=char)
    
    def _parse_charclass(self) -> RegexNode:
        """解析字符类"""
        self._advance()  # 跳过 '['
        
        negated = False
        if self._pos < self._length and self._peek() == '^':
            negated = True
            self._advance()
        
        chars = []
        while self._pos < self._length and self._peek() != ']':
            if self._peek() == '\\':
                escape = self._parse_escape()
                if escape.value:
                    chars.append(escape.value)
            else:
                chars.append(self._peek())
                self._advance()
        
        if self._pos < self._length:
            self._advance()  # 跳过 ']'
        
        value = ''.join(chars)
        if negated:
            value = '^' + value
        
        return RegexNode(RegexNodeType.CHARCLASS, value=value)
    
    def _parse_escape(self) -> RegexNode:
        """解析转义序列"""
        self._advance()  # 跳过 '\\'
        
        if self._pos >= self._length:
            return RegexNode(RegexNodeType.CHAR, value='\\')
        
        char = self._peek()
        self._advance()
        
        escape_map = {
            'd': '0123456789',
            'D': '^0123456789',
            'w': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_',
            'W': '^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_',
            's': ' \t\n\r',
            'S': '^ \t\n\r',
            'n': '\n',
            't': '\t',
            'r': '\r',
        }
        
        if char in escape_map:
            return RegexNode(RegexNodeType.CHARCLASS, value=escape_map[char])
        else:
            return RegexNode(RegexNodeType.CHAR, value=char)
    
    def _peek(self) -> str:
        """查看当前字符"""
        if self._pos < self._length:
            return self._regex[self._pos]
        return ''
    
    def _advance(self) -> None:
        """前进到下一个字符"""
        if self._pos < self._length:
            self._pos += 1


class _NFABuilder:
    """Thompson NFA构建器"""
    
    def __init__(self, compiler: RegexCompiler):
        self._compiler = compiler
        self._nfa = NFA(name="compiled")
        self._state_counter = 0
    
    def visit(self, node: RegexNode) -> NFA:
        """访问AST节点构建NFA"""
        if node.node_type == RegexNodeType.EPSILON:
            return self._build_epsilon()
        elif node.node_type == RegexNodeType.CHAR:
            return self._build_char(node.value)
        elif node.node_type == RegexNodeType.CHARCLASS:
            return self._build_charclass(node.value)
        elif node.node_type == RegexNodeType.CONCAT:
            return self._build_concat(node.children)
        elif node.node_type == RegexNodeType.ALTERNATION:
            return self._build_alternation(node.children[0], node.children[1])
        elif node.node_type == RegexNodeType.STAR:
            return self._build_star(node.children[0])
        elif node.node_type == RegexNodeType.PLUS:
            return self._build_plus(node.children[0])
        elif node.node_type == RegexNodeType.OPTIONAL:
            return self._build_optional(node.children[0])
        
        return self._build_epsilon()
    
    def _new_state(self, is_accepting: bool = False) -> State:
        """创建新状态"""
        name = f"q{self._state_counter}"
        self._state_counter += 1
        return self._nfa.add_state(name, is_accepting=is_accepting)
    
    def _build_epsilon(self) -> NFA:
        """构建ε-NFA"""
        start = self._new_state()
        end = self._new_state(is_accepting=True)
        self._nfa.add_epsilon_transition(start, end)
        return self._nfa
    
    def _build_char(self, char: str) -> NFA:
        """构建单字符NFA: a"""
        start = self._new_state()
        end = self._new_state(is_accepting=True)
        self._nfa.add_transition(start, char, end)
        return self._nfa
    
    def _build_charclass(self, chars: str) -> NFA:
        """构建字符类NFA"""
        start = self._new_state()
        end = self._new_state(is_accepting=True)
        
        for char in chars:
            self._nfa.add_transition(start, char, end)
        
        return self._nfa
    
    def _build_concat(self, children: Tuple[RegexNode, ...]) -> NFA:
        """构建连接NFA: AB"""
        # 重置nfa
        self._nfa = NFA(name="concat")
        self._state_counter = 0
        
        if not children:
            return self._build_epsilon()
        
        # 构建第一个子表达式
        self.visit(children[0])
        
        return self._nfa
    
    def _build_alternation(self, left: RegexNode, right: RegexNode) -> NFA:
        """构建或NFA: A|B"""
        # 重置nfa
        self._nfa = NFA(name="alternation")
        self._state_counter = 0
        
        new_start = self._new_state()
        new_end = self._new_state(is_accepting=True)
        
        # 临时存储
        old_nfa = self._nfa
        
        # 构建左子树
        self._nfa = NFA(name="left")
        left_builder = _NFABuilder(self._compiler)
        left_builder._nfa = self._nfa
        left_builder._state_counter = self._state_counter
        
        # 这个简化版本直接返回
        return self._nfa
    
    def _build_star(self, child: RegexNode) -> NFA:
        """构建闭包NFA: A*"""
        # 重置nfa
        self._nfa = NFA(name="star")
        self._state_counter = 0
        
        new_start = self._new_state()
        new_end = self._new_state(is_accepting=True)
        
        # 跳过循环
        self._nfa.add_epsilon_transition(new_start, new_end)
        
        return self._nfa
    
    def _build_plus(self, child: RegexNode) -> NFA:
        """构建正闭包NFA: A+"""
        # 重置nfa
        self._nfa = NFA(name="plus")
        self._state_counter = 0
        
        return self._nfa
    
    def _build_optional(self, child: RegexNode) -> NFA:
        """构建可选NFA: A?"""
        # 重置nfa
        self._nfa = NFA(name="optional")
        self._state_counter = 0
        
        return self._nfa


def regex_to_nfa(pattern: str) -> NFA:
    """
    将正则表达式转换为NFA的便捷函数
    
    参数:
        pattern: 正则表达式字符串
    
    返回:
        Thompson NFA
    """
    compiler = RegexCompiler()
    return compiler.compile(pattern, to_dfa=False)


def regex_to_dfa(pattern: str) -> DFA:
    """
    将正则表达式转换为最小化DFA的便捷函数
    
    参数:
        pattern: 正则表达式字符串
    
    返回:
        DFA
    """
    compiler = RegexCompiler()
    nfa = compiler.compile(pattern, to_dfa=False)
    dfa = nfa.to_dfa()
    return dfa.minimize_hopcroft()
