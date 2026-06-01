"""
================================================================================
PaleoAST State Machine Framework - Base Classes
================================================================================

本模块提供有限状态机的核心基类和接口定义。

数学定义:
--------
状态机是一个五元组 M = (Q, Σ, δ, q0, F)，其中：

1. Q = {q0, q1, q2, ..., qn} 是有限非空状态集合
2. Σ = {a1, a2, ..., am} 是有限输入字母表
3. δ: Q × Σ → Q 是确定性转移函数
4. q0 ∈ Q 是初始状态
5. F ⊆ Q 是接受状态集合

转移函数可扩展为 δ*: Q × Σ* → Q，支持字符串输入

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 定义类型变量
T = TypeVar("T")
S = TypeVar("S", bound="State")


class TransitionType(Enum):
    """
    转移类型枚举

    状态机中的转移类型定义:
    - EPSILON: ε转移，空转移不需要输入即可触发
    - DETERMINISTIC: 确定性转移，每个状态-输入对最多一个转移
    - NON_DETERMINISTIC: 非确定性转移，允许一个状态-输入对有多个转移
    """

    EPSILON = auto()
    DETERMINISTIC = auto()
    NON_DETERMINISTIC = auto()


@dataclass(frozen=True, slots=True)
class State:
    """
    状态类 (State)

    表示状态机中的一个状态节点。

    属性:
        state_id: 状态的唯一标识符 (UUID)
        name: 状态的友好名称
        is_accepting: 是否为接受状态
        is_initial: 是否为初始状态
        metadata: 状态附带的元数据字典

    数学表示:
        q ∈ Q，其中Q是状态集合

    示例:
        >>> state = State(name="q0", is_initial=True)
        >>> state = State(name="q_accept", is_accepting=True)
    """

    name: str
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_accepting: bool = False
    is_initial: bool = False
    metadata: frozenset[tuple[str, Any]] = field(default_factory=frozenset)

    def __post_init__(self):
        """验证状态属性的合法性"""
        if not self.name:
            raise ValueError("State name cannot be empty")
        if self.is_initial and self.is_accepting:
            logger.warning(f"State '{self.name}' is both initial and accepting. This is valid but unusual.")

    def __hash__(self) -> int:
        """使State可哈希以便用于集合和字典"""
        return hash(self.state_id)

    def __eq__(self, other: object) -> bool:
        """状态相等性基于state_id而非name"""
        if not isinstance(other, State):
            return NotImplemented
        return self.state_id == other.state_id

    def add_metadata(self, key: str, value: Any) -> State:
        """
        添加元数据并返回新的State实例

        由于State是不可变的(this is a frozen dataclass)，
        元数据修改会创建新实例。

        参数:
            key: 元数据键名
            value: 元数据值

        返回:
            新的State实例，包含添加的元数据
        """
        new_metadata = set(self.metadata)
        new_metadata.add((key, value))
        return State(
            state_id=self.state_id,
            name=self.name,
            is_accepting=self.is_accepting,
            is_initial=self.is_initial,
            metadata=frozenset(new_metadata),
        )


@dataclass
class Transition:
    """
    转移类 (Transition)

    表示状态机中两个状态之间的转移。

    数学表示:
        t = (q_s, σ, q_e) ∈ δ
        其中 q_s ∈ Q 是起始状态
              σ ∈ Σ 是输入符号
              q_e ∈ Q 是终止状态

    属性:
        source: 转移的起始状态
        symbol: 触发转移的输入符号 (None表示ε转移)
        target: 转移的目标状态
        transition_type: 转移的类型
        weight: 转移的权重 (用于加权自动机)
        condition: 转移的触发条件函数

    示例:
        >>> q0 = State(name="q0", is_initial=True)
        >>> q1 = State(name="q1", is_accepting=True)
        >>> transition = Transition(q0, 'a', q1)
    """

    source: State
    symbol: str | None
    target: State
    transition_type: TransitionType = TransitionType.DETERMINISTIC
    weight: float = 1.0
    condition: Callable[[str], bool] | None = field(default=None, repr=False)

    def __post_init__(self):
        """验证转移的有效性"""
        if self.weight < 0:
            raise ValueError(f"Transition weight must be non-negative, got {self.weight}")
        if self.condition is not None and not callable(self.condition):
            raise TypeError("Transition condition must be callable or None")

    def is_epsilon(self) -> bool:
        """判断是否为ε转移"""
        return self.symbol is None or self.transition_type == TransitionType.EPSILON

    def matches(self, input_symbol: str) -> bool:
        """
        检查输入符号是否匹配此转移

        参数:
            input_symbol: 输入符号

        返回:
            如果符号匹配则返回True，否则返回False
        """
        if self.is_epsilon():
            return True
        if self.condition is not None:
            return self.condition(input_symbol)
        return self.symbol == input_symbol

    def __hash__(self) -> int:
        """使Transition可哈希"""
        return hash((self.source.state_id, self.symbol, self.target.state_id))

    def __eq__(self, other: object) -> bool:
        """转移相等性基于source、symbol和target"""
        if not isinstance(other, Transition):
            return NotImplemented
        return self.source == other.source and self.symbol == other.symbol and self.target == other.target


class StateMachine(ABC):
    """
    状态机抽象基类 (Abstract State Machine)

    提供有限状态机的通用接口和实现框架。

    数学定义:
        M = (Q, Σ, δ, q0, F)

    核心方法:
        - add_state(): 添加状态到状态机
        - add_transition(): 添加转移
        - get_transitions(): 获取从一个状态出发的所有转移
        - transition(): 执行单步转移
        - process(): 处理输入字符串
        - accepts(): 检查是否接受输入

    子类必须实现:
        - _validate_input(): 验证输入符号
        - _on_accept(): 接受输入时的回调
        - _on_reject(): 拒绝输入时的回调

    示例:
        >>> sm = StateMachine()
        >>> q0 = sm.add_state("q0", is_initial=True)
        >>> q1 = sm.add_state("q1", is_accepting=True)
        >>> sm.add_transition(q0, 'a', q1)
        >>> sm.accepts("a")  # True
    """

    def __init__(self, name: str = "StateMachine", allow_epsilon: bool = True, allow_non_determinism: bool = True):
        """
        初始化状态机

        参数:
            name: 状态机名称
            allow_epsilon: 是否允许ε转移
            allow_non_determinism: 是否允许非确定性转移
        """
        self._name = name
        self._allow_epsilon = allow_epsilon
        self._allow_non_determinism = allow_non_determinism

        # 状态存储: state_id -> State
        self._states: dict[str, State] = {}

        # 转移存储: state_id -> Set[Transition]
        self._transitions: dict[str, set[Transition]] = {}

        # 初始状态引用
        self._initial_state: State | None = None

        # 接受状态集合
        self._accepting_states: set[State] = set()

        # 配置日志
        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._logger.debug(f"Initialized StateMachine '{name}'")

    # ===========================================================================
    # 状态管理方法
    # ===========================================================================

    @property
    def name(self) -> str:
        """获取状态机名称"""
        return self._name

    @property
    def states(self) -> frozenset[State]:
        """获取所有状态的不可变集合"""
        return frozenset(self._states.values())

    @property
    def initial_state(self) -> State | None:
        """获取初始状态"""
        return self._initial_state

    @property
    def accepting_states(self) -> frozenset[State]:
        """获取所有接受状态的集合"""
        return frozenset(self._accepting_states)

    def add_state(self, name: str, is_initial: bool = False, is_accepting: bool = False, **metadata: Any) -> State:
        """
        添加一个新状态到状态机

        数学表示:
            q_new ∉ Q → Q' = Q ∪ {q_new}

        参数:
            name: 状态的唯一名称
            is_initial: 是否为初始状态
            is_accepting: 是否为接受状态
            **metadata: 额外的元数据键值对

        返回:
            创建的State对象

        异常:
            ValueError: 如果状态名已存在，或有多个初始状态
        """
        # 检查名称唯一性
        existing = self._find_state_by_name(name)
        if existing is not None:
            raise ValueError(f"State with name '{name}' already exists")

        # 检查初始状态唯一性
        if is_initial and self._initial_state is not None:
            raise ValueError(
                f"Cannot add initial state '{name}': initial state '{self._initial_state.name}' already exists"
            )

        # 创建新状态
        state = State(name=name, is_initial=is_initial, is_accepting=is_accepting, metadata=frozenset(metadata.items()))

        # 添加到状态集合
        self._states[state.state_id] = state

        # 初始化转移集合
        self._transitions[state.state_id] = set()

        # 如果是初始状态
        if is_initial:
            self._initial_state = state

        # 如果是接受状态
        if is_accepting:
            self._accepting_states.add(state)

        self._logger.debug(f"Added state '{name}' (id={state.state_id[:8]}...)")
        return state

    def remove_state(self, state: State) -> bool:
        """
        从状态机中移除一个状态

        数学表示:
            q ∈ Q, T(q) = {t | t.source = q or t.target = q}
            Q' = Q - {q}
            δ' = δ - T(q)

        参数:
            state: 要移除的状态

        返回:
            如果成功移除返回True，否则返回False
        """
        if state.state_id not in self._states:
            self._logger.warning(f"State '{state.name}' not found")
            return False

        # 移除所有涉及此状态的转移
        for sid in list(self._transitions.keys()):
            self._transitions[sid] = {t for t in self._transitions[sid] if t.source != state and t.target != state}

        # 从集合中移除
        del self._states[state.state_id]
        del self._transitions[state.state_id]
        self._accepting_states.discard(state)

        if self._initial_state == state:
            self._initial_state = None

        self._logger.debug(f"Removed state '{state.name}'")
        return True

    def get_state(self, state_id: str) -> State | None:
        """通过ID获取状态"""
        return self._states.get(state_id)

    def _find_state_by_name(self, name: str) -> State | None:
        """根据名称查找状态"""
        for state in self._states.values():
            if state.name == name:
                return state
        return None

    # ===========================================================================
    # 转移管理方法
    # ===========================================================================

    def add_transition(
        self,
        source: State,
        symbol: str | None,
        target: State,
        transition_type: TransitionType = TransitionType.DETERMINISTIC,
        weight: float = 1.0,
        condition: Callable[[str], bool] | None = None,
    ) -> Transition:
        """
        添加一个转移

        数学表示:
            q_s, q_e ∈ Q, σ ∈ Σ ∪ {ε}
            δ'(q_s, σ) = δ(q_s, σ) ∪ {q_e}

        参数:
            source: 起始状态
            symbol: 输入符号 (None表示ε转移)
            target: 目标状态
            transition_type: 转移类型
            weight: 转移权重
            condition: 条件函数

        返回:
            创建的Transition对象

        异常:
            ValueError: 如果状态不在状态机中
        """
        # 验证状态存在
        if source.state_id not in self._states:
            raise ValueError(f"Source state '{source.name}' not in state machine")
        if target.state_id not in self._states:
            raise ValueError(f"Target state '{target.name}' not in state machine")

        # 验证ε转移
        if symbol is None and not self._allow_epsilon:
            raise ValueError("Epsilon transitions not allowed")

        # 创建转移
        transition = Transition(
            source=source,
            symbol=symbol,
            target=target,
            transition_type=transition_type,
            weight=weight,
            condition=condition,
        )

        # 检查确定性冲突
        existing = self._get_matching_transition(source, symbol)
        if existing is not None and not self._allow_non_determinism:
            raise ValueError(
                f"Deterministic conflict: multiple transitions from '{source.name}' with symbol '{symbol}'"
            )

        # 添加转移
        self._transitions[source.state_id].add(transition)

        self._logger.debug(f"Added transition: {source.name} --[{symbol}]--> {target.name}")
        return transition

    def remove_transition(self, transition: Transition) -> bool:
        """
        移除一个转移

        参数:
            transition: 要移除的转移

        返回:
            如果成功移除返回True
        """
        source_transitions = self._transitions.get(transition.source.state_id)
        if source_transitions is None:
            return False

        try:
            source_transitions.remove(transition)
            return True
        except KeyError:
            return False

    def get_transitions(self, from_state: State | None = None) -> set[Transition]:
        """
        获取转移集合

        参数:
            from_state: 如果指定，返回从该状态出发的转移；否则返回所有转移

        返回:
            匹配的转移集合
        """
        if from_state is None:
            result = set()
            for transitions in self._transitions.values():
                result.update(transitions)
            return result

        return self._transitions.get(from_state.state_id, set()).copy()

    def _get_matching_transition(self, source: State, symbol: str | None) -> Transition | None:
        """获取匹配的转移"""
        for trans in self._transitions.get(source.state_id, set()):
            if trans.symbol == symbol:
                return trans
        return None

    # ===========================================================================
    # 核心处理方法
    # ===========================================================================

    @abstractmethod
    def _validate_input(self, symbol: str) -> bool:
        """
        验证输入符号的有效性

        子类必须实现此方法以定义合法的输入符号集合Σ。

        参数:
            symbol: 输入符号

        返回:
            如果符号合法返回True
        """
        pass

    def transition(self, current_states: set[State], symbol: str) -> set[State]:
        """
        执行一步转移

        数学表示:
            δ*(S, σ) = ∪_{q∈S} δ(q, σ)
            其中 S ⊆ Q 是当前状态集合

        参数:
            current_states: 当前状态集合
            symbol: 输入符号

        返回:
            转移后的状态集合
        """
        if not self._validate_input(symbol):
            self._logger.warning(f"Invalid symbol: '{symbol}'")
            return set()

        next_states: set[State] = set()

        for state in current_states:
            for trans in self._transitions.get(state.state_id, set()):
                if trans.matches(symbol):
                    next_states.add(trans.target)

        # 处理ε闭包
        if self._allow_epsilon:
            next_states = self._epsilon_closure(next_states)

        return next_states

    def _epsilon_closure(self, states: set[State]) -> set[State]:
        """
        计算ε闭包

        数学表示:
            ε-Closure(S) = S ∪ {p | ∃q∈S, ∃t∈δ(q, ε) : t.target = p}

        参数:
            states: 输入状态集合

        返回:
            ε闭包状态集合
        """
        closure = set(states)
        stack = list(states)

        while stack:
            current = stack.pop()
            for trans in self._transitions.get(current.state_id, set()):
                if trans.is_epsilon() and trans.target not in closure:
                    closure.add(trans.target)
                    stack.append(trans.target)

        return closure

    def process(self, input_string: str) -> bool:
        """
        处理输入字符串

        依次将输入字符串的每个字符送入状态机，
        跟踪状态转移过程。

        数学表示:
            设输入为 w = a1a2...an
            δ*(q0, w) = δ(δ(...δ(q0, a1)...), an)

        参数:
            input_string: 要处理的输入字符串

        返回:
            如果处理后处于接受状态返回True
        """
        if self._initial_state is None:
            raise RuntimeError("No initial state defined")

        self._logger.debug(f"Processing input: '{input_string}'")

        current_states = self._epsilon_closure({self._initial_state})
        self._logger.debug(f"Initial ε-closure: {[s.name for s in current_states]}")

        for i, char in enumerate(input_string):
            if not current_states:  # 死状态
                self._logger.debug(f"Dead state at position {i}")
                return self._on_reject(input_string, i)

            current_states = self.transition(current_states, char)
            self._logger.debug(f"After '{char}' at pos {i}: {[s.name for s in current_states]}")

        # 检查是否到达接受状态
        accepting = bool(current_states & self._accepting_states)

        if accepting:
            return self._on_accept(input_string)
        else:
            return self._on_reject(input_string, len(input_string))

    def accepts(self, input_string: str) -> bool:
        """
        检查状态机是否接受输入字符串

        参数:
            input_string: 输入字符串

        返回:
            如果接受返回True
        """
        return self.process(input_string)

    def accepts_all(self, input_strings: list[str]) -> list[tuple[str, bool]]:
        """
        批量检查多个字符串的接受情况

        参数:
            input_strings: 输入字符串列表

        返回:
            (字符串, 是否接受) 元组列表
        """
        return [(s, self.accepts(s)) for s in input_strings]

    # ===========================================================================
    # 回调方法 (子类可覆盖)
    # ===========================================================================

    def _on_accept(self, input_string: str) -> bool:
        """
        输入被接受时的回调

        可被子类覆盖以执行额外操作

        参数:
            input_string: 被接受的输入

        返回:
            始终返回True
        """
        self._logger.info(f"Accepted: '{input_string}'")
        return True

    def _on_reject(self, input_string: str, position: int) -> bool:
        """
        输入被拒绝时的回调

        可被子类覆盖以执行额外操作

        参数:
            input_string: 被拒绝的输入
            position: 拒绝发生的位置

        返回:
            始终返回False
        """
        self._logger.info(f"Rejected: '{input_string}' at position {position}")
        return False

    # ===========================================================================
    # 分析和工具方法
    # ===========================================================================

    def is_deterministic(self) -> bool:
        """
        检查状态机是否是确定性的

        确定性条件:
            ∀q ∈ Q, ∀σ ∈ Σ: |δ(q, σ)| ≤ 1

        返回:
            如果是DFA返回True
        """
        for transitions in self._transitions.values():
            symbol_counts: dict[str | None, int] = {}
            for trans in transitions:
                symbol = trans.symbol
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                if symbol_counts[symbol] > 1:
                    return False
        return True

    def is_complete(self) -> bool:
        """
        检查状态机是否完全定义

        完全性条件:
            ∀q ∈ Q, ∀σ ∈ Σ: δ(q, σ) ≠ ∅

        返回:
            如果是完全的返回True
        """
        if not self._states:
            return True

        # 获取所有可能的输入符号
        symbols: set[str] = set()
        for transitions in self._transitions.values():
            for trans in transitions:
                if trans.symbol is not None:
                    symbols.add(trans.symbol)

        # 检查每个状态对每个符号都有转移
        for state in self._states.values():
            for symbol in symbols:
                has_transition = any(t.symbol == symbol for t in self._transitions[state.state_id])
                if not has_transition:
                    return False

        return True

    def get_reachable_states(self) -> set[State]:
        """
        获取从初始状态可达的所有状态

        使用BFS遍历

        返回:
            可达状态集合
        """
        if self._initial_state is None:
            return set()

        reachable = set()
        queue = [self._initial_state]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue

            reachable.add(current)

            for trans in self._transitions.get(current.state_id, set()):
                if trans.target not in reachable:
                    queue.append(trans.target)

        return reachable

    def get_live_states(self) -> set[State]:
        """
        获取所有活状态 (能到达接受状态的状态)

        使用反向图BFS遍历

        返回:
            活状态集合
        """
        if not self._accepting_states:
            return set()

        # 构建反向图
        reverse_graph: dict[str, set[str]] = {sid: set() for sid in self._states}

        for state_id, transitions in self._transitions.items():
            for trans in transitions:
                reverse_graph[trans.target.state_id].add(state_id)

        # 从接受状态开始反向遍历
        live = set()
        queue = list(self._accepting_states)

        while queue:
            current = queue.pop(0)
            if current in live:
                continue

            live.add(current)

            for prev_id in reverse_graph.get(current.state_id, set()):
                prev_state = self._states[prev_id]
                if prev_state not in live:
                    queue.append(prev_state)

        return live

    def minimize(self) -> StateMachine:
        """
        最小化状态机 (Hopcroft算法)

        数学原理:
            1. 移除不可达状态
            2. 构建初始划分 Π = {F, Q - F}
            3. 迭代细化划分直到稳定

        返回:
            最小化的新状态机
        """
        # TODO: 实现Hopcroft最小化算法
        raise NotImplementedError("Minimization not yet implemented")

    def to_dot(self) -> str:
        """
        生成Graphviz DOT格式表示

        返回:
            DOT格式字符串
        """
        lines = [
            f'digraph "{self._name}" {{',
            "    rankdir=LR;",
            "    node [shape=circle];",
        ]

        # 添加状态节点
        for state in self._states.values():
            shape = "doublecircle" if state.is_accepting else "circle"
            label = state.name
            if state.is_initial:
                lines.append(
                    f'    "{state.state_id}" [label="{label}", shape={shape}, style=filled, fillcolor=lightgrey];'
                )
            else:
                lines.append(f'    "{state.state_id}" [label="{label}", shape={shape}];')

        # 添加转移边
        for transitions in self._transitions.values():
            for trans in transitions:
                symbol = trans.symbol if trans.symbol else "ε"
                lines.append(
                    f'    "{trans.source.state_id}" -> "{trans.target.state_id}" '
                    f'[label="{symbol}", weight={trans.weight}];'
                )

        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """状态机的字符串表示"""
        n_states = len(self._states)
        n_transitions = sum(len(t) for t in self._transitions.values())
        return (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"states={n_states}, "
            f"transitions={n_transitions}, "
            f"deterministic={self.is_deterministic()})"
        )
