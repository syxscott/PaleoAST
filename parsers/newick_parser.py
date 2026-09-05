"""
================================================================================
PaleoAST Parsers - Newick Tree Format Parser
================================================================================

本模块实现Newick格式树文件的解析器。

Newick格式是系统发育树的标准文本表示格式，
由Ohlert在1986年提出，以Newick Tree Restaurant命名。

Newick格式数学定义:
==============================================================================

Newick格式由以下递归文法定义:

    Tree        → Subtree ";"?
    Subtree     → Name ":" BranchLength Subtrees?
                 | Name Subtrees?
                 | BranchLength Subtrees?
    Subtrees    → "(" Subtree ("," Subtree)* ")"
    Name        → string
    BranchLength → ":" number

形式化表示:
    T ::= N | (T{,T})N:L | (T{,T})N | (T{,T}):L | N:L | :L

其中:
    - N: 节点名称 (可选)
    - L: 枝长 (可选)
    - {}: 可选的重复

示例:
    (A:0.1,B:0.2)C:0.3;

    表示树:
            C
           / \
          A   B
         0.1 0.2
        |
       0.3

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """
    树节点数据结构

    表示Newick树中的一个节点（内部节点或叶节点）。

    属性:
        name: 节点名称 (叶节点为分类单元名，内部节点可能无名)
        branch_length: 枝长 (从父节点到本节点的分支长度)
        parent: 父节点引用
        children: 子节点列表
        support: 支持率 (如Bootstrap值)
        metadata: 额外元数据

    数学表示:
        节点 v 具有:
        - 名称 n(v)
        - 枝长 l(v) ∈ ℝ≥0
        - 度 deg(v) = |children(v)|

    示例:
        >>> leaf = TreeNode(name="Homo_sapiens", branch_length=0.05)
        >>> internal = TreeNode(name="Mammalia")
    """

    name: str
    branch_length: float | None = None
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)
    support: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证节点属性"""
        if self.branch_length is not None and self.branch_length < 0:
            raise ValueError(f"Branch length must be non-negative, got {self.branch_length}")

    @property
    def is_leaf(self) -> bool:
        """检查是否为叶节点"""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """检查是否为根节点"""
        return self.parent is None

    @property
    def is_internal(self) -> bool:
        """检查是否为内部节点"""
        return len(self.children) > 0

    @property
    def depth(self) -> int:
        """计算从根节点到本节点的深度"""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    @property
    def leaf_count(self) -> int:
        """计算以本节点为根的子树中的叶节点数"""
        if self.is_leaf:
            return 1
        return sum(child.leaf_count for child in self.children)

    @property
    def total_length(self) -> float:
        """计算从根节点到叶节点的总枝长"""
        length = self.branch_length or 0.0
        if self.parent is not None:
            length += self.parent.total_length
        return length

    def get_leaves(self) -> list[TreeNode]:
        """
        获取所有叶节点

        返回:
            叶节点列表
        """
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_ancestors(self) -> list[TreeNode]:
        """
        获取所有祖先节点

        返回:
            从根到父节点的节点列表
        """
        ancestors = []
        node = self.parent
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors

    def prune_leaf(self, leaf_name: str) -> bool:
        """
        剪除指定名称的叶节点

        参数:
            leaf_name: 要剪除的叶节点名称

        返回:
            如果成功剪除返回True
        """
        for i, child in enumerate(self.children):
            if child.name == leaf_name:
                self.children.pop(i)
                return True
            if child.prune_leaf(leaf_name):
                return True
        return False

    def reroot(self, new_root_name: str) -> TreeNode | None:
        """
        将树重新根植于指定节点

        参数:
            new_root_name: 新根节点名称

        返回:
            新的根节点或None
        """
        # 找到新根节点
        target = self._find_by_name(new_root_name)
        if target is None:
            return None

        # 重建树结构
        return self._reroute(target)

    def _find_by_name(self, name: str) -> TreeNode | None:
        """递归查找指定名称的节点"""
        if self.name == name:
            return self
        for child in self.children:
            found = child._find_by_name(name)
            if found:
                return found
        return None

    def _reroute(self, new_root: TreeNode) -> TreeNode:
        """重新路由树结构，使new_root成为新的根节点"""
        # 如果new_root就是当前根，直接返回拷贝
        if new_root is self:

            def copy_subtree(node: TreeNode, parent: TreeNode | None) -> TreeNode:
                new_node = TreeNode(
                    name=node.name, branch_length=node.branch_length, parent=parent, support=node.support,
                    metadata=dict(node.metadata),
                )
                for child in node.children:
                    new_child = copy_subtree(child, new_node)
                    new_node.children.append(new_child)
                return new_node

            return copy_subtree(self, None)

        # 找到从self到new_root的路径
        path = []
        node = new_root
        while node is not None and node is not self:
            path.append(node)
            node = node.parent

        if node is not self:
            # new_root不在以self为根的树中
            return self._reroute(self)  # fallback: return copy

        # 反转路径: [new_root, ..., self]
        path.reverse()  # 现在是 [self, ..., new_root]

        # 创建新的根节点 (new_root的拷贝)
        rerooted = TreeNode(name=new_root.name, support=new_root.support)

        # 沿路径反向重建树
        # new_root原来的子树（除了路径上的父节点方向）成为新根的子节点
        prev_node = None  # 在新树中，path[i-1]的对应节点

        for i, old_node in enumerate(path):
            if i == 0:
                # old_node = self (old root)
                # 在新树中，self变成new_root路径上的一个子节点
                new_node = TreeNode(name=old_node.name, support=old_node.support)
                if i == len(path) - 1:
                    # self就是new_root (已在上面处理)
                    pass
                else:
                    # self的非路径子节点保持不变
                    next_on_path = path[i + 1]
                    for child in old_node.children:
                        if child is not next_on_path:
                            child_copy = self._copy_subtree_full(child, new_node)
                            new_node.children.append(child_copy)
                    prev_node = new_node
            elif i == len(path) - 1:
                # old_node = new_root, 这是新根
                # 新根获得: 原来的非路径子节点 + 路径上的前一个节点
                for child in old_node.children:
                    if i > 0 and child is path[i - 1]:
                        # 路径上的前一个节点，已处理
                        if prev_node is not None:
                            prev_node.branch_length = (
                                child.branch_length if child.branch_length is not None else None
                            )
                            rerooted.children.append(prev_node)
                            prev_node.parent = rerooted
                    else:
                        child_copy = self._copy_subtree_full(child, rerooted)
                        rerooted.children.append(child_copy)
            else:
                # 中间节点: 反转父子关系
                new_node = TreeNode(name=old_node.name, support=old_node.support)
                next_on_path = path[i + 1]
                for child in old_node.children:
                    if child is next_on_path:
                        # 路径上的下一个节点，稍后处理
                        pass
                    elif i > 0 and child is path[i - 1]:
                        # 路径上的前一个节点
                        if prev_node is not None:
                            prev_node.branch_length = (
                                child.branch_length if child.branch_length is not None else None
                            )
                            new_node.children.append(prev_node)
                            prev_node.parent = new_node
                    else:
                        child_copy = self._copy_subtree_full(child, new_node)
                        new_node.children.append(child_copy)
                prev_node = new_node

        return rerooted

    def _copy_subtree_full(self, node: TreeNode, parent: TreeNode | None) -> TreeNode:
        """完整复制子树"""
        new_node = TreeNode(
            name=node.name, branch_length=node.branch_length, parent=parent, support=node.support,
            metadata=dict(node.metadata),
        )
        for child in node.children:
            new_child = self._copy_subtree_full(child, new_node)
            new_node.children.append(new_child)
        return new_node

    @staticmethod
    def _format_label(name: str) -> str:
        """
        将节点标签格式化为合法 Newick token。

        含空白或特殊字符的标签用单引号包裹 (内部单引号翻倍),
        与解析器的引号标签支持配对, 保证 to_newick → parse 往返一致。
        """
        if not name:
            return ""
        if any(ch in name for ch in "()[],;:'\" \t\n\r"):
            return "'" + name.replace("'", "''") + "'"
        return name

    def to_newick(self, include_lengths: bool = True) -> str:
        """
        将节点转换为Newick格式字符串

        参数:
            include_lengths: 是否包含枝长

        返回:
            Newick格式字符串
        """
        if self.is_leaf:
            label = self._format_label(self.name)
            if include_lengths and self.branch_length is not None:
                return f"{label}:{self.branch_length}"
            return label

        # 内部节点
        children_strs = []
        for child in self.children:
            children_strs.append(child.to_newick(include_lengths))

        subtree = f"({','.join(children_strs)})"

        if self.name:
            subtree += self._format_label(self.name)

        if include_lengths and self.branch_length is not None:
            subtree += f":{self.branch_length}"

        return subtree

    def to_dict(self) -> dict[str, Any]:
        """
        将节点转换为字典表示

        返回:
            节点字典
        """
        result = {
            "name": self.name,
        }

        if self.branch_length is not None:
            result["branch_length"] = self.branch_length

        if self.support is not None:
            result["support"] = self.support

        if self.children:
            result["children"] = [child.to_dict() for child in self.children]

        return result

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"TreeNode('{self.name}')"
        return f"TreeNode('{self.name}', children={len(self.children)})"


@dataclass
class NewickTree:
    """
    Newick格式树容器

    包含一或多棵树。

    属性:
        trees: 树列表 (通常只有一棵)
        root: 根节点引用
        metadata: 树的元数据 (如树名称)

    示例:
        >>> tree = NewickTree()
        >>> tree.parse("(A:0.1,B:0.2)C:0.3;")
    """

    trees: list[TreeNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> TreeNode | None:
        """获取根节点"""
        return self.trees[0] if self.trees else None

    @property
    def leaf_names(self) -> list[str]:
        """获取所有叶节点名称"""
        if not self.root:
            return []
        return [leaf.name for leaf in self.root.get_leaves()]

    @property
    def leaf_count(self) -> int:
        """获取叶节点数量"""
        if not self.root:
            return 0
        return self.root.leaf_count

    def to_newick(self, include_lengths: bool = True) -> str:
        """
        转换为Newick格式字符串

        参数:
            include_lengths: 是否包含枝长

        返回:
            Newick格式字符串
        """
        if not self.trees:
            return ""

        newicks = []
        for tree in self.trees:
            newick = tree.to_newick(include_lengths)
            newicks.append(newick + ";")

        return "\n".join(newicks)

    def get_lca(self, name1: str, name2: str) -> TreeNode | None:
        """
        获取两个节点的最近公共祖先(LCA)

        数学定义:
            LCA(n1, n2) = arg max_{v ∈ Ancestors(n1) ∩ Ancestors(n2)} depth(v)

        参数:
            name1: 第一个节点名称
            name2: 第二个节点名称

        返回:
            LCA节点或None
        """
        if not self.root:
            return None

        node1 = self.root._find_by_name(name1)
        node2 = self.root._find_by_name(name2)

        if node1 is None or node2 is None:
            return None

        # 获取两个节点的祖先集合
        ancestors1 = set(node1.get_ancestors())
        ancestors2 = set(node2.get_ancestors())

        # 交集即为公共祖先
        common = ancestors1 & ancestors2

        if not common:
            return None

        # 返回深度最大的（最接近两个节点）
        return max(common, key=lambda n: n.depth)


class NewickParser:
    """
    Newick格式解析器

    使用递归下降解析算法。

    解析算法:
    --------
        1. 识别子树: '(' ... ')'
        2. 解析子节点: 用 ',' 分隔
        3. 解析节点名称: 名称后跟 ':'
        4. 解析枝长: ':' 后跟数字

    状态机:
    --------
        State: INITIAL → NAME → LENGTH → COMMA/SIBLING
                         ↓
                      SUBTREE

    性能考虑:
    --------
        - 使用正则预处理
        - 递归深度受Python调用栈限制
        - 大树可能需要迭代实现

    示例:
        >>> parser = NewickParser()
        >>> tree = parser.parse("(A:0.1,B:0.2)C:0.3;")
        >>> print(tree.leaf_names)
        ['A', 'B']
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.NewickParser")

        # 解析状态
        self._input: str = ""
        self._pos: int = 0
        self._length: int = 0
        # Track recursion depth for the descent parser; the public
        # ``_parse_subtree_with_children`` increments / decrements it
        # so that malicious or pathological input can't crash the
        # interpreter with ``RecursionError``.
        self._depth: int = 0
        self._MAX_DEPTH: int = 1000

        # 正则表达式
        # 注: 中括号/引号不属于未加引号的名字——'[' 开启注释或 NHX 元数据,
        # 否则 "[&&NHX:..." 会被吞进节点名并使后续解析错位 (曾引发死循环)。
        self._name_pattern = re.compile(r"^([^():,\[\]'\"\s;]+)")
        self._number_pattern = re.compile(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
        self._whitespace_pattern = re.compile(r"^\s+")
        # 括号配对计数: parse() 结束时非零说明括号不匹配
        self._paren_balance: int = 0

    def parse(self, newick_string: str) -> NewickTree:
        """
        解析Newick格式字符串

        参数:
            newick_string: Newick格式树字符串

        返回:
            NewickTree对象

        异常:
            ValueError: 格式错误

        示例:
            >>> parser = NewickParser()
            >>> tree = parser.parse("((A,B)C,(D,E)F)G;")
        """
        self._input = newick_string.strip()
        self._pos = 0
        self._length = len(self._input)
        self._paren_balance = 0

        self._logger.debug(f"Parsing Newick: {self._input[:50]}...")

        trees: list[TreeNode] = []

        while self._pos < self._length:
            # 跳过空白
            self._skip_whitespace()

            if self._pos >= self._length:
                break

            # 检查分号结束
            if self._current() == ";":
                self._advance()
                break

            # 跳过 Newick 注释 [comment]: 扫描至配对的 ']' (支持嵌套)。
            # 旧实现扫描到行尾, 会把同一行注释之后的树一并吞掉。
            if self._current() == "[":
                self._consume_bracket_block()
                continue

            # 游离的 ')': 若不报错, 顶层循环将无法消耗任何字符而无限追加空节点
            if self._current() == ")":
                raise ValueError(
                    f"Unmatched ')' in Newick string at position {self._pos}"
                )

            # 解析树
            tree = self._parse_subtree()
            if tree:
                trees.append(tree)

            # 跳过空白
            self._skip_whitespace()

            # 处理分号
            if self._current() == ";":
                self._advance()
                break

            # 检查逗号分隔多棵树
            if self._current() == ",":
                self._advance()

        # 检查括号配对 (如 "((A,B);" 这类未闭合输入此前被静默接受)
        if self._paren_balance > 0:
            raise ValueError(
                f"Unbalanced parentheses in Newick string: "
                f"{self._paren_balance} unclosed '('"
            )

        # 检查是否解析成功
        if not trees:
            raise ValueError("Failed to parse Newick tree: no valid tree found")

        tree_obj = NewickTree(trees=trees)

        self._logger.info(f"Parsed Newick tree with {tree_obj.leaf_count} leaves")

        return tree_obj

    def parse_file(self, filepath: str) -> list[NewickTree]:
        """
        从文件解析Newick树

        参数:
            filepath: 文件路径

        返回:
            NewickTree列表
        """
        # Cap the file size at 50 MiB. Newick files are typically
        # small (a few KB); anything larger is almost certainly
        # malformed or an attempted resource exhaustion. Without this
        # guard, a malicious 10 GiB file would consume all available
        # memory before we even start parsing.
        import os

        size = os.path.getsize(filepath)
        max_size = 50 * 1024 * 1024
        if size > max_size:
            raise ValueError(
                f"Newick file too large: {size} bytes (limit {max_size}); "
                "refusing to read to prevent memory exhaustion."
            )
        # Use utf-8-sig to strip UTF-8 BOM if present
        with open(filepath, encoding="utf-8-sig") as f:
            content = f.read()

        return self.parse_multi(content)

    def parse_multi(self, content: str) -> list[NewickTree]:
        """
        解析包含多棵树的字符串

        参数:
            content: 包含多棵树的字符串

        返回:
            NewickTree列表
        """
        trees: list[NewickTree] = []

        # 分割各棵树
        tree_strings = content.split(";")

        for ts in tree_strings:
            ts = ts.strip()
            if ts:
                try:
                    tree = self.parse(ts + ";")
                    trees.append(tree)
                except ValueError as e:
                    self._logger.warning(f"Failed to parse tree: {e}")

        return trees

    def _parse_subtree(self) -> TreeNode | None:
        """
        解析子树

        递归下降解析入口。

        递归文法:
            Subtree → Name? Subtrees? ":" BranchLength?
                    | Subtrees ":" BranchLength?
                    | Name ":" BranchLength?

        返回:
            TreeNode对象
        """
        self._skip_whitespace()

        if self._pos >= self._length:
            return None

        # 检查是否开始子节点列表
        if self._current() == "(":
            return self._parse_subtree_with_children()
        else:
            return self._parse_simple_node()

    def _parse_subtree_with_children(self) -> TreeNode:
        """
        解析带子节点的子树

        格式: (child1, child2, ...)name:length

        返回:
            TreeNode对象
        """
        # Recursion-depth guard. Newick allows arbitrarily deep
        # nesting of parentheses; without this cap a malicious or
        # accidentally pathological input of ``((((...`` depth 10^5
        # would crash the interpreter with ``RecursionError`` (or
        # worse, exhaust the C stack). 1000 is well above any
        # reasonable biological tree and matches the effective
        # default Python recursion limit.
        self._depth += 1
        try:
            if self._depth > self._MAX_DEPTH:
                raise ValueError(
                    f"Newick recursion depth exceeded {self._MAX_DEPTH}; "
                    "tree is too deeply nested or malformed."
                )
            return self._parse_subtree_with_children_impl()
        finally:
            self._depth -= 1

    def _parse_subtree_with_children_impl(self) -> TreeNode:
        """Implementation of ``_parse_subtree_with_children``; the
        public method wraps this in a depth counter."""
        # 消耗 '('
        self._advance()
        self._paren_balance += 1

        children: list[TreeNode] = []

        while True:
            self._skip_whitespace()

            # 子节点前的注释 (如 (A,[comment]B))
            while self._current() == "[":
                self._consume_bracket_block()
                self._skip_whitespace()

            # 解析第一个子节点
            child = self._parse_subtree()
            if child:
                children.append(child)

            self._skip_whitespace()

            # 子节点后的注释
            while self._current() == "[":
                self._consume_bracket_block()
                self._skip_whitespace()

            # 检查分隔符
            if self._current() == ",":
                self._advance()
                continue
            elif self._current() == ")":
                self._advance()
                self._paren_balance -= 1
                break
            else:
                # 可能到达字符串末尾
                break

        # 解析节点名称和枝长
        name = self._parse_name() or ""

        # 名称之后、枝长之前的 NHX 元数据: (...)[&&NHX:x=1]:0.5
        metadata: dict[str, Any] = {}
        if self._current() == "[":
            nhx_content = self._consume_bracket_block()
            if nhx_content is not None:
                metadata.update(self._parse_nhx_pairs(nhx_content))

        branch_length = None
        if self._current() == ":":
            self._advance()
            branch_length = self._parse_number()

        # 枝长之后的 NHX 元数据: (...):0.5[&&NHX:x=1]
        if self._current() == "[":
            nhx_content = self._consume_bracket_block()
            if nhx_content is not None:
                metadata.update(self._parse_nhx_pairs(nhx_content))

        # 创建内部节点
        node = TreeNode(name=name, branch_length=branch_length)
        if metadata:
            node.metadata.update(metadata)

        # 设置子节点
        for child in children:
            child.parent = node
            node.children.append(child)

        return node

    def _parse_simple_node(self) -> TreeNode:
        """
        解析简单节点 (叶节点)

        格式: name[length][:branch_length]

        返回:
            TreeNode对象
        """
        name = self._parse_name() or ""

        # 叶节点 NHX 元数据 (名称之后、枝长之前): A[&&NHX:S=human]:0.1
        metadata: dict[str, Any] = {}
        if self._current() == "[":
            nhx_content = self._consume_bracket_block()
            if nhx_content is not None:
                metadata.update(self._parse_nhx_pairs(nhx_content))

        branch_length = None
        if self._current() == ":":
            self._advance()
            branch_length = self._parse_number()

        # 枝长之后的 NHX 元数据: A:0.1[&&NHX:S=human]
        if self._current() == "[":
            nhx_content = self._consume_bracket_block()
            if nhx_content is not None:
                metadata.update(self._parse_nhx_pairs(nhx_content))

        node = TreeNode(name=name, branch_length=branch_length)
        if metadata:
            node.metadata.update(metadata)
        return node

    def _parse_name(self) -> str | None:
        """
        解析节点名称

        支持:
        - 普通名称: 非空白、非特殊字符序列
        - 单引号名称: 'Homo sapiens' (允许包含空格)
        - 双引号名称: "Homo sapiens"

        返回:
            名称字符串 (引号已剥离)
        """
        self._skip_whitespace()

        if self._pos >= self._length:
            return None

        # Support quoted labels: 'name with spaces' or "name with spaces"
        quote_char = self._current()
        if quote_char in ("'", '"'):
            quote_char_typed = quote_char
            self._advance()
            start = self._pos
            # Scan until matching quote
            while self._pos < self._length and self._current() != quote_char_typed:
                # Allow escaped quotes: \' or \"
                if self._current() == "\\" and self._pos + 1 < self._length:
                    self._advance()
                self._advance()
            if self._pos >= self._length:
                raise ValueError(
                    f"Unterminated quoted name starting at position {start}"
                )
            name = self._input[start : self._pos]
            self._advance()  # consume closing quote
            # Unescape internal escaped quotes
            name = name.replace("\\" + quote_char_typed, quote_char_typed)
            return name

        # Unquoted name: non-whitespace, non-special chars
        match = self._name_pattern.match(self._input[self._pos :])
        if match:
            name = match.group(1)
            self._pos += match.end()
            return name

        return None

    def _parse_number(self) -> float | None:
        """
        解析数字 (枝长)

        支持整数、浮点数、科学计数法。

        返回:
            数字值
        """
        self._skip_whitespace()

        match = self._number_pattern.match(self._input[self._pos :])
        if match:
            value = float(match.group(1))
            self._pos += match.end()
            return value

        return None

    def _consume_bracket_block(self) -> str | None:
        """
        消费当前位置的 [ ... ] 块 (平衡嵌套扫描)。

        返回:
            若块以 "&&NHX" 开头则返回其内容字符串 (不含括号);
            普通注释返回 None。

        异常:
            ValueError: 块未闭合
        """
        if self._current() != "[":
            return None
        start_pos = self._pos
        self._advance()  # consume '['

        is_nhx = self._input[self._pos : self._pos + 5] == "&&NHX"
        if is_nhx:
            self._advance(5)  # consume '&&NHX'

        content_start = self._pos
        depth = 1
        while self._pos < self._length and depth > 0:
            ch = self._current()
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            self._advance()

        if depth > 0:
            raise ValueError(
                f"Unterminated bracket block at position {start_pos}"
            )

        if not is_nhx:
            return None
        # self._pos 位于 ']' 之后
        return self._input[content_start : self._pos - 1]

    @staticmethod
    def _parse_nhx_pairs(content: str) -> dict[str, str]:
        """
        解析 NHX 属性串。

        规范格式 (Zmasek & Eddy 2001) 以 ':' 分隔属性:
            [&&NHX:spec=Human:bug=Tyro]
        亦兼容逗号分隔 (部分工具输出):
            [&&NHX B=95,D=N]

        返回:
            dict of metadata key -> value strings
        """
        content = content.strip()
        if not content:
            return {}

        if ":" in content:
            tokens = content.split(":")
        else:
            tokens = content.split(",")

        metadata: dict[str, str] = {}
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                key, _, value = token.partition("=")
                metadata[key.strip()] = value.strip()
            else:
                # 裸标志位 (无值), 记录为空串
                metadata[token] = ""
        return metadata

    def _skip_whitespace(self) -> None:
        """跳过空白字符"""
        while self._pos < self._length:
            char = self._input[self._pos]
            if char in " \t\n\r":
                self._pos += 1
            else:
                break

    def _current(self) -> str:
        """获取当前位置字符"""
        if self._pos < self._length:
            return self._input[self._pos]
        return ""

    def _advance(self, count: int = 1) -> None:
        """前进指定字符数"""
        self._pos = min(self._pos + count, self._length)


class TreeComparator:
    """
    树比较工具类

    提供树的比较、距离计算等功能。
    """

    @staticmethod
    def rf_distance(tree1: NewickTree, tree2: NewickTree) -> int:
        """
        计算Robinson-Foulds距离

        数学定义:
            RF(T1, T2) = |P(T1) Δ P(T2)|
            其中 P(T) 是树T的所有分割集合
                  Δ 是对称差

        参数:
            tree1: 第一棵树
            tree2: 第二棵树

        返回:
            RF距离 (整数)
        """
        splits1 = TreeComparator._get_splits(tree1.root)
        splits2 = TreeComparator._get_splits(tree2.root)

        # 对称差
        only1 = splits1 - splits2
        only2 = splits2 - splits1

        return len(only1) + len(only2)

    @staticmethod
    def _get_splits(node: TreeNode) -> set[tuple[str, ...]]:
        """
        获取树的所有分割

        参数:
            node: 根节点

        返回:
            分割集合
        """
        splits = set()

        if node.is_leaf:
            return splits

        # 递归获取子节点分割
        for child in node.children:
            child_splits = TreeComparator._get_splits(child)
            splits.update(child_splits)

        # 当前节点的分割
        # 处理 polytomy (多叉树): 为每个子节点生成一个分割
        # 分割 = (该子节点的叶节点, 所有其他子节点的叶节点)
        if len(node.children) >= 2:
            all_leaves = node.get_leaves()
            for i, child in enumerate(node.children):
                child_leaves = frozenset(leaf.name for leaf in child.get_leaves())
                other_leaves = frozenset(leaf.name for leaf in all_leaves if leaf not in child.get_leaves())
                # 确保 group1 < group2 以保持唯一性
                if child_leaves > other_leaves:
                    child_leaves, other_leaves = other_leaves, child_leaves
                splits.add((child_leaves, other_leaves))
        elif len(node.children) == 1:
            # 单子节点: 无分割 (退化情况)
            pass

        return splits


def parse_newick(newick_string: str) -> NewickTree:
    """
    解析Newick格式字符串的便捷函数

    参数:
        newick_string: Newick格式字符串

    返回:
        NewickTree对象
    """
    parser = NewickParser()
    return parser.parse(newick_string)


def read_newick_file(filepath: str) -> list[NewickTree]:
    """
    从文件读取Newick树

    参数:
        filepath: 文件路径

    返回:
        NewickTree列表
    """
    parser = NewickParser()
    return parser.parse_file(filepath)
