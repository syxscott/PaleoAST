"""
================================================================================
PaleoAST - AST Source Code Auditor
================================================================================

本模块实现基于AST的Python源码审计器，检查：
- 所有函数是否包含Docstring
- 非法import循环
- 代码复杂度
- 潜在bug模式

作者: PaleoAST Development Team
"""

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """函数信息"""

    name: str
    lineno: int
    end_lineno: int
    docstring: str | None
    parameters: list[str]
    returns: str | None
    is_async: bool
    is_method: bool
    class_name: str | None = None
    decorators: list[str] = field(default_factory=list)
    complexity: int = 1


@dataclass
class ImportInfo:
    """Import信息"""

    module: str
    names: list[str]
    lineno: int
    is_from: bool


@dataclass
class AuditIssue:
    """审计问题"""

    file_path: str
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    lineno: int | None = None
    line_content: str | None = None


@dataclass
class AuditReport:
    """审计报告"""

    timestamp: str
    project_root: str
    total_files: int
    total_functions: int
    files_with_issues: int
    issues: list[AuditIssue] = field(default_factory=list)
    functions_missing_docstring: list[FunctionInfo] = field(default_factory=list)
    import_cycles: list[list[str]] = field(default_factory=list)
    complexity_warnings: list[tuple] = field(default_factory=list)

    def get_issues_by_severity(self, severity: str) -> list[AuditIssue]:
        """按严重性获取问题"""
        return [i for i in self.issues if i.severity == severity]

    def get_summary(self) -> str:
        """获取摘要"""
        return (
            f"Audit Report Summary:\n"
            f"  - Timestamp: {self.timestamp}\n"
            f"  - Total Files: {self.total_files}\n"
            f"  - Total Functions: {self.total_functions}\n"
            f"  - Files with Issues: {self.files_with_issues}\n"
            f"  - Errors: {len(self.get_issues_by_severity('error'))}\n"
            f"  - Warnings: {len(self.get_issues_by_severity('warning'))}\n"
            f"  - Info: {len(self.get_issues_by_severity('info'))}\n"
            f"  - Missing Docstrings: {len(self.functions_missing_docstring)}"
        )

    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        lines = [
            "# PaleoAST Code Audit Report",
            "",
            f"**Generated**: {self.timestamp}",
            f"**Project Root**: {self.project_root}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Files | {self.total_files} |",
            f"| Total Functions | {self.total_functions} |",
            f"| Files with Issues | {self.files_with_issues} |",
            f"| Errors | {len(self.get_issues_by_severity('error'))} |",
            f"| Warnings | {len(self.get_issues_by_severity('warning'))} |",
            f"| Missing Docstrings | {len(self.functions_missing_docstring)} |",
            "",
            "## Issues by Severity",
            "",
            "### Errors",
        ]

        errors = self.get_issues_by_severity("error")
        if errors:
            for issue in errors:
                lines.append(f"- **{issue.file_path}:{issue.lineno}** - {issue.message}")
        else:
            lines.append("*No errors found.*")

        lines.extend(["", "### Warnings"])

        warnings = self.get_issues_by_severity("warning")
        if warnings:
            for issue in warnings[:50]:  # 限制输出
                lines.append(f"- {issue.file_path}:{issue.lineno} - {issue.message}")
        else:
            lines.append("*No warnings found.*")

        lines.extend(["", "## Missing Docstrings", ""])

        if self.functions_missing_docstring:
            lines.append("| File | Function | Line |")
            lines.append("|------|----------|------|")
            for func in self.functions_missing_docstring[:30]:
                loc = f"{func.class_name}.{func.name}" if func.class_name else func.name
                lines.append(
                    f"| {Path(func.name).parent if ':' not in str(func.lineno) else 'N/A'} | {loc} | {func.lineno} |"
                )
        else:
            lines.append("*All functions have docstrings.*")

        return "\n".join(lines)


class ASTAuditor:
    """
    AST源码审计器

    使用示例:
        >>> auditor = ASTAuditor()
        >>> report = auditor.audit_directory()
        >>> print(report.to_markdown())
    """

    # 忽略检查的目录
    IGNORED_DIRS = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "venv",
        "env",
        ".venv",
        "build",
        "dist",
        "*.egg-info",
        "node_modules",
    }

    # 忽略检查的文件
    IGNORED_FILES = {"setup.py", "__init__.py"}

    # 复杂度阈值
    COMPLEXITY_THRESHOLD = 10

    # 循环import检测
    DETECT_IMPORT_CYCLES = True

    def __init__(self, project_root: str = None):
        """
        初始化审计器

        参数:
            project_root: 项目根目录
        """
        if project_root is None:
            self._project_root = Path.cwd()
        else:
            self._project_root = Path(project_root)

        self._report = AuditReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            project_root=str(self._project_root),
            total_files=0,
            total_functions=0,
            files_with_issues=0,
            issues=[],
            functions_missing_docstring=[],
            import_cycles=[],
            complexity_warnings=[],
        )

        self._import_graph: dict[str, set[str]] = defaultdict(set)
        self._functions_by_file: dict[str, list[FunctionInfo]] = defaultdict(list)

        self._logger = logging.getLogger(f"{__name__}.ASTAuditor")

    def audit_directory(self, extensions: list[str] = None, recursive: bool = True) -> AuditReport:
        """
        审计整个目录

        参数:
            extensions: 要检查的文件扩展名
            recursive: 是否递归

        返回:
            AuditReport
        """
        if extensions is None:
            extensions = [".py"]

        self._logger.info(f"Starting audit of {self._project_root}")

        # 收集所有Python文件
        py_files = self._collect_python_files(extensions, recursive)
        self._report.total_files = len(py_files)

        # 审计每个文件
        for py_file in py_files:
            try:
                self._audit_file(py_file)
            except Exception as e:
                self._report.issues.append(
                    AuditIssue(
                        file_path=str(py_file),
                        issue_type="parse_error",
                        severity="error",
                        message=f"Failed to parse: {e!s}",
                    )
                )
                self._logger.error(f"Error auditing {py_file}: {e}")

        # 检测循环import
        if self.DETECT_IMPORT_CYCLES:
            self._detect_import_cycles()

        # 统计有问题的文件
        files_with_issues = set(i.file_path for i in self._report.issues)
        self._report.files_with_issues = len(files_with_issues)

        # 更新总函数数
        self._report.total_functions = sum(len(funcs) for funcs in self._functions_by_file.values())

        self._logger.info(self._report.get_summary())
        return self._report

    def _collect_python_files(self, extensions: list[str], recursive: bool) -> list[Path]:
        """收集所有Python文件"""
        py_files = []

        if recursive:
            for ext in extensions:
                py_files.extend(self._project_root.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                py_files.extend(self._project_root.glob(f"*{ext}"))

        # 过滤忽略的目录和文件
        filtered = []
        for f in py_files:
            rel_path = f.relative_to(self._project_root)

            # 检查是否在忽略目录中
            if any(part in self.IGNORED_DIRS for part in rel_path.parts):
                continue

            # 检查文件名
            if f.name in self.IGNORED_FILES:
                continue

            filtered.append(f)

        return filtered

    def _audit_file(self, file_path: Path) -> None:
        """审计单个文件"""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            self._report.issues.append(
                AuditIssue(
                    file_path=str(file_path),
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Syntax error: {e.msg}",
                    lineno=e.lineno,
                    line_content=e.text,
                )
            )
            return

        rel_path = file_path.relative_to(self._project_root)

        # 收集import
        self._collect_imports(tree, str(rel_path))

        # 检查函数
        self._check_functions(tree, str(rel_path), content)

        # 检查复杂度
        self._check_complexity(tree, str(rel_path))

        # 检查其他问题
        self._check_other_issues(tree, str(rel_path), content)

    def _collect_imports(self, tree: ast.AST, file_path: str) -> None:
        """收集import信息"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._import_graph[file_path].add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._import_graph[file_path].add(node.module)

    def _check_functions(self, tree: ast.AST, file_path: str, content: str) -> None:
        """检查函数定义"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 确定类名
                class_name = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if any(
                            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is node
                            for child in parent.body
                        ):
                            class_name = parent.name
                            break

                # 提取docstring
                docstring = ast.get_docstring(node)

                func_info = FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    docstring=docstring,
                    parameters=[arg.arg for arg in node.args.args],
                    returns=None,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    is_method=class_name is not None,
                    class_name=class_name,
                    decorators=[d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                )

                # 检查返回值
                for node_child in ast.walk(node):
                    if isinstance(node_child, ast.Return):
                        if node_child.value:
                            func_info.returns = "specified"
                        else:
                            func_info.returns = "implicit"
                        break

                self._functions_by_file[file_path].append(func_info)

                # 缺少docstring检查 (跳过特殊方法)
                if not docstring and not node.name.startswith("_"):
                    self._report.functions_missing_docstring.append(func_info)
                    self._report.issues.append(
                        AuditIssue(
                            file_path=file_path,
                            issue_type="missing_docstring",
                            severity="warning",
                            message=f"Function '{node.name}' lacks docstring",
                            lineno=node.lineno,
                        )
                    )

    def _check_complexity(self, tree: ast.AST, file_path: str) -> None:
        """检查代码复杂度"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)

                if complexity > self.COMPLEXITY_THRESHOLD:
                    func_name = node.name
                    self._report.complexity_warnings.append((file_path, func_name, complexity))
                    self._report.issues.append(
                        AuditIssue(
                            file_path=file_path,
                            issue_type="high_complexity",
                            severity="info",
                            message=f"Function '{func_name}' has complexity {complexity} (threshold: {self.COMPLEXITY_THRESHOLD})",
                            lineno=node.lineno,
                        )
                    )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ExceptHandler, ast.Try)):
                complexity += 1

        return complexity

    def _check_other_issues(self, tree: ast.AST, file_path: str, content: str) -> None:
        """检查其他问题"""
        lines = content.split("\n")

        for node in ast.walk(tree):
            # 检查 TODO/FIXME
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    value = node.value.value.upper()
                    if "TODO" in value or "FIXME" in value:
                        self._report.issues.append(
                            AuditIssue(
                                file_path=file_path,
                                issue_type="todo_found",
                                severity="info",
                                message="TODO/FIXME comment found",
                                lineno=node.lineno,
                            )
                        )

            # 检查过于长的行
            if hasattr(node, "lineno"):
                line_idx = node.lineno - 1
                if line_idx < len(lines):
                    line = lines[line_idx]
                    if len(line) > 120:
                        self._report.issues.append(
                            AuditIssue(
                                file_path=file_path,
                                issue_type="line_too_long",
                                severity="info",
                                message=f"Line exceeds 120 characters ({len(line)} chars)",
                                lineno=node.lineno,
                                line_content=line[:80] + "...",
                            )
                        )

    def _detect_import_cycles(self) -> None:
        """检测循环import"""
        # 使用DFS检测循环
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._import_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # 发现循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node in self._import_graph.keys():
            if node not in visited:
                dfs(node, [])

        self._report.import_cycles = cycles

        for cycle in cycles:
            self._report.issues.append(
                AuditIssue(
                    file_path=cycle[0],
                    issue_type="import_cycle",
                    severity="error",
                    message=f"Import cycle detected: {' -> '.join(cycle[:5])}",
                )
            )


def audit_paleoast(project_root: str = None) -> AuditReport:
    """
    便捷函数：审计PaleoAST源码

    参数:
        project_root: 项目根目录

    返回:
        AuditReport
    """
    auditor = ASTAuditor(project_root)
    return auditor.audit_directory()


if __name__ == "__main__":
    # 直接运行时的测试
    report = audit_paleoast()
    print(report.to_markdown())
