# =============================================================================
# FILE: utils/script_session.py
# =============================================================================
"""
The execution session behind the script console.

WHY THIS IS NOT IN THE VIEWS PACKAGE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAST3's scripting is its single largest advantage over a menu-driven
tool: with fifty samples and twenty variables, no amount of clicking
gets you through the grid. What it costs is a separate little language.

This session has no language. It is Python, because the application is
Python, and the analyses are reachable by name through the plugin
registry. So the value added over "embed a REPL" is the part that
knows about the application: ``analyses()`` lists what can be run,
``run("mantel", ...)`` runs one and prints its summary, and ``data`` is
always the current spreadsheet.

Keeping the logic here rather than in views/script_console.py means it
is testable without a QApplication, which on this platform also means
testable without a display. The Qt dialog is a shell over this class and
holds no logic of its own.

WHAT IS IN THE NAMESPACE
~~~~~~~~~~~~~~~~~~~~~~~~
``data``
    The current spreadsheet, refreshed on every access rather than
    captured at construction, so a script that runs after the user edits
    a cell sees the edit.
``controller``
    The StatisticsController, for anything not in the registry.
``registry``
    The plugin registry.
``analyses()`` / ``run(name, ...)``
    The two conveniences worth having.
``np``
    numpy. Not matplotlib: plotting from a script would need a figure
    manager and a canvas, and the application already has a plot canvas
    with its own R export path. Scripts that want a figure should use
    the application's own plotting.

Execution is synchronous, and a long-running script freezes the window.
That is the same behaviour as running a long analysis from the menu, and
it is stated in the dialog rather than worked around with a thread,
because a thread would need the results marshalled back to the GUI
thread and the marshal is where the bugs would be.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """What one execution produced.

    Attributes:
        ok: False when the source raised. The session survives either
            way -- a script with a typo in line 40 should keep the
            variables it built in lines 1-39.
        output: Anything the source printed.
        value: The value of a trailing expression, formatted, or None.
        error: The formatted traceback when ``ok`` is False.
    """

    ok: bool
    output: str = ""
    value: str | None = None
    error: str | None = None
    namespace_keys: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Everything worth showing, in the order a REPL would."""
        parts: list[str] = []
        if self.output:
            parts.append(self.output.rstrip("\n"))
        if self.error:
            parts.append(self.error.rstrip("\n"))
        if self.value is not None:
            parts.append(self.value)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "ok": self.ok,
            "output": self.output,
            "value": self.value,
            "error": self.error,
            "namespace_keys": list(self.namespace_keys),
        }


def _split_trailing_expression(source: str) -> tuple[str, str | None]:
    """Split a source string into (body, trailing expression source).

    A REPL echoes the value of a final bare expression, which is how
    ``x = 1`` and then ``x`` differ in practice. When the last statement
    is an Expression, it is rewritten to print its repr instead. When it
    is anything else -- an assignment, a loop, an import -- nothing is
    added, because there is no value to echo.

    Returns ``(source_to_exec, expression_source_or_None)``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Let exec() raise, so the caller reports the real SyntaxError
        # with its line number rather than this function inventing one.
        return source, None
    if not tree.body:
        return source, None
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return source, None

    # A bare print() has no value to echo. Evaluating it returns None and
    # the printed text has already gone to stdout, so echoing "None" after
    # it is worse than no echo. Leave the block as written.
    if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name):
        if last.value.func.id == "print":
            return source, None

    # Anything earlier on the same line means a line-based split would
    # take the assignment with it: in "p = 2; p ** 5" the final
    # expression is on line 1, so head would be empty and p would be
    # undefined. Column offsets do not rescue it -- a semicolon puts the
    # expression mid-line. When the expression DOES start at column 0 it
    # is alone on its line, which is the case worth splitting.
    lines = source.splitlines(keepends=True)
    line_start = lines[last.lineno - 1]
    if last.col_offset > 0 and line_start[: last.col_offset].strip():
        return source, None

    expr_src = ast.get_source_segment(source, last)
    if expr_src is None:
        return source, None
    head = "".join(lines[: last.lineno - 1])
    return head, expr_src


class DataProxy:
    """A live view of the spreadsheet, bound to the name ``data``.

    A plain injection would be a snapshot: the session would keep
    whatever the sheet held when the console opened, and a script run
    after the user edits a cell would analyse the old numbers while
    reporting new ones. That failure is invisible, which is why this
    forwards to the provider on every operation instead.

    So ``data.shape``, ``data[0]``, ``data.mean(axis=0)``,
    ``np.corrcoef(data.T)`` and ``len(data)`` all reach the current
    sheet. Only :meth:`snapshot` materialises a fixed array, for a
    script that genuinely wants one.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: Callable[[], Any]) -> None:
        """Bind to a zero-argument data provider."""
        self._provider = provider

    def _current(self) -> Any:
        """The live array, or None when nothing is loaded."""
        return self._provider()

    def snapshot(self) -> Any:
        """Materialise the current data as a fixed array.

        Use this when a loop must not see the sheet change underneath
        it, or when handing the array to something that needs a real
        ndarray rather than something array-like.
        """
        current = self._current()
        return None if current is None else np.asarray(current)

    def __getattr__(self, name: str) -> Any:
        current = object.__getattribute__(self, "_provider")()
        if current is None:
            raise AttributeError(
                f"no data is loaded, so data.{name} has nothing to read. "
                f"Open a spreadsheet, or build an array in the console."
            )
        if hasattr(current, "raw_data"):  # a DataMatrix, not an array
            current = current.raw_data
        try:
            return getattr(current, name)
        except AttributeError:
            raise AttributeError(
                f"data.{name} is not available; the loaded object is a {type(current).__name__}"
            ) from None

    def __getitem__(self, key: Any) -> Any:
        current = self._current()
        if current is None:
            raise IndexError("no data is loaded, so data[...] has nothing to read")
        if hasattr(current, "raw_data"):
            current = current.raw_data
        return current[key]

    def __len__(self) -> int:
        current = self._current()
        if current is None:
            return 0
        if hasattr(current, "raw_data"):
            current = current.raw_data
        return len(current)

    def __array__(self, dtype: Any = None, copy: Any = None) -> npt.NDArray:
        current = self.snapshot()
        if current is None:
            raise ValueError("no data is loaded, so data cannot be converted to an array")
        return current if dtype is None else current.astype(dtype, copy=False)

    def __iter__(self) -> Any:
        current = self._current()
        if current is None:
            return iter(())
        if hasattr(current, "raw_data"):
            current = current.raw_data
        return iter(current)

    def __bool__(self) -> bool:
        """False when nothing is loaded, so ``if data:`` reads naturally."""
        current = self._current()
        return current is not None and bool(np.asarray(current).size)

    def __repr__(self) -> str:
        current = self._current()
        if current is None:
            return "<data: nothing loaded>"
        return f"<data {tuple(np.asarray(current).shape)} (live)>"

    def to_dict(self) -> dict[str, Any]:
        """Describe the binding, for a script that wants to introspect."""
        current = self._current()
        return {
            "loaded": current is not None,
            "shape": None if current is None else tuple(np.asarray(current).shape),
            "live": True,
        }


class ScriptSession:
    """A persistent Python namespace wired to the running application.

    Parameters
    ----------
    data_provider:
        Zero-argument callable returning the current data, or None. It
        is called on every access to :attr:`data`, so the session does
        not hold a stale copy of the spreadsheet.
    controller:
        The StatisticsController, exposed as ``controller``.
    """

    def __init__(
        self,
        data_provider: Callable[[], Any] | None = None,
        controller: Any = None,
    ) -> None:
        """Create a session with an initial namespace."""
        self._data_provider = data_provider
        self._controller = controller
        self._registry: Any = None
        self._catalog_loaded = False
        self.namespace: dict[str, Any] = {
            "__name__": "paleoast_script",
            "__builtins__": __builtins__,
        }
        self._refresh_registry()
        self._install_helpers()

    # -- namespace wiring -------------------------------------------------

    def _refresh_registry(self) -> None:
        """Point ``registry`` at the live singleton, tolerating absence."""
        try:
            from plugins.registry import get_plugin_registry

            self._registry = get_plugin_registry()
        except Exception:  # pragma: no cover - plugins always import
            self._registry = None
        self.namespace["registry"] = self._registry

    def _install_helpers(self) -> None:
        """Define the application-aware conveniences."""
        self.namespace["np"] = np
        self.namespace["analyses"] = self.analyses
        self.namespace["run"] = self.run
        self.namespace["categories"] = self.categories
        self.namespace["controller"] = self._controller
        self.namespace["session"] = self
        # A live view of the sheet, not a snapshot of it. See DataProxy for
        # why injecting the array itself would be wrong.
        self.namespace["data"] = DataProxy(self._read_data)

    @property
    def data(self) -> Any:
        """The current data, or None when nothing is loaded."""
        return self._read_data()

    def _read_data(self) -> Any:
        """Call the provider, unwrapping a DataMatrix, never raising.

        Unwrapping here means ``run()`` and the ``data`` proxy agree on
        what "the data" means: both hand the analysis an ndarray, not a
        wrapper whose ``raw_data`` the analysis would have to know about.
        """
        if self._data_provider is None:
            return None
        try:
            current = self._data_provider()
        except Exception:  # a provider that raises must not kill the console
            logger.debug("ScriptSession data provider raised", exc_info=True)
            return None
        if current is not None and hasattr(current, "raw_data"):
            return current.raw_data
        return current

    @property
    def controller(self) -> Any:
        """The StatisticsController, or None."""
        return self._controller

    # -- the conveniences -------------------------------------------------

    def _ensure_catalog_loaded(self) -> None:
        """Fill the plugin registry from the built-in catalog, once.

        The session does this for itself rather than borrowing the
        controller's lazy registration. That matters for the headless
        path, which has no controller at all: with the registration
        attached to the controller, ``run("mantel")`` worked in the GUI
        console and reported "Available (0)" from the command line.

        Idempotent, and the flag is set even on failure so a catalog
        that cannot load once does not log a traceback per call.
        """
        if self._catalog_loaded:
            return
        try:
            from plugins.catalog import register_builtin_analyses

            register_builtin_analyses()
        except Exception:  # a broken catalog must not break the console
            logger.warning(
                "could not register built-in analyses; run() will find none",
                exc_info=True,
            )
        finally:
            self._catalog_loaded = True

    def analyses(self, category: str | None = None) -> list[str]:
        """Names of every runnable analysis.

        This forces the catalog to load, so a session that has never run
        anything still lists the full set -- which is the point of a
        discovery helper.
        """
        self._ensure_catalog_loaded()
        if self._registry is None:
            return []
        if category is None:
            return sorted(self._registry.list_plugins())
        return sorted(self._registry.list_plugins(category=category))

    def categories(self) -> list[str]:
        """Every category name present in the catalog."""
        self._ensure_catalog_loaded()
        if self._registry is None:
            return []
        return sorted(self._registry.list_categories())

    def run(
        self,
        name: str,
        data: Any = None,
        show: bool = True,
        **kwargs: Any,
    ) -> str:
        """Run a catalogued analysis and return its summary.

        ``data`` defaults to the current spreadsheet, as an ndarray. The
        result is formatted through the analysis's own ``summary()``
        when it has one, because every analyzer in this codebase writes
        its summary for humans and that text is better than a repr.

        Parameters
        ----------
        name:
            A name from :meth:`analyses`.
        data:
            Override the current data.
        show:
            Print the summary as well as returning it.
        **kwargs:
            Forwarded to the analysis.

        Returns
        -------
        str
            The formatted result, or the error text on failure.
        """
        self._ensure_catalog_loaded()
        if self._registry is None or self._registry.get(name) is None:
            available = ", ".join(self.analyses()[:20])
            return f"No analysis named {name!r}.\nAvailable ({len(self.analyses())}): {available} ..."
        payload = self.data if data is None else data
        if payload is not None and hasattr(payload, "raw_data"):
            payload = payload.raw_data
        try:
            outcome = self._registry.get(name).analyze(payload, **kwargs)
        except Exception:  # the analysis's own error, rendered not swallowed
            return traceback.format_exc()
        text = _format_result(outcome)
        if show:
            print(text)
        return text

    # -- execution --------------------------------------------------------

    def execute(self, source: str) -> ExecutionResult:
        """Run a block of Python in the session namespace.

        The namespace persists across calls, so the second block sees
        what the first defined. An exception does not roll that back.

        Parameters
        ----------
        source:
            Python source, possibly several lines.

        Returns
        -------
        ExecutionResult
        """
        if not source.strip():
            return ExecutionResult(ok=True, namespace_keys=self._public_keys())

        body, trailing = _split_trailing_expression(source)
        buffer = io.StringIO()
        value_text: str | None = None
        try:
            with contextlib.redirect_stdout(buffer):
                if body.strip():
                    # exec on user-typed source is the entire point of
                    # this module; the S102 advisory is not enabled here.
                    exec(compile(body, "<script>", "exec"), self.namespace)
                if trailing is not None:
                    value = eval(  # S307: same, this is a REPL
                        compile(trailing, "<script>", "eval"), self.namespace
                    )
                    value_text = _format_result(value)
        except Exception:  # a console reports an error, it does not crash
            return ExecutionResult(
                ok=False,
                output=buffer.getvalue(),
                error=traceback.format_exc(),
                namespace_keys=self._public_keys(),
            )
        return ExecutionResult(
            ok=True,
            output=buffer.getvalue(),
            value=value_text,
            namespace_keys=self._public_keys(),
        )

    def _public_keys(self) -> list[str]:
        """Names a user typed, excluding the injected helpers."""
        injected = {
            "__name__",
            "__builtins__",
            "np",
            "analyses",
            "run",
            "categories",
            "session",
            "registry",
            "controller",
            "data",
        }
        return sorted(k for k in self.namespace if k not in injected)

    def reset(self) -> None:
        """Clear everything the user defined and re-seed the helpers."""
        self.namespace.clear()
        self.namespace["__name__"] = "paleoast_script"
        self.namespace["__builtins__"] = __builtins__
        self._refresh_registry()
        self._install_helpers()


def _format_result(value: Any) -> str:
    """Render a result for a human.

    An analysis result gets its own ``summary()``: every analyzer in this
    codebase writes that text for display, and it beats a dataclass repr
    by a wide margin. ``to_dict()`` is used when there is no summary but
    there is a dict. Everything else falls back to ``repr``.

    NumPy scalars are unwrapped first, and this is not cosmetic. Under
    numpy 1.26 ``repr(np.int64(3))`` is ``3``; under numpy 2 it is
    ``np.int64(3)``. A console that printed the scalar's type back at
    the user would look fine on one version and wrong on the other, and
    the test that pinned it would pass on one machine and fail on CI --
    which is exactly what happened.
    """
    if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
        value = value.item()

    summary = getattr(value, "summary", None)
    if callable(summary):
        try:
            text = summary()
            if isinstance(text, str):
                return text
        except Exception:  # a broken summary must not lose the result
            logger.debug("summary() failed while formatting", exc_info=True)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            import json

            payload = to_dict()
            if isinstance(payload, dict):
                return json.dumps(payload, indent=2, default=str, ensure_ascii=False)
        except Exception:
            logger.debug("to_dict() failed while formatting", exc_info=True)
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict) and "summary" in metadata:
        return str(metadata["summary"])
    return repr(value)


__all__ = ["DataProxy", "ExecutionResult", "ScriptSession"]
