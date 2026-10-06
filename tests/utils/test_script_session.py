# =============================================================================
# FILE: tests/utils/test_script_session.py
# =============================================================================
"""
Tests for the script console's execution session.

No QApplication is involved: ScriptSession is deliberately free of Qt so
that the part worth testing can be tested on a headless runner. What is
asserted here is the behaviour a user of the console depends on -- the
namespace survives, a traceback does not end the session, and the value
of a trailing expression is echoed the way a REPL echoes it.

The run() tests go through a real controller and a real registry on
purpose. A stub registry would pass whether or not the catalog can
actually be filled, and filling the catalog from an empty start is the
one thing this console depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.script_session import (
    DataProxy,
    ExecutionResult,
    ScriptSession,
    _format_result,
)


@pytest.fixture
def session() -> ScriptSession:
    """A session with no application attached."""
    return ScriptSession()


@pytest.fixture(scope="module")
def wired():
    """A session bound to a real controller and a real data provider."""
    from controllers.statistics_controller import StatisticsController

    controller = StatisticsController()
    data = np.random.default_rng(0).normal(size=(20, 4))
    return ScriptSession(data_provider=lambda: data, controller=controller)


# ---------------------------------------------------------------------------
# Namespace behaviour
# ---------------------------------------------------------------------------


def test_empty_input_is_accepted(session: ScriptSession) -> None:
    """Pressing Run on a blank editor is not an error."""
    result = session.execute("   \n  ")
    assert result.ok
    assert result.value is None
    assert result.error is None


def test_assignment_defines_a_variable(session: ScriptSession) -> None:
    """The basic case."""
    result = session.execute("x = 6 * 7")
    assert result.ok
    assert session.namespace["x"] == 42


def test_trailing_expression_is_echoed(session: ScriptSession) -> None:
    """What a REPL does, and what makes a console usable."""
    session.execute("x = 6 * 7")
    result = session.execute("x")
    assert result.value == "42"


def test_trailing_expression_survives_a_stateless_prefix(
    session: ScriptSession,
) -> None:
    """A loop that ends in an expression still echoes it."""
    result = session.execute("total = 0\nfor i in range(5):\n    total += i\ntotal")
    assert result.ok
    assert result.value == "10"


def test_assignment_does_not_echo(session: ScriptSession) -> None:
    """An assignment has no value; echoing None would be noise."""
    assert session.execute("y = 1").value is None


def test_print_goes_to_output_not_value(session: ScriptSession) -> None:
    """The two channels are separate, so a caller can show them apart."""
    result = session.execute("print('hello')")
    assert result.output.strip() == "hello"
    assert result.value is None


def test_print_and_echo_together(session: ScriptSession) -> None:
    """Both channels populate in one block."""
    result = session.execute("z = 3\nprint('side effect')\nz * 2")
    assert result.output.strip() == "side effect"
    assert result.value == "6"


def test_namespace_persists_across_calls(session: ScriptSession) -> None:
    """The second block sees the first."""
    session.execute("a = 1")
    session.execute("b = a + 1")
    assert session.execute("b").value == "2"


def test_error_does_not_end_the_session(session: ScriptSession) -> None:
    """A typo on line 40 must not discard lines 1-39.

    This is the behaviour that makes a console usable for a long script:
    a REPL that resets on every exception is a calculator.
    """
    session.execute("kept = 'still here'")
    result = session.execute("boom = 1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in (result.error or "")
    assert session.execute("kept").value == "'still here'"


def test_output_before_an_error_is_kept(session: ScriptSession) -> None:
    """Partial output survives a later failure."""
    result = session.execute("print('before')\n1 / 0")
    assert not result.ok
    assert "before" in result.output
    assert "ZeroDivisionError" in (result.error or "")


def test_syntax_error_is_reported_not_raised(session: ScriptSession) -> None:
    """Malformed source must come back as a result."""
    result = session.execute("def (:")
    assert not result.ok
    assert "SyntaxError" in (result.error or "")


def test_unclosed_quote_does_not_wedge_the_session(session: ScriptSession) -> None:
    """An unterminated string is a syntax error, and the next block works."""
    assert not session.execute("s = 'unterminated").ok
    assert session.execute("1 + 1").value == "2"


def test_indented_block_with_trailing_expression(session: ScriptSession) -> None:
    """A function definition followed by a call still echoes the call."""
    result = session.execute("def double(v):\n    return v * 2\ndouble(21)")
    assert result.ok
    assert result.value == "42"


def test_semicolon_separated_tail_runs_without_an_echo(
    session: ScriptSession,
) -> None:
    """A trailing expression sharing a line is run, not split.

    Splitting "p = 2; p ** 5" by line would drop the assignment and fail
    with a NameError, so the block executes whole and prints nothing
    extra. The assertion is that the assignment survived and the
    expression was evaluated.
    """
    result = session.execute("p = 2; p ** 5")
    assert result.ok
    assert session.namespace["p"] == 2
    assert result.value is None


def test_comment_only_input(session: ScriptSession) -> None:
    """A comment is not an expression and must not be echoed."""
    result = session.execute("# just a note")
    assert result.ok
    assert result.value is None


# ---------------------------------------------------------------------------
# The injected namespace
# ---------------------------------------------------------------------------


def test_helpers_are_present(session: ScriptSession) -> None:
    """The conveniences the docstring promises."""
    for name in ("np", "analyses", "run", "categories", "session"):
        assert name in session.namespace, f"{name} missing from the namespace"
    assert session.namespace["np"] is np


def test_numpy_is_usable(session: ScriptSession) -> None:
    """np is the same numpy, not a copy."""
    assert session.execute("np.arange(3).sum()").value == "3"


def test_reset_clears_user_names_but_keeps_helpers(
    session: ScriptSession,
) -> None:
    """Reset is a namespace reset, not a session teardown."""
    session.execute("mine = 1")
    assert "mine" in session._public_keys()
    session.reset()
    assert "mine" not in session.namespace
    assert "run" in session.namespace
    assert "np" in session.namespace


def test_public_keys_excludes_injected_names(session: ScriptSession) -> None:
    """The key list is what a user typed, not what we injected."""
    session.execute("a = 1\nb = 2")
    assert session._public_keys() == ["a", "b"]


def test_a_raising_data_provider_yields_none() -> None:
    """A provider that throws must not kill the console."""

    def boom() -> object:
        raise RuntimeError("no data")

    session = ScriptSession(data_provider=boom)
    assert session.data is None
    assert session.execute("1").ok


# ---------------------------------------------------------------------------
# The live data proxy
# ---------------------------------------------------------------------------


@pytest.fixture
def sheet() -> dict:
    """A mutable stand-in for the spreadsheet."""
    return {"v": np.arange(12, dtype=float).reshape(3, 4)}


@pytest.fixture
def sheet_session(sheet: dict) -> ScriptSession:
    """A session reading from the mutable sheet."""
    return ScriptSession(data_provider=lambda: sheet["v"])


def test_data_is_in_the_namespace(sheet_session: ScriptSession) -> None:
    """The docstring promises it; it used not to be there."""
    assert "data" in sheet_session.namespace
    assert isinstance(sheet_session.namespace["data"], DataProxy)


def test_data_supports_the_ordinary_array_operations(
    sheet_session: ScriptSession,
) -> None:
    """What a script actually does with it."""
    assert sheet_session.execute("data.shape").value == "(3, 4)"
    assert sheet_session.execute("len(data)").value == "3"
    assert sheet_session.execute("data.mean()").value == "5.5"
    assert sheet_session.execute("np.corrcoef(data.T).shape").value == "(4, 4)"
    assert sheet_session.execute("data[0, 2]").value == "2.0"


def test_data_is_live_not_a_snapshot(sheet_session: ScriptSession, sheet: dict) -> None:
    """The property that justifies the proxy's existence.

    A snapshot would report the array the sheet held when the console
    opened, and the script would silently analyse numbers the user has
    already replaced.
    """
    before = sheet_session.execute("data.sum()").value
    sheet["v"] = np.ones((3, 4)) * 100
    after = sheet_session.execute("data.sum()").value
    assert before == "66.0"
    assert after == "1200.0"


def test_snapshot_materialises_a_fixed_array(sheet_session: ScriptSession, sheet: dict) -> None:
    """An explicit request for a fixed array gets one."""
    fixed = sheet_session.namespace["data"].snapshot()
    assert isinstance(fixed, np.ndarray)
    sheet["v"] = np.zeros((3, 4))
    assert fixed.sum() == 66.0


def test_data_proxy_unwraps_a_data_matrix(sheet: dict) -> None:
    """A DataMatrix is handed over as its array.

    run() and the proxy have to agree, or a script would pass a wrapper
    to one analysis and an ndarray to another.
    """

    class FakeMatrix:
        def __init__(self) -> None:
            self.data = np.ones((2, 3))
            self.raw_data = self.data

    session = ScriptSession(data_provider=lambda: FakeMatrix())
    assert session.execute("data.shape").value == "(2, 3)"
    assert isinstance(session.data, np.ndarray)


def test_empty_data_degrades_with_a_readable_message() -> None:
    """Nothing loaded is a state, not a crash."""
    session = ScriptSession()
    proxy = session.namespace["data"]
    assert "nothing loaded" in repr(proxy)
    assert bool(proxy) is False
    assert len(proxy) == 0
    result = session.execute("data.shape")
    assert not result.ok
    assert "no data is loaded" in (result.error or "")


def test_data_proxy_is_excluded_from_user_keys(sheet_session: ScriptSession) -> None:
    """It is injected, not typed by the user."""
    sheet_session.execute("mine = 1")
    assert sheet_session._public_keys() == ["mine"]


def test_reset_restores_the_data_proxy(sheet_session: ScriptSession) -> None:
    """Reset must not lose the binding."""
    sheet_session.execute("mine = 1")
    sheet_session.reset()
    assert isinstance(sheet_session.namespace["data"], DataProxy)
    assert sheet_session.execute("data.shape").value == "(3, 4)"


# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------


def test_analyses_fills_and_lists_the_catalog(wired: ScriptSession) -> None:
    """analyses() must work on a session that has never run anything.

    That is the whole point of a discovery helper, and it is the case
    that fails if registration only happens as a side effect of running
    an analysis.
    """
    names = wired.analyses()
    assert len(names) > 50
    for expected in ("mantel", "dca", "kmeans", "pca", "two_way_anova"):
        assert expected in names
    assert names == sorted(names)


def test_analyses_can_filter_by_category(wired: ScriptSession) -> None:
    """Category filtering agrees with the unfiltered list."""
    everything = set(wired.analyses())
    seen: set[str] = set()
    for category in wired.categories():
        subset = set(wired.analyses(category=category))
        assert subset
        seen.update(subset)
    assert seen == everything


def test_categories_are_non_empty(wired: ScriptSession) -> None:
    """The catalog carries real categories."""
    categories = wired.categories()
    assert categories
    assert all(c.strip() for c in categories)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_executes_a_real_analysis(wired: ScriptSession) -> None:
    """run() reaches a real analyzer and formats it for a human."""
    text = wired.run("mantel", n_permutations=99, random_seed=1, show=False)
    assert "Mantel" in text
    # Formatted through summary(), not a dataclass repr.
    assert "MantelResult(" not in text


def test_run_uses_the_current_data(wired: ScriptSession) -> None:
    """The default payload is the spreadsheet, not None."""
    text = wired.run("pca", n_components=2, show=False)
    assert text.strip()


def test_run_accepts_an_explicit_payload(wired: ScriptSession) -> None:
    """An override is passed straight through."""
    other = np.random.default_rng(1).normal(size=(16, 3))
    text = wired.run("kmeans", data=other, n_clusters=2, random_seed=0, show=False)
    assert "K-Means" in text


def test_run_reports_an_unknown_name_without_raising(
    wired: ScriptSession,
) -> None:
    """A typo lists what is available instead of a traceback."""
    text = wired.run("not_an_analysis", show=False)
    assert "No analysis named" in text
    assert "Available" in text


def test_run_surfaces_the_analysis_error(wired: ScriptSession) -> None:
    """A validation failure is rendered, not swallowed.

    run() returns the traceback text so the user sees which parameter
    was wrong; swallowing it would make the console look like the
    analysis is broken.
    """
    text = wired.run("kmeans", data=np.ones((6, 2)), n_clusters=99, show=False)
    assert "Error" in text or "error" in text


def test_run_can_print(wired: ScriptSession, capsys) -> None:
    """show=True echoes to stdout as well as returning."""
    text = wired.run("mantel", n_permutations=49, random_seed=1, show=True)
    captured = capsys.readouterr()
    assert "Mantel" in captured.out
    assert "Mantel" in text


def test_a_detached_session_still_sees_the_process_registry() -> None:
    """The registry is a process singleton, so a detached session is not blind.

    Whether a controller has filled it is process state, not session
    state -- which is why a session constructed without a controller can
    still list analyses once anything has registered them. The
    guarantee a detached session can make is that it never raises.
    """
    detached = ScriptSession()
    names = detached.analyses()
    assert isinstance(names, list)
    assert detached.categories() == sorted(detached.categories())
    # And the fallback path when nothing has registered still degrades
    # quietly rather than raising.
    assert "No analysis named" in detached.run("definitely_not_registered", show=False)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def test_format_prefers_summary() -> None:
    """Every analyzer writes a summary for display; use it."""

    class WithSummary:
        def summary(self) -> str:
            return "the summary"

        def to_dict(self) -> dict:
            return {"unused": True}

    assert _format_result(WithSummary()) == "the summary"


def test_format_falls_back_to_to_dict() -> None:
    """No summary, but a dict is still better than a repr."""

    class WithDict:
        def to_dict(self) -> dict:
            return {"a": 1}

    text = _format_result(WithDict())
    assert '"a": 1' in text


def test_format_survives_a_broken_summary() -> None:
    """A summary that raises must not lose the result."""

    class Broken:
        def summary(self) -> str:
            raise RuntimeError("bad summary")

        def to_dict(self) -> dict:
            return {"recovered": True}

    text = _format_result(Broken())
    assert "recovered" in text


def test_format_falls_back_to_repr() -> None:
    """A bare object still prints something."""
    assert _format_result([1, 2, 3]) == "[1, 2, 3]"


def test_execution_result_text_combines_channels() -> None:
    """text() is what a console would display."""
    result = ExecutionResult(ok=False, output="printed\n", value="42", error="Traceback...\n")
    text = result.text
    assert "printed" in text
    assert "42" in text
    assert "Traceback" in text


def test_execution_result_serialises() -> None:
    """to_dict stays JSON-friendly."""
    payload = ExecutionResult(ok=True, output="x", value="1").to_dict()
    assert payload["ok"] is True
    assert payload["value"] == "1"
    assert isinstance(payload["namespace_keys"], list)
