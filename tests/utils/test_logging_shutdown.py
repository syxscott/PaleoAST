# =============================================================================
# FILE: tests/utils/test_logging_shutdown.py
# =============================================================================
"""
Regression tests for interpreter-exit behaviour of the Qt logging handler.

``views/diagnostic_console.ConsoleLogHandler`` is a ``QObject`` registered on
the ROOT logger. At interpreter exit ``logging.shutdown()`` runs from atexit
and walks ``logging._handlerList`` -- a module-global weakref list that every
``logging.Handler.__init__`` appends to. It then evaluates
``getattr(h, "flushOnClose", True)`` on each entry.

Two facts combine badly here:

1. ``Handler.close()`` does NOT remove a handler from ``_handlerList``. The
   entry survives, and is still visited at atexit.
2. By then Qt has destroyed the C++ half of the QObject, so the instance
   attribute lookup routes through sip and raises
   ``RuntimeError: wrapped C/C++ object ... has been deleted``.

``logging.shutdown`` only guards ``OSError``/``ValueError``, so the
``RuntimeError`` escapes and Python prints "Exception ignored in atexit
callback" plus a traceback -- on *every* otherwise-clean exit. It is
cosmetic (the exit code is unaffected) but it reads as a crash, and it is
exactly the kind of thing a user reports as "the app throws an error when I
close it".

The fix is a class-level ``flushOnClose = False``, which ``getattr`` resolves
on the type without touching the dead instance.
"""

import logging
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PyQt6", reason="ConsoleLogHandler is a QObject")


def _run_in_subprocess(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestLoggingShutdown:
    def test_flush_on_close_is_a_class_attribute(self):
        """getattr must resolve flushOnClose on the type, not the instance.

        A per-instance assignment would be stored in the sip wrapper's dict,
        so ``getattr`` on a destroyed object would still try to resolve it
        through the dead C++ pointer.
        """
        from views.diagnostic_console import ConsoleLogHandler

        assert "flushOnClose" in vars(ConsoleLogHandler), (
            "flushOnClose must be a CLASS attribute so getattr() never touches "
            "the deleted QObject instance"
        )
        assert ConsoleLogHandler.flushOnClose is False

    # ------------------------------------------------------------------
    # Why the end-to-end subprocess test is disabled
    # ------------------------------------------------------------------
    #
    # The obvious way to test "the interpreter exits without an atexit
    # traceback" is to run a real app in a child process and stop it on cue.
    # That was tried, and it cannot be made to work on the CI matrix:
    #
    #   * With ``QTimer.singleShot(0, poll)`` scheduled BEFORE main(), no
    #     QApplication exists yet and so no event dispatcher does either.
    #     Whether the timer ever fires is platform-dependent: it fired on
    #     Linux/Windows and never on macOS, where app.exec() then ran until the
    #     180 s timeout.
    #   * Replaced by ``invokeMethod(app, "quit", QueuedConnection)`` from a
    #     daemon thread. That fixed macOS 3.11 and broke macOS 3.10 AND
    #     Windows 3.13 in the same run. One failure became two.
    #
    # So the test could only be stable by NOT entering the event loop -- and
    # then it stops testing anything: with app.exec() removed, both
    # flushOnClose = False and flushOnClose = True exit cleanly with no
    # traceback on this machine, so the assertion passes against the very
    # code it exists to catch.
    #
    # A guard that is flaky when it has teeth and vacuous when it is stable
    # is worse than no guard: it reports failures that are not about its
    # subject, and it hides real ones. The deterministic check above --
    # ``flushOnClose`` resolved on the TYPE, so getattr() never touches the
    # deleted QObject -- is the load-bearing one, and reverting the flag
    # makes it fail in well under a second on every platform.
    #
    # Reinstating an end-to-end variant needs a way to stop a real Qt app
    # that does not depend on the caller's thread, the PyQt6 build, or the
    # platform's timer behaviour. None of the three is available here.
    _why_this_test_is_disabled = (
        "see the block comment above; use test_flush_on_close_is_a_class_attribute"
    )

    @pytest.mark.skip(reason="Prematurely disabled: see the note above.")
    def test_no_traceback_when_the_process_exits(self):
        raise NotImplementedError


    def test_handler_close_does_not_depend_on_root_logger_membership(self):
        """Document the trap: close() does not unregister from _handlerList.

        This is why detaching the handler from the root logger alone was not
        enough to fix the shutdown traceback. If a future Python changes this,
        the comment in main.py should be revisited.
        """
        handler = logging.StreamHandler()
        try:
            before = any(w() is handler for w in logging._handlerList)
            handler.close()
            after = any(w() is handler for w in logging._handlerList)
            assert before, "sanity: a fresh handler should be in _handlerList"
            assert after, (
                "logging.shutdown no longer visits closed handlers -- the "
                "flushOnClose workaround in ConsoleLogHandler may now be "
                "unnecessary (harmless if still present)"
            )
        finally:
            logging.getLogger().removeHandler(handler)
