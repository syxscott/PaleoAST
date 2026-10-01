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

    def test_no_traceback_when_the_process_exits(self):
        """A real ``main.main()`` session must exit without an atexit traceback.

        This drives the actual production entry point rather than a
        hand-assembled QApplication, because the failure only reproduces when
        the process still holds a live reference to the handler at exit --
        exactly the state a running app is in. A snippet that builds its own
        QApplication and drops the console lets the weakref in
        ``logging._handlerList`` die first, so the bad entry is skipped and
        the test passes against buggy code.
        """
        proc = _run_in_subprocess(
            """
            import os, sys
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            import faulthandler; faulthandler.enable()

            import main
            import PyQt6.QtCore as QtCore

            # main() builds its own QApplication, so poll for the instance
            # before scheduling the quit.
            def _poll():
                app = QtCore.QCoreApplication.instance()
                if app is None:
                    QtCore.QTimer.singleShot(200, _poll)
                    return
                QtCore.QTimer.singleShot(4000, app.quit)

            QtCore.QTimer.singleShot(0, _poll)
            rc = main.main()
            print("SURVIVED rc=%s" % rc)
            """
        )
        assert proc.returncode == 0, (
            f"process died (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        combined = proc.stdout + proc.stderr
        assert "SURVIVED" in combined, f"script did not finish:\n{combined}"
        assert "Exception ignored in atexit callback" not in combined, (
            f"logging.shutdown() still raised at exit:\n{combined}"
        )
        assert "wrapped C/C++ object" not in combined, (
            f"a deleted QObject was touched during shutdown:\n{combined}"
        )

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
