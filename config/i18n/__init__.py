"""
PaleoAST Internationalization Module

Provides Chinese/English language switching for the application.

Usage:
    from config.i18n import _

    # Simple translation
    label = _("PCA Scores Plot")

    # With format arguments
    label = _("Method: {0}").format(method_name)

    # Language switching
    from config.i18n import get_translator
    get_translator().set_language("zh")

Author: PaleoAST Development Team
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

# PyQt6 imports - must be available (i18n initialized after QApplication)
try:
    from PyQt6.QtCore import QObject, pyqtSignal

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


class _TranslatorBase:
    """Base translator without Qt dependency."""

    def __init__(self):
        self._lock = threading.RLock()
        self._current_lang: str = "en"
        self._dictionaries: dict[str, dict[str, str]] = {}

    def register_dictionary(self, lang: str, dictionary: dict[str, str]) -> None:
        with self._lock:
            self._dictionaries[lang] = dictionary

    def set_language(self, lang: str) -> None:
        with self._lock:
            self._current_lang = lang

    def get_language(self) -> str:
        with self._lock:
            return self._current_lang

    def translate(self, key: str, *args) -> str:
        with self._lock:
            d = self._dictionaries.get(self._current_lang, {})
            text = d.get(key, key)
            if args:
                return text.format(*args)
            return text


if _HAS_QT:

    class _Translator(QObject, _TranslatorBase):
        """Qt-enabled translator with language change signal.

        QObject initialization is deferred until a QApplication exists,
        so the singleton can be created early (e.g. at import time)
        without crashing.
        """

        language_changed = pyqtSignal(str)

        def __init__(self):
            _TranslatorBase.__init__(self)
            self._qt_init_done = False
            self._try_init_qt()

        def _try_init_qt(self) -> None:
            if self._qt_init_done:
                return
            try:
                from PyQt6.QtWidgets import QApplication

                if QApplication.instance() is not None:
                    QObject.__init__(self)
                    self._qt_init_done = True
            except Exception:
                pass

        def set_language(self, lang: str) -> None:
            self._try_init_qt()
            with self._lock:
                old = self._current_lang
                self._current_lang = lang
            if old != lang and self._qt_init_done:
                self.language_changed.emit(lang)
else:

    class _Translator(_TranslatorBase):
        pass


# Module-level singleton
_translator: _Translator | None = None
_init_lock = threading.Lock()


def _get_or_create_translator() -> _Translator:
    global _translator
    if _translator is None:
        with _init_lock:
            if _translator is None:
                _translator = _Translator()
    return _translator


def _reset_translator() -> None:
    """Reset the singleton so it re-initializes with Qt on next access.

    Called after QApplication is created to ensure the translator
    picks up the QObject base class.
    """
    global _translator
    with _init_lock:
        _translator = None


def get_translator() -> _Translator:
    """Get the singleton translator instance."""
    return _get_or_create_translator()


def _(key: str, *args) -> str:
    """
    Translate a key to the current language.

    Args:
        key: English text to translate (also serves as fallback)
        *args: Positional format arguments

    Returns:
        Translated text, or the key itself if no translation found.

    Examples:
        >>> _("PCA Scores Plot")
        'PCA Scores Plot'  # (English)
        >>> get_translator().set_language("zh")
        >>> _("PCA Scores Plot")
        'PCA 得分图'  # (Chinese)
        >>> _("Method: {0}").format("correlation")
        'Method: correlation'
    """
    return _get_or_create_translator().translate(key, *args)


def get_language() -> str:
    """Get the current language code."""
    return _get_or_create_translator().get_language()


def set_language(lang: str) -> None:
    """Set the current language and emit change signal."""
    _get_or_create_translator().set_language(lang)


def register_translations() -> None:
    """Register built-in translation dictionaries."""
    from .translations_en import TRANSLATIONS as en_dict
    from .translations_zh import TRANSLATIONS as zh_dict

    translator = _get_or_create_translator()
    translator.register_dictionary("en", en_dict)
    translator.register_dictionary("zh", zh_dict)
