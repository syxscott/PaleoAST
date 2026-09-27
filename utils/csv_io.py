"""
CSV text-encoding fallback for PaleoAST.

Windows users overwhelmingly save CSVs as CP936/GBK, not UTF-8. Reading
such a file with ``pd.read_csv(..., encoding="utf-8", errors="replace")``
does **not** raise -- it substitutes U+FFFD and silently destroys every column
header and every text row label. Falling back across candidate encodings is
the only way to notice.

This lives in ``utils`` rather than in a feature package because both the
example-data loader (``data/loader.py``) and the interactive import path
(``controllers/data_controller.py``) need it, and ``utils`` is the only
package every layer already depends on.

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Tried in order. ``utf-8-sig`` also strips a UTF-8 BOM; ``gbk`` covers the
# CP936 files produced by Chinese-locale Excel, which is the overwhelmingly
# common real-world case; ``latin-1`` decodes every possible byte and is
# therefore a true last resort (its contents may be mojibake, hence the
# warning).
CSV_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "gbk", "latin-1")


def read_csv_with_fallback(path: str, **kwargs: Any) -> tuple[pd.DataFrame, str]:
    """Read ``path`` as CSV, trying candidate encodings in order.

    Parameters:
        path: File to read.
        **kwargs: Forwarded verbatim to :func:`pandas.read_csv` (e.g. ``sep``,
            ``header``, ``na_values``, ``index_col``, ``low_memory``).

    Returns:
        ``(dataframe, encoding)``. The encoding is reported so callers can
        surface it; it is the first candidate that decoded without error.

    Raises:
        UnicodeDecodeError: If no candidate encoding could decode the file.
    """
    last_error: Exception | None = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:  # try the next candidate
            last_error = exc
            continue
        if encoding != CSV_ENCODING_CANDIDATES[0]:
            logger.warning(
                "CSV %s could not be read as %s; fell back to encoding '%s'. Non-ASCII text may be misdecoded.",
                path,
                CSV_ENCODING_CANDIDATES[0],
                encoding,
            )
        return df, encoding
    raise (
        last_error
        if last_error is not None
        else UnicodeDecodeError("utf-8", b"", 0, 1, "could not decode CSV with any known encoding")
    )
