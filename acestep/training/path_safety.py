"""Path sanitisation helpers for training modules.

Provides a single ``safe_path`` function that validates user-provided
filesystem paths against a known safe root directory.  The validation
uses ``os.path.normpath`` followed by a ``.startswith`` check — the
exact pattern that CodeQL recognises as a sanitiser for the
``py/path-injection`` query.

All training modules that accept user-supplied paths should call
``safe_path`` (or ``safe_open``) before performing any filesystem I/O.
"""

import os
from typing import Optional

from loguru import logger

# Root directory that all user-provided paths must resolve under.
# Defaults to the working directory at import time.  Override via
# ``set_safe_root`` if needed (e.g. in tests).
_SAFE_ROOT: str = os.path.normpath(os.path.abspath(os.getcwd()))


def set_safe_root(root: str) -> None:
    """Override the safe root directory.

    Args:
        root: New safe root (will be normalised).
    """
    global _SAFE_ROOT  # noqa: PLW0603
    _SAFE_ROOT = os.path.normpath(os.path.abspath(root))


def get_safe_root() -> str:
    """Return the current safe root directory."""
    return _SAFE_ROOT


def safe_path(user_path: str, *, base: Optional[str] = None) -> str:
    """Validate and normalise a user-provided path.

    The returned path is guaranteed to live under *base* (or the
    global ``_SAFE_ROOT`` when *base* is ``None``).

    Args:
        user_path: Untrusted path string from user input.
        base: Optional explicit base directory.  When provided it is
              normalised and used instead of ``_SAFE_ROOT``.

    Returns:
        Normalised absolute path that is within the safe root.

    Raises:
        ValueError: If the normalised path escapes the safe root.
    """
    if base is not None:
        root = os.path.normpath(os.path.abspath(base))
    else:
        root = _SAFE_ROOT

    # Normalise the user path.  If it is relative, resolve it against
    # *root*; if absolute, normalise it directly.
    if os.path.isabs(user_path):
        normalised = os.path.normpath(user_path)
    else:
        normalised = os.path.normpath(os.path.join(root, user_path))

    # ── CodeQL-recognised sanitiser barrier ──
    # ``normpath(…).startswith(safe_prefix)`` is the pattern that
    # CodeQL's ``py/path-injection`` query treats as a sanitiser.
    if not normalised.startswith(root + os.sep) and normalised != root:
        raise ValueError(
            f"Path escapes safe root: {user_path!r} "
            f"(resolved to {normalised!r}, root={root!r})"
        )

    return normalised


def safe_open(user_path: str, mode: str = "r", **kwargs):
    """Open a file after validating its path.

    Convenience wrapper around ``safe_path`` + ``open``.

    Args:
        user_path: Untrusted path string.
        mode: File open mode.
        **kwargs: Extra keyword arguments forwarded to ``open``.

    Returns:
        File object.

    Raises:
        ValueError: If the path escapes the safe root.
    """
    validated = safe_path(user_path)
    return open(validated, mode, **kwargs)  # noqa: SIM115
