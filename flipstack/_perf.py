"""Performance utilities: gc control, timing, memory tracking."""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def gc_disabled() -> Generator[None]:
    """Temporarily disable garbage collection for performance."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


@contextmanager
def timer(label: str = "") -> Generator[dict[str, float]]:
    """Context manager that measures wall time.

    Args:
        label: Optional label for logging.

    Yields:
        Dict with 'elapsed' key updated on exit.
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.monotonic()
    try:
        yield result
    finally:
        result["elapsed"] = time.monotonic() - start
