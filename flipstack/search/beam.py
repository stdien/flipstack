"""Beam search for pancake sorting.

Expands states by filtered moves, scores children, keeps top-k.
Supports simple and iterated modes with dedup and gc control.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.heuristics.composite import composite_h
from flipstack.search.move_filter import all_moves, filter_moves

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@contextmanager
def _gc_disabled() -> Generator[None]:
    """Temporarily disable garbage collection for performance."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


def beam_search(  # noqa: C901
    perm: np.ndarray,
    beam_width: int = 1024,
    max_steps: int = 200,
    max_moves: int = 3,
    use_filter: bool = True,
    scorer: Callable[[np.ndarray], float] | None = None,
    batch_scorer: Callable[[np.ndarray], np.ndarray] | None = None,
    timeout: float = 300.0,
) -> list[int] | None:
    """Beam search to find a flip sequence sorting perm to identity.

    Args:
        perm: Starting permutation (int8).
        beam_width: Number of states to keep per step.
        max_steps: Maximum number of flip steps.
        max_moves: Moves per state when filtering (ignored if use_filter=False).
        use_filter: If True, use move filtering; otherwise expand all n-1 moves.
        scorer: Per-state scoring function (lower = better). Defaults to composite_h.
        batch_scorer: Batch scoring function (2D array -> 1D scores). If provided,
            used instead of scorer for efficiency.
        timeout: Maximum wall time in seconds.

    Returns:
        List of flip sizes that sort perm, or None if not found.
    """
    if is_sorted(perm):
        return []

    if scorer is None and batch_scorer is None:
        scorer = composite_h

    n = len(perm)
    start_time = time.monotonic()

    initial_bytes = perm.tobytes()
    beam: list[tuple[np.ndarray, list[int]]] = [(perm.copy(), [])]
    visited: dict[bytes, int] = {initial_bytes: 0}

    with _gc_disabled():
        for _step in range(max_steps):
            if time.monotonic() - start_time > timeout:
                break

            # Expand all candidates
            children: list[np.ndarray] = []
            children_flips: list[list[int]] = []

            for state, flips in beam:
                moves = filter_moves(state, max_moves) if use_filter else all_moves(n)

                for k in moves:
                    child = apply_flip(state, k)
                    child_flips = [*flips, k]

                    if is_sorted(child):
                        return child_flips

                    child_bytes = child.tobytes()
                    depth = len(child_flips)

                    # Dedup: keep only if new or shorter path
                    prev_depth = visited.get(child_bytes)
                    if prev_depth is not None and prev_depth <= depth:
                        continue
                    visited[child_bytes] = depth

                    children.append(child)
                    children_flips.append(child_flips)

            if not children:
                break

            # Score all candidates at once
            if batch_scorer is not None:
                batch = np.array(children)
                scores = batch_scorer(batch)
                candidates = list(zip(scores.tolist(), children, children_flips, strict=False))
            elif scorer is not None:
                candidates = [(scorer(c), c, f) for c, f in zip(children, children_flips, strict=False)]
            else:
                candidates = [(composite_h(c), c, f) for c, f in zip(children, children_flips, strict=False)]

            # Keep top beam_width by score (lower is better)
            candidates.sort(key=lambda x: x[0])
            beam = [(c[1], c[2]) for c in candidates[:beam_width]]

            # Evict old visited entries if map grows too large
            if len(visited) > 10_000_000:
                visited.clear()
                for state, flips in beam:
                    visited[state.tobytes()] = len(flips)

    # Fallback: retry with full moves at reduced beam width
    if use_filter:
        fallback_width = max(64, beam_width // 4)
        remaining_time = timeout - (time.monotonic() - start_time)
        if remaining_time > 1.0:
            return beam_search(
                perm,
                beam_width=fallback_width,
                max_steps=max_steps,
                use_filter=False,
                scorer=scorer,
                batch_scorer=batch_scorer,
                timeout=remaining_time,
            )

    return None


def iterated_beam_search(
    perm: np.ndarray,
    beam_widths: list[int] | None = None,
    max_steps: int = 200,
    max_moves: int = 3,
    scorer: Callable[[np.ndarray], float] | None = None,
    batch_scorer: Callable[[np.ndarray], np.ndarray] | None = None,
    timeout: float = 300.0,
) -> list[int] | None:
    """Iterated beam search with increasing beam widths.

    Args:
        perm: Starting permutation.
        beam_widths: List of beam widths to try (default: powers of 2).
        max_steps: Maximum steps per attempt.
        max_moves: Moves per state when filtering.
        scorer: Per-state scoring function.
        batch_scorer: Batch scoring function for efficiency.
        timeout: Total timeout across all attempts.

    Returns:
        Shortest solution found, or None.
    """
    if beam_widths is None:
        beam_widths = [64, 256, 1024, 4096]

    start_time = time.monotonic()
    best: list[int] | None = None

    for bw in beam_widths:
        remaining = timeout - (time.monotonic() - start_time)
        if remaining < 1.0:
            break
        result = beam_search(
            perm,
            beam_width=bw,
            max_steps=max_steps,
            max_moves=max_moves,
            scorer=scorer,
            batch_scorer=batch_scorer,
            timeout=remaining,
        )
        if result is not None and (best is None or len(result) < len(best)):
            best = result

    return best
