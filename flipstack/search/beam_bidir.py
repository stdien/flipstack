"""Bidirectional beam search for pancake sorting.

Forward beam from start permutation, backward beam from identity.
Meet-in-the-middle via hash intersection.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.heuristics.composite import composite_h
from flipstack.search.move_filter import all_moves, filter_moves

if TYPE_CHECKING:
    from collections.abc import Callable


def bidir_beam_search(
    perm: np.ndarray,
    beam_width: int = 1024,
    max_steps: int = 100,
    max_moves: int = 3,
    use_filter: bool = True,
    scorer: Callable[[np.ndarray], float] | None = None,
    timeout: float = 300.0,
) -> list[int] | None:
    """Bidirectional beam search.

    Forward beam expands from perm toward identity.
    Backward beam expands from identity toward perm.
    When beams intersect, reconstruct the combined path.

    Args:
        perm: Starting permutation (int8).
        beam_width: States per beam per step.
        max_steps: Max steps per direction.
        max_moves: Moves when filtering.
        use_filter: Whether to use move filtering.
        scorer: Scoring function (lower = better).
        timeout: Wall time limit.

    Returns:
        Flip sequence sorting perm, or None.
    """
    if is_sorted(perm):
        return []

    if scorer is None:
        scorer = composite_h

    n = len(perm)
    start_time = time.monotonic()

    # Forward: perm -> identity
    fwd_beam: list[tuple[np.ndarray, list[int]]] = [(perm.copy(), [])]
    fwd_visited: dict[bytes, list[int]] = {perm.tobytes(): []}

    # Backward: identity -> perm (we expand from identity)
    identity = np.arange(n, dtype=np.int8)
    bwd_beam: list[tuple[np.ndarray, list[int]]] = [(identity.copy(), [])]
    bwd_visited: dict[bytes, list[int]] = {identity.tobytes(): []}

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _step in range(max_steps):
            if time.monotonic() - start_time > timeout:
                break

            # Expand forward beam
            fwd_beam, fwd_visited, fwd_progress = _expand_beam(
                fwd_beam, fwd_visited, n, scorer, beam_width, max_moves, use_filter
            )

            # Check intersection
            result = _check_intersection(fwd_visited, bwd_visited)
            if result is not None:
                return result

            if time.monotonic() - start_time > timeout:
                break

            # Expand backward beam
            bwd_beam, bwd_visited, bwd_progress = _expand_beam(
                bwd_beam, bwd_visited, n, scorer, beam_width, max_moves, use_filter
            )

            # Check intersection again
            result = _check_intersection(fwd_visited, bwd_visited)
            if result is not None:
                return result

            # Stop if neither direction made progress
            if not fwd_progress and not bwd_progress:
                break

    finally:
        if gc_was_enabled:
            gc.enable()

    return None


def _expand_beam(
    beam: list[tuple[np.ndarray, list[int]]],
    visited: dict[bytes, list[int]],
    n: int,
    scorer: Callable[[np.ndarray], float],
    beam_width: int,
    max_moves: int,
    use_filter: bool,
) -> tuple[list[tuple[np.ndarray, list[int]]], dict[bytes, list[int]], bool]:
    """Expand a beam by one step.

    Args:
        beam: Current beam states.
        visited: State hash -> flip path.
        n: Permutation size.
        scorer: Scoring function.
        beam_width: Max states to keep.
        max_moves: Moves per state.
        use_filter: Whether to filter moves.

    Returns:
        Tuple of (beam, visited, made_progress).
    """
    candidates: list[tuple[float, np.ndarray, list[int]]] = []

    for state, flips in beam:
        moves = filter_moves(state, max_moves) if use_filter else all_moves(n)
        for k in moves:
            child = apply_flip(state, k)
            child_bytes = child.tobytes()
            child_flips = [*flips, k]

            if child_bytes in visited and len(visited[child_bytes]) <= len(child_flips):
                continue
            visited[child_bytes] = child_flips

            score = scorer(child)
            candidates.append((score, child, child_flips))

    if not candidates:
        return beam, visited, False

    candidates.sort(key=lambda x: x[0])
    new_beam = [(c[1], c[2]) for c in candidates[:beam_width]]
    return new_beam, visited, True


def _check_intersection(
    fwd_visited: dict[bytes, list[int]],
    bwd_visited: dict[bytes, list[int]],
) -> list[int] | None:
    """Check if forward and backward beams have met.

    Args:
        fwd_visited: Forward beam visited states.
        bwd_visited: Backward beam visited states.

    Returns:
        Combined flip sequence, or None.
    """
    best: list[int] | None = None
    best_len = float("inf")

    # Check smaller set against larger
    smaller, larger = (fwd_visited, bwd_visited) if len(fwd_visited) < len(bwd_visited) else (bwd_visited, fwd_visited)
    is_fwd_smaller = len(fwd_visited) <= len(bwd_visited)

    for state_bytes in smaller:
        if state_bytes in larger:
            if is_fwd_smaller:
                fwd_flips = fwd_visited[state_bytes]
                bwd_flips = bwd_visited[state_bytes]
            else:
                fwd_flips = fwd_visited[state_bytes]
                bwd_flips = bwd_visited[state_bytes]
            # Forward flips get us from start to meeting point
            # Backward flips get us from identity to meeting point
            # Combined: forward_flips + reversed(backward_flips)
            combined = fwd_flips + list(reversed(bwd_flips))
            if len(combined) < best_len:
                best = combined
                best_len = len(combined)

    return best
