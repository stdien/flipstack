"""Move filtering for beam search — select promising flips instead of all n-1.

Strategy (from Antonina's approach), in priority order:
1. Flips bringing element with value top±1 to front
2. Flips at longest gap positions
3. Max-length flip (k=n) as last resort
"""

from __future__ import annotations

import numpy as np

from flipstack.core.permutation import as_compute_dtype


def filter_moves(perm: np.ndarray, max_moves: int = 3) -> list[int]:
    """Select promising flip sizes for a permutation.

    Returns at most ``max_moves`` flips, chosen by priority: top±1 element
    flips, gap-position flips, then max-length flip as last resort.

    Args:
        perm: Permutation array.
        max_moves: Strict upper bound on number of moves returned (default 3).

    Returns:
        Sorted list of flip sizes k (2 <= k <= n), deduplicated, 1..max_moves entries.
    """
    p = as_compute_dtype(perm)
    n = len(p)

    if n < 2:
        return []

    # Collect candidates in priority order (duplicates handled by seen set)
    candidates: list[int] = []
    seen: set[int] = set()

    def _add(k: int) -> None:
        if 2 <= k <= n and k not in seen:
            seen.add(k)
            candidates.append(k)

    # 1. Flips bringing elements with value top±1 to front
    top = int(p[0])
    for target in (top - 1, top + 1):
        if 0 <= target < n:
            pos = int(np.argwhere(p == target).item())
            if pos >= 1:
                _add(pos + 1)

    # 2. Fill remaining slots with flips at gap positions (largest gaps first)
    extended = np.empty(n + 1, dtype=np.int16)
    extended[:n] = p
    extended[n] = n
    diffs = np.abs(np.diff(extended))
    gap_positions = np.where(diffs != 1)[0]
    gap_order = gap_positions[np.argsort(-diffs[gap_positions])]
    for pos in gap_order:
        if len(candidates) >= max_moves:
            break
        k = int(pos) + 1
        _add(k)

    # 3. Max-length flip only if we still have room
    if len(candidates) < max_moves:
        _add(n)

    # Guarantee at least 1 valid move, enforce cap
    if not candidates:
        candidates.append(2)

    return sorted(candidates[:max_moves])


def filter_moves_batch(perms: np.ndarray, max_moves: int = 3) -> list[list[int]]:
    """Filter moves for a batch of permutations.

    Args:
        perms: 2D array of shape (batch, n).
        max_moves: Maximum moves per permutation.

    Returns:
        List of move lists, one per permutation.
    """
    return [filter_moves(perms[i], max_moves) for i in range(perms.shape[0])]


def all_moves(n: int) -> list[int]:
    """Return all valid flip sizes for a permutation of length n.

    Args:
        n: Permutation length.

    Returns:
        List [2, 3, ..., n].
    """
    return list(range(2, n + 1))
