"""Lock detection heuristic: gap_h + 1 when no gap-decreasing move exists."""

from __future__ import annotations

import numpy as np

from flipstack.core.permutation import apply_flip
from flipstack.heuristics.gap import gap_h, gap_h_batch


def ld_h(perm: np.ndarray) -> int:
    """Lock detection heuristic.

    If there exists a flip that decreases the gap count, returns gap_h(perm).
    Otherwise (locked state), returns gap_h(perm) + 1.

    Args:
        perm: Permutation array.

    Returns:
        Heuristic value (admissible lower bound).
    """
    g = gap_h(perm)
    if g == 0:
        return 0
    n = len(perm)
    for k in range(2, n + 1):
        child = apply_flip(perm, k)
        if gap_h(child) < g:
            return g
    return g + 1


def ld_h_batch(perms: np.ndarray) -> np.ndarray:
    """Lock detection heuristic for a batch.

    Args:
        perms: 2D array of shape (batch, n).

    Returns:
        1D int array of heuristic values.
    """
    batch_size, n = perms.shape
    gaps = gap_h_batch(perms)
    result = gaps.copy()
    for i in range(batch_size):
        if gaps[i] == 0:
            continue
        locked = True
        for k in range(2, n + 1):
            child = apply_flip(perms[i], k)
            if gap_h(child) < gaps[i]:
                locked = False
                break
        if locked:
            result[i] = gaps[i] + 1
    return result
