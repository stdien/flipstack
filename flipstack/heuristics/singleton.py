"""Singleton tie-breaking heuristic.

A singleton is an element with gaps on both sides. States with more singletons
give more freedom for subsequent flips.
"""

from __future__ import annotations

import numpy as np

from flipstack.core.permutation import as_compute_dtype


def count_singletons(perm: np.ndarray) -> int:
    """Count singleton elements (gap on both left and right).

    Args:
        perm: Permutation array.

    Returns:
        Number of singletons.
    """
    p = as_compute_dtype(perm)
    n = len(p)
    extended = np.empty(n + 1, dtype=np.int16)
    extended[:n] = p
    extended[n] = n
    diffs = np.abs(np.diff(extended))
    # gap_right[i] = True if perm[i] has gap to its right neighbor
    gap_right = diffs != 1  # shape (n,)
    # gap_left[i] = True if perm[i] has gap to its left neighbor
    # For i=0, there's no left neighbor — not a singleton by convention
    gap_left = np.empty(n, dtype=np.bool_)
    gap_left[0] = False
    gap_left[1:] = gap_right[:-1]
    # Exclude the sentinel position: gap_right has n entries, last is sentinel gap
    # Singleton = gap_left AND gap_right, but only for actual elements (indices 0..n-2)
    # gap_right[n-1] is perm[n-1] vs sentinel n
    return int(np.sum(gap_left & gap_right[:n]))


def singleton_tiebreak(base_score: float, perm: np.ndarray, eps: float = 0.001) -> float:
    """Apply singleton tie-breaking to a base heuristic score.

    Args:
        base_score: Score from the base heuristic.
        perm: Permutation array.
        eps: Weight for singleton bonus (subtracted from score).

    Returns:
        Adjusted score favoring states with more singletons.
    """
    return base_score - eps * count_singletons(perm)
