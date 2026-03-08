"""Gap heuristic for pancake sorting.

The gap heuristic counts positions where adjacent elements differ by more than 1,
using a right sentinel [perm, n]. A single prefix reversal can remove at most 1 gap,
so the gap count is a lower bound on the number of flips needed (admissible).

Note: Using only a right sentinel (not left) preserves admissibility — validated
via BFS on S_5 through S_8.
"""

from __future__ import annotations

import numpy as np

from flipstack.core.permutation import as_compute_dtype


def gap_h(perm: np.ndarray) -> int:
    """Count gaps in permutation with right sentinel [perm, n].

    A gap exists at position i if |extended[i] - extended[i+1]| != 1.

    Args:
        perm: Permutation array (int8 or int16).

    Returns:
        Number of gaps (lower bound on flip distance).
    """
    p = as_compute_dtype(perm)
    n = len(p)
    extended = np.empty(n + 1, dtype=np.int16)
    extended[:n] = p
    extended[n] = n
    diffs = np.abs(np.diff(extended))
    return int(np.sum(diffs != 1))


def gap_h_batch(perms: np.ndarray) -> np.ndarray:
    """Compute gap heuristic for a batch of permutations.

    Args:
        perms: 2D array of shape (batch, n), dtype int8.

    Returns:
        1D int array of gap counts, shape (batch,).
    """
    batch_size, n = perms.shape
    p = perms.astype(np.int16)
    extended = np.empty((batch_size, n + 1), dtype=np.int16)
    extended[:, :n] = p
    extended[:, n] = n
    diffs = np.abs(np.diff(extended, axis=1))
    return np.sum(diffs != 1, axis=1).astype(np.int32)


def k_gap(perm: np.ndarray, k: int = 2) -> int:
    """Compute k-gap: count positions where |perm[i] - perm[i+k]| != k.

    Args:
        perm: Permutation array.
        k: Gap distance to check (default 2).

    Returns:
        Number of k-gaps.
    """
    p = as_compute_dtype(perm)
    if len(p) <= k:
        return 0
    return int(np.sum(np.abs(p[:-k] - p[k:]) != k))


def gap_features(perm: np.ndarray) -> np.ndarray:
    """Compute gap-based features: gap_1, gap_2, gap_3.

    Args:
        perm: Permutation array.

    Returns:
        Array of [gap_1, gap_2, gap_3] as float32.
    """
    return np.array([gap_h(perm), k_gap(perm, 2), k_gap(perm, 3)], dtype=np.float32)
