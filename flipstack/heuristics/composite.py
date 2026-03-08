"""Composite heuristic combining multiple scoring functions."""

from __future__ import annotations

import numpy as np

from flipstack.heuristics.gap import gap_h, gap_h_batch
from flipstack.heuristics.lock_detect import ld_h, ld_h_batch
from flipstack.heuristics.singleton import count_singletons


def composite_h(perm: np.ndarray, use_ld: bool = True, singleton_eps: float = 0.001) -> float:
    """Composite heuristic: lock detection + singleton tie-breaking.

    Args:
        perm: Permutation array.
        use_ld: If True, use lock detection; otherwise plain gap_h.
        singleton_eps: Weight for singleton bonus.

    Returns:
        Composite heuristic score.
    """
    base = float(ld_h(perm) if use_ld else gap_h(perm))
    return base - singleton_eps * count_singletons(perm)


def composite_h_batch(
    perms: np.ndarray,
    use_ld: bool = True,
    singleton_eps: float = 0.001,
) -> np.ndarray:
    """Composite heuristic for a batch of permutations.

    Args:
        perms: 2D array of shape (batch, n).
        use_ld: If True, use lock detection.
        singleton_eps: Weight for singleton bonus.

    Returns:
        1D float array of scores.
    """
    base = ld_h_batch(perms).astype(np.float64) if use_ld else gap_h_batch(perms).astype(np.float64)
    batch_size = perms.shape[0]
    for i in range(batch_size):
        base[i] -= singleton_eps * count_singletons(perms[i])
    return base
