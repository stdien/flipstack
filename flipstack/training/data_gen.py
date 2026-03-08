"""Random walk data generation for training value models."""

from __future__ import annotations

import numpy as np


def generate_random_walks(
    n: int,
    num_samples: int = 10000,
    max_walk_length: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data via random walks from identity.

    Each sample is a permutation reached by a random walk of random length,
    labeled with the walk distance (proxy for true distance).

    Args:
        n: Permutation size.
        num_samples: Number of samples to generate.
        max_walk_length: Maximum walk length (default: 2*n).
        seed: Random seed.

    Returns:
        Tuple of (X, y) where X is (num_samples, n) int8 and
        y is (num_samples,) float32 walk distances.
    """
    if max_walk_length is None:
        max_walk_length = 2 * n

    rng = np.random.default_rng(seed)
    x_data = np.empty((num_samples, n), dtype=np.int8)
    y_data = np.empty(num_samples, dtype=np.float32)

    identity = np.arange(n, dtype=np.int8)

    for i in range(num_samples):
        walk_len = rng.integers(1, max_walk_length + 1)
        perm = identity.copy()
        for _step in range(walk_len):
            k = int(rng.integers(2, n + 1))
            apply_flip_inplace_raw(perm, k)
        x_data[i] = perm
        y_data[i] = float(walk_len)

    return x_data, y_data


def apply_flip_inplace_raw(perm: np.ndarray, k: int) -> None:
    """Fast in-place flip without function call overhead.

    Args:
        perm: Permutation array (modified in-place).
        k: Number of elements to reverse.
    """
    perm[:k] = perm[k - 1 :: -1] if k == len(perm) else perm[:k][::-1]


def generate_features(
    perms: np.ndarray,
) -> np.ndarray:
    """Generate features for a batch of permutations.

    Features: raw permutation values + gap_1/gap_2/gap_3 + positional features.

    Args:
        perms: 2D array of shape (batch, n), dtype int8.

    Returns:
        2D float32 feature array of shape (batch, n + 6).
    """
    from flipstack.heuristics.gap import gap_h_batch, k_gap

    batch_size, n = perms.shape
    p = perms.astype(np.float32)

    # Gap features
    gap1 = gap_h_batch(perms).reshape(-1, 1).astype(np.float32)

    # k-gap features
    gap2 = np.array([k_gap(perms[i], 2) for i in range(batch_size)], dtype=np.float32).reshape(-1, 1)
    gap3 = np.array([k_gap(perms[i], 3) for i in range(batch_size)], dtype=np.float32).reshape(-1, 1)

    # Positional: how far each element is from its home position
    positions = np.arange(n, dtype=np.float32)
    displacement = np.abs(p - positions).mean(axis=1, keepdims=True)
    max_displacement = np.abs(p - positions).max(axis=1, keepdims=True)
    sorted_ratio = np.array(
        [np.sum(np.diff(perms[i].astype(np.int16)) == 1) / max(n - 1, 1) for i in range(batch_size)],
        dtype=np.float32,
    ).reshape(-1, 1)

    return np.hstack([p, gap1, gap2, gap3, displacement, max_displacement, sorted_ratio])
