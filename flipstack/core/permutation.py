"""Permutation operations for pancake sorting."""

from __future__ import annotations

import numpy as np


def apply_flip(perm: np.ndarray, k: int) -> np.ndarray:
    """Apply prefix reversal of first k elements.

    Args:
        perm: Permutation array.
        k: Number of elements to reverse (2 <= k <= n).

    Returns:
        New array with first k elements reversed.
    """
    result = perm.copy()
    result[:k] = result[k - 1 :: -1] if k == len(perm) else result[:k][::-1]
    return result


def apply_flip_inplace(perm: np.ndarray, k: int) -> None:
    """Apply prefix reversal in-place.

    Args:
        perm: Permutation array (modified in-place).
        k: Number of elements to reverse (2 <= k <= n).
    """
    perm[:k] = perm[k - 1 :: -1] if k == len(perm) else perm[:k][::-1]


def apply_flip_batch(perms: np.ndarray, k: int) -> np.ndarray:
    """Apply prefix reversal to a batch of permutations.

    Args:
        perms: 2D array of shape (batch, n).
        k: Number of elements to reverse.

    Returns:
        New 2D array with first k columns reversed in each row.
    """
    result = perms.copy()
    result[:, :k] = result[:, k - 1 :: -1] if k == perms.shape[1] else result[:, :k][:, ::-1]
    return result


def is_sorted(perm: np.ndarray) -> bool:
    """Check if permutation is identity [0, 1, 2, ..., n-1].

    Args:
        perm: Permutation array.

    Returns:
        True if perm is the identity permutation.
    """
    n = len(perm)
    return bool(np.array_equal(perm, np.arange(n, dtype=perm.dtype)))


def validate_perm(perm: np.ndarray, expected_n: int | None = None) -> None:
    """Validate a permutation array.

    Args:
        perm: Array to validate.
        expected_n: If given, check length matches.

    Raises:
        ValueError: If permutation is invalid.
    """
    if perm.ndim != 1:
        msg = f"Permutation must be 1D, got {perm.ndim}D"
        raise ValueError(msg)
    n = len(perm)
    if expected_n is not None and n != expected_n:
        msg = f"Expected length {expected_n}, got {n}"
        raise ValueError(msg)
    if n == 0:
        msg = "Empty permutation"
        raise ValueError(msg)
    # Cast to int16 for safe arithmetic
    vals = perm.astype(np.int16)
    if vals.min() < 0 or vals.max() >= n:
        msg = f"Values must be in [0, {n - 1}], got [{vals.min()}, {vals.max()}]"
        raise ValueError(msg)
    if len(np.unique(vals)) != n:
        msg = "Duplicate values in permutation"
        raise ValueError(msg)


def as_compute_dtype(perm: np.ndarray) -> np.ndarray:
    """Cast int8 array to int16 for safe arithmetic.

    Args:
        perm: Permutation array (typically int8).

    Returns:
        Array with int16 dtype.
    """
    return perm.astype(np.int16)


def perm_to_index(perm: np.ndarray) -> int:
    """Encode permutation as Lehmer code index.

    Args:
        perm: Permutation array [0..n-1].

    Returns:
        Integer index in [0, n!).
    """
    n = len(perm)
    available = list(range(n))
    index = 0
    for i in range(n):
        k = available.index(int(perm[i]))
        index = index * (n - i) + k
        available.pop(k)
    return index


def index_to_perm(index: int, n: int, dtype: np.typing.DTypeLike = np.int8) -> np.ndarray:
    """Decode Lehmer code index to permutation.

    Args:
        index: Integer index in [0, n!).
        n: Permutation size.
        dtype: Output array dtype.

    Returns:
        Permutation array [0..n-1].
    """
    available = list(range(n))
    perm = np.empty(n, dtype=dtype)
    for i in range(n):
        fact = 1
        for j in range(2, n - i):
            fact *= j
        k = index // fact
        index = index % fact
        perm[i] = available[k]
        available.pop(k)
    return perm


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    """Compute inverse permutation.

    Args:
        perm: Permutation array where perm[i] = j means element j is at position i.

    Returns:
        Inverse permutation where result[j] = i.
    """
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv
