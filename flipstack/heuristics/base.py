"""Base protocol for heuristic functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Heuristic(Protocol):
    """Protocol for heuristic scoring functions.

    Lower scores indicate states closer to the goal (identity permutation).
    """

    def __call__(self, perm: np.ndarray) -> float:
        """Evaluate a single permutation.

        Args:
            perm: Permutation array (int8).

        Returns:
            Heuristic score (lower = closer to sorted).
        """
        ...

    def batch(self, perms: np.ndarray) -> np.ndarray:
        """Evaluate a batch of permutations.

        Args:
            perms: 2D array of shape (batch, n), dtype int8.

        Returns:
            1D float array of scores.
        """
        ...
