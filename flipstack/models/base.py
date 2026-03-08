"""Base protocol for value models."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseModel(Protocol):
    """Protocol for value prediction models."""

    def predict(self, perms: np.ndarray) -> np.ndarray:
        """Predict distance estimates for a batch of permutations.

        Args:
            perms: 2D array of shape (batch, n), dtype int8.

        Returns:
            1D float array of predicted distances.
        """
        ...

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save to.
        """
        ...

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load from.
        """
        ...
