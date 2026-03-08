"""Core types for pancake sorting solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SolverResult:
    """Result of solving a single permutation.

    Attributes:
        perm_id: Competition row ID.
        flips: Sequence of flip sizes (each k means reverse first k elements).
        original_perm: The original permutation that was solved.
    """

    perm_id: int
    flips: list[int]
    original_perm: np.ndarray = field(repr=False)

    @property
    def length(self) -> int:
        """Number of flips in the solution."""
        return len(self.flips)

    def solution_string(self) -> str:
        """Format as submission string like 'R4.R2'."""
        if not self.flips:
            return ""
        return ".".join(f"R{k}" for k in self.flips)
