"""Portfolio strategy: assign solver per n-range, try multiple, keep shortest."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from flipstack.core.permutation import inverse_perm, is_sorted
from flipstack.core.types import SolverResult
from flipstack.search.beam import beam_search, iterated_beam_search
from flipstack.search.beam_bidir import bidir_beam_search
from flipstack.search.gap_reduce import gap_reduce_solve

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def solve_single(
    perm_id: int,
    n: int,
    perm: np.ndarray,
    beam_width: int = 1024,
    max_moves: int = 3,
    timeout: float = 300.0,
    scorer: Callable[[np.ndarray], float] | None = None,
) -> SolverResult:
    """Solve a single permutation using portfolio strategy.

    Strategy by n-range:
    - n <= 12: bidirectional beam (near-optimal)
    - n = 13-30: iterated beam with model scorer
    - n > 30: beam search + gap reduce, keep shortest

    Also tries solving the inverse permutation.

    Args:
        perm_id: Competition row ID.
        n: Permutation size.
        perm: Permutation array.
        beam_width: Base beam width.
        max_moves: Move filter count.
        timeout: Total time budget for this permutation.
        scorer: Optional model-based scorer.

    Returns:
        Best SolverResult found.
    """
    if is_sorted(perm):
        return SolverResult(perm_id=perm_id, flips=[], original_perm=perm)

    start_time = time.monotonic()
    candidates: list[list[int]] = []

    def _remaining() -> float:
        return max(0.0, timeout - (time.monotonic() - start_time))

    # Strategy 1: Beam search (always try)
    if _remaining() > 1.0:
        if n <= 12:
            result = bidir_beam_search(
                perm,
                beam_width=beam_width,
                max_moves=max_moves,
                scorer=scorer,
                timeout=_remaining() * 0.4,
            )
        elif n <= 30:
            result = iterated_beam_search(
                perm,
                beam_widths=[256, 1024, 4096],
                max_moves=max_moves,
                scorer=scorer,
                timeout=_remaining() * 0.4,
            )
        else:
            result = beam_search(
                perm,
                beam_width=beam_width,
                max_moves=max_moves,
                scorer=scorer,
                timeout=_remaining() * 0.4,
            )
        if result is not None:
            candidates.append(result)

    # Strategy 2: Gap reduce (complementary approach)
    if _remaining() > 1.0:
        result = gap_reduce_solve(perm, timeout=_remaining() * 0.3)
        if result is not None:
            candidates.append(result)

    # Strategy 3: Inverse solving
    if _remaining() > 1.0:
        inv_perm = inverse_perm(perm)
        result = beam_search(
            inv_perm,
            beam_width=beam_width,
            max_moves=max_moves,
            scorer=scorer,
            timeout=_remaining() * 0.3,
        )
        if result is not None:
            # Inverse solution: reverse the flip sequence
            candidates.append(list(reversed(result)))

    if not candidates:
        # Emergency: greedy
        from flipstack.cli import _greedy_solve

        result = _greedy_solve(perm)
        if result is not None:
            candidates.append(result)

    if not candidates:
        logger.warning("Failed to solve id=%d n=%d", perm_id, n)
        return SolverResult(perm_id=perm_id, flips=[], original_perm=perm)

    best: list[int] = min(candidates, key=len)  # type: ignore[assignment]
    return SolverResult(perm_id=perm_id, flips=best, original_perm=perm)
