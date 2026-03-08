"""Multi-strategy orchestrator: run multiple strategies per permutation, keep shortest."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from flipstack.core.permutation import inverse_perm, is_sorted
from flipstack.core.types import SolverResult
from flipstack.search.beam import beam_search, iterated_beam_search
from flipstack.search.beam_bidir import bidir_beam_search
from flipstack.search.gap_reduce import gap_reduce_solve

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Type alias for strategy config
StratConfig = dict[str, Any]


def multi_strategy_solve(
    perm_id: int,
    n: int,
    perm: np.ndarray,
    strategies: list[StratConfig] | None = None,
    timeout: float = 300.0,
    scorer: Callable[[np.ndarray], float] | None = None,
) -> SolverResult:
    """Solve a permutation using multiple strategies, keep shortest.

    Args:
        perm_id: Competition row ID.
        n: Permutation size.
        perm: Permutation array.
        strategies: List of strategy configs. If None, uses defaults per n.
        timeout: Total time budget.
        scorer: Optional model-based scorer.

    Returns:
        Best SolverResult found.
    """
    if is_sorted(perm):
        return SolverResult(perm_id=perm_id, flips=[], original_perm=perm)

    start_time = time.monotonic()
    candidates: list[list[int]] = []

    def remaining() -> float:
        return max(0.0, timeout - (time.monotonic() - start_time))

    if strategies is None:
        strategies = _default_strategies(n)

    for strat in strategies:
        if remaining() < 1.0:
            break

        name = str(strat.get("name", "unknown"))
        time_frac = float(strat.get("time_fraction", 0.25))
        strat_timeout = remaining() * time_frac

        result = _run_strategy(name, perm, n, strat_timeout, scorer, strat)
        if result is not None:
            candidates.append(result)
            logger.debug("id=%d strategy=%s len=%d", perm_id, name, len(result))

    # Try inverse
    if remaining() > 1.0:
        inv_perm = inverse_perm(perm)
        inv_bw = int(strat.get("beam_width", 1024)) if strategies else 1024
        result = beam_search(
            inv_perm,
            beam_width=inv_bw,
            max_moves=3,
            scorer=scorer,
            timeout=remaining() * 0.3,
        )
        if result is not None:
            candidates.append(list(reversed(result)))

    if not candidates:
        from flipstack.cli import _greedy_solve

        result = _greedy_solve(perm)
        if result is not None:
            candidates.append(result)

    if not candidates:
        logger.warning("Failed to solve id=%d n=%d", perm_id, n)
        return SolverResult(perm_id=perm_id, flips=[], original_perm=perm)

    best: list[int] = min(candidates, key=len)  # type: ignore[assignment]
    return SolverResult(perm_id=perm_id, flips=best, original_perm=perm)


def _default_strategies(n: int) -> list[StratConfig]:
    """Get default strategies for a given n.

    Args:
        n: Permutation size.

    Returns:
        List of strategy config dicts.
    """
    if n <= 12:
        return [
            {"name": "bidir", "beam_width": 2048, "max_moves": 3, "time_fraction": 0.5},
            {"name": "gap_reduce", "time_fraction": 0.3},
        ]
    if n <= 30:
        return [
            {"name": "iterated", "beam_widths": [256, 1024, 4096], "max_moves": 3, "time_fraction": 0.4},
            {"name": "beam", "beam_width": 4096, "max_moves": 4, "time_fraction": 0.3},
            {"name": "gap_reduce", "time_fraction": 0.2},
        ]
    return [
        {"name": "beam", "beam_width": 1024, "max_moves": 3, "time_fraction": 0.4},
        {"name": "gap_reduce", "time_fraction": 0.3},
    ]


def _run_strategy(
    name: str,
    perm: np.ndarray,
    n: int,
    timeout: float,
    scorer: Callable[[np.ndarray], float] | None,
    config: StratConfig,
) -> list[int] | None:
    """Run a single strategy.

    Args:
        name: Strategy name.
        perm: Permutation to solve.
        n: Permutation size.
        timeout: Time budget for this strategy.
        scorer: Optional scorer.
        config: Strategy configuration.

    Returns:
        Flip sequence or None.
    """
    if name == "bidir":
        return bidir_beam_search(
            perm,
            beam_width=int(config.get("beam_width", 1024)),
            max_moves=int(config.get("max_moves", 3)),
            scorer=scorer,
            timeout=timeout,
        )
    if name == "iterated":
        widths_raw = config.get("beam_widths", [256, 1024, 4096])
        widths = [int(w) for w in widths_raw] if isinstance(widths_raw, list) else [1024]
        return iterated_beam_search(
            perm,
            beam_widths=widths,
            max_moves=int(config.get("max_moves", 3)),
            scorer=scorer,
            timeout=timeout,
        )
    if name == "beam":
        return beam_search(
            perm,
            beam_width=int(config.get("beam_width", 1024)),
            max_moves=int(config.get("max_moves", 3)),
            scorer=scorer,
            timeout=timeout,
        )
    if name == "gap_reduce":
        return gap_reduce_solve(perm, timeout=timeout)
    logger.warning("Unknown strategy: %s", name)
    return None
