"""GPU-accelerated beam search using PyTorch."""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from flipstack.core.permutation import is_sorted

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def apply_flip_gpu(states: torch.Tensor, k: int) -> torch.Tensor:
    """Apply flip to all states on GPU.

    Args:
        states: (batch, n) int8 tensor on GPU.
        k: Number of elements to reverse.

    Returns:
        New states tensor with first k elements reversed.
    """
    result = states.clone()
    result[:, :k] = states[:, :k].flip(1)
    return result


def gap_h_gpu(states: torch.Tensor) -> torch.Tensor:
    """Vectorized gap heuristic on GPU.

    Args:
        states: (batch, n) tensor, any integer dtype.

    Returns:
        (batch,) float32 tensor of gap counts.
    """
    n = states.shape[1]
    p = states.to(torch.int16)

    # Right sentinel: append n to each row
    sentinel = torch.full((p.shape[0], 1), n, dtype=torch.int16, device=p.device)
    extended = torch.cat([p, sentinel], dim=1)

    # Count positions where |p[i] - p[i+1]| > 1
    diffs = torch.abs(extended[:, 1:] - extended[:, :-1])
    return (diffs > 1).sum(dim=1).float()


def _expand_beam_gpu(
    states: torch.Tensor,
    paths: list[list[int]],
    moves_per_state: list[list[int]],
    visited: dict[bytes, int],
    scorer_fn: Callable[[np.ndarray], float] | None,
    new_visited_keys: list[bytes],
) -> tuple[list[torch.Tensor], list[list[int]], list[float]] | list[int] | None:
    """Expand beam states by one step on GPU.

    Args:
        states: Current beam states on GPU.
        paths: Flip paths for each state.
        moves_per_state: Valid moves per state.
        visited: Visited state dedup map (mutated in place).
        scorer_fn: Optional custom scorer.
        new_visited_keys: Caller-owned list; keys added to visited are appended
            here so the caller can roll back on OOM without copying the dict.

    Returns:
        Tuple of (child_states, child_paths, child_scores),
        or a solution list[int] if solved, or None if no children.
    """
    child_states: list[torch.Tensor] = []
    child_paths: list[list[int]] = []
    child_scores: list[float] = []

    for i in range(states.shape[0]):
        for k in moves_per_state[i]:
            child = apply_flip_gpu(states[i : i + 1], k)
            child_np = child[0].cpu().numpy()

            if is_sorted(child_np):
                return [*paths[i], k]

            child_bytes = child_np.tobytes()
            child_depth = len(paths[i]) + 1
            if child_bytes in visited and visited[child_bytes] <= child_depth:
                continue
            visited[child_bytes] = child_depth
            new_visited_keys.append(child_bytes)

            score = scorer_fn(child_np) if scorer_fn is not None else float(gap_h_gpu(child)[0])
            child_states.append(child)
            child_paths.append([*paths[i], k])
            child_scores.append(score)

    if not child_states:
        return None
    return child_states, child_paths, child_scores


def gpu_beam_search(
    perm: np.ndarray,
    beam_width: int = 4096,
    max_steps: int = 200,
    max_moves: int = 0,
    scorer_fn: Callable[[np.ndarray], float] | None = None,
    timeout: float = 300.0,
    device: str = "cuda",
) -> list[int] | None:
    """GPU-accelerated beam search.

    Uses GPU for state expansion and scoring, CPU for bookkeeping.
    Includes deduplication, OOM recovery, and optional custom scorer.

    Args:
        perm: Starting permutation (int8).
        beam_width: Max states per step.
        max_steps: Maximum search steps.
        max_moves: If > 0, filter to top moves. If 0, use all moves.
        scorer_fn: Optional CPU scorer for ranking (lower = better).
            Falls back to GPU gap heuristic when None.
        timeout: Wall time limit.
        device: PyTorch device.

    Returns:
        Flip sequence sorting perm, or None.
    """
    if is_sorted(perm):
        return []

    n = len(perm)
    dev = torch.device(device)
    start_time = time.monotonic()

    states = torch.from_numpy(perm.copy()).unsqueeze(0).to(dev)
    paths: list[list[int]] = [[]]
    visited: dict[bytes, int] = {perm.tobytes(): 0}

    if max_moves > 0 and max_moves < n - 1:
        from flipstack.search.move_filter import filter_moves

        use_filter = True
    else:
        use_filter = False

    min_beam = min(64, beam_width)
    current_beam_width = beam_width

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        step = 0
        while step < max_steps:
            if time.monotonic() - start_time > timeout:
                break

            batch_size = states.shape[0]
            if use_filter:
                states_np = states.cpu().numpy()
                moves_per_state = [filter_moves(states_np[i], max_moves) for i in range(batch_size)]
            else:
                moves_per_state = [list(range(2, n + 1))] * batch_size

            result = None
            new_keys: list[bytes] = []
            child_states_list = None
            child_paths_list = None
            child_scores_list = None
            all_children = None
            all_scores = None
            try:
                result = _expand_beam_gpu(
                    states,
                    paths,
                    moves_per_state,
                    visited,
                    scorer_fn,
                    new_keys,
                )

                # Solution found
                if isinstance(result, list):
                    return result
                # No children
                if result is None:
                    break

                child_states_list, child_paths_list, child_scores_list = result
                all_children = torch.cat(child_states_list, dim=0)
                all_scores = torch.tensor(child_scores_list, device=dev)

                k_select = min(current_beam_width, len(all_scores))
                _, top_indices = all_scores.topk(k_select, largest=False)
                top_indices = top_indices.cpu().tolist()

                states = all_children[top_indices]
                paths = [child_paths_list[i] for i in top_indices]

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Roll back visited using tracked keys (avoids dict copy)
                    for key in new_keys:
                        visited.pop(key, None)
                    # Release all tensor references before clearing cache
                    del result, child_states_list, child_paths_list, child_scores_list
                    del all_children, all_scores
                    torch.cuda.empty_cache()
                    current_beam_width = max(min_beam, current_beam_width // 2)
                    logger.warning("OOM: shrinking beam to %d", current_beam_width)
                    states = states[:current_beam_width]
                    paths = paths[:current_beam_width]
                    # Don't increment step — retry at same depth
                    continue
                raise

            step += 1

    finally:
        if gc_was_enabled:
            gc.enable()

    return None
