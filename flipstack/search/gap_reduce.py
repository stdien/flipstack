"""Gap-reduction algorithm.

Each move either places an element correctly or creates a ±1 pair.
On deadlock, use probabilistic restarts.

Extended with inverse solving and first-move enumeration for better solutions.
"""

from __future__ import annotations

import time

import numpy as np

from flipstack.core.permutation import apply_flip, inverse_perm, is_sorted


def _flip_gap_delta(p: np.ndarray, k: int, n: int) -> int:
    """Compute gap change from flipping first k elements, in O(1).

    A flip(k) only changes the boundary pair at position k-1:
    old pair: (p[k-1], p[k]) where p[n] = sentinel = n
    new pair: (p[0], p[k])

    Returns:
        -1, 0, or +1 — the change in gap count.
    """
    right = n if k == n else int(p[k])
    old_is_gap = abs(int(p[k - 1]) - right) != 1
    new_is_gap = abs(int(p[0]) - right) != 1
    return int(new_is_gap) - int(old_is_gap)


def _count_gap_reducing(p: np.ndarray, n: int) -> int:
    """Count how many gap-reducing moves exist from state p."""
    count = 0
    for k in range(2, n + 1):
        if _flip_gap_delta(p, k, n) < 0:
            count += 1
    return count


def _pick_best_lookahead(
    p: np.ndarray,
    candidates: list[int],
    n: int,
    rng: np.random.Generator,
) -> int:
    """Pick move from candidates that maximizes gap-reducing options after."""
    if len(candidates) == 1:
        return candidates[0]

    best_score = -1
    best_moves: list[int] = []
    for k in candidates:
        child = apply_flip(p, k)
        score = _count_gap_reducing(child, n)
        if score > best_score:
            best_score = score
            best_moves = [k]
        elif score == best_score:
            best_moves.append(k)

    return int(rng.choice(best_moves))


def gap_reduce_solve(
    perm: np.ndarray,
    max_restarts: int = 50,
    max_steps: int = 500,
    seed: int = 42,
    timeout: float = 60.0,
) -> list[int] | None:
    """Solve via gap-reduction with 2-step lookahead and probabilistic restarts.

    Strategy: greedily reduce gaps. Among gap-reducing moves, pick the one
    that leaves the most gap-reducing options for the next step (2-step
    lookahead). When stuck, pick the neutral move with the best lookahead.
    Uses O(1) incremental gap computation; lookahead is O(n^2) per step.

    Args:
        perm: Starting permutation.
        max_restarts: Number of restart attempts.
        max_steps: Max steps per attempt.
        seed: Random seed.
        timeout: Wall time limit.

    Returns:
        Flip sequence, or None if unsolved.
    """
    if is_sorted(perm):
        return []

    rng = np.random.default_rng(seed)
    n = len(perm)
    start_time = time.monotonic()
    best: list[int] | None = None

    for _restart in range(max_restarts):
        if time.monotonic() - start_time > timeout:
            break

        p = perm.copy()
        flips: list[int] = []

        for _step in range(max_steps):
            if is_sorted(p):
                if best is None or len(flips) < len(best):
                    best = list(flips)
                break

            # Use O(1) gap delta to classify moves
            gap_reducing: list[int] = []
            gap_neutral: list[int] = []
            for k in range(2, n + 1):
                delta = _flip_gap_delta(p, k, n)
                if delta < 0:
                    gap_reducing.append(k)
                elif delta == 0:
                    gap_neutral.append(k)

            if gap_reducing:
                if n >= 40 and len(gap_reducing) > 1:
                    k = _pick_best_lookahead(p, gap_reducing, n, rng)
                else:
                    k = int(rng.choice(gap_reducing))
            elif gap_neutral:
                if n >= 40 and len(gap_neutral) > 1:
                    k = _pick_best_lookahead(p, gap_neutral, n, rng)
                else:
                    k = int(rng.choice(gap_neutral))
            else:
                k = int(rng.integers(2, n + 1))

            p = apply_flip(p, k)
            flips.append(k)

    return best


def shorten_solution(
    perm: np.ndarray,
    solution: list[int],
    max_restarts: int = 200,
    timeout: float = 10.0,
) -> list[int]:
    """Try to shorten a solution by re-solving from intermediate states.

    For each prefix of the solution, run gap_reduce_solve from that
    intermediate state. If a shorter suffix is found, the total solution
    is shortened.

    Args:
        perm: Original permutation.
        solution: Known solution (flip sequence).
        max_restarts: Restarts per intermediate re-solve.
        timeout: Total wall time limit.

    Returns:
        Potentially shorter solution (or original if no improvement found).
    """
    if len(solution) <= 1:
        return solution

    best = list(solution)
    start_time = time.monotonic()

    # Build intermediate states
    states = [perm.copy()]
    p = perm.copy()
    for k in solution:
        p = apply_flip(p, k)
        states.append(p.copy())

    # Try re-solving from each intermediate state
    per_split = timeout / len(solution)
    for i in range(1, len(best)):
        if time.monotonic() - start_time > timeout:
            break

        suffix_len = len(best) - i
        if suffix_len <= 1:
            break

        state = states[i]
        t = min(per_split, timeout - (time.monotonic() - start_time))
        result = gap_reduce_solve(
            state, max_restarts=max_restarts, max_steps=suffix_len, seed=i, timeout=t,
        )
        if result is not None and len(result) < suffix_len:
            best = list(best[:i]) + result
            # Rebuild states for the new solution
            states = [perm.copy()]
            p = perm.copy()
            for k in best:
                p = apply_flip(p, k)
                states.append(p.copy())

    return best


def gap_reduce_inverse(
    perm: np.ndarray,
    max_restarts: int = 50,
    max_steps: int = 500,
    seed: int = 42,
    timeout: float = 60.0,
) -> list[int] | None:
    """Solve by gap-reducing the inverse permutation, then reversing flips.

    If inv(p) is sorted by flips [f1, ..., fk], then p is sorted by [fk, ..., f1].

    Args:
        perm: Starting permutation.
        max_restarts: Number of restart attempts.
        max_steps: Max steps per attempt.
        seed: Random seed.
        timeout: Wall time limit.

    Returns:
        Flip sequence sorting perm, or None if unsolved.
    """
    if is_sorted(perm):
        return []

    inv = inverse_perm(perm)
    result = gap_reduce_solve(inv, max_restarts, max_steps, seed, timeout)
    if result is None:
        return None

    # Reverse the flip sequence
    solution = list(reversed(result))

    # Verify correctness
    p = perm.copy()
    for k in solution:
        p = apply_flip(p, k)
    if not is_sorted(p):
        return None

    return solution


def gap_reduce_fme(
    perm: np.ndarray,
    max_restarts: int = 200,
    max_steps: int = 500,
    seeds: int = 3,
    timeout: float = 60.0,
) -> list[int] | None:
    """First-move enumeration: try each first flip, then gap-reduce.

    For each k in 2..n, apply flip(k) to perm, then run gap_reduce_solve
    on the result. Returns the shortest solution found.

    Args:
        perm: Starting permutation.
        max_restarts: Restarts per first-move attempt.
        max_steps: Max steps per attempt.
        seeds: Number of random seeds per first move.
        timeout: Total wall time limit.

    Returns:
        Flip sequence sorting perm, or None if unsolved.
    """
    if is_sorted(perm):
        return []

    n = len(perm)
    start_time = time.monotonic()
    best: list[int] | None = None
    per_move_timeout = timeout / (n * seeds)

    for first_k in range(2, n + 1):
        if time.monotonic() - start_time > timeout:
            break

        child = apply_flip(perm, first_k)
        if is_sorted(child):
            if best is None or len(best) > 1:
                best = [first_k]
            continue

        for seed in range(seeds):
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                break
            t = min(per_move_timeout, remaining)
            result = gap_reduce_solve(
                child,
                max_restarts=max_restarts,
                max_steps=max_steps,
                seed=seed * 100 + first_k,
                timeout=t,
            )
            if result is not None:
                full = [first_k, *result]
                if best is None or len(full) < len(best):
                    best = full

    return best
