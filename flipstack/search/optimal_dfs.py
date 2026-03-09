"""Numba-JIT optimal DFS solver for pancake sorting.

Replaces C++ scripts (gap_only_dfs, gap_slack_dfs, gap_count_all) with
pure-Python Numba equivalents. Uses iterative deepening with gap heuristic.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numba as nb
import numpy as np

logger = logging.getLogger(__name__)

@nb.njit
def _flip_inplace(p: np.ndarray, k: int) -> None:
    """Reverse first k elements in-place."""
    for i in range(k // 2):
        p[i], p[k - 1 - i] = p[k - 1 - i], p[i]


@nb.njit
def _is_sorted(p: np.ndarray, n: int) -> bool:
    """Check if permutation is identity [0..n-1]."""
    for i in range(n):  # noqa: SIM110
        if p[i] != i:
            return False
    return True


@nb.njit
def gap_h(p: np.ndarray, n: int) -> int:
    """Compute gap heuristic (number of breakpoints).

    Args:
        p: Permutation array.
        n: Permutation size.

    Returns:
        Number of gaps (adjacent pairs with |diff| != 1).
    """
    gaps = 0
    for i in range(n - 1):
        if abs(int(p[i]) - int(p[i + 1])) != 1:
            gaps += 1
    if abs(int(p[n - 1]) - n) != 1:
        gaps += 1
    return gaps


@nb.njit
def _flip_gap_delta(p: np.ndarray, k: int, n: int) -> int:
    """Compute gap change from flipping first k elements, O(1)."""
    right = n if k == n else int(p[k])
    old_gap = 1 if abs(int(p[k - 1]) - right) != 1 else 0
    new_gap = 1 if abs(int(p[0]) - right) != 1 else 0
    return new_gap - old_gap


@nb.njit
def _dfs_exists(
    p: np.ndarray, depth: int, max_depth: int, n: int, gaps: int, slack: int,
) -> bool:
    """DFS Phase 1: check if a solution of given length exists."""
    if depth == max_depth:
        return _is_sorted(p, n)
    if gaps < 0 or gaps > max_depth - depth:
        return False
    for k in range(2, n + 1):
        delta = _flip_gap_delta(p, k, n)
        if delta < 0:
            _flip_inplace(p, k)
            if _dfs_exists(p, depth + 1, max_depth, n, gaps + delta, slack):
                _flip_inplace(p, k)
                return True
            _flip_inplace(p, k)
        elif slack > 0:
            _flip_inplace(p, k)
            if _dfs_exists(p, depth + 1, max_depth, n, gaps + delta, slack - 1):
                _flip_inplace(p, k)
                return True
            _flip_inplace(p, k)
    return False


@nb.njit
def _dfs_count(
    p: np.ndarray,
    depth: int,
    max_depth: int,
    n: int,
    gaps: int,
    slack: int,
    limit: np.int64,
) -> np.int64:
    """DFS Phase 1.5: count solutions, stopping early at limit."""
    if depth == max_depth:
        return np.int64(1) if _is_sorted(p, n) else np.int64(0)
    if gaps < 0 or gaps > max_depth - depth:
        return np.int64(0)
    total = np.int64(0)
    for k in range(2, n + 1):
        delta = _flip_gap_delta(p, k, n)
        if delta < 0:
            _flip_inplace(p, k)
            total += _dfs_count(p, depth + 1, max_depth, n, gaps + delta, slack, limit)
            _flip_inplace(p, k)
            if total > limit:
                return total
        elif slack > 0:
            _flip_inplace(p, k)
            total += _dfs_count(
                p, depth + 1, max_depth, n, gaps + delta, slack - 1, limit,
            )
            _flip_inplace(p, k)
            if total > limit:
                return total
    return total


@nb.njit
def _dfs_enumerate(
    p: np.ndarray,
    depth: int,
    max_depth: int,
    n: int,
    gaps: int,
    slack: int,
    cur_path: np.ndarray,
    solutions: np.ndarray,
    sol_idx: int,
    max_solutions: int,
) -> int:
    """DFS Phase 2: write all solution paths into pre-allocated array.

    Returns updated sol_idx. If sol_idx >= max_solutions, stops early.
    """
    if depth == max_depth:
        if _is_sorted(p, n) and sol_idx < max_solutions:
            for i in range(max_depth):
                solutions[sol_idx, i] = cur_path[i]
            return sol_idx + 1
        return sol_idx
    if gaps < 0 or gaps > max_depth - depth:
        return sol_idx
    for k in range(2, n + 1):
        if sol_idx >= max_solutions:
            return sol_idx
        delta = _flip_gap_delta(p, k, n)
        if delta < 0:
            cur_path[depth] = k
            _flip_inplace(p, k)
            sol_idx = _dfs_enumerate(
                p, depth + 1, max_depth, n, gaps + delta, slack,
                cur_path, solutions, sol_idx, max_solutions,
            )
            _flip_inplace(p, k)
        elif slack > 0:
            cur_path[depth] = k
            _flip_inplace(p, k)
            sol_idx = _dfs_enumerate(
                p, depth + 1, max_depth, n, gaps + delta, slack - 1,
                cur_path, solutions, sol_idx, max_solutions,
            )
            _flip_inplace(p, k)
    return sol_idx


@nb.njit
def solve_one(perm: np.ndarray, n: int, max_slack: int) -> tuple[int, np.int64]:
    """Find optimal solution length and count solutions.

    Args:
        perm: Permutation array (int8).
        n: Permutation size.
        max_slack: Maximum slack above gap_h to try.

    Returns:
        (solution_length, count). Returns (-1, 0) if unsolved.
    """
    gh = gap_h(perm, n)
    if gh == 0:
        return (0, np.int64(1))
    work = perm.copy()
    slack = 0
    while max_slack < 0 or slack <= max_slack:
        for i in range(n):
            work[i] = perm[i]
        if _dfs_exists(work, 0, gh + slack, n, gh, slack):
            for i in range(n):
                work[i] = perm[i]
            count = _dfs_count(
                work, 0, gh + slack, n, gh, slack, np.int64(2**62),
            )
            return (gh + slack, count)
        slack += 1
    return (-1, np.int64(0))


@nb.njit
def solve_and_enumerate(
    perm: np.ndarray, n: int, max_slack: int, max_solutions: int,
) -> tuple[int, np.int64, np.ndarray, bool]:
    """Find optimal solutions and enumerate paths.

    Args:
        perm: Permutation array (int8).
        n: Permutation size.
        max_slack: Maximum slack above gap_h.
        max_solutions: Max paths to enumerate; if count exceeds this,
            mark as truncated.

    Returns:
        (solution_length, count, solutions_array, truncated).
        solutions_array shape: (actual_count, solution_length).
        If unsolved: (-1, 0, empty, False).
    """
    gh = gap_h(perm, n)
    if gh == 0:
        empty = np.empty((0, 0), dtype=np.int16)
        return (0, np.int64(1), empty, False)

    work = perm.copy()
    slack = 0
    while max_slack < 0 or slack <= max_slack:
        for i in range(n):
            work[i] = perm[i]
        if not _dfs_exists(work, 0, gh + slack, n, gh, slack):
            slack += 1
            continue

        sol_len = gh + slack

        # Count with early-stop at max_solutions + 1
        for i in range(n):
            work[i] = perm[i]
        count = _dfs_count(
            work, 0, sol_len, n, gh, slack, np.int64(max_solutions + 1),
        )
        truncated = count > max_solutions

        if truncated:
            # count is a lower bound (early-stopped), report -1 for unknown
            empty = np.empty((0, sol_len), dtype=np.int16)
            return (sol_len, np.int64(-1), empty, True)

        # Enumerate all solutions
        solutions = np.empty((int(count), sol_len), dtype=np.int16)
        cur_path = np.empty(sol_len, dtype=np.int16)
        for i in range(n):
            work[i] = perm[i]
        actual = _dfs_enumerate(
            work, 0, sol_len, n, gh, slack,
            cur_path, solutions, 0, int(count),
        )
        return (sol_len, np.int64(actual), solutions[:actual], False)

    empty = np.empty((0, 1), dtype=np.int16)
    return (-1, np.int64(0), empty, False)


def warmup_jit(n: int = 5, max_slack: int = 2) -> None:
    """Trigger Numba JIT compilation with a small problem."""
    p = np.arange(n, dtype=np.int8)
    p[0], p[1] = p[1], p[0]
    solve_one(p, n, max_slack)
    solve_and_enumerate(p, n, max_slack, 100)


def _solve_worker(
    args: tuple[np.ndarray, int, int, int, bool],
) -> dict:
    """Multiprocessing worker for solve_optimal CLI.

    Args:
        args: (perm, n, max_slack, max_solutions, enumerate_flag)

    Returns:
        Dict with perm, sol_len, count, solutions (if enumerate), truncated.
    """
    perm, n, max_slack, max_solutions, do_enumerate = args
    perm_list = perm.tolist()

    if do_enumerate:
        sol_len, count, solutions, truncated = solve_and_enumerate(
            perm, n, max_slack, max_solutions,
        )
        result: dict = {
            "perm": perm_list,
            "sol_len": int(sol_len),
            "count": int(count),
            "truncated": truncated,
        }
        if not truncated and sol_len >= 0:
            result["solutions"] = solutions.tolist()
        return result
    sol_len, count = solve_one(perm, n, max_slack)
    return {
            "perm": perm_list,
            "sol_len": int(sol_len),
            "count": int(count),
        }


def solve_optimal_batch(
    n: int,
    num_perms: int,
    max_slack: int = -1,
    max_solutions: int = 10000,
    do_enumerate: bool = True,
    output_path: Path | None = None,
    save_interval: int = 100,
    seed: int = 42,
    num_workers: int | None = None,
) -> list[dict]:
    """Solve random permutations optimally with multiprocessing.

    Args:
        n: Permutation size.
        num_perms: Number of random permutations to generate.
        max_slack: Maximum slack above gap_h.
        max_solutions: Max solutions to enumerate per perm.
        do_enumerate: Whether to enumerate all solution paths.
        output_path: JSON file for incremental saving.
        save_interval: Save every N permutations.
        seed: Random seed.
        num_workers: Number of worker processes (default: cpu_count).

    Returns:
        List of result dicts.
    """
    from multiprocessing import Pool, cpu_count

    if num_workers is None:
        num_workers = min(cpu_count() or 1, 128)

    # Generate permutations
    total = _factorial(n)
    if num_perms >= total:
        num_perms = total
        from itertools import permutations as iter_perms

        perms = [np.array(p, dtype=np.int8) for p in iter_perms(range(n))]
    else:
        rng = np.random.default_rng(seed)
        seen: set[bytes] = set()
        perms: list[np.ndarray] = []
        base = np.arange(n, dtype=np.int8)
        while len(perms) < num_perms:
            p = rng.permutation(base).astype(np.int8)
            key = p.tobytes()
            if key not in seen:
                seen.add(key)
                perms.append(p)

    logger.info("Solving %d permutations of size %d (slack=%d, workers=%d)",
                num_perms, n, max_slack, num_workers)

    # Warm up JIT in main process
    warmup_jit(min(n, 5), 2)

    work = [(p, n, max_slack, max_solutions, do_enumerate) for p in perms]
    results: list[dict] = []
    last_save = 0
    t0 = time.monotonic()

    with Pool(num_workers) as pool:
        for i, result in builtins_enumerate(
            pool.imap_unordered(_solve_worker, work),
        ):
            results.append(result)

            is_final = i + 1 == num_perms
            is_log = save_interval > 0 and (i + 1) % save_interval == 0
            if is_final or is_log:
                elapsed = time.monotonic() - t0
                solved = sum(1 for r in results if r["sol_len"] >= 0)
                truncated = sum(1 for r in results if r.get("truncated", False))
                logger.info(
                    "Progress: %d/%d solved=%d truncated=%d (%.1fs)",
                    i + 1, num_perms, solved, truncated, elapsed,
                )
            if output_path is not None and (is_final or (is_log and i + 1 >= last_save + save_interval)):
                _save_json(results, n, max_slack, output_path)
                last_save = i + 1

    return results


def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# Alias to avoid shadowing built-in enumerate in numba context
builtins_enumerate = enumerate


def _save_json(results: list[dict], n: int, max_slack: int, path: Path) -> None:
    """Save results to JSON with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n": n,
        "max_slack": max_slack,
        "num_perms": len(results),
        "num_solved": sum(1 for r in results if r["sol_len"] >= 0),
        "num_truncated": sum(1 for r in results if r.get("truncated", False)),
        "results": results,
    }
    with path.open("w") as f:
        json.dump(data, f)
