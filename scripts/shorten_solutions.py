"""Solution shortening via BFS transformation tables.

For each window of consecutive moves in a solution:
1. Compute net transformation (apply window flips to identity)
2. Look up in BFS table (shortest producing sequence)
3. If shorter sequence exists, replace window

Also detects and removes cycles (repeated intermediate states).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted
from scripts._common import load_existing_solutions


def build_bfs_table(
    n: int, max_depth: int,
) -> dict[bytes, tuple[int, list[int]]]:
    """Build BFS table from identity permutation.

    Returns:
        Dict mapping perm.tobytes() -> (distance, sorting_path).
        sorting_path sorts that permutation (brings it to identity).
    """
    identity = np.arange(n, dtype=np.int8)
    table: dict[bytes, tuple[int, list[int]]] = {identity.tobytes(): (0, [])}
    frontier: deque[tuple[np.ndarray, int, list[int]]] = deque()
    frontier.append((identity, 0, []))

    while frontier:
        state, depth, path = frontier.popleft()
        if depth >= max_depth:
            continue
        for k in range(2, n + 1):
            child = apply_flip(state, k)
            key = child.tobytes()
            if key not in table:
                child_path = [k, *path]
                table[key] = (depth + 1, child_path)
                frontier.append((child, depth + 1, child_path))

    return table


def _producing_sequence(
    bfs_table: dict[bytes, tuple[int, list[int]]],
    key: bytes,
) -> list[int] | None:
    """Get the shortest sequence of flips that produces permutation from identity.

    The BFS table stores sorting paths. The producing path is the reverse.
    """
    entry = bfs_table.get(key)
    if entry is None:
        return None
    _, sorting_path = entry
    return list(reversed(sorting_path))


def shorten_solution_windows(
    solution: list[int],
    n: int,
    bfs_table: dict[bytes, tuple[int, list[int]]],
    max_window: int = 10,
) -> list[int]:
    """Shorten a solution by replacing windows with shorter BFS equivalents."""
    sol = list(solution)
    improved = True

    while improved:
        improved = False
        best_saving = 0
        best_pos = -1
        best_win_size = -1
        best_replacement: list[int] = []

        # Scan all windows, find the one with biggest saving
        for w in range(2, min(max_window + 1, len(sol) + 1)):
            for i in range(len(sol) - w + 1):
                # Compute net transformation of window
                state = np.arange(n, dtype=np.int8)
                for k in sol[i : i + w]:
                    state = apply_flip(state, k)

                key = state.tobytes()
                entry = bfs_table.get(key)
                if entry is None:
                    continue

                dist, sorting_path = entry
                if dist < w:
                    saving = w - dist
                    if saving > best_saving:
                        best_saving = saving
                        best_pos = i
                        best_win_size = w
                        best_replacement = list(reversed(sorting_path))

        if best_saving > 0:
            sol = sol[:best_pos] + best_replacement + sol[best_pos + best_win_size :]
            improved = True

    return sol


def remove_cycles(
    perm_list: list[int], solution: list[int],
) -> list[int]:
    """Remove cycles from solution (repeated intermediate states)."""
    state = np.array(perm_list, dtype=np.int8)
    states: dict[bytes, int] = {state.tobytes(): 0}

    for i, k in enumerate(solution):
        state = apply_flip(state, k)
        key = state.tobytes()
        if key in states:
            prev_pos = states[key]
            new_solution = solution[:prev_pos] + solution[i + 1 :]
            return remove_cycles(perm_list, new_solution)
        states[key] = i + 1

    return solution


def _choose_bfs_depth(n: int) -> int:
    """Choose BFS depth based on n to keep table size manageable."""
    if n <= 10:
        return 7
    if n <= 20:
        return 5
    if n <= 50:
        return 4
    return 3


# Worker globals
_worker_bfs: dict[bytes, tuple[int, list[int]]] = {}
_worker_n: int = 0


def _init_worker(bfs_table: dict[bytes, tuple[int, list[int]]], n: int) -> None:
    """Initialize worker with shared BFS table."""
    global _worker_bfs, _worker_n
    _worker_bfs = bfs_table
    _worker_n = n


def _shorten_one(
    args: tuple[int, list[int], list[int]],
) -> tuple[int, list[int], int]:
    """Shorten one solution. Returns (idx, shortened, original_len)."""
    idx, perm_list, solution = args
    n = len(perm_list)
    original_len = len(solution)

    # First remove cycles
    sol = remove_cycles(perm_list, solution)

    # Then shorten windows
    sol = shorten_solution_windows(sol, n, _worker_bfs, max_window=10)

    # Remove cycles again (window replacement might create new ones)
    sol = remove_cycles(perm_list, sol)

    return (idx, sol, original_len)


def main() -> None:
    """Shorten all existing solutions using BFS transformation tables."""
    base = Path(__file__).resolve().parent.parent
    data_path = base / "data" / "test.csv"
    sol_dir = base / "experiments" / "solutions"
    merged_path = sol_dir / "merged_best.csv"
    output_path = sol_dir / "shortened.json"

    # Load competition data
    perms: dict[int, tuple[int, list[int]]] = {}
    perms_by_n: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    with data_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["id"])
            n = int(row["n"])
            perm = [int(x) for x in row["permutation"].split(",")]
            perms[idx] = (n, perm)
            perms_by_n[n].append((idx, perm))

    # Load existing solutions
    existing = load_existing_solutions(merged_path)
    print(f"Loaded {len(existing)} existing solutions")

    improvements: dict[str, list[int]] = {}
    total_saved = 0

    for n in sorted(perms_by_n.keys()):
        items = perms_by_n[n]
        depth = _choose_bfs_depth(n)
        print(f"\n=== n={n}: {len(items)} perms, BFS depth={depth} ===", flush=True)

        t0 = time.time()
        bfs_table = build_bfs_table(n, max_depth=depth)
        print(f"  BFS table: {len(bfs_table):,} states in {time.time()-t0:.1f}s", flush=True)

        # Build work items
        work = []
        for idx, perm_list in items:
            if idx not in existing:
                continue
            _, sol = existing[idx]
            work.append((idx, perm_list, sol))

        if not work:
            print("  No solutions to shorten")
            continue

        n_improved = 0
        n_saved = 0

        # Use multiprocessing for large workloads
        ncpu = min(cpu_count() or 1, 16)
        with Pool(ncpu, initializer=_init_worker, initargs=(bfs_table, n)) as pool:
            for idx, shortened, original_len in pool.imap_unordered(_shorten_one, work):
                new_len = len(shortened)
                if new_len < original_len:
                    # Verify the shortened solution
                    perm = np.array(perms[idx][1], dtype=np.int8)
                    for k in shortened:
                        perm = apply_flip(perm, k)
                    if is_sorted(perm):
                        improvements[str(idx)] = shortened
                        n_improved += 1
                        n_saved += original_len - new_len
                    else:
                        print(f"  WARNING: id={idx} shortened solution INVALID!", flush=True)

        total_saved += n_saved
        print(
            f"  n={n}: {n_improved}/{len(work)} improved, {n_saved} flips saved",
            flush=True,
        )

        # Save incrementally
        with output_path.open("w") as f:
            json.dump(improvements, f)

    print(f"\nDone: {len(improvements)} improved, {total_saved} flips saved")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    set_start_method("forkserver")
    main()
