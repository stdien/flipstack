"""Merge all solution sources into merged_best.csv.

Reads all CSV solution files + JSON improvement files in experiments/solutions/,
verifies each solution, keeps the shortest valid solution per permutation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted


def verify_solution(perm_list: list[int], solution: list[int]) -> bool:
    """Verify a solution actually sorts the permutation with legal moves."""
    n = len(perm_list)
    perm = np.array(perm_list, dtype=np.int8)
    for k in solution:
        if not (2 <= k <= n):
            return False
        perm = apply_flip(perm, k)
    return bool(is_sorted(perm))


def _try_merge(
    best: dict[int, list[int]],
    idx: int,
    solution: list[int],
    perm_list: list[int],
    source: str,
) -> bool:
    """Try to merge a solution if it's valid and shorter than existing."""
    if not verify_solution(perm_list, solution):
        print(f"  WARNING: {source} id={idx} INVALID solution!")
        return False

    old_len = len(best[idx]) if idx in best else 9999
    if len(solution) < old_len:
        best[idx] = solution
        return True
    return False


def _parse_moves(moves_str: str) -> list[int]:
    """Parse move string in either 'R4.R2' or '4.2' format."""
    return [int(tok.lstrip("R")) for tok in moves_str.split(".")]


def _load_csv_solutions(
    sol_dir: Path,
    perms: dict[int, tuple[int, list[int]]],
    best: dict[int, list[int]],
) -> int:
    """Load solutions from all CSV files in sol_dir, return count of improvements."""
    total = 0
    for csv_path in sorted(sol_dir.glob("*.csv")):
        merged_from_file = 0
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            # Detect format: "Id,Moves" vs "id,permutation,solution"
            id_key = "Id" if "Id" in headers else "id"
            moves_key = "Moves" if "Moves" in headers else "solution"
            for row in reader:
                idx = int(row[id_key])
                moves_str = row.get(moves_key, "").strip()
                if not moves_str or idx not in perms:
                    continue
                solution = _parse_moves(moves_str)
                _, perm_list = perms[idx]
                if _try_merge(best, idx, solution, perm_list, csv_path.name):
                    merged_from_file += 1
        if merged_from_file > 0:
            print(f"  {csv_path.name}: {merged_from_file} improvements")
            total += merged_from_file
    return total


def _load_json_solutions(
    sol_dir: Path,
    perms: dict[int, tuple[int, list[int]]],
    best: dict[int, list[int]],
) -> int:
    """Load solutions from all JSON files in sol_dir, return count of improvements."""
    total = 0
    for json_path in sorted(sol_dir.glob("*.json")):
        with json_path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        merged_from_file = 0
        for idx_str, solution in data.items():
            # Skip non-solution entries (e.g., benchmark metadata)
            if not isinstance(solution, list) or not solution or not isinstance(solution[0], int):
                continue
            idx = int(idx_str)
            if idx not in perms:
                continue
            _, perm_list = perms[idx]
            if _try_merge(best, idx, solution, perm_list, json_path.name):
                merged_from_file += 1
        if merged_from_file > 0:
            print(f"  {json_path.name}: {merged_from_file} improvements")
            total += merged_from_file
    return total


def main() -> None:
    """Merge all solution files into merged_best.csv."""
    base = Path(__file__).resolve().parent.parent
    data_path = base / "data" / "test.csv"
    sol_dir = base / "experiments" / "solutions"
    output_path = sol_dir / "merged_best.csv"

    # Load competition data
    perms: dict[int, tuple[int, list[int]]] = {}
    with data_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["id"])
            n = int(row["n"])
            perm = [int(x) for x in row["permutation"].split(",")]
            perms[idx] = (n, perm)

    print(f"Loaded {len(perms)} competition permutations")

    best: dict[int, list[int]] = {}
    total_merged = _load_csv_solutions(sol_dir, perms, best)
    total_merged += _load_json_solutions(sol_dir, perms, best)

    print(f"\nTotal solutions: {len(best)}/{len(perms)}")

    # Fail fast if any permutation is missing a solution
    missing = sorted(idx for idx in perms if idx not in best)
    if missing:
        print(f"FATAL: {len(missing)} permutations have no solution: {missing[:10]}...")
        sys.exit(1)

    # Write output
    total_score = 0
    per_n_totals: dict[int, tuple[int, int]] = {}

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "permutation", "solution"])

        for idx in sorted(perms.keys()):
            _, perm_list = perms[idx]
            perm_str = ",".join(str(x) for x in perm_list)
            sol_str = ".".join(f"R{k}" for k in best[idx])
            sol_len = len(best[idx])
            writer.writerow([idx, perm_str, sol_str])
            total_score += sol_len

            n = perms[idx][0]
            count, total = per_n_totals.get(n, (0, 0))
            per_n_totals[n] = (count + 1, total + sol_len)

    print(f"\nMerged {total_merged} new improvements")
    print(f"Total score: {total_score}")
    print("\nPer-n breakdown:")
    for n in sorted(per_n_totals):
        count, total = per_n_totals[n]
        avg = total / count if count else 0
        print(f"  n={n:3d}: {count:3d} perms, total={total:6d}, avg={avg:.1f}")

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
