"""Shared utilities for solver scripts."""
from __future__ import annotations

import csv
from pathlib import Path


def load_existing_lengths(merged_path: Path) -> dict[int, int]:
    """Load existing solution lengths from merged CSV.

    Handles both formats:
    - 'id,permutation,solution' with R-prefixed moves (e.g., R4.R2)
    - 'Id,Moves' with plain integers (e.g., 4.2)

    Args:
        merged_path: Path to merged_best.csv or similar.

    Returns:
        Dict mapping perm id to solution length.
    """
    existing: dict[int, int] = {}
    if not merged_path.exists():
        return existing

    with merged_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        id_key = "Id" if "Id" in headers else "id"
        moves_key = "Moves" if "Moves" in headers else "solution"

        for row in reader:
            idx = int(row[id_key])
            sol_str = row.get(moves_key, "").strip()
            if sol_str:
                existing[idx] = len(sol_str.split("."))
            else:
                existing[idx] = 9999

    return existing


def load_existing_solutions(merged_path: Path) -> dict[int, tuple[int, list[int]]]:
    """Load existing solutions (with flip sequences) from merged CSV.

    Args:
        merged_path: Path to merged_best.csv or similar.

    Returns:
        Dict mapping perm id to (length, flips) tuple.
    """
    existing: dict[int, tuple[int, list[int]]] = {}
    if not merged_path.exists():
        return existing

    with merged_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        id_key = "Id" if "Id" in headers else "id"
        moves_key = "Moves" if "Moves" in headers else "solution"

        for row in reader:
            idx = int(row[id_key])
            sol_str = row.get(moves_key, "").strip()
            if sol_str:
                moves = [int(tok.lstrip("R")) for tok in sol_str.split(".")]
                existing[idx] = (len(moves), moves)

    return existing
