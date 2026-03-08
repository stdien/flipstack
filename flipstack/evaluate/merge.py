"""Merge multiple submission files, keeping shortest valid solution per permutation."""

from __future__ import annotations

import logging
from pathlib import Path

from flipstack.core.io import load_competition_data, load_submission, write_submission
from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.core.types import SolverResult

logger = logging.getLogger(__name__)


def merge_submissions(
    paths: list[str | Path],
    data_path: str | Path = "data/test.csv",
    output_path: str | Path | None = None,
) -> dict[int, list[int]]:
    """Merge submissions keeping shortest valid solution per permutation.

    Re-verifies each solution before accepting it.

    Args:
        paths: Paths to submission CSVs.
        data_path: Path to competition test.csv.
        output_path: If provided, write merged submission here.

    Returns:
        Dict mapping row_id -> best flip sequence.
    """
    data = load_competition_data(str(data_path))
    perm_by_id = {row_id: (n, perm) for row_id, n, perm in data}

    best: dict[int, list[int]] = {}

    for path in paths:
        solutions = load_submission(str(path))
        accepted = 0
        for row_id, flips in solutions.items():
            if row_id not in perm_by_id:
                continue

            n, perm = perm_by_id[row_id]

            # Verify solution
            p = perm.copy()
            valid = True
            for k in flips:
                if k < 2 or k > n:
                    valid = False
                    break
                p = apply_flip(p, k)
            if not valid or not is_sorted(p):
                continue

            if row_id not in best or len(flips) < len(best[row_id]):
                best[row_id] = flips
                accepted += 1

        logger.info("Merged %s: %d solutions accepted", path, accepted)

    if output_path is not None:
        results = []
        for row_id, _n, perm in data:
            flips = best.get(row_id, [])
            results.append(SolverResult(perm_id=row_id, flips=flips, original_perm=perm))
        write_submission(results, str(data_path), str(output_path))

    total = sum(len(f) for f in best.values())
    logger.info("Merge total: %d flips across %d permutations", total, len(best))
    return best
