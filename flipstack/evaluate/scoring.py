"""Scoring utilities for evaluating solutions."""

from __future__ import annotations

from typing import Any

import numpy as np

from flipstack.core.io import load_competition_data, load_submission
from flipstack.core.permutation import apply_flip, is_sorted


def score_submission(
    submission_path: str,
    data_path: str = "data/test.csv",
) -> dict[str, Any]:
    """Score a submission file and return detailed breakdown.

    Args:
        submission_path: Path to submission CSV.
        data_path: Path to competition test.csv.

    Returns:
        Dict with total_score, per_n breakdown, and validity info.
    """
    data = load_competition_data(data_path)
    solutions = load_submission(submission_path)

    total = 0
    per_n: dict[int, dict[str, Any]] = {}
    invalid_ids: list[int] = []

    for row_id, n, perm in data:
        flips = solutions.get(row_id, [])

        # Verify
        p = perm.copy()
        valid = True
        for k in flips:
            if k < 2 or k > n:
                valid = False
                break
            p = apply_flip(p, k)
        if not valid or not is_sorted(p):
            invalid_ids.append(row_id)
            continue

        length = len(flips)
        total += length

        if n not in per_n:
            per_n[n] = {"count": 0, "lengths": []}
        per_n[n]["count"] += 1
        per_n[n]["lengths"].append(length)

    # Compute stats per n
    per_n_stats: dict[int, dict[str, Any]] = {}
    for n, info in sorted(per_n.items()):
        lengths = info["lengths"]
        arr = np.array(lengths)
        per_n_stats[n] = {
            "count": info["count"],
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": round(float(arr.mean()), 1),
            "total": int(arr.sum()),
        }

    return {
        "total_score": total,
        "per_n": per_n_stats,
        "invalid_count": len(invalid_ids),
        "invalid_ids": invalid_ids,
    }
