"""Competition I/O for loading data and writing submissions."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from flipstack.core.permutation import is_sorted, validate_perm
from flipstack.core.types import SolverResult

EXPECTED_ROWS = 2405


def load_competition_data(path: Path | str) -> list[tuple[int, int, np.ndarray]]:
    """Load competition test.csv.

    Args:
        path: Path to test.csv file.

    Returns:
        List of (id, n, perm_array_int8) tuples.

    Raises:
        ValueError: If row count != 2405 or any permutation is invalid.
    """
    path = Path(path)
    rows: list[tuple[int, int, np.ndarray]] = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["id", "n", "permutation"]:
            msg = f"Unexpected header: {header}"
            raise ValueError(msg)
        for row in reader:
            row_id = int(row[0])
            n = int(row[1])
            perm_str = row[2]
            values = [int(x) for x in perm_str.split(",")]
            perm = np.array(values, dtype=np.int8)
            validate_perm(perm, expected_n=n)
            rows.append((row_id, n, perm))

    if len(rows) != EXPECTED_ROWS:
        msg = f"Expected {EXPECTED_ROWS} rows, got {len(rows)}"
        raise ValueError(msg)
    return rows


def write_submission(
    results: list[SolverResult],
    input_path: Path | str,
    output_path: Path | str,
) -> None:
    """Write submission CSV with atomic rename.

    Args:
        results: Solver results for all permutations.
        input_path: Path to original test.csv (for ID/perm validation).
        output_path: Path to write submission CSV.

    Raises:
        ValueError: If results don't cover all IDs or any solution is invalid.
    """
    input_data = load_competition_data(input_path)
    input_by_id = {row_id: (n, perm) for row_id, n, perm in input_data}

    if len(results) != EXPECTED_ROWS:
        msg = f"Expected {EXPECTED_ROWS} results, got {len(results)}"
        raise ValueError(msg)

    result_ids = {r.perm_id for r in results}
    expected_ids = set(input_by_id.keys())
    if result_ids != expected_ids:
        missing = expected_ids - result_ids
        msg = f"Missing IDs: {missing}"
        raise ValueError(msg)

    # Verify each solution
    for r in results:
        _verify_solution(r, input_by_id[r.perm_id][1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp, then rename
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        suffix=".csv",
        delete=False,
        newline="",
    ) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.writer(tmp)
        writer.writerow(["id", "permutation", "solution"])
        results_sorted = sorted(results, key=lambda r: r.perm_id)
        for r in results_sorted:
            _n, auth_perm = input_by_id[r.perm_id]
            perm_str = ",".join(str(x) for x in auth_perm)
            writer.writerow([r.perm_id, perm_str, r.solution_string()])

    tmp_path.rename(output_path)


def _verify_solution(result: SolverResult, original_perm: np.ndarray) -> None:
    """Verify a solution actually sorts the permutation.

    Args:
        result: Solver result to verify.
        original_perm: The original permutation.

    Raises:
        ValueError: If the solution doesn't sort the permutation.
    """
    n = len(original_perm)
    perm = original_perm.copy()
    for k in result.flips:
        if k < 2 or k > n:
            msg = f"Invalid flip size k={k} for n={n} in ID {result.perm_id}"
            raise ValueError(msg)
        perm[:k] = perm[k - 1 :: -1] if k == n else perm[:k][::-1]
    if not is_sorted(perm):
        msg = f"Solution for ID {result.perm_id} does not sort the permutation"
        raise ValueError(msg)


def load_submission(path: Path | str) -> dict[int, list[int]]:
    """Load a submission CSV into {id: flip_sequence}.

    Args:
        path: Path to submission CSV.

    Returns:
        Dict mapping perm_id to list of flip sizes.
    """
    path = Path(path)
    results: dict[int, list[int]] = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            row_id = int(row[0])
            solution_str = row[2]
            if not solution_str:
                results[row_id] = []
            else:
                results[row_id] = [int(s[1:]) for s in solution_str.split(".")]
    return results
