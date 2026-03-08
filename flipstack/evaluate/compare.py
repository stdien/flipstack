"""Side-by-side comparison of experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def compare_runs(log_paths: list[str | Path]) -> None:
    """Print side-by-side comparison of experiment logs.

    Args:
        log_paths: Paths to JSON experiment log files.
    """
    console = Console()
    logs: list[dict[str, Any]] = []
    for p in log_paths:
        with Path(p).open() as f:
            logs.append(json.load(f))

    table = Table(title="Run Comparison")
    table.add_column("Metric", style="bold")
    for log in logs:
        table.add_column(log.get("run_id", "unknown")[:30])

    # Total score
    table.add_row("Total Score", *[str(log["results"].get("total_score", "?")) for log in logs])
    table.add_row("Wall Time (s)", *[str(log["results"].get("wall_time_seconds", "?")) for log in logs])

    # Find all n values across runs
    all_ns: set[int] = set()
    for log in logs:
        per_n = log["results"].get("per_n", {})
        all_ns.update(int(k) for k in per_n)

    for n in sorted(all_ns):
        row = [f"n={n} mean"]
        for log in logs:
            per_n = log["results"].get("per_n", {})
            n_data = per_n.get(str(n), {})
            row.append(str(n_data.get("mean", "?")))
        table.add_row(*row)

    console.print(table)
