"""Typer CLI for flipstack pancake sorting solver."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from flipstack.core.io import load_competition_data, write_submission
from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.core.types import SolverResult
from flipstack.search.beam import beam_search

app = typer.Typer(help="Flipstack: Pancake sorting solver")
console = Console()

DEFAULT_DATA = Path("data/test.csv")


def _default_output() -> Path:
    import uuid
    from datetime import UTC, datetime

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    return Path(f"experiments/solutions/{ts}_{short_id}.csv")


@app.command()
def solve(
    data_path: Annotated[Path, typer.Option(help="Path to test.csv")] = DEFAULT_DATA,
    output: Annotated[Path | None, typer.Option(help="Output submission CSV")] = None,
    beam_width: Annotated[int, typer.Option(help="Beam search width")] = 1024,
    max_moves: Annotated[int, typer.Option(help="Move filter count")] = 3,
    max_steps: Annotated[int, typer.Option(help="Max steps per permutation")] = 200,
    timeout: Annotated[float, typer.Option(help="Timeout per permutation (seconds)")] = 300.0,
    n_min: Annotated[int, typer.Option(help="Minimum n to solve")] = 5,
    n_max: Annotated[int, typer.Option(help="Maximum n to solve")] = 100,
) -> None:
    """Solve all competition permutations using beam search."""
    if output is None:
        output = _default_output()
    data = load_competition_data(data_path)
    console.print(f"Loaded {len(data)} permutations from {data_path}")

    results: list[SolverResult] = []
    total_flips = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Solving", total=len(data))
        for row_id, n, perm in data:
            progress.update(task, description=f"n={n} id={row_id}")

            if is_sorted(perm):
                flips: list[int] = []
            elif n_min <= n <= n_max:
                result = beam_search(
                    perm,
                    beam_width=beam_width,
                    max_steps=max_steps,
                    max_moves=max_moves,
                    timeout=timeout,
                )
                if result is not None:
                    flips = result
                else:
                    flips = _greedy_solve(perm) or []
                    if not flips and not is_sorted(perm):
                        console.print(f"[red]FAILED: id={row_id} n={n}[/red]")
                        failed += 1
            else:
                flips = _greedy_solve(perm) or []

            total_flips += len(flips)
            results.append(SolverResult(perm_id=row_id, flips=flips, original_perm=perm))
            progress.advance(task)

    console.print(f"\nTotal flips: [bold]{total_flips}[/bold], Failed: {failed}")

    write_submission(results, data_path, output)
    console.print(f"Submission written to {output}")


@app.command()
def evaluate(
    submission: Annotated[Path, typer.Argument(help="Path to submission CSV")],
    data_path: Annotated[Path, typer.Option(help="Path to test.csv")] = DEFAULT_DATA,
) -> None:
    """Score a submission CSV."""
    from flipstack.core.io import load_submission

    data = load_competition_data(data_path)
    solutions = load_submission(submission)

    total = 0
    per_n: dict[int, list[int]] = {}
    invalid = 0

    for row_id, n, perm in data:
        flips = solutions.get(row_id, [])
        p = perm.copy()
        valid = True
        for k in flips:
            if k < 2 or k > n:
                valid = False
                break
            p = apply_flip(p, k)
        if not valid or not is_sorted(p):
            invalid += 1
            continue

        length = len(flips)
        total += length
        per_n.setdefault(n, []).append(length)

    if invalid:
        console.print(f"[red]{invalid} invalid/missing solutions — score is incomplete[/red]")

    table = Table(title=f"Score: {total} ({'INCOMPLETE' if invalid else 'valid'})")
    table.add_column("n", justify="right")
    table.add_column("count", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("total", justify="right")

    for n in sorted(per_n):
        lengths = per_n[n]
        table.add_row(
            str(n),
            str(len(lengths)),
            str(min(lengths)),
            str(max(lengths)),
            f"{np.mean(lengths):.1f}",
            str(sum(lengths)),
        )

    console.print(table)


@app.command()
def merge(
    files: Annotated[list[Path], typer.Argument(help="Submission CSVs to merge")],
    output: Annotated[Path | None, typer.Option(help="Output path")] = None,
    data_path: Annotated[Path, typer.Option(help="Path to test.csv")] = DEFAULT_DATA,
) -> None:
    """Merge multiple submissions keeping shortest solution per permutation."""
    if output is None:
        output = _default_output()

    from flipstack.core.io import load_submission

    data = load_competition_data(data_path)
    best: dict[int, list[int]] = {}

    for fpath in files:
        solutions = load_submission(fpath)
        for row_id, flips in solutions.items():
            if row_id not in best or len(flips) < len(best[row_id]):
                best[row_id] = flips

    results = []
    for row_id, _n, perm in data:
        flips = best.get(row_id, [])
        results.append(SolverResult(perm_id=row_id, flips=flips, original_perm=perm))

    write_submission(results, data_path, output)
    total = sum(len(r.flips) for r in results)
    console.print(f"Merged {len(files)} files -> {output}, total: {total}")


@app.command()
def portfolio(
    data_path: Annotated[Path, typer.Option(help="Path to test.csv")] = DEFAULT_DATA,
    output: Annotated[Path | None, typer.Option(help="Output submission CSV")] = None,
    timeout: Annotated[float, typer.Option(help="Timeout per permutation (seconds)")] = 30.0,
    model_dir: Annotated[Path | None, typer.Option(help="XGBoost model directory")] = None,
) -> None:
    """Solve using multi-strategy portfolio (beam + bidir + gap-reduce + inverse)."""
    if output is None:
        output = _default_output()
    data = load_competition_data(data_path)
    console.print(f"Loaded {len(data)} permutations from {data_path}")

    # Load XGBoost models per n if available
    from collections.abc import Callable

    scorers: dict[int, Callable[[np.ndarray], float]] = {}
    if model_dir is not None and model_dir.exists():
        import re

        from flipstack.models.predictor import make_scorer
        from flipstack.models.xgboost_model import XGBoostValueModel

        for mf in model_dir.glob("xgb_n*.json"):
            m = re.search(r"xgb_n(\d+)\.json$", mf.name)
            if m:
                model_n = int(m.group(1))
                model = XGBoostValueModel()
                model.load(str(mf))
                scorers[model_n] = make_scorer(model, alpha=0.3)
        if scorers:
            console.print(f"Loaded {len(scorers)} XGBoost models from {model_dir}")

    from flipstack.solver.merger import multi_strategy_solve

    results: list[SolverResult] = []
    total_flips = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Portfolio solving", total=len(data))
        for row_id, n, perm in data:
            progress.update(task, description=f"n={n} id={row_id}")

            # Use n-matched scorer or None
            scorer = scorers.get(n)

            sr = multi_strategy_solve(
                perm_id=row_id,
                n=n,
                perm=perm,
                timeout=timeout,
                scorer=scorer,
            )
            if not sr.flips and not is_sorted(perm):
                failed += 1

            total_flips += len(sr.flips)
            results.append(sr)
            progress.advance(task)

    console.print(f"\nTotal flips: [bold]{total_flips}[/bold], Failed: {failed}")

    write_submission(results, data_path, output)
    console.print(f"Submission written to {output}")


def _greedy_solve(perm: np.ndarray) -> list[int] | None:
    """Simple greedy: place largest element first, then next, etc.

    Args:
        perm: Permutation to solve.

    Returns:
        Flip sequence, or None on failure.
    """
    p = perm.copy()
    n = len(p)
    flips: list[int] = []
    for target_pos in range(n - 1, 0, -1):
        target_val = target_pos
        if p[target_pos] == target_val:
            continue
        pos = int(np.argwhere(p == target_val).item())
        if pos != 0:
            flips.append(pos + 1)
            p[: pos + 1] = p[pos::-1]
        flips.append(target_pos + 1)
        p[: target_pos + 1] = p[target_pos::-1]
    if is_sorted(p):
        return flips
    return None


if __name__ == "__main__":
    app()
