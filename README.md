# Flipstack

Pancake sorting solver — finding shortest flip sequences on pancake graphs.

## What is this?

Given a permutation of `[0, 1, ..., n-1]`, find the shortest sequence of **prefix reversals** (flips) that sorts it. This is the [pancake sorting problem](https://en.wikipedia.org/wiki/Pancake_sorting) — equivalent to finding shortest paths in Cayley graphs of symmetric groups.

## Approach

- **Gap heuristic** with O(1) incremental updates
- **Filtered beam search** — only gap-reducing + structural moves (adjacent-value, max breakpoint, full reverse)
- **Multi-pass solving** — 3-step lookahead → beam search → suffix re-solving → segment replacement
- **First-move enumeration (FME)** — try all/best first moves, beam search each child
- **Numba JIT** for 10–100x speedup on inner loops
- **Inverse solving** — solve `inv(perm)`, reverse the flip sequence
- **Gap-reduce with restarts** — massive random restarts with deadlock recovery

## Tech Stack

Python 3.14 · NumPy · Numba · PyTorch · XGBoost · Typer + Rich CLI

## Setup

```bash
uv sync
```

## Usage

```bash
# Solve competition permutations
uv run python -m flipstack.cli solve

# Evaluate a solution file
uv run python -m flipstack.cli evaluate --solution path/to/solution.csv

# Merge multiple solutions (keep best per permutation)
uv run python -m flipstack.cli merge
```

## Scripts

Utilities and standalone solvers in `scripts/`:

| Script | Description |
|--------|-------------|
| `merge_solutions.py` | Merge multiple solution files, keep best per permutation |
| `shorten_solutions.py` | Shorten existing solutions via suffix re-solving |
| `gap_only_dfs.cpp` | C++ DFS solver using gap-reducing moves only (OpenMP) |
| `gap_slack_dfs.cpp` | C++ DFS solver with configurable slack for non-reducing moves (OpenMP) |

## Project Structure

```
flipstack/
  core/          # types, permutation ops, I/O
  heuristics/    # gap, singleton, lock detection, composite
  search/        # beam search, bidirectional, GPU beam, gap-reduce, move filtering
  models/        # XGBoost, ResNet-MLP, move prediction
  training/      # random walk data generation, training loops
  evaluate/      # scoring, comparison, merge
  solver/        # portfolio strategy, orchestrator
  tracking/      # JSON + W&B logging
  cli.py         # Typer CLI
  _perf.py       # gc control, timing, memory utils
scripts/         # standalone solver scripts
config/          # TOML configuration
tests/           # pytest suite
```

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ty check
```
