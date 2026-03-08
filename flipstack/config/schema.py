"""Frozen config dataclasses, loadable from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BeamConfig:
    """Beam search configuration."""

    beam_width: int = 1024
    max_steps: int = 200
    max_moves: int = 3
    use_filter: bool = True
    timeout: float = 300.0


@dataclass(frozen=True)
class GpuBeamConfig:
    """GPU beam search configuration."""

    beam_width: int = 4096
    max_steps: int = 200
    max_moves: int = 0
    timeout: float = 300.0
    device: str = "cuda"


@dataclass(frozen=True)
class TrainingConfig:
    """Model training configuration."""

    num_samples: int = 100_000
    epochs: int = 1000
    batch_size: int = 1024
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class SolverConfig:
    """Top-level solver configuration."""

    data_path: str = "data/test.csv"
    output_dir: str = "experiments/solutions"
    beam: BeamConfig = field(default_factory=BeamConfig)
    gpu_beam: GpuBeamConfig = field(default_factory=GpuBeamConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    n_min: int = 5
    n_max: int = 100
    model_dir: str = "experiments/checkpoints"


def load_config(path: str | Path) -> SolverConfig:
    """Load config from TOML file.

    Args:
        path: Path to TOML config file.

    Returns:
        Parsed SolverConfig.
    """
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> SolverConfig:
    """Parse raw TOML dict into SolverConfig.

    Args:
        raw: Raw dict from TOML parser.

    Returns:
        SolverConfig with nested configs.
    """
    beam = BeamConfig(**raw.get("beam", {}))
    gpu_beam = GpuBeamConfig(**raw.get("gpu_beam", {}))
    training = TrainingConfig(**raw.get("training", {}))

    section_keys = {"beam", "gpu_beam", "training"}
    top_keys = {"data_path", "output_dir", "n_min", "n_max", "model_dir"}
    unknown = set(raw) - section_keys - top_keys
    if unknown:
        msg = f"Unknown config keys: {sorted(unknown)}"
        raise ValueError(msg)

    top = {k: v for k, v in raw.items() if k in top_keys}

    return SolverConfig(beam=beam, gpu_beam=gpu_beam, training=training, **top)
