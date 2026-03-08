"""Tests for config schema and loading."""

from __future__ import annotations

from pathlib import Path

from flipstack.config.schema import SolverConfig, load_config


class TestConfig:
    def test_defaults(self):
        cfg = SolverConfig()
        assert cfg.beam.beam_width == 1024
        assert cfg.gpu_beam.device == "cuda"
        assert cfg.training.seed == 42

    def test_frozen(self):
        cfg = SolverConfig()
        try:
            cfg.n_min = 10  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_load_toml(self):
        path = Path(__file__).parent.parent / "config" / "default.toml"
        cfg = load_config(path)
        assert cfg.beam.beam_width == 1024
        assert cfg.data_path == "data/test.csv"
