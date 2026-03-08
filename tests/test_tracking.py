"""Tests for experiment tracking and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from flipstack.tracking.json_logger import _sanitize_config, create_log, save_log


class TestJsonLogger:
    def test_create_log_structure(self):
        log = create_log(
            run_name="test_run",
            config={"beam_width": 1024, "max_moves": 3},
            results={"total_score": 90000},
        )
        assert "run_id" in log
        assert "timestamp" in log
        assert log["config"]["beam_width"] == 1024
        assert log["results"]["total_score"] == 90000

    def test_sanitize_strips_home(self):
        config = {"path": "/home/user/flipstack/data/test.csv"}
        sanitized = _sanitize_config(config)
        assert "/home/" not in sanitized["path"]
        assert "data" in sanitized["path"]

    def test_sanitize_nested(self):
        config = {"outer": {"path": "/Users/me/flipstack/experiments/foo.csv"}}
        sanitized = _sanitize_config(config)
        assert "/Users/" not in str(sanitized)

    def test_save_log(self, tmp_path: Path):
        log = create_log(
            run_name="save_test",
            config={"x": 1},
            results={"score": 42},
        )
        path = save_log(log, log_dir=tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["results"]["score"] == 42


class TestWandbLogger:
    def test_graceful_no_run(self):
        """log_metrics and finish should not raise when no run is active."""
        from flipstack.tracking.wandb_logger import finish, log_metrics

        log_metrics({"x": 1})
        finish()
