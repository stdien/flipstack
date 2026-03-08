"""Structured JSON experiment logging."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_log(
    run_name: str,
    config: dict[str, Any],
    results: dict[str, Any],
    system: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a structured experiment log.

    Args:
        run_name: Short name for the run.
        config: Full configuration dict.
        results: Results dict with total_score, per_n, wall_time_seconds.
        system: Optional system info (device, memory).

    Returns:
        Complete log dict ready for JSON serialization.
    """
    import uuid

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    short_id = uuid.uuid4().hex[:6]
    run_id = f"{timestamp}_{run_name}_{short_id}"
    log: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": _sanitize_config(config),
        "results": results,
    }
    if system:
        log["system"] = system
    return log


def save_log(log: dict[str, Any], log_dir: Path | str = Path("experiments/logs")) -> Path:
    """Save experiment log to JSON file.

    Args:
        log: Log dict from create_log.
        log_dir: Directory to save in.

    Returns:
        Path to saved file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = log["run_id"]
    # Sanitize filename
    safe_name = run_id.replace(":", "-").replace("/", "-")
    path = log_dir / f"{safe_name}.json"
    with path.open("w") as f:
        json.dump(log, f, indent=2, default=str)
    return path


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Strip absolute paths and sensitive values from config.

    Args:
        config: Raw config dict.

    Returns:
        Sanitized copy.
    """
    sanitized: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_config(value)
        elif isinstance(value, Path | str):
            s = str(value)
            # Strip home directory prefix
            if "/Users/" in s or "/home/" in s:
                parts = Path(s).parts
                # Find the project-relative part
                for i, part in enumerate(parts):
                    if part in ("flipstack", "data", "experiments", "config"):
                        sanitized[key] = str(Path(*parts[i:]))
                        break
                else:
                    sanitized[key] = Path(s).name
            else:
                sanitized[key] = s
        else:
            sanitized[key] = value
    return sanitized
