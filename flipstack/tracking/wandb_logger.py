"""W&B experiment logging with graceful fallback."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_wandb_available = False
_wandb_error_cls: type = OSError  # placeholder
try:
    import wandb

    _wandb_available = True
    _wandb_error_cls = wandb.Error
except ImportError:
    wandb = None  # type: ignore[assignment]

_WANDB_ERRORS = (OSError, RuntimeError, ValueError, _wandb_error_cls)


def init_run(
    project: str = "flipstack",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> Any | None:
    """Initialize a W&B run.

    Args:
        project: W&B project name.
        run_name: Optional run name.
        config: Config dict to log.

    Returns:
        wandb.Run or None if wandb unavailable.
    """
    if not _wandb_available:
        logger.info("wandb not available, skipping")
        return None
    try:
        return wandb.init(
            project=project,
            name=run_name,
            config=config,
            settings=wandb.Settings(silent=True),
        )
    except _WANDB_ERRORS:
        logger.warning("Failed to init wandb run", exc_info=True)
        return None


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to active W&B run.

    Args:
        metrics: Dict of metric name -> value.
        step: Optional step number.
    """
    if not _wandb_available or wandb.run is None:
        return
    try:
        wandb.log(metrics, step=step)
    except _WANDB_ERRORS:
        logger.warning("Failed to log to wandb", exc_info=True)


def finish() -> None:
    """Finish the active W&B run."""
    if not _wandb_available or wandb.run is None:
        return
    try:
        wandb.finish()
    except _WANDB_ERRORS:
        logger.warning("Failed to finish wandb run", exc_info=True)
