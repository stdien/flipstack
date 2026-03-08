"""Training loop for XGBoost models with data regeneration."""

from __future__ import annotations

import logging
from pathlib import Path

from flipstack.models.xgboost_model import XGBoostValueModel
from flipstack.training.data_gen import generate_random_walks

logger = logging.getLogger(__name__)


def train_xgboost_per_n(
    n_values: list[int],
    samples_per_n: int = 50000,
    checkpoint_dir: str = "experiments/checkpoints",
    device: str = "cpu",
    seed: int = 42,
) -> dict[int, XGBoostValueModel]:
    """Train one XGBoost model per permutation size.

    Args:
        n_values: List of permutation sizes to train for.
        samples_per_n: Training samples per model.
        checkpoint_dir: Directory to save model checkpoints.
        device: Device for XGBoost ('cpu' or 'cuda').
        seed: Random seed.

    Returns:
        Dict mapping n -> trained model.
    """
    models: dict[int, XGBoostValueModel] = {}
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for n in n_values:
        logger.info("Training XGBoost for n=%d with %d samples", n, samples_per_n)

        x, y = generate_random_walks(n, num_samples=samples_per_n, seed=seed + n)

        model = XGBoostValueModel(device=device)
        model.train(x, y)

        path = str(ckpt_dir / f"xgb_n{n}.json")
        model.save(path)
        logger.info("Saved model to %s", path)

        models[n] = model

    return models
