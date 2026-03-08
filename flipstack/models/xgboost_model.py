"""XGBoost value model for pancake sorting distance prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from flipstack.training.data_gen import generate_features

logger = logging.getLogger(__name__)


class XGBoostValueModel:
    """XGBoost-based value predictor.

    Trains on (permutation features, walk_distance) pairs to predict
    how far a permutation is from being sorted.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            device: Device to use ('cpu' or 'cuda').
        """
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "device": device,
            "tree_method": "hist",
        }
        self.model: xgb.XGBRegressor | None = None  # type: ignore[possibly-missing-attribute]
        self._n: int | None = None

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the model.

        Args:
            x: Permutations array of shape (samples, n), dtype int8.
            y: Walk distances of shape (samples,), dtype float32.
        """
        self._n = x.shape[1]
        features = generate_features(x)
        self.model = xgb.XGBRegressor(**self.params)  # type: ignore[possibly-missing-attribute]
        self.model.fit(features, y)
        logger.info("XGBoost trained: %d samples, n=%d", len(x), self._n)

    def predict(self, perms: np.ndarray) -> np.ndarray:
        """Predict distance estimates.

        Args:
            perms: 2D array of shape (batch, n), dtype int8.

        Returns:
            1D float array of predicted distances.
        """
        if self.model is None:
            msg = "Model not trained"
            raise RuntimeError(msg)
        features = generate_features(perms)
        return self.model.predict(features).astype(np.float32)

    def predict_single(self, perm: np.ndarray) -> float:
        """Predict distance for a single permutation.

        Args:
            perm: 1D permutation array.

        Returns:
            Predicted distance.
        """
        return float(self.predict(perm.reshape(1, -1))[0])

    def save(self, path: str) -> None:
        """Save model to JSON file.

        Args:
            path: Output path.
        """
        if self.model is None:
            msg = "No model to save"
            raise RuntimeError(msg)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from JSON file.

        Args:
            path: Path to saved model.
        """
        self.model = xgb.XGBRegressor()  # type: ignore[possibly-missing-attribute]
        self.model.load_model(path)
