"""Unified predictor wrapping models or heuristics into beam search scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flipstack.heuristics.composite import composite_h
from flipstack.heuristics.gap import gap_h, gap_h_batch

if TYPE_CHECKING:
    from collections.abc import Callable

    from flipstack.models.xgboost_model import XGBoostValueModel


def make_scorer(
    model: XGBoostValueModel | None = None,
    alpha: float = 0.5,
    use_gap: bool = True,
) -> Callable[[np.ndarray], float]:
    """Create a per-state scoring function for beam search.

    When both model and gap are used, score = alpha * model + (1-alpha) * gap_h.

    Args:
        model: Trained value model, or None for pure heuristic.
        alpha: Weight for model prediction (0 = pure heuristic, 1 = pure model).
        use_gap: Whether to include gap heuristic.

    Returns:
        Scoring function taking a single permutation, returning float.
    """
    if model is None:
        return composite_h

    def _scorer(perm: np.ndarray) -> float:
        model_score = model.predict_single(perm)
        if use_gap:
            gap_score = float(gap_h(perm))
            return alpha * model_score + (1 - alpha) * gap_score
        return model_score

    return _scorer


def make_batch_scorer(
    model: XGBoostValueModel | None = None,
    alpha: float = 0.5,
    use_gap: bool = True,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Create a batch scoring function for efficient beam search.

    Args:
        model: Trained value model, or None (returns None = use per-state scorer).
        alpha: Weight for model prediction.
        use_gap: Whether to include gap heuristic.

    Returns:
        Batch scoring function (2D int8 array -> 1D float scores), or None.
    """
    if model is None:
        return None

    def _batch_scorer(perms: np.ndarray) -> np.ndarray:
        model_scores = model.predict(perms)
        if use_gap:
            gap_scores = gap_h_batch(perms).astype(np.float32)
            return alpha * model_scores + (1 - alpha) * gap_scores
        return model_scores

    return _batch_scorer
