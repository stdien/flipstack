"""Tests for predictor and scoring functions."""

from __future__ import annotations

import numpy as np

from flipstack.heuristics.composite import composite_h
from flipstack.models.predictor import make_batch_scorer, make_scorer


class TestMakeScorer:
    def test_default_is_composite(self):
        scorer = make_scorer()
        perm = np.array([3, 1, 4, 0, 2], dtype=np.int8)
        assert scorer(perm) == composite_h(perm)

    def test_returns_float(self):
        scorer = make_scorer()
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = scorer(perm)
        assert isinstance(result, float)


class TestMakeBatchScorer:
    def test_none_without_model(self):
        result = make_batch_scorer()
        assert result is None
