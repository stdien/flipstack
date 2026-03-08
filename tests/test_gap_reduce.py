"""Tests for gap-reduce solver variants."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.heuristics.gap import gap_h
from flipstack.search.gap_reduce import (
    _flip_gap_delta,
    gap_reduce_fme,
    gap_reduce_inverse,
    gap_reduce_solve,
    shorten_solution,
)


def _verify(perm: np.ndarray, flips: list[int]) -> bool:
    p = perm.copy()
    for k in flips:
        p = apply_flip(p, k)
    return is_sorted(p)


class TestGapReduce:
    def test_identity(self):
        perm = np.arange(5, dtype=np.int8)
        assert gap_reduce_solve(perm) == []

    def test_simple(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = gap_reduce_solve(perm, max_restarts=50, timeout=5.0)
        assert result is not None
        assert _verify(perm, result)

    def test_hard_perm(self):
        perm = np.array([3, 0, 4, 1, 2], dtype=np.int8)
        result = gap_reduce_solve(perm, max_restarts=100, timeout=5.0)
        assert result is not None
        assert _verify(perm, result)


class TestGapReduceInverse:
    def test_identity(self):
        perm = np.arange(5, dtype=np.int8)
        assert gap_reduce_inverse(perm) == []

    def test_simple(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = gap_reduce_inverse(perm, max_restarts=50, timeout=5.0)
        assert result is not None
        assert _verify(perm, result)

    def test_produces_valid_solution(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            perm = rng.permutation(8).astype(np.int8)
            result = gap_reduce_inverse(perm, max_restarts=100, timeout=5.0)
            if result is not None:
                assert _verify(perm, result), f"Invalid solution for {perm}"


class TestGapReduceFME:
    def test_identity(self):
        perm = np.arange(5, dtype=np.int8)
        assert gap_reduce_fme(perm) == []

    def test_simple(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = gap_reduce_fme(perm, max_restarts=50, seeds=2, timeout=5.0)
        assert result is not None
        assert _verify(perm, result)

    def test_finds_one_flip_solution(self):
        # [1, 0, 2, 3, 4] is solved by flip(2)
        perm = np.array([1, 0, 2, 3, 4], dtype=np.int8)
        result = gap_reduce_fme(perm, timeout=2.0)
        assert result is not None
        assert _verify(perm, result)
        assert len(result) == 1

    @pytest.mark.parametrize("seed", range(5))
    def test_random_perm_n10(self, seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(10).astype(np.int8)
        result = gap_reduce_fme(perm, max_restarts=100, seeds=2, timeout=10.0)
        assert result is not None
        assert _verify(perm, result)


class TestShortenSolution:
    def test_shortens_or_same(self):
        rng = np.random.default_rng(99)
        for _ in range(5):
            perm = rng.permutation(10).astype(np.int8)
            sol = gap_reduce_solve(perm, max_restarts=50, timeout=2.0)
            assert sol is not None
            shortened = shorten_solution(perm, sol, max_restarts=100, timeout=2.0)
            assert len(shortened) <= len(sol)
            assert _verify(perm, shortened)

    def test_identity_solution(self):
        perm = np.arange(5, dtype=np.int8)
        assert shorten_solution(perm, []) == []


class TestFlipGapDelta:
    @pytest.mark.parametrize("n", [5, 6, 7])
    def test_matches_full_gap_h(self, n):
        """Verify O(1) gap delta matches full gap_h for all S_n states and flips."""
        for state_tuple in permutations(range(n)):
            perm = np.array(state_tuple, dtype=np.int8)
            base_gap = gap_h(perm)
            for k in range(2, n + 1):
                delta = _flip_gap_delta(perm, k, n)
                child = apply_flip(perm, k)
                child_gap = gap_h(child)
                assert delta == child_gap - base_gap, (
                    f"perm={state_tuple} k={k}: delta={delta} but gap_h diff={child_gap - base_gap}"
                )
