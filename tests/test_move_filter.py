"""Tests for move filtering."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from flipstack.search.move_filter import all_moves, filter_moves


class TestFilterMoves:
    def test_output_valid_range(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        moves = filter_moves(perm)
        for k in moves:
            assert 2 <= k <= len(perm)

    def test_no_duplicates(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        moves = filter_moves(perm)
        assert len(moves) == len(set(moves))

    def test_non_empty(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        moves = filter_moves(perm)
        assert len(moves) >= 1

    def test_subset_of_all(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        moves = filter_moves(perm)
        valid = set(all_moves(len(perm)))
        assert set(moves).issubset(valid)

    def test_identity_moves(self):
        perm = np.arange(5, dtype=np.int8)
        moves = filter_moves(perm)
        assert len(moves) >= 1

    def test_all_s5_valid(self):
        for state in permutations(range(5)):
            perm = np.array(state, dtype=np.int8)
            moves = filter_moves(perm)
            assert len(moves) >= 1
            assert len(moves) == len(set(moves))
            for k in moves:
                assert 2 <= k <= 5

    def test_max_moves_respected(self):
        perm = np.array([4, 0, 3, 1, 2], dtype=np.int8)
        for cap in (1, 2, 3, 4):
            moves = filter_moves(perm, max_moves=cap)
            assert len(moves) <= cap

    def test_includes_max_flip(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        moves = filter_moves(perm)
        assert len(perm) in moves


class TestAllMoves:
    def test_range(self):
        assert all_moves(5) == [2, 3, 4, 5]
        assert all_moves(2) == [2]
