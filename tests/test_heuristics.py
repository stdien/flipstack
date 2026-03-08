"""Tests for lock detection, singleton, and composite heuristics."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from flipstack.heuristics.composite import composite_h
from flipstack.heuristics.gap import gap_h
from flipstack.heuristics.lock_detect import ld_h
from flipstack.heuristics.singleton import count_singletons
from tests.test_gap import get_bfs_distances


class TestLockDetect:
    def test_identity(self):
        assert ld_h(np.arange(5, dtype=np.int8)) == 0

    def test_ld_geq_gap(self):
        for state in permutations(range(5)):
            perm = np.array(state, dtype=np.int8)
            assert ld_h(perm) >= gap_h(perm)

    @pytest.mark.parametrize("n", [5, 6, 7])
    def test_admissibility_bfs(self, n):
        distances = get_bfs_distances(n)
        violations = 0
        for state_tuple, true_dist in distances.items():
            perm = np.array(state_tuple, dtype=np.int8)
            h = ld_h(perm)
            if h > true_dist:
                violations += 1
        assert violations == 0, f"Found {violations} ld_h admissibility violations in S_{n}"


class TestSingleton:
    def test_identity_no_singletons(self):
        assert count_singletons(np.arange(5, dtype=np.int8)) == 0

    def test_known_singletons(self):
        # [2, 0, 4, 1, 3] with sentinel [2,0,4,1,3,5]
        # diffs: |2-0|=2, |0-4|=4, |4-1|=3, |1-3|=2, |3-5|=2
        # gap_right: [T, T, T, T, T]
        # gap_left: [F, T, T, T, T]
        # singletons (both L+R): indices 1,2,3,4 -> 4
        perm = np.array([2, 0, 4, 1, 3], dtype=np.int8)
        assert count_singletons(perm) == 4


class TestComposite:
    def test_composite_leq_ld(self):
        # Composite subtracts singleton bonus, so should be <= ld_h
        for state in permutations(range(5)):
            perm = np.array(state, dtype=np.int8)
            assert composite_h(perm) <= ld_h(perm) + 1e-9
